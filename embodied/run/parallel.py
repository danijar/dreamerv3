import collections
import threading
import time
from functools import partial as bind

import cloudpickle
import elements
import embodied
import numpy as np
import portal

prefix = lambda d, p: {f'{p}/{k}': v for k, v in d.items()}


def combined(
    make_agent,
    make_replay_train,
    make_replay_eval,
    make_env_train,
    make_env_eval,
    make_stream,
    make_logger,
    args):

  if args.actor_batch <= 0:
    args = args.update(actor_batch=max(1, args.envs // 2))
  assert args.actor_batch <= args.envs, (args.actor_batch, args.envs)
  for key in ('actor_addr', 'replay_addr', 'logger_addr'):
    if '{auto}' in args[key]:
      args = args.update({key: args[key].format(auto=portal.free_port())})

  make_agent = cloudpickle.dumps(make_agent)
  make_replay_train = cloudpickle.dumps(make_replay_train)
  make_replay_eval = cloudpickle.dumps(make_replay_eval)
  make_env_train = cloudpickle.dumps(make_env_train)
  make_env_eval = cloudpickle.dumps(make_env_eval)
  make_stream = cloudpickle.dumps(make_stream)
  make_logger = cloudpickle.dumps(make_logger)

  workers = []
  if args.agent_process:
    workers.append(portal.Process(parallel_agent, make_agent, args))
  else:
    workers.append(portal.Thread(parallel_agent, make_agent, args))
  workers.append(portal.Process(parallel_logger, make_logger, args))

  if not args.remote_envs:
    for i in range(args.envs):
      workers.append(portal.Process(parallel_env, make_env_train, i, args))
    for i in range(args.envs, args.envs + args.eval_envs):
      workers.append(portal.Process(
          parallel_env, make_env_eval, i, args, True))

  if not args.remote_replay:
    workers.append(portal.Process(
        parallel_replay, make_replay_train, make_replay_eval,
        make_stream, args))

  portal.run(workers)


def parallel_agent(make_agent, args):
  if isinstance(make_agent, bytes):
    make_agent = cloudpickle.loads(make_agent)
  agent = make_agent()
  barrier = threading.Barrier(2)
  workers = []
  workers.append(portal.Thread(parallel_actor, agent, barrier, args))
  workers.append(portal.Thread(parallel_learner, agent, barrier, args))
  portal.run(workers)


@elements.timer.section('actor')
def parallel_actor(agent, barrier, args):

  islist = lambda x: isinstance(x, list)
  initial = agent.init_policy(args.actor_batch)
  initial = elements.tree.map(lambda x: x[0], initial, isleaf=islist)
  carries = collections.defaultdict(lambda: initial)
  barrier.wait()  # Do not collect data before learner restored checkpoint.
  fps = elements.FPS()

  should_log = embodied.LocalClock(args.log_every)
  backlog = 8 * args.actor_threads
  logger = portal.Client(args.logger_addr, 'ActorLogger', maxinflight=backlog)
  replay = portal.Client(args.replay_addr, 'ActorReplay', maxinflight=backlog)

  @elements.timer.section('workfn')
  def workfn(obs):
    envid = obs.pop('envid')
    assert envid.shape == (args.actor_batch,)
    is_eval = obs.pop('is_eval')
    fps.step(obs['is_first'].size)
    with elements.timer.section('get_states'):
      carry = [carries[a] for a in envid]
      carry = elements.tree.map(lambda *xs: list(xs), *carry)
    logs = {k: v for k, v in obs.items() if k.startswith('log/')}
    obs = {k: v for k, v in obs.items() if not k.startswith('log/')}
    carry, acts, outs = agent.policy(carry, obs)
    assert all(k not in acts for k in outs), (
        list(outs.keys()), list(acts.keys()))
    with elements.timer.section('put_states'):
      for i, a in enumerate(envid):
        carries[a] = elements.tree.map(lambda x: x[i], carry, isleaf=islist)
    trans = {'envid': envid, 'is_eval': is_eval, **obs, **acts, **outs, **logs}
    [x.setflags(write=False) for x in trans.values()]
    acts = {**acts, 'reset': obs['is_last'].copy()}
    return acts, trans

  @elements.timer.section('donefn')
  def postfn(trans):
    logs = {k: v for k, v in trans.items() if k.startswith('log/')}
    trans = {k: v for k, v in trans.items() if not k.startswith('log/')}
    replay.add_batch(trans)
    logger.tran({**trans, **logs})
    if should_log():
      stats = {}
      stats['fps/policy'] = fps.result()
      stats['parallel/ep_states'] = len(carries)
      stats.update(prefix(server.stats(), 'server/actor'))
      stats.update(prefix(logger.stats(), 'client/actor_logger'))
      stats.update(prefix(replay.stats(), 'client/actor_replay'))
      logger.add(stats)

  server = portal.BatchServer(args.actor_addr, name='Actor')
  server.bind('act', workfn, postfn, args.actor_batch, args.actor_threads)
  server.start()


@elements.timer.section('learner')
def parallel_learner(agent, barrier, args):

  agg = elements.Agg()
  usage = elements.Usage(**args.usage)
  should_log = embodied.GlobalClock(args.log_every)
  should_report = embodied.GlobalClock(args.report_every)
  should_save = embodied.GlobalClock(args.save_every)
  fps = elements.FPS()
  batch_steps = args.batch_size * args.batch_length

  cp = elements.Checkpoint(elements.Path(args.logdir) / 'ckpt/agent')
  cp.agent = agent
  if args.from_checkpoint:
    elements.checkpoint.load(args.from_checkpoint, dict(
        agent=bind(agent.load, regex=args.from_checkpoint_regex)))
  cp.load_or_save()
  logger = portal.Client(args.logger_addr, 'LearnerLogger', maxinflight=1)
  updater = portal.Client(
      args.replay_addr, 'LearnerReplayUpdater', maxinflight=8)
  barrier.wait()

  replays = {}
  received = collections.defaultdict(int)
  def parallel_stream(source, prefetch=2):
    replay = portal.Client(args.replay_addr, f'LearnerReplay{source.title()}')
    replays[source] = replay
    call = getattr(replay, f'sample_batch_{source}')
    futures = collections.deque([call() for _ in range(prefetch)])
    while True:
      futures.append(call())
      with elements.timer.section(f'stream_{source}_response'):
        data = futures.popleft().result()
      received[source] += 1
      yield data

  def evaluate(stream):
    carry = agent.init_report(args.batch_size)
    agg = elements.Agg()
    for _ in range(args.consec_report * args.report_batches):
      batch = next(stream)
      carry, metrics = agent.report(carry, batch)
      agg.add(metrics)
    return agg.result()

  stream_train = iter(agent.stream(
      embodied.streams.Stateless(parallel_stream('train'))))
  stream_report = iter(agent.stream(
      embodied.streams.Stateless(parallel_stream('report'))))
  stream_eval = iter(agent.stream(
      embodied.streams.Stateless(parallel_stream('eval'))))
  carry = agent.init_train(args.batch_size)

  while True:

    with elements.timer.section('batch_next'):
      batch = next(stream_train)
    with elements.timer.section('train_step'):
      carry, outs, mets = agent.train(carry, batch)
    if 'replay' in outs:
      with elements.timer.section('replay_update'):
        updater.update(outs['replay'])

    time.sleep(0.0001)
    agg.add(mets)
    fps.step(batch_steps)

    if should_report(skip=not received['report']):
      print('Report started...')
      with elements.timer.section('report'):
        logger.add(prefix(evaluate(stream_report), 'report'))
        if args.eval_envs and received['eval']:
          logger.add(prefix(evaluate(stream_eval), 'eval'))
      print('Report finished!')

    if should_log():
      with elements.timer.section('metrics'):
        stats = {}
        stats['fps/train'] = fps.result()
        stats['timer/agent'] = elements.timer.stats()['summary']
        stats.update(prefix(agg.result(), 'train'))
        stats.update(prefix(usage.stats(), 'usage/agent'))
        stats.update(prefix(logger.stats(), 'client/learner_logger'))
        for source, client in replays.items():
          stats.update(prefix(client.stats(), f'client/replay_{source}'))
      logger.add(stats)

    if should_save():
      cp.save()


def parallel_replay(make_replay_train, make_replay_eval, make_stream, args):
  if isinstance(make_replay_train, bytes):
    make_replay_train = cloudpickle.loads(make_replay_train)
  if isinstance(make_replay_eval, bytes):
    make_replay_eval = cloudpickle.loads(make_replay_eval)
  if isinstance(make_stream, bytes):
    make_stream = cloudpickle.loads(make_stream)

  replay_train = make_replay_train()
  replay_eval = make_replay_eval()

  stream_train = iter(make_stream(replay_train, 'train'))
  stream_report = iter(make_stream(replay_train, 'report'))
  stream_eval = iter(make_stream(replay_eval, 'eval'))

  should_log = embodied.LocalClock(args.log_every)
  logger = portal.Client(args.logger_addr, 'ReplayLogger', maxinflight=1)
  usage = elements.Usage(**args.usage.update(nvsmi=False))
  limit_agg = elements.Agg()
  active = elements.Counter()

  limiter = embodied.limiters.SamplesPerInsert(
      args.train_ratio / args.batch_length,
      tolerance=4 * args.batch_size,
      minsize=args.batch_size * replay_train.length)

  def add_batch(data):
    active.increment()
    for i, envid in enumerate(data.pop('envid')):
      tran = {k: v[i] for k, v in data.items()}
      if tran.pop('is_eval', False):
        replay_eval.add(tran, envid)
        continue
      with elements.timer.section('replay_insert_wait'):
        dur = embodied.limiters.wait(
            limiter.want_insert, 'Replay insert waiting',
            limiter.__dict__)
        limit_agg.add('insert_wait_dur', dur, agg='sum')
        limit_agg.add('insert_wait_count', dur > 0, agg='sum')
        limit_agg.add('insert_wait_frac', dur > 0, agg='avg')
        limiter.insert()
        replay_train.add(tran, envid)
    return {}

  def sample_batch_train():
    active.increment()
    with elements.timer.section('replay_sample_wait'):
      for _ in range(args.batch_size):
        dur = embodied.limiters.wait(
            limiter.want_sample, 'Replay sample waiting',
            limiter.__dict__)
        limit_agg.add('sample_wait_dur', dur, agg='sum')
        limit_agg.add('sample_wait_count', dur > 0, agg='sum')
        limit_agg.add('sample_wait_frac', dur > 0, agg='avg')
        limiter.sample()
    return next(stream_train)

  def sample_batch_report():
    active.increment()
    return next(stream_report)

  def sample_batch_eval():
    active.increment()
    return next(stream_eval)

  should_save = embodied.LocalClock(args.save_every)
  cp = elements.Checkpoint(elements.Path(args.logdir) / 'ckpt/replay')
  cp.replay_train = replay_train
  cp.replay_eval = replay_eval
  cp.limiter = limiter
  cp.load_or_save()

  server = portal.Server(args.replay_addr, name='Replay')
  server.bind('add_batch', add_batch, workers=1)
  server.bind('sample_batch_train', sample_batch_train, workers=1)
  server.bind('sample_batch_report', sample_batch_report, workers=1)
  server.bind('sample_batch_eval', sample_batch_eval, workers=1)
  server.bind('update', lambda data: replay_train.update(data), workers=1)
  server.start(block=False)
  while True:
    if should_save() and active > 0:
      active.reset()
      cp.save()
    if should_log():
      stats = {}
      stats['timer/replay'] = elements.timer.stats()['summary']
      stats.update(prefix(limit_agg.result(), 'limiter'))
      stats.update(prefix(replay_train.stats(), 'replay'))
      stats.update(prefix(replay_eval.stats(), 'replay_eval'))
      stats.update(prefix(usage.stats(), 'usage/replay'))
      stats.update(prefix(logger.stats(), 'client/replay_logger'))
      stats.update(prefix(server.stats(), 'server/replay'))
      logger.add(stats)
    time.sleep(1)


@elements.timer.section('logger')
def parallel_logger(make_logger, args):
  if isinstance(make_logger, bytes):
    make_logger = cloudpickle.loads(make_logger)

  logger = make_logger()
  should_log = embodied.LocalClock(args.log_every)
  usage = elements.Usage(**args.usage.update(nvsmi=False))

  active = elements.Counter()
  should_save = embodied.LocalClock(args.save_every)
  cp = elements.Checkpoint(elements.Path(args.logdir) / 'ckpt/logger')
  cp.step = logger.step
  cp.load_or_save()

  parallel = elements.Agg()
  epstats = elements.Agg()
  episodes = collections.defaultdict(elements.Agg)
  updated = collections.defaultdict(lambda: None)
  dones = collections.defaultdict(lambda: True)

  @elements.timer.section('addfn')
  def addfn(metrics):
    active.increment()
    logger.add(metrics)

  @elements.timer.section('tranfn')
  def tranfn(trans):
    active.increment()
    now = time.time()
    envid = trans.pop('envid')
    logger.step.increment((~trans['is_eval']).sum())
    parallel.add('ep_starts', trans['is_first'].sum(), agg='sum')
    parallel.add('ep_ends', trans['is_last'].sum(), agg='sum')

    for i, addr in enumerate(envid):
      tran = {k: v[i] for k, v in trans.items()}

      updated[addr] = now
      episode = episodes[addr]
      if tran['is_first']:
        episode.reset()
        parallel.add('ep_abandoned', int(not dones[addr]), agg='sum')
      dones[addr] = tran['is_last']

      episode.add('score', tran['reward'], agg='sum')
      episode.add('length', 1, agg='sum')
      episode.add('rewards', tran['reward'], agg='stack')

      first_addr = next(iter(episodes.keys()))
      for key, value in tran.items():
        if value.dtype == np.uint8 and value.ndim == 3:
          if addr == first_addr:
            episode.add(f'policy_{key}', value, agg='stack')
        elif key.startswith('log/'):
          assert value.ndim == 0, (key, value.shape, value.dtype)
          episode.add(key + '/avg', value, agg='avg')
          episode.add(key + '/max', value, agg='max')
          episode.add(key + '/sum', value, agg='sum')
      if tran['is_last']:
        result = episode.result()
        logger.add({
            'score': result.pop('score'),
            'length': result.pop('length') - 1,
        }, prefix='episode')
        rew = result.pop('rewards')
        if len(rew) > 1:
          result['reward_rate'] = (np.abs(rew[1:] - rew[:-1]) >= 0.01).mean()
        epstats.add(result)

    for addr, last in list(updated.items()):
      if now - last >= args.episode_timeout:
        print('Dropping episode statistics due to timeout.')
        del episodes[addr]
        del updated[addr]

  server = portal.Server(args.logger_addr, 'Logger')
  server.bind('add', addfn)
  server.bind('tran', tranfn)
  server.start(block=False)
  last_step = int(logger.step)
  while True:
    time.sleep(1)
    if should_log() and active > 0:
      active.reset()
      with elements.timer.section('metrics'):
        logger.add({'timer/logger': elements.timer.stats()['summary']})
        logger.add(parallel.result(), prefix='parallel')
        logger.add(epstats.result(), prefix='epstats')
        logger.add(usage.stats(), prefix='usage/logger')
        logger.add(server.stats(), prefix='server/logger')
      if logger.step == last_step:
        continue
      logger.write()
      last_step = int(logger.step)
    if should_save():
      cp.save()


@elements.timer.section('env')
def parallel_env(make_env, envid, args, is_eval=False):
  if isinstance(make_env, bytes):
    make_env = cloudpickle.loads(make_env)
  assert envid >= 0, envid
  name = f'Env{envid:05}'
  print = lambda x: elements.print(f'[{name}] {x}', flush=True)

  should_log = embodied.LocalClock(args.log_every)
  fps = elements.FPS()
  if envid == 0:
    logger = portal.Client(args.logger_addr, f'{name}Logger', maxinflight=1)
    usage = elements.Usage(**args.usage.update(nvsmi=False))

  print('Make env')
  env = make_env(envid)
  actor = portal.Client(args.actor_addr, name, autoconn=False)
  actor.connect()

  done = True
  while True:

    if done:
      act = {k: v.sample() for k, v in env.act_space.items()}
      act['reset'] = True
      score, length = 0, 0

    scope_name = 'reset' if act['reset'] else 'step'
    with elements.timer.section(scope_name):
      obs = env.step(act)
    obs = {k: np.asarray(v, order='C') for k, v in obs.items()}
    obs['is_eval'] = is_eval
    score += obs['reward']
    length += 1
    fps.step(1)
    done = obs['is_last']
    if done and envid == 0:
      print(f'Episode of length {length} with score {score:.2f}')

    try:
      with elements.timer.section('request'):
        future = actor.act({'envid': envid, **obs})
      with elements.timer.section('response'):
        act = future.result()
    except portal.Disconnected:
      print('Env lost connection to agent')
      actor.connect()
      done = True

    if should_log() and envid == 0:
      stats = {}
      stats['fps/env'] = fps.result()
      stats['timer/env'] = elements.timer.stats()['summary']
      stats.update(prefix(usage.stats(), 'usage/env'))
      stats.update(prefix(logger.stats(), 'client/env_logger'))
      stats.update(prefix(actor.stats(), 'client/env_actor'))
      logger.add(stats)


def parallel_envs(make_env, make_env_eval, args):
  workers = []
  for i in range(args.envs):
    workers.append(portal.Process(parallel_env, make_env, i, args))
  for i in range(args.envs, args.envs + args.eval_envs):
    workers.append(portal.Process(parallel_env, make_env_eval, i, args, True))
  portal.run(workers)
