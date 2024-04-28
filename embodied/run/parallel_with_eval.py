import re
import sys
import threading
import time
from collections import defaultdict, deque
from functools import partial as bind

import cloudpickle
import embodied
import numpy as np

prefix = lambda d, p: {f'{p}/{k}': v for k, v in d.items()}


def combined(
    make_agent, make_replay, make_replay_eval, make_env, make_env_eval,
    make_logger, args):
  if args.num_envs:
    assert args.actor_batch <= args.num_envs, (args.actor_batch, args.num_envs)
  for key in ('actor_addr', 'replay_addr', 'logger_addr'):
    if '{auto}' in args[key]:
      port = embodied.distr.get_free_port()
      args = args.update({key: args[key].format(auto=port)})

  make_agent = cloudpickle.dumps(make_agent)
  make_replay = cloudpickle.dumps(make_replay)
  make_replay_eval = cloudpickle.dumps(make_replay_eval)
  make_env = cloudpickle.dumps(make_env)
  make_env_eval = cloudpickle.dumps(make_env_eval)
  make_logger = cloudpickle.dumps(make_logger)

  workers = []
  for i in range(args.num_envs):
    workers.append(embodied.distr.Process(
        parallel_env, make_env, i, args, True))
  for i in range(args.num_envs_eval):
    workers.append(embodied.distr.Process(
        parallel_env, make_env_eval, args.num_envs + i, args, True, True))
  if args.agent_process:
    workers.append(embodied.distr.Process(parallel_agent, make_agent, args))
  else:
    workers.append(embodied.distr.Thread(parallel_agent, make_agent, args))
  if not args.remote_replay:
    workers.append(embodied.distr.Process(
        parallel_replay, make_replay, make_replay_eval, args))
  workers.append(embodied.distr.Process(parallel_logger, make_logger, args))
  embodied.distr.run(workers, args.duration, exit_after=True)


def parallel_agent(make_agent, args):
  if isinstance(make_agent, bytes):
    make_agent = cloudpickle.loads(make_agent)
  agent = make_agent()
  barrier = threading.Barrier(2)
  workers = []
  workers.append(embodied.distr.Thread(parallel_actor, agent, barrier, args))
  workers.append(embodied.distr.Thread(parallel_learner, agent, barrier, args))
  embodied.distr.run(workers, args.duration)


def parallel_actor(agent, barrier, args):

  islist = lambda x: isinstance(x, list)
  initial = agent.init_policy(args.actor_batch)
  initial = embodied.tree.map(lambda x: x[0], initial, isleaf=islist)
  allstates = defaultdict(lambda: initial)
  barrier.wait()  # Do not collect data before learner restored checkpoint.
  fps = embodied.FPS()

  should_log = embodied.when.Clock(args.log_every)
  logger = embodied.distr.Client(
      args.logger_addr, 'ActorLogger', args.ipv6,
      maxinflight=8 * args.actor_threads, connect=True)
  replay = embodied.distr.Client(
      args.replay_addr, 'ActorReplay', args.ipv6,
      maxinflight=8 * args.actor_threads, connect=True)

  @embodied.timer.section('actor_workfn')
  def workfn(obs):
    envids = obs.pop('envid')
    fps.step(obs['is_first'].size)
    with embodied.timer.section('get_states'):
      states = [allstates[a] for a in envids]
      states = embodied.tree.map(lambda *xs: list(xs), *states)
    acts, outs, states = agent.policy(obs, states)
    assert all(k not in acts for k in outs), (
        list(outs.keys()), list(acts.keys()))
    acts['reset'] = obs['is_last'].copy()
    with embodied.timer.section('put_states'):
      for i, a in enumerate(envids):
        allstates[a] = embodied.tree.map(lambda x: x[i], states, isleaf=islist)
    trans = {'envids': envids, **obs, **acts, **outs}
    [x.setflags(write=False) for x in trans.values()]
    return acts, trans

  @embodied.timer.section('actor_donefn')
  def donefn(trans):
    replay.add_batch(trans)
    logger.trans(trans)
    if should_log():
      stats = {}
      stats['fps/policy'] = fps.result()
      stats['parallel/ep_states'] = len(allstates)
      stats.update(prefix(server.stats(), 'server/actor'))
      stats.update(prefix(logger.stats(), 'client/actor_logger'))
      stats.update(prefix(replay.stats(), 'client/actor_replay'))
      logger.add(stats)

  server = embodied.distr.ProcServer(args.actor_addr, 'Actor', args.ipv6)
  server.bind('act', workfn, donefn, args.actor_threads, args.actor_batch)
  server.run()


def parallel_learner(agent, barrier, args):

  logdir = embodied.Path(args.logdir)
  agg = embodied.Agg()
  usage = embodied.Usage(**args.usage)
  should_log = embodied.when.Clock(args.log_every)
  should_eval = embodied.when.Clock(args.eval_every)
  should_save = embodied.when.Clock(args.save_every)
  fps = embodied.FPS()
  batch_steps = args.batch_size * (args.batch_length - args.replay_context)

  checkpoint = embodied.Checkpoint(logdir / 'checkpoint.ckpt')
  checkpoint.agent = agent
  if args.from_checkpoint:
    checkpoint.load(args.from_checkpoint)
  checkpoint.load_or_save()
  logger = embodied.distr.Client(
      args.logger_addr, 'LearnerLogger', args.ipv6,
      maxinflight=1, connect=True)
  updater = embodied.distr.Client(
      args.replay_addr, 'LearnerReplayUpdater', args.ipv6,
      maxinflight=8, connect=True)
  barrier.wait()

  replays = []
  received = defaultdict(int)
  def parallel_dataset(source, prefetch=2):
    replay = embodied.distr.Client(
        args.replay_addr, f'LearnerReplay{len(replays)}', args.ipv6,
        connect=True)
    replays.append(replay)
    call = getattr(replay, f'sample_batch_{source}')
    futures = deque([call({}) for _ in range(prefetch)])
    while True:
      futures.append(call({}))
      batch = futures.popleft().result()
      received[source] += 1
      yield batch

  def evaluate(dataset):
    num_batches = args.replay_length_eval // args.batch_length_eval
    carry = agent.init_report(args.batch_size)
    agg = embodied.Agg()
    for _ in range(num_batches):
      batch = next(dataset)
      metrics, carry = agent.report(batch, carry)
      agg.add(metrics)
    return agg.result()

  dataset_train = agent.dataset(bind(parallel_dataset, 'train'))
  dataset_report = agent.dataset(bind(parallel_dataset, 'report'))
  dataset_eval = agent.dataset(bind(parallel_dataset, 'eval'))
  carry = agent.init_train(args.batch_size)
  should_save()  # Delay first save.
  should_eval()  # Delay first eval.

  while True:

    with embodied.timer.section('learner_batch_next'):
      batch = next(dataset_train)
    with embodied.timer.section('learner_train_step'):
      outs, carry, mets = agent.train(batch, carry)
    if 'replay' in outs:
      with embodied.timer.section('learner_replay_update'):
        updater.update(outs['replay'])
    time.sleep(0.0001)
    agg.add(mets)
    fps.step(batch_steps)

    if should_eval():
      with embodied.timer.section('learner_eval'):
        if received['report'] > 0:
          logger.add(prefix(evaluate(dataset_report), 'report'))
        if received['eval'] > 0:
          logger.add(prefix(evaluate(dataset_eval), 'eval'))

    if should_log():
      with embodied.timer.section('learner_metrics'):
        stats = {}
        stats.update(prefix(agg.result(), 'train'))
        stats.update(prefix(embodied.timer.stats(), 'timer/agent'))
        stats.update(prefix(usage.stats(), 'usage/agent'))
        stats.update(prefix(logger.stats(), 'client/learner_logger'))
        stats.update(prefix(replays[0].stats(), 'client/learner_replay0'))
        stats.update({'fps/train': fps.result()})
      logger.add(stats)

    if should_save():
      checkpoint.save()


def parallel_replay(make_replay, make_replay_eval, args):
  if isinstance(make_replay, bytes):
    make_replay = cloudpickle.loads(make_replay)
  if isinstance(make_replay_eval, bytes):
    make_replay_eval = cloudpickle.loads(make_replay_eval)

  replay = make_replay()
  replay_eval = make_replay_eval()
  dataset_train = iter(replay.dataset(
      args.batch_size, args.batch_length))
  dataset_report = iter(replay.dataset(
      args.batch_size, args.batch_length_eval))
  dataset_eval = iter(replay_eval.dataset(
      args.batch_size, args.batch_length_eval))

  should_log = embodied.when.Clock(args.log_every)
  logger = embodied.distr.Client(
      args.logger_addr, 'ReplayLogger', args.ipv6,
      maxinflight=1, connect=True)
  usage = embodied.Usage(**args.usage.update(nvsmi=False))

  should_save = embodied.when.Clock(args.save_every)
  cp = embodied.Checkpoint(embodied.Path(args.logdir) / 'replay.ckpt')
  cp.replay = replay
  cp.load_or_save()

  def add_batch(data):
    for i, envid in enumerate(data.pop('envids')):
      tran = {k: v[i] for k, v in data.items()}
      if tran.pop('is_eval', False):
        replay_eval.add(tran, envid)
      else:
        replay.add(tran, envid)
    return {}

  server = embodied.distr.Server(args.replay_addr, 'Replay', args.ipv6)
  server.bind('add_batch', add_batch, workers=1)
  server.bind('sample_batch_train', lambda _: next(dataset_train), workers=1)
  server.bind('sample_batch_report', lambda _: next(dataset_report), workers=1)
  server.bind('sample_batch_eval', lambda _: next(dataset_eval), workers=1)
  server.bind('update', lambda data: replay.update(data), workers=1)
  with server:
    while True:
      server.check()
      should_save() and cp.save()
      time.sleep(1)
      if should_log():
        stats = {}
        stats.update(prefix(replay.stats(), 'replay'))
        stats.update(prefix(replay_eval.stats(), 'replay_eval'))
        stats.update(prefix(embodied.timer.stats(), 'timer/replay'))
        stats.update(prefix(usage.stats(), 'usage/replay'))
        stats.update(prefix(logger.stats(), 'client/replay_logger'))
        stats.update(prefix(server.stats(), 'server/replay'))
        logger.add(stats)


def parallel_logger(make_logger, args):
  if isinstance(make_logger, bytes):
    make_logger = cloudpickle.loads(make_logger)

  logger = make_logger()
  should_log = embodied.when.Clock(args.log_every)
  usage = embodied.Usage(**args.usage.update(nvsmi=False))

  should_save = embodied.when.Clock(args.save_every)
  cp = embodied.Checkpoint(embodied.Path(args.logdir) / 'logger.ckpt')
  cp.step = logger.step
  cp.load_or_save()

  parallel = embodied.Agg()
  epstats = embodied.Agg()
  episodes = defaultdict(embodied.Agg)
  updated = defaultdict(lambda: None)
  dones = defaultdict(lambda: True)

  log_keys_max = re.compile(args.log_keys_max)
  log_keys_sum = re.compile(args.log_keys_sum)
  log_keys_avg = re.compile(args.log_keys_avg)

  @embodied.timer.section('logger_addfn')
  def addfn(metrics):
    logger.add(metrics)

  @embodied.timer.section('logger_transfn')
  def transfn(trans):
    now = time.time()
    envids = trans.pop('envids')
    logger.step.increment(len(trans['is_first']))
    parallel.add('ep_starts', trans['is_first'].sum(), agg='sum')
    parallel.add('ep_ends', trans['is_last'].sum(), agg='sum')

    for i, addr in enumerate(envids):
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

      video_addrs = list(episodes.keys())[:args.log_video_streams]
      if addr in video_addrs:
        for key in args.log_keys_video:
          if key in tran:
            episode.add(f'policy_{key}', tran[key], agg='stack')

      for key in trans.keys():
        if log_keys_max.match(key):
          episode.add(key, tran[key], agg='max')
        if log_keys_sum.match(key):
          episode.add(key, tran[key], agg='sum')
        if log_keys_avg.match(key):
          episode.add(key, tran[key], agg='avg')

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
      if now - last >= args.log_episode_timeout:
        print('Dropping episode statistics due to timeout.')
        del episodes[addr]
        del updated[addr]

  server = embodied.distr.Server(args.logger_addr, 'Logger', args.ipv6)
  server.bind('add', addfn)
  server.bind('trans', transfn)
  with server:
    while True:
      server.check()
      should_save() and cp.save()
      time.sleep(1)
      if should_log():
        with embodied.timer.section('logger_metrics'):
          logger.add(parallel.result(), prefix='parallel')
          logger.add(epstats.result(), prefix='epstats')
          logger.add(embodied.timer.stats(), prefix='timer/logger')
          logger.add(usage.stats(), prefix='usage/logger')
          logger.add(server.stats(), prefix='server/logger')
        logger.write()


def parallel_env(make_env, envid, args, logging=False, is_eval=False):
  if isinstance(make_env, bytes):
    make_env = cloudpickle.loads(make_env)
  assert envid >= 0, envid
  name = f'Env{envid}'

  _print = lambda x: embodied.print(f'[{name}] {x}', flush=True)
  should_log = embodied.when.Clock(args.log_every)
  if logging and envid == 0:
    logger = embodied.distr.Client(
        args.logger_addr, f'{name}Logger', args.ipv6,
        maxinflight=1, connect=True)
  fps = embodied.FPS()
  if envid == 0:
    usage = embodied.Usage(**args.usage.update(nvsmi=False))

  _print('Make env')
  env = make_env(envid)
  actor = embodied.distr.Client(
      args.actor_addr, name, args.ipv6, identity=envid,
      pings=10, maxage=60, connect=True)

  done = True
  while True:

    if done:
      act = {k: v.sample() for k, v in env.act_space.items()}
      act['reset'] = True
      score, length = 0, 0

    with embodied.timer.section('env_step'):
      obs = env.step(act)
    obs = {k: np.asarray(v, order='C') for k, v in obs.items()}
    obs['is_eval'] = is_eval
    score += obs['reward']
    length += 1
    fps.step(1)
    done = obs['is_last']
    if done:
      _print(f'Episode of length {length} with score {score:.2f}')

    with embodied.timer.section('env_request'):
      future = actor.act({'envid': envid, **obs})
    try:
      with embodied.timer.section('env_response'):
        act = future.result()
    except embodied.distr.NotAliveError:
      # Wait until we are connected again, so we don't unnecessarily reset the
      # environment hundreds of times while the server is unavailable.
      _print('Lost connection to server')
      actor.connect()
      done = True
    except embodied.distr.RemoteError as e:
      _print(f'Shutting down env due to agent error: {e}')
      sys.exit(0)

    if should_log() and logging and envid == 0:
      stats = {f'fps/env{envid}': fps.result()}
      stats.update(prefix(usage.stats(), f'usage/env{envid}'))
      stats.update(prefix(logger.stats(), f'client/env{envid}_logger'))
      stats.update(prefix(actor.stats(), f'client/env{envid}_actor'))
      stats.update(prefix(embodied.timer.stats(), f'timer/env{envid}'))
      logger.add(stats)
