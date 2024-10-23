import re
from collections import defaultdict
from functools import partial as bind

import numpy as np

from ... import embodied

def train_eval(
    make_agent, make_train_replay, make_eval_replay,
    make_train_env, make_eval_env, make_logger, args):

  agent = make_agent()
  train_replay = make_train_replay()
  eval_replay = make_eval_replay()
  logger = make_logger()

  logdir = embodied.Path(args.logdir)
  logdir.mkdir()
  print('Logdir', logdir)
  step = logger.step
  usage = embodied.Usage(**args.usage)
  agg = embodied.Agg()
  train_episodes = defaultdict(embodied.Agg)
  train_epstats = embodied.Agg()
  eval_episodes = defaultdict(embodied.Agg)
  eval_epstats = embodied.Agg()
  policy_fps = embodied.FPS()
  train_fps = embodied.FPS()

  batch_steps = args.batch_size * (args.batch_length - args.replay_context)
  should_expl = embodied.when.Until(args.expl_until)
  should_train = embodied.when.Ratio(args.train_ratio / batch_steps)

  @embodied.timer.section('log_step')
  def log_step(tran, worker, mode):
    episodes = dict(train=train_episodes, eval=eval_episodes)[mode]
    epstats = dict(train=train_epstats, eval=eval_epstats)[mode]

    episode = episodes[worker]
    episode.add('score', tran['reward'], agg='sum')
    episode.add('length', 1, agg='sum')
    episode.add('rewards', tran['reward'], agg='stack')

    if tran['is_first']:
      episode.reset()

    if worker < args.log_video_streams:
      for key in args.log_keys_video:
        if key in tran:
          episode.add(f'policy_{key}', tran[key], agg='stack')
    for key, value in tran.items():
      if re.match(args.log_keys_sum, key):
        episode.add(key, value, agg='sum')
      if re.match(args.log_keys_avg, key):
        episode.add(key, value, agg='avg')
      if re.match(args.log_keys_max, key):
        episode.add(key, value, agg='max')

    if tran['is_last']:
      result = episode.result()
      logger.add({
          'score': result.pop('score'),
          'length': result.pop('length'),
      }, prefix='episode')
      rew = result.pop('rewards')
      if len(rew) > 1:
        result['reward_rate'] = (np.abs(rew[1:] - rew[:-1]) >= 0.01).mean()
      epstats.add(result)

  fns = [bind(make_train_env, i) for i in range(args.num_envs)]
  train_driver = embodied.Driver(fns, args.driver_parallel)
  train_driver.on_step(lambda tran, _: step.increment())
  train_driver.on_step(lambda tran, _: policy_fps.step())
  train_driver.on_step(train_replay.add)
  train_driver.on_step(bind(log_step, mode='train'))

  fns = [bind(make_eval_env, i) for i in range(args.num_envs)]
  eval_driver = embodied.Driver(fns, args.driver_parallel)
  eval_driver.on_step(eval_replay.add)
  eval_driver.on_step(bind(log_step, mode='eval'))
  eval_driver.on_step(lambda tran, _: policy_fps.step())

  dataset_train = agent.dataset(
      bind(train_replay.dataset, args.batch_size, args.batch_length))
  dataset_report = agent.dataset(
      bind(train_replay.dataset, args.batch_size, args.batch_length_eval))
  dataset_eval = agent.dataset(
      bind(eval_replay.dataset, args.batch_size, args.batch_length_eval))
  carry = [agent.init_train(args.batch_size)]
  carry_report = agent.init_report(args.batch_size)

  def train_step(tran, worker):
    if len(train_replay) < args.batch_size or step < args.train_fill:
      return
    for _ in range(should_train(step)):
      with embodied.timer.section('dataset_next'):
        batch = next(dataset_train)
      outs, carry[0], mets = agent.train(batch, carry[0])
      train_fps.step(batch_steps)
      if 'replay' in outs:
        train_replay.update(outs['replay'])
      agg.add(mets, prefix='train')
  train_driver.on_step(train_step)

  checkpoint = embodied.Checkpoint(logdir / 'checkpoint.ckpt')
  checkpoint.step = step
  checkpoint.agent = agent
  checkpoint.train_replay = train_replay
  checkpoint.eval_replay = eval_replay
  if args.from_checkpoint:
    checkpoint.load(args.from_checkpoint)
  checkpoint.load_or_save()
  should_save(step)  # Register that we just saved.

  print('Start training loop')
  train_policy = lambda *args: agent.policy(
      *args, mode='explore' if should_expl(step) else 'train')
  eval_policy = lambda *args: agent.policy(*args, mode='eval')
  train_driver.reset(agent.init_policy)
  while step < args.steps:

    if should_eval(step):
      print('Start evaluation')
      eval_driver.reset(agent.init_policy)
      eval_driver(eval_policy, episodes=args.eval_eps)
      logger.add(eval_epstats.result(), prefix='epstats')
      if len(train_replay):
        mets, _ = agent.report(next(dataset_report), carry_report)
        logger.add(mets, prefix='report')
      if len(eval_replay):
        mets, _ = agent.report(next(dataset_eval), carry_report)
        logger.add(mets, prefix='eval')

    train_driver(train_policy, steps=10)

    if should_log(step):
      logger.add(agg.result())
      logger.add(train_epstats.result(), prefix='epstats')
      logger.add(embodied.timer.stats(), prefix='timer')
      logger.add(train_replay.stats(), prefix='replay')
      logger.add(usage.stats(), prefix='usage')
      logger.add({'fps/policy': policy_fps.result()})
      logger.add({'fps/train': train_fps.result()})
      logger.write()

    if should_save(step):
      checkpoint.save()
  logger.close()
