import re
from collections import defaultdict
from functools import partial as bind

import numpy as np

from ... import embodied


def train(make_agent, make_replay, make_env, make_logger, args):

  agent = make_agent()
  replay = make_replay()
  logger = make_logger()

  logdir = embodied.Path(args.logdir)
  logdir.mkdir()
  print('Logdir', logdir)
  step = logger.step
  usage = embodied.Usage(**args.usage)
  agg = embodied.Agg()
  epstats = embodied.Agg()
  episodes = defaultdict(embodied.Agg)
  policy_fps = embodied.FPS()
  train_fps = embodied.FPS()

  batch_steps = args.batch_size * (args.batch_length - args.replay_context)
  should_expl = embodied.when.Until(args.expl_until)
  should_train = embodied.when.Ratio(args.train_ratio / batch_steps)
  log_when_klass = embodied.when.Clock if args.log_units == "seconds" else embodied.when.Every

  should_log = log_when_klass(args.log_every)
  should_eval = log_when_klass(args.eval_every)
  should_save = log_when_klass(args.save_every)

  @embodied.timer.section('log_step')
  def log_step(tran, worker):

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

  fns = [bind(make_env, i) for i in range(args.num_envs)]
  driver = embodied.Driver(fns, args.driver_parallel)
  driver.on_step(lambda tran, _: step.increment())
  driver.on_step(lambda tran, _: policy_fps.step())
  driver.on_step(replay.add)
  driver.on_step(log_step)

  dataset_train = iter(agent.dataset(bind(
      replay.dataset, args.batch_size, args.batch_length)))
  dataset_report = iter(agent.dataset(bind(
      replay.dataset, args.batch_size, args.batch_length_eval)))
  carry = [agent.init_train(args.batch_size)]
  carry_report = agent.init_report(args.batch_size)

  def train_step(tran, worker):
    if len(replay) < args.batch_size or step < args.train_fill:
      return
    for _ in range(should_train(step)):
      with embodied.timer.section('dataset_next'):
        batch = next(dataset_train)
      outs, carry[0], mets = agent.train(batch, carry[0])
      train_fps.step(batch_steps)
      if 'replay' in outs:
        replay.update(outs['replay'])
      agg.add(mets, prefix='train')
  driver.on_step(train_step)

  checkpoint = embodied.Checkpoint(logdir / 'checkpoint.ckpt')
  checkpoint.step = step
  checkpoint.agent = agent
  checkpoint.replay = replay
  if args.from_checkpoint:
    checkpoint.load(args.from_checkpoint)
  checkpoint.load_or_save()
  should_save(step)  # Register that we just saved.

  print('Start training loop')
  policy = lambda *args: agent.policy(
      *args, mode='explore' if should_expl(step) else 'train')
  driver.reset(agent.init_policy)
  while step < args.steps:

    driver(policy, steps=10)

    if should_eval(step) and len(replay):
      mets, _ = agent.report(next(dataset_report), carry_report)
      logger.add(mets, prefix='report')

    if should_log(step):
      logger.add(agg.result())
      logger.add(epstats.result(), prefix='epstats')
      logger.add(embodied.timer.stats(), prefix='timer')
      logger.add(replay.stats(), prefix='replay')
      logger.add(usage.stats(), prefix='usage')
      logger.add({'fps/policy': policy_fps.result()})
      logger.add({'fps/train': train_fps.result()})
      logger.write()

    if should_save(step):
      checkpoint.save()
      if args.save_intermediate_ckpt:
        ckpt_step_path = logdir / f"checkpoint_{step.value}.ckpt"
        checkpoint.save(ckpt_step_path)
  logger.close()
