import collections
import re
import warnings

import embodied
import numpy as np


def train_holdout(agent, env, train_replay, eval_replay, logger, args):

  logdir = embodied.Path(args.logdir)
  logdir.mkdirs()
  print('Logdir', logdir)
  should_expl = embodied.when.Until(args.expl_until)
  should_train = embodied.when.Ratio(args.train_ratio / args.batch_steps)
  should_log = embodied.when.Clock(args.log_every)
  should_save = embodied.when.Clock(args.save_every)
  step = logger.step
  metrics = embodied.Metrics()
  print('Observation space:', env.obs_space)
  print('Action space:', env.act_space)

  timer = embodied.Timer()
  timer.wrap('agent', agent, ['policy', 'train', 'report', 'save'])
  timer.wrap('env', env, ['step'])
  if hasattr(train_replay, '_sample'):
    timer.wrap('replay', train_replay, ['_sample'])

  nonzeros = set()
  def per_episode(ep):
    length = len(ep['reward']) - 1
    score = float(ep['reward'].astype(np.float64).sum())
    logger.add({
        'length': length, 'score': score,
        'reward_rate': (ep['reward'] - ep['reward'].min() >= 0.1).mean(),
    }, prefix='episode')
    print(f'Episode has {length} steps and return {score:.1f}.')
    stats = {}
    for key in args.log_keys_video:
      if key in ep:
        stats[f'policy_{key}'] = ep[key]
    for key, value in ep.items():
      if not args.log_zeros and key not in nonzeros and (value == 0).all():
        continue
      nonzeros.add(key)
      if re.match(args.log_keys_sum, key):
        stats[f'sum_{key}'] = ep[key].sum()
      if re.match(args.log_keys_mean, key):
        stats[f'mean_{key}'] = ep[key].mean()
      if re.match(args.log_keys_max, key):
        stats[f'max_{key}'] = ep[key].max(0).mean()
    metrics.add(stats, prefix='stats')

  fill = max(0, args.eval_fill - len(eval_replay))
  if fill:
    print(f'Fill eval dataset ({fill} steps).')
    eval_driver = embodied.Driver(env)
    eval_driver.on_step(eval_replay.add)
    random_agent = embodied.RandomAgent(env.act_space)
    eval_driver(random_agent.policy, steps=fill)
    del eval_driver

  driver = embodied.Driver(env)
  driver.on_episode(lambda ep, worker: per_episode(ep))
  driver.on_step(lambda tran, _: step.increment())
  driver.on_step(train_replay.add)
  fill = max(args.batch_steps, args.train_fill - len(train_replay))
  if fill:
    print(f'Fill train dataset ({fill} steps).')
    random_agent = embodied.RandomAgent(env.act_space)
    driver(random_agent.policy, steps=fill)
  logger.write()

  dataset_train = iter(agent.dataset(train_replay.dataset))
  dataset_eval = iter(agent.dataset(eval_replay.dataset))
  state = [None]  # To be writable from train step function below.
  assert args.pretrain > 0  # At least one step to initialize variables.
  for _ in range(args.pretrain):
    with timer.scope('dataset_train'):
      batch = next(dataset_train)
    _, state[0], _ = agent.train(batch, state[0])

  batch = [None]
  def train_step(tran, worker):
    for _ in range(should_train(step)):
      with timer.scope('dataset_train'):
        batch[0] = next(dataset_train)
      outs, state[0], mets = agent.train(batch[0], state[0])
      metrics.add(mets, prefix='train')
      if 'priority' in outs:
        train_replay.prioritize(outs['key'], outs['priority'])
    if should_log(step):
      logger.add(metrics.result())
      logger.add(agent.report(batch[0]), prefix='report')
      with timer.scope('dataset_eval'):
        eval_batch = next(dataset_eval)
      logger.add(agent.report(eval_batch), prefix='eval')
      logger.add(train_replay.stats, prefix='replay')
      logger.add(eval_replay.stats, prefix='eval_replay')
      logger.add(timer.stats(), prefix='timer')
      logger.write(fps=True)
  driver.on_step(train_step)

  checkpoint = embodied.Checkpoint(logdir / 'checkpoint.pkl')
  checkpoint.step = step
  checkpoint.agent = agent
  checkpoint.train_replay = train_replay
  checkpoint.eval_replay = eval_replay
  checkpoint.load_or_save()
  should_save(step)  # Register that we jused saved.

  print('Start training loop.')
  policy = lambda *args: agent.policy(
      *args, mode='explore' if should_expl(step) else 'train')
  while step < args.steps:
    # scalars = collections.defaultdict(list)
    # for _ in range(args.eval_samples):
    #   for key, value in agent.report(next(dataset_eval)).items():
    #     if value.shape == ():
    #       scalars[key].append(value)
    # for name, values in scalars.items():
    #   logger.scalar(f'eval/{name}', np.array(values, np.float64).mean())
    # logger.write()
    driver(policy, steps=1000)
    if should_save(step):
      checkpoint.save()
  logger.write()
  logger.write()
