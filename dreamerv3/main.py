import importlib
import os
import pathlib
import sys
import warnings
from functools import partial as bind

directory = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(directory.parent))
sys.path.insert(0, str(directory.parent.parent))
__package__ = directory.name

warnings.filterwarnings('ignore', '.*box bound precision lowered.*')
warnings.filterwarnings('ignore', '.*using stateful random seeds*')
warnings.filterwarnings('ignore', '.*is a deprecated alias for.*')
warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')

import embodied
from embodied import wrappers


def main(argv=None):

  embodied.print(r"---  ___                           __   ______ ---")
  embodied.print(r"--- |   \ _ _ ___ __ _ _ __  ___ _ \ \ / /__ / ---")
  embodied.print(r"--- | |) | '_/ -_) _` | '  \/ -_) '/\ V / |_ \ ---")
  embodied.print(r"--- |___/|_| \___\__,_|_|_|_\___|_|  \_/ |___/ ---")

  from . import agent as agt
  parsed, other = embodied.Flags(configs=['defaults']).parse_known(argv)
  config = embodied.Config(agt.Agent.configs['defaults'])
  for name in parsed.configs:
    config = config.update(agt.Agent.configs[name])
  config = embodied.Flags(config).parse(other)
  config = config.update(
      logdir=config.logdir.format(timestamp=embodied.timestamp()),
      replay_length=config.replay_length or config.batch_length,
      replay_length_eval=config.replay_length_eval or config.batch_length_eval)
  args = embodied.Config(
      **config.run,
      logdir=config.logdir,
      batch_size=config.batch_size,
      batch_length=config.batch_length,
      batch_length_eval=config.batch_length_eval,
      replay_length=config.replay_length,
      replay_length_eval=config.replay_length_eval,
      replay_context=config.replay_context)
  print('Run script:', args.script)
  print('Logdir:', args.logdir)

  logdir = embodied.Path(args.logdir)
  if not args.script.endswith(('_env', '_replay')):
    logdir.mkdir()
    config.save(logdir / 'config.yaml')

  def init():
    embodied.timer.global_timer.enabled = args.timer
  embodied.distr.Process.initializers.append(init)
  init()

  if args.script == 'train':
    embodied.run.train(
        bind(make_agent, config),
        bind(make_replay, config, 'replay'),
        bind(make_env, config),
        bind(make_logger, config), args)

  elif args.script == 'train_eval':
    embodied.run.train_eval(
        bind(make_agent, config),
        bind(make_replay, config, 'replay'),
        bind(make_replay, config, 'eval_replay', is_eval=True),
        bind(make_env, config),
        bind(make_env, config),
        bind(make_logger, config), args)

  elif args.script == 'train_holdout':
    assert config.eval_dir
    embodied.run.train_holdout(
        bind(make_agent, config),
        bind(make_replay, config, 'replay'),
        bind(make_replay, config, config.eval_dir),
        bind(make_env, config),
        bind(make_logger, config), args)

  elif args.script == 'eval_only':
    embodied.run.eval_only(
        bind(make_agent, config),
        bind(make_env, config),
        bind(make_logger, config), args)

  elif args.script == 'parallel':
    embodied.run.parallel.combined(
        bind(make_agent, config),
        bind(make_replay, config, 'replay', rate_limit=True),
        bind(make_env, config),
        bind(make_logger, config), args)

  elif args.script == 'parallel_env':
    envid = args.env_replica
    if envid < 0:
      envid = int(os.environ['JOB_COMPLETION_INDEX'])
    embodied.run.parallel.env(
        bind(make_env, config), envid, args, False)

  elif args.script == 'parallel_replay':
    embodied.run.parallel.replay(
        bind(make_replay, config, 'replay', rate_limit=True), args)

  elif args.script == 'parallel_with_eval':
    embodied.run.parallel_with_eval.combined(
        bind(make_agent, config),
        bind(make_replay, config, 'replay', rate_limit=True),
        bind(make_replay, config, 'replay_eval', is_eval=True),
        bind(make_env, config),
        bind(make_env, config),
        bind(make_logger, config), args)

  elif args.script == 'parallel_with_eval_env':
    envid = args.env_replica
    if envid < 0:
      envid = int(os.environ['JOB_COMPLETION_INDEX'])
    is_eval = envid >= args.num_envs
    embodied.run.parallel_with_eval.parallel_env(
        bind(make_env, config), envid, args, True, is_eval)

  elif args.script == 'parallel_with_eval_replay':
    embodied.run.parallel_with_eval.parallel_replay(
        bind(make_replay, config, 'replay', rate_limit=True),
        bind(make_replay, config, 'replay_eval', is_eval=True), args)

  else:
    raise NotImplementedError(args.script)


def make_agent(config):
  from . import agent as agt
  env = make_env(config, 0)
  if config.random_agent:
    agent = embodied.RandomAgent(env.obs_space, env.act_space)
  else:
    agent = agt.Agent(env.obs_space, env.act_space, config)
  env.close()
  return agent


def make_logger(config):
  step = embodied.Counter()
  logdir = config.logdir
  multiplier = config.env.get(config.task.split('_')[0], {}).get('repeat', 1)
  logger = embodied.Logger(step, [
      embodied.logger.TerminalOutput(config.filter, 'Agent'),
      embodied.logger.JSONLOutput(logdir, 'metrics.jsonl'),
      embodied.logger.JSONLOutput(logdir, 'scores.jsonl', 'episode/score'),
      embodied.logger.TensorBoardOutput(
          logdir, config.run.log_video_fps, config.tensorboard_videos),
      # embodied.logger.WandbOutput(logdir.name, ...),
  ], multiplier)
  return logger


def make_replay(config, directory=None, is_eval=False, rate_limit=False):
  directory = directory and embodied.Path(config.logdir) / directory
  size = int(config.replay.size / 10 if is_eval else config.replay.size)
  length = config.replay_length_eval if is_eval else config.replay_length
  kwargs = {}
  kwargs['online'] = config.replay.online
  if rate_limit and config.run.train_ratio > 0:
    kwargs['samples_per_insert'] = config.run.train_ratio / (
        length - config.replay_context)
    kwargs['tolerance'] = 5 * config.batch_size
    kwargs['min_size'] = min(
        max(config.batch_size, config.run.train_fill), size)
  selectors = embodied.replay.selectors
  if config.replay.fracs.uniform < 1 and not is_eval:
    assert config.jax.compute_dtype in ('bfloat16', 'float32'), (
        'Gradient scaling for low-precision training can produce invalid loss '
        'outputs that are incompatible with prioritized replay.')
    import numpy as np
    recency = 1.0 / np.arange(1, size + 1) ** config.replay.recexp
    kwargs['selector'] = selectors.Mixture(dict(
        uniform=selectors.Uniform(),
        priority=selectors.Prioritized(**config.replay.prio),
        recency=selectors.Recency(recency),
    ), config.replay.fracs)
  kwargs['chunksize'] = config.replay.chunksize
  replay = embodied.replay.Replay(length, size, directory, **kwargs)
  return replay


def make_env(config, index, **overrides):
  suite, task = config.task.split('_', 1)
  if suite == 'memmaze':
    from embodied.envs import from_gym
    import memory_maze  # noqa
  ctor = {
      'dummy': 'embodied.envs.dummy:Dummy',
      'gym': 'embodied.envs.from_gym:FromGym',
      'dm': 'embodied.envs.from_dmenv:FromDM',
      'crafter': 'embodied.envs.crafter:Crafter',
      'dmc': 'embodied.envs.dmc:DMC',
      'atari': 'embodied.envs.atari:Atari',
      'atari100k': 'embodied.envs.atari:Atari',
      'dmlab': 'embodied.envs.dmlab:DMLab',
      'minecraft': 'embodied.envs.minecraft:Minecraft',
      'loconav': 'embodied.envs.loconav:LocoNav',
      'pinpad': 'embodied.envs.pinpad:PinPad',
      'langroom': 'embodied.envs.langroom:LangRoom',
      'procgen': 'embodied.envs.procgen:ProcGen',
      'bsuite': 'embodied.envs.bsuite:BSuite',
      'memmaze': lambda task, **kw: from_gym.FromGym(
          f'MemoryMaze-{task}-ExtraObs-v0', **kw),
  }[suite]
  if isinstance(ctor, str):
    module, cls = ctor.split(':')
    module = importlib.import_module(module)
    ctor = getattr(module, cls)
  kwargs = config.env.get(suite, {})
  kwargs.update(overrides)
  if kwargs.pop('use_seed', False):
    kwargs['seed'] = hash((config.seed, index)) % (2 ** 32 - 1)
  if kwargs.pop('use_logdir', False):
    kwargs['logdir'] = embodied.Path(config.logdir) / f'env{index}'
  env = ctor(task, **kwargs)
  return wrap_env(env, config)


def wrap_env(env, config):
  args = config.wrapper
  for name, space in env.act_space.items():
    if name == 'reset':
      continue
    elif not space.discrete:
      env = wrappers.NormalizeAction(env, name)
      if args.discretize:
        env = wrappers.DiscretizeAction(env, name, args.discretize)
  env = wrappers.ExpandScalars(env)
  if args.length:
    env = wrappers.TimeLimit(env, args.length, args.reset)
  if args.checks:
    env = wrappers.CheckSpaces(env)
  for name, space in env.act_space.items():
    if not space.discrete:
      env = wrappers.ClipAction(env, name)
  return env


if __name__ == '__main__':
  main()
