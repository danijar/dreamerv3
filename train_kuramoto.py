def main():

  import warnings
  import dreamerv3
  from dreamerv3 import embodied
  warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')

  # See configs.yaml for all options.
  config = embodied.Config(dreamerv3.configs['defaults'])
  config = config.update(dreamerv3.configs['medium'])
  config = config.update({
      'logdir': '~/logdir/kuramoto5',
      'run.log_every': 30,  # Seconds
      'encoder.mlp_keys': '^$',
      'decoder.mlp_keys': '^$',
      'encoder.cnn_keys': 'correlogram',
      'decoder.cnn_keys': 'correlogram',
  })
  config = embodied.Flags(config).parse()

  logdir = embodied.Path(config.logdir)
  step = embodied.Counter()
  logger = embodied.Logger(step, [
      embodied.logger.TerminalOutput(),
      embodied.logger.JSONLOutput(logdir, 'metrics.jsonl'),
      embodied.logger.TensorBoardOutput(logdir),
  ])

  from embodied.envs import kuramoto
  from embodied.envs import from_gym
  import numpy as np
  target_matrix = np.random.rand(64,64)
  target_matrix = (target_matrix + target_matrix.T) / 2 # Ensure symmetry
  np.fill_diagonal(target_matrix, 0) # No self-coupling
  env = kuramoto.KuramotoEnv(target_matrix)  # Replace this with your Gym env.
  env = from_gym.FromGym(env, obs_key='correlogram') 
  env = dreamerv3.wrap_env(env, config)
  env = embodied.BatchEnv([env], parallel=False)

  agent = dreamerv3.Agent(env.obs_space, env.act_space, step, config)
  replay = embodied.replay.Uniform(
      config.batch_length, config.replay_size, logdir / 'replay')
  args = embodied.Config(
      **config.run, logdir=config.logdir,
      batch_steps=config.batch_size * config.batch_length)
  embodied.run.train(agent, env, replay, logger, args)
  # embodied.run.eval_only(agent, env, logger, args)

if __name__ == '__main__':
    main()
