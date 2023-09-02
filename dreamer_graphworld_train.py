def main():

  import warnings
  import argparse
  import dreamerv3
  from dreamerv3 import embodied
  warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')

  # Add argument parsing
  graphworld_parser = argparse.ArgumentParser(
    description="""This script trains an agent in a GraphWorld environment 
    using DreamerV3. It takes the number of rooms, a random seed, and the number 
    of steps as command-line arguments. It saves logs and metrics in a directory 
    whose name is composed of these arguments."""
  )
  graphworld_parser.add_argument("--N_ROOMS", type=int, required=True, help="Number of rooms")
  graphworld_parser.add_argument("--SEED", type=int, required=True, help="Seed for random number generator")
  graphworld_parser.add_argument("--N_STEPS", type=int, required=True, help="Number of steps")
  
  graphworld_args, unknown_args = graphworld_parser.parse_known_args()

  # See configs.yaml for all options.
  config = embodied.Config(dreamerv3.configs['defaults'])
  config = config.update(dreamerv3.configs['small'])
  config = config.update({
      'logdir': f'~/logdir/run_{graphworld_args.N_ROOMS}_{graphworld_args.SEED}_{graphworld_args.N_STEPS}',
      'run.log_every': 30,  # Seconds
      'run.steps': 1e7,
      'encoder.mlp_keys': 'vector',
      'decoder.mlp_keys': 'vector',
      'encoder.cnn_keys': '$^',
      'decoder.cnn_keys': '$^',
  })
  config = embodied.Flags(config).parse(argv=unknown_args)

  logdir = embodied.Path(config.logdir)
  step = embodied.Counter()
  logger = embodied.Logger(step, [
      embodied.logger.TerminalOutput(),
      embodied.logger.JSONLOutput(logdir, 'metrics.jsonl'),
      embodied.logger.TensorBoardOutput(logdir),
  ])

  import graphworld
  from embodied.envs import from_gym
  env = graphworld.GraphWorld(graphworld_args.N_ROOMS, graphworld_args.SEED)
  env = from_gym.FromGym(env, obs_key='vector')
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
