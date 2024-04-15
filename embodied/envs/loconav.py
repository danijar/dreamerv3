import functools
import os
import warnings

import embodied
import numpy as np


class LocoNav(embodied.Env):

  DEFAULT_CAMERAS = dict(
      ant=4,
      quadruped=5,
  )

  def __init__(
      self, name, repeat=1, size=(64, 64), camera=-1, again=False,
      termination=False, weaker=1.0):
    if name.endswith('hz'):
      name, freq = name.rsplit('_', 1)
      freq = int(freq.strip('hz'))
    else:
      freq = 50
    if 'MUJOCO_GL' not in os.environ:
      os.environ['MUJOCO_GL'] = 'egl'
    from dm_control import composer
    from dm_control.locomotion.props import target_sphere
    from dm_control.locomotion.tasks import random_goal_maze
    walker, arena = name.split('_', 1)
    if camera == -1:
      camera = self.DEFAULT_CAMERAS.get(walker, 0)
    self._walker = self._make_walker(walker)
    arena = self._make_arena(arena)
    target = target_sphere.TargetSphere(radius=1.2, height_above_ground=0.0)
    task = random_goal_maze.RepeatSingleGoalMaze(
        walker=self._walker, maze_arena=arena, target=target,
        max_repeats=1000 if again else 1,
        randomize_spawn_rotation=True,
        target_reward_scale=1.0,
        aliveness_threshold=-0.5 if termination else -1.0,
        contact_termination=False,
        physics_timestep=min(1 / freq / 4, 0.02),
        control_timestep=1 / freq)
    if not again:
      def after_step(self, physics, random_state):
        super(random_goal_maze.RepeatSingleGoalMaze, self).after_step(
            physics, random_state)
        self._rewarded_this_step = self._target.activated
        self._targets_obtained = int(self._target.activated)
      task.after_step = functools.partial(after_step, task)
    env = composer.Environment(
        time_limit=60, task=task, random_state=None,
        strip_singleton_obs_buffer_dim=True)
    from . import dmc
    self._env = dmc.DMC(env, repeat, size=size, camera=camera, image=False)
    self._visited = None
    self._weaker = weaker

  @property
  def obs_space(self):
    spaces = self._env.obs_space.copy()
    spaces['log_coverage'] = embodied.Space(np.int32, low=-1)
    return spaces

  @property
  def act_space(self):
    return self._env.act_space

  def step(self, action):
    with warnings.catch_warnings():
      warnings.filterwarnings('ignore', '.*is a deprecated alias for.*')
      action = action.copy()
      action['action'] *= self._weaker
      obs = self._env.step(action)
    if obs['is_first']:
      self._visited = set()
    global_pos = self._walker.get_pose(
        self._env._dmenv._physics)[0].reshape(-1)
    self._visited.add(tuple(np.round(global_pos[:2]).astype(int).tolist()))
    obs['log_coverage'] = np.int32(len(self._visited))
    return obs

  def _make_walker(self, name):
    if name == 'ant':
      from dm_control.locomotion.walkers import ant
      return ant.Ant()
    elif name == 'quadruped':
      from . import loconav_quadruped
      return loconav_quadruped.Quadruped()
    else:
      raise NotImplementedError(name)

  def _make_arena(self, name):
    import labmaze
    from dm_control import mjcf
    from dm_control.locomotion.arenas import labmaze_textures
    from dm_control.locomotion.arenas import mazes
    import matplotlib.pyplot as plt
    class WallTexture(labmaze_textures.WallTextures):
      def _build(self, color=[0.8, 0.8, 0.8], model='labmaze_style_01'):
        self._mjcf_root = mjcf.RootElement(model=model)
        self._textures = [self._mjcf_root.asset.add(
            'texture', type='2d', name='wall', builtin='flat',
            rgb1=color, width=100, height=100)]
    wall_textures = {'*': WallTexture([0.8, 0.8, 0.8])}
    cmap = plt.get_cmap('tab10')
    for index in range(9):
      wall_textures[str(index + 1)] = WallTexture(cmap(index)[:3])
    layout = ''.join([
        line[::2].replace('.', ' ') + '\n' for line in MAPS[name]])
    maze = labmaze.FixedMazeWithRandomGoals(
        entity_layer=layout,
        num_spawns=1, num_objects=1, random_state=None)
    arena = mazes.MazeWithTargets(
        maze, xy_scale=1.2, z_height=2.0, aesthetic='default',
        wall_textures=wall_textures, name='maze')
    return arena


MAPS = {

    'maze_s': (
        '            6 6 6 6 6',
        '            6 . . . 6',
        '            6 . G . 6',
        '            6 . . . 6',
        '            5 . . . 4',
        '            5 . . . 4',
        '1 1 1 1 5 5 5 . . . 4',
        '1 . . . . . . . . . 3',
        '1 . P . . . . . . . 3',
        '1 . . . . . . . . . 3',
        '1 1 1 1 2 2 2 3 3 3 3',
    ),

    'maze_m': (
        '6 6 6 6 8 8 8 7 7 7 7',
        '6 . . . . . . . . . 7',
        '6 . G . . . . . . . 7',
        '6 . . . . . . . . . 7',
        '6 6 6 5 5 5 5 . . . 4',
        '            5 . . . 4',
        '1 1 1 1 5 5 5 . . . 4',
        '1 . . . . . . . . . 3',
        '1 . P . . . . . . . 3',
        '1 . . . . . . . . . 3',
        '1 1 1 1 2 2 2 3 3 3 3',
    ),

    'maze_l': (
        '8 8 8 8 7 7 7 6 6 6 6 . . .',
        '8 . . . . . . . . . 6 . . .',
        '8 . G . . . . . . . 6 . . .',
        '8 . . . . . . . . . 6 5 5 5',
        '8 8 8 8 7 7 7 . . . . . . 5',
        '. . . . . . 7 . . . . . . 5',
        '1 1 1 1 1 . 7 . . . . . . 5',
        '1 . . . 1 . 7 9 9 9 . . . 5',
        '1 . . . 1 . . . . 9 . . . 5',
        '1 . . . 1 1 1 9 9 9 . . . 5',
        '2 . . . . . . . . . . . . 4',
        '2 . . . . P . . . . . . . 4',
        '2 . . . . . . . . . . . . 4',
        '2 2 2 2 3 3 3 3 3 3 4 4 4 4',
    ),

    'maze_xl': (
        '9 9 9 9 9 9 9 8 8 8 8 . 4 4 4 4 4',
        '9 . . . . . . . . . 8 . 4 . . . 4',
        '9 . . . . . . . G . 8 . 4 . . . 4',
        '9 . . . . . . . . . 8 . 4 . . . 4',
        '6 . . . 7 7 7 8 8 8 8 . 5 . . . 3',
        '6 . . . 7 . . . . . . . 5 . . . 3',
        '6 . . . 7 7 7 5 5 5 5 5 5 . . . 3',
        '5 . . . . . . . . . . . . . . . 3',
        '5 . . . . . . . . . . . . . . . 3',
        '5 . . . . . . . . . . . . . . . 3',
        '5 5 5 5 4 4 4 . . . 6 6 6 . . . 3',
        '. . . . . . 4 . . . 6 . 6 . . . 3',
        '1 1 1 1 4 4 4 . . . 6 . 6 . . . 3',
        '1 . . . . . . . . . 2 . 1 . . . 1',
        '1 . P . . . . . . . 2 . 1 . . . 1',
        '1 . . . . . . . . . 2 . 1 . . . 1',
        '1 1 1 1 1 1 1 2 2 2 2 . 1 1 1 1 1',
    ),

    'maze_xxl': (
        '7 7 7 7 * * * 6 6 6 * * * 9 9 9 9',
        '7 . . . . . . . . . . . . . . . 9',
        '7 . . . . . . . . . . . . . G . 9',
        '7 . . . . . . . . . . . . . . . 9',
        '* . . . 5 5 5 * * * * * * 9 9 9 9',
        '* . . . 5 . . . . . . . . . . . .',
        '* . . . 5 5 5 * * * * * * 3 3 3 3',
        '8 . . . . . . . . . . . . . . . 3',
        '8 . . . . . . . . . . . . . . . 3',
        '8 . . . . . . . . . . . . . . . 3',
        '8 8 8 8 * * * * * * 4 4 4 . . . *',
        '. . . . . . . . . . . . 4 . . . *',
        '1 1 1 1 * * * * * * 4 4 4 . . . *',
        '1 . . . . . . . . . . . . . . . 2',
        '1 . P . . . . . . . . . . . . . 2',
        '1 . . . . . . . . . . . . . . . 2',
        '1 1 1 1 * * * 6 6 6 * * * 2 2 2 2',
    ),

    'empty': (
        '. . . . . . . . . . . . . . . . .',
        '. . . . . . . . . . . . . . . . .',
        '. . . . . . . . . . . . . . . . .',
        '. . . . . . . . . . . . . . . . .',
        '. . . . . . . . . . . . . . . . .',
        '. . . . . . . . . . . . . . . . .',
        '. . . . . . . . . . . . . . . . .',
        '. . . . . . . . . . . . . . . . .',
        '. . . . . . . . . . . . . . . . .',
        '. . . . . . . . . . . . . . . . .',
        '. . . . . . . . . . . . . . . . .',
        '. . . . . . . . . . . . . . . . .',
        '. . . . . . . . . . . . . . . . .',
        '. . . . . . . . . . . . . . . . .',
        '. . . . . . . . . . . . . . . . .',
        '. . . . . . . . . . . . . . . . .',
        '. . . . . . . . . . . . . . . . .',
    ),

}
