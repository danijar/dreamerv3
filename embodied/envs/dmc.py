import functools
import os

import embodied
import numpy as np


class DMC(embodied.Env):

  DEFAULT_CAMERAS = dict(
      quadruped=2,
      locom_rodent=4,
  )

  def __init__(
      self, env, repeat=1, size=(64, 64), image=True, camera=-1):
    if 'MUJOCO_GL' not in os.environ:
      os.environ['MUJOCO_GL'] = 'egl'
    if isinstance(env, str):
      domain, task = env.split('_', 1)
      if camera == -1:
        camera = self.DEFAULT_CAMERAS.get(domain, 0)
      if domain == 'cup':  # Only domain with multiple words.
        domain = 'ball_in_cup'
      if domain == 'manip':
        from dm_control import manipulation
        env = manipulation.load(task + '_vision')
      elif domain == 'locom':
        # camera 0: topdown map
        # camera 2: shoulder
        # camera 4: topdown tracking
        # camera 5: eyes
        from dm_control.locomotion.examples import basic_rodent_2020
        env = getattr(basic_rodent_2020, task)()
      else:
        from dm_control import suite
        env = suite.load(domain, task)
    self._dmenv = env
    from . import from_dm
    self._env = from_dm.FromDM(self._dmenv)
    self._env = embodied.wrappers.ExpandScalars(self._env)
    self._env = embodied.wrappers.ActionRepeat(self._env, repeat)
    self._size = size
    self._image = image
    self._camera = camera

  @functools.cached_property
  def obs_space(self):
    spaces = self._env.obs_space.copy()
    key = 'image' if self._image else 'log_image'
    spaces[key] = embodied.Space(np.uint8, self._size + (3,))
    return spaces

  @functools.cached_property
  def act_space(self):
    return self._env.act_space

  def step(self, action):
    for key, space in self.act_space.items():
      if not space.discrete:
        assert np.isfinite(action[key]).all(), (key, action[key])
    obs = self._env.step(action)
    key = 'image' if self._image else 'log_image'
    obs[key] = self._dmenv.physics.render(*self._size, camera_id=self._camera)
    for key, space in self.obs_space.items():
      if np.issubdtype(space.dtype, np.floating):
        assert np.isfinite(obs[key]).all(), (key, obs[key])
    return obs
