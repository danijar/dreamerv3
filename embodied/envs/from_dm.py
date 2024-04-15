import functools

import embodied
import numpy as np


class FromDM(embodied.Env):

  def __init__(self, env, obs_key='observation', act_key='action'):
    self._env = env
    obs_spec = self._env.observation_spec()
    act_spec = self._env.action_spec()
    self._obs_dict = isinstance(obs_spec, dict)
    self._act_dict = isinstance(act_spec, dict)
    self._obs_key = not self._obs_dict and obs_key
    self._act_key = not self._act_dict and act_key
    self._obs_empty = []
    self._done = True

  @functools.cached_property
  def obs_space(self):
    spec = self._env.observation_spec()
    spec = spec if self._obs_dict else {self._obs_key: spec}
    if 'reward' in spec:
      spec['obs_reward'] = spec.pop('reward')
    for key, value in spec.copy().items():
      if int(np.prod(value.shape)) == 0:
        self._obs_empty.append(key)
        del spec[key]
    spaces = {
        'reward': embodied.Space(np.float32),
        'is_first': embodied.Space(bool),
        'is_last': embodied.Space(bool),
        'is_terminal': embodied.Space(bool),
    }
    for key, value in spec.items():
      key = key.replace('/', '_')
      spaces[key] = self._convert(value)
    return spaces

  @functools.cached_property
  def act_space(self):
    spec = self._env.action_spec()
    spec = spec if self._act_dict else {self._act_key: spec}
    return {
        'reset': embodied.Space(bool),
        **{k or self._act_key: self._convert(v) for k, v in spec.items()},
    }

  def step(self, action):
    action = action.copy()
    reset = action.pop('reset')
    if reset or self._done:
      time_step = self._env.reset()
    else:
      action = action if self._act_dict else action[self._act_key]
      time_step = self._env.step(action)
    self._done = time_step.last()
    return self._obs(time_step)

  def _obs(self, time_step):
    if not time_step.first():
      assert time_step.discount in (0, 1), time_step.discount
    obs = time_step.observation
    obs = dict(obs) if self._obs_dict else {self._obs_key: obs}
    if 'reward' in obs:
      obs['obs_reward'] = obs.pop('reward')
    for key in self._obs_empty:
      del obs[key]
    obs = {k.replace('/', '_'): v for k, v in obs.items()}
    return dict(
        reward=np.float32(0.0 if time_step.first() else time_step.reward),
        is_first=time_step.first(),
        is_last=time_step.last(),
        is_terminal=False if time_step.first() else time_step.discount == 0,
        **obs,
    )

  def _convert(self, space):
    if hasattr(space, 'num_values'):
      return embodied.Space(space.dtype, (), 0, space.num_values)
    elif hasattr(space, 'minimum'):
      assert np.isfinite(space.minimum).all(), space.minimum
      assert np.isfinite(space.maximum).all(), space.maximum
      return embodied.Space(
          space.dtype, space.shape, space.minimum, space.maximum)
    else:
      return embodied.Space(space.dtype, space.shape, None, None)
