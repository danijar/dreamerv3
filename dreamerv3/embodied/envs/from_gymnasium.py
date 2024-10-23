# Source: https://github.com/qxcv/dreamerv3
# 
# From_gym.py converted to work with Gymnasium. Differences:
#
# - gym.* -> gymnasium.*
# - Deals with .step() returning a tuple of (obs, reward, terminated, truncated,
#   info) rather than (obs, reward, done, info).
# - Also deals with .reset() returning a tuple of (obs, info) rather than just
#   obs.
# - Passes render_mode='rgb_array' to gymnasium.make() rather than .render().
# - A bunch of minor/irrelevant type checking changes that stopped pyright from
#   complaining (these have no functional purpose, I'm just a completionist who
#   doesn't like red squiggles).

import functools
from typing import Any, Generic, TypeVar, Union, cast, Dict

import gymnasium # type: ignore # Users should install gymnasium themselves
import numpy as np

from ... import embodied

U = TypeVar('U')
V = TypeVar('V')


class FromGymnasium(embodied.Env, Generic[U, V]):
  def __init__(self, env: Union[str, gymnasium.Env[U, V]], obs_key='image', act_key='action', **kwargs):
    seed = kwargs.pop('seed', None)
    if isinstance(env, str):
      self._env: gymnasium.Env[U, V] = gymnasium.make(env, render_mode="rgb_array", **kwargs)
    else:
      assert not kwargs, kwargs
      assert env.render_mode == "rgb_array", f"render_mode must be rgb_array, got {env.render_mode}"
      self._env = env
    self._env.reset(seed=seed)
    self._obs_dict = hasattr(self._env.observation_space, 'spaces')
    self._act_dict = hasattr(self._env.action_space, 'spaces')
    self._obs_key = obs_key
    self._act_key = act_key
    self._done = True
    self._info = None

  @property
  def info(self):
    return self._info

  @functools.cached_property # type: ignore
  def obs_space(self):
    if self._obs_dict:
      # cast is here to stop type checkers from complaining (we already check
      # that .spaces attr exists in __init__ as a proxy for the type check)
      obs_space = cast(gymnasium.spaces.Dict, self._env.observation_space)
      spaces = obs_space.spaces
    else:
      spaces = {self._obs_key: self._env.observation_space}
    spaces = {k: self._convert(v) for k, v in spaces.items()}
    return {
        **spaces,
        'reward': embodied.Space(np.float32),
        'is_first': embodied.Space(bool),
        'is_last': embodied.Space(bool),
        'is_terminal': embodied.Space(bool),
    }

  @functools.cached_property # type: ignore
  def act_space(self):
    if self._act_dict:
      act_space = cast(gymnasium.spaces.Dict, self._env.action_space)
      spaces = act_space.spaces
    else:
      spaces = {self._act_key: self._env.action_space}
    spaces = {k: self._convert(v) for k, v in spaces.items()}
    spaces['reset'] = embodied.Space(bool)
    return spaces

  def step(self, action):
    if action['reset'] or self._done:
      self._done = False
      obs, self._info = self._env.reset()
      return self._obs(obs, 0.0, is_first=True)
    if self._act_dict:
      gymnasium_action = cast(V, self._unflatten(action))
    else:
      gymnasium_action = cast(V, action[self._act_key])
    obs, reward, terminated, truncated, self._info = self._env.step(gymnasium_action)
    self._done = terminated or truncated
    return self._obs(
        obs, reward,
        is_last=bool(self._done),
        is_terminal=bool(terminated))

  def _obs(
      self, obs, reward, is_first=False, is_last=False, is_terminal=False):
    if not self._obs_dict:
      obs = {self._obs_key: obs}
    obs = self._flatten(obs)
    np_obs: Dict[str, Any] = {k: np.asarray(v) for k, v in obs.items()}
    np_obs.update(
        reward=np.float32(reward),
        is_first=is_first,
        is_last=is_last,
        is_terminal=is_terminal)
    return np_obs

  def render(self):
    image = self._env.render()
    assert image is not None
    return image

  def close(self):
    try:
      self._env.close()
    except Exception:
      pass

  def _flatten(self, nest, prefix=None):
    result = {}
    for key, value in nest.items():
      key = prefix + '/' + key if prefix else key
      if isinstance(value, gymnasium.spaces.Dict):
        value = value.spaces
      if isinstance(value, dict):
        result.update(self._flatten(value, key))
      else:
        result[key] = value
    return result

  def _unflatten(self, flat):
    result = {}
    for key, value in flat.items():
      parts = key.split('/')
      node = result
      for part in parts[:-1]:
        if part not in node:
          node[part] = {}
        node = node[part]
      node[parts[-1]] = value
    return result

  def _convert(self, space):
    if hasattr(space, 'n'):
      return embodied.Space(np.int32, (), 0, space.n)
    return embodied.Space(space.dtype, space.shape, space.low, space.high)
