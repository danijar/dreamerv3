import functools

import elements
import embodied
import numpy as np


def _import_myosuite():
  import importlib
  import pathlib
  import sys
  import myosuite
  if getattr(myosuite, '__file__', None) is not None:
    return myosuite
  bad_parents = {
      str(pathlib.Path(p).parent) for p in getattr(myosuite, '__path__', [])}
  saved_path = sys.path[:]
  try:
    sys.path = [p for p in sys.path if p not in bad_parents]
    for key in [
        k for k in sys.modules
        if k == 'myosuite' or k.startswith('myosuite.')]:
      del sys.modules[key]
    return importlib.import_module('myosuite')
  finally:
    sys.path = saved_path


class MyoSuite(embodied.Env):

  def __init__(
      self,
      task,
      repeat=1,
      proprio=True,
      image=False,
      log_image=False,
      size=(64, 64),
      camera=None,
      device_id=0,
      seed=None,
      **kwargs,
  ):
    assert proprio or image, 'At least one of proprio or image must be True.'
    myosuite = _import_myosuite()
    from myosuite.utils import gym

    registry = getattr(gym.envs, 'registry', {})
    if task not in registry and f'myo{task}' in registry:
      task = f'myo{task}'

    self._task = task
    if hasattr(myosuite, 'make_env'):
      self._env = myosuite.make_env(task, **kwargs)
    else:
      self._env = gym.make(task, **kwargs)
    self._repeat = int(repeat)
    self._proprio = bool(proprio)
    self._image = bool(image)
    self._log_image = bool(log_image)
    self._size = tuple(size)
    self._camera = camera if camera not in (-1, '') else None
    self._device_id = int(device_id)
    self._seed = seed
    self._renderer = None
    self._done = True

  @property
  def env(self):
    return self._env

  @functools.cached_property
  def obs_space(self):
    spaces = {
        'reward': elements.Space(np.float32),
        'is_first': elements.Space(bool),
        'is_last': elements.Space(bool),
        'is_terminal': elements.Space(bool),
        'log/solved': elements.Space(np.float32),
        'log/rwd_dense': elements.Space(np.float32),
        'log/rwd_sparse': elements.Space(np.float32),
    }
    if self._proprio:
      obs_sp = self._env.observation_space
      if hasattr(obs_sp, 'spaces'):
        for key, sp in obs_sp.spaces.items():
          spaces[key.replace('/', '_')] = elements.Space(np.float32, sp.shape)
      else:
        spaces['proprio'] = elements.Space(np.float32, obs_sp.shape)
    if self._image:
      spaces['image'] = elements.Space(np.uint8, (*self._size, 3))
    elif self._log_image:
      spaces['log/image'] = elements.Space(np.uint8, (*self._size, 3))
    return spaces

  @functools.cached_property
  def act_space(self):
    act_sp = self._env.action_space
    return {
        'action': elements.Space(
            np.float32, act_sp.shape,
            np.asarray(act_sp.low, np.float32),
            np.asarray(act_sp.high, np.float32)),
        'reset': elements.Space(bool),
    }

  def step(self, action):
    if action['reset'] or self._done:
      reset_kw = {'seed': int(self._seed)} if self._seed is not None else {}
      self._seed = None
      res = self._env.reset(**reset_kw)
      if isinstance(res, tuple) and len(res) == 2 and isinstance(res[1], dict):
        obs, info = res
      else:
        obs, info = res, {}
      self._done = False
      return self._obs(obs, 0.0, info, is_first=True)

    act = np.asarray(action['action'], self._env.action_space.dtype)
    reward = 0.0
    terminated = False
    truncated = False
    info = {}
    for _ in range(self._repeat):
      res = self._env.step(act)
      if len(res) == 5:
        obs, rew, term, trunc, info = res
        terminated = terminated or bool(term)
        truncated = truncated or bool(trunc)
      else:
        obs, rew, done, info = res
        terminated = terminated or bool(info.get('is_terminal', done))
        truncated = truncated or (bool(done) and not terminated)
      reward += float(rew)
      if terminated or truncated:
        break

    self._done = terminated or truncated
    return self._obs(
        obs, reward, info,
        is_last=self._done,
        is_terminal=terminated)

  def _obs(
      self, obs, reward, info,
      is_first=False, is_last=False, is_terminal=False):
    info = info or {}
    rwd_dict = info.get('rwd_dict', {})
    solved = info.get('solved', rwd_dict.get('solved', 0.0))
    rwd_dense = info.get('rwd_dense', rwd_dict.get('dense', reward))
    rwd_sparse = info.get('rwd_sparse', rwd_dict.get('sparse', 0.0))
    out = {
        'reward': np.float32(reward),
        'is_first': bool(is_first),
        'is_last': bool(is_last),
        'is_terminal': bool(is_terminal),
        'log/solved': np.float32(np.asarray(solved).mean()),
        'log/rwd_dense': np.float32(np.asarray(rwd_dense).mean()),
        'log/rwd_sparse': np.float32(np.asarray(rwd_sparse).mean()),
    }
    if self._proprio:
      if isinstance(obs, dict):
        for key, val in obs.items():
          out[key.replace('/', '_')] = np.asarray(val, np.float32)
      else:
        out['proprio'] = np.asarray(obs, np.float32)
    if self._image:
      out['image'] = self._render()
    elif self._log_image:
      out['log/image'] = self._render()
    return out

  def _render(self):
    unwrapped = self._env.unwrapped
    height, width = self._size
    if hasattr(unwrapped, 'mj_renderer'):
      frame = unwrapped.mj_renderer.render_offscreen(
          width=width,
          height=height,
          camera_id=self._camera,
          device_id=self._device_id,
      )
    elif hasattr(unwrapped, 'sim') and hasattr(unwrapped.sim, 'renderer'):
      kw = {'width': width, 'height': height, 'device_id': self._device_id}
      if self._camera is not None:
        kw['camera_id'] = self._camera
      frame = unwrapped.sim.renderer.render_offscreen(**kw)
    elif hasattr(unwrapped, 'model') and hasattr(unwrapped, 'data'):
      import mujoco
      if self._renderer is None:
        self._renderer = mujoco.Renderer(
            unwrapped.model, height=height, width=width)
      cam = -1 if self._camera is None else self._camera
      self._renderer.update_scene(unwrapped.data, camera=cam)
      frame = self._renderer.render()
    else:
      frame = self._env.render()
    return np.asarray(frame, np.uint8)

  def close(self):
    if self._renderer is not None:
      try:
        self._renderer.close()
      except Exception:
        pass
      self._renderer = None
    try:
      unwrapped = self._env.unwrapped
      if hasattr(unwrapped, 'mj_renderer') and hasattr(
          unwrapped.mj_renderer, 'close'):
        unwrapped.mj_renderer.close()
    except Exception:
      pass
    try:
      self._env.close()
    except Exception:
      pass
