import elements
import embodied
import numpy as np


class NetHack(embodied.Env):

  def __init__(self, task, size=(64, 64), max_episode_steps=5000, seed=None):
    import gymnasium as gym
    import nle
    from gymnasium.wrappers import TimeLimit

    env_name = f'NetHack{task.replace("_", "-")}-v0'
    try:
      base_env = gym.make(env_name)
    except Exception:
      base_env = gym.make('NetHackChallenge-v0')

    self._env = TimeLimit(base_env, max_episode_steps=max_episode_steps)
    self._seed = seed
    self._size = size
    self._done = True
    obs_space = self._env.observation_space

    self._blstats_shape = tuple(obs_space["blstats"].shape)

  @property
  def obs_space(self):
    return {
        'image': elements.Space(np.uint8, (*self._size, 3)),
        'blstats': elements.Space(np.float32, self._blstats_shape),
        'reward': elements.Space(np.float32),
        'is_first': elements.Space(bool),
        'is_last': elements.Space(bool),
        'is_terminal': elements.Space(bool),
    }

  @property
  def act_space(self):
    return {
        'action': elements.Space(np.int32, (), 0, self._env.action_space.n),
        'reset': elements.Space(bool),
    }

  def step(self, action):
    if action['reset'] or self._done:
      self._done = False
      obs, info = self._env.reset(seed=self._seed)
      return self._obs(obs, 0.0, is_first=True)

    obs, reward, terminated, truncated, info = self._env.step(action['action'])
    self._done = bool(terminated or truncated)
    return self._obs(
        obs, reward,
        is_last=self._done,
        is_terminal=bool(terminated),
    )

  def _obs(self, obs, reward, is_first=False, is_last=False, is_terminal=False):
    image = self._render_glyphs(obs['glyphs'])
    return dict(
        image=image,
        blstats=obs['blstats'].astype(np.float32),
        reward=np.float32(reward),
        is_first=is_first,
        is_last=is_last,
        is_terminal=is_terminal,
    )

  def _render_glyphs(self, glyphs):
    from PIL import Image
    h, w = glyphs.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[:, :, 0] = ((glyphs >> 0) & 0xFF).astype(np.uint8)
    rgb[:, :, 1] = ((glyphs >> 8) & 0xFF).astype(np.uint8)
    rgb[:, :, 2] = ((glyphs >> 4) & 0xFF).astype(np.uint8)
    image = Image.fromarray(rgb)
    image = image.resize(self._size, Image.BILINEAR)
    return np.array(image)

  def close(self):
    try:
      self._env.close()
    except Exception:
      pass