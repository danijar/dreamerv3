import numpy as np

from . import base


class BatchEnv(base.Env):

  def __init__(self, envs, parallel):
    assert all(len(env) == 0 for env in envs)
    assert len(envs) > 0
    self._envs = envs
    self._parallel = parallel
    self._keys = list(self.obs_space.keys())

  @property
  def obs_space(self):
    return self._envs[0].obs_space

  @property
  def act_space(self):
    return self._envs[0].act_space

  def __len__(self):
    return len(self._envs)

  def step(self, action):
    assert all(len(v) == len(self._envs) for v in action.values()), (
        len(self._envs), {k: v.shape for k, v in action.items()})
    obs = []
    for i, env in enumerate(self._envs):
      act = {k: v[i] for k, v in action.items()}
      obs.append(env.step(act))
    if self._parallel:
      obs = [ob() for ob in obs]
    return {k: np.array([ob[k] for ob in obs]) for k in obs[0]}

  def render(self):
    return np.stack([env.render() for env in self._envs])

  def close(self):
    for env in self._envs:
      try:
        env.close()
      except Exception:
        pass
