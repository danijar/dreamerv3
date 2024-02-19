import collections

import numpy as np

from .basics import convert


class Driver:

  _CONVERSION = {
      np.floating: np.float32,
      np.signedinteger: np.int32,
      np.uint8: np.uint8,
      bool: bool,
  }

  def __init__(self, env, **kwargs):
    assert len(env) > 0
    self._env = env
    self._kwargs = kwargs
    self._on_steps = []
    self._on_episodes = []
    self.reset()

  def reset(self):
    self._acts = {
        k: convert(np.zeros((len(self._env),) + v.shape, v.dtype))
        for k, v in self._env.act_space.items()}
    self._acts['reset'] = np.ones(len(self._env), bool)
    self._eps = [collections.defaultdict(list) for _ in range(len(self._env))]
    self._state = None

  def on_step(self, callback):
    self._on_steps.append(callback)

  def on_episode(self, callback):
    self._on_episodes.append(callback)

  def __call__(self, policy, steps=0, episodes=0):
    step, episode = 0, 0
    while step < steps or episode < episodes:
      step, episode = self._step(policy, step, episode)

  def _step(self, policy, step, episode):
    assert all(len(x) == len(self._env) for x in self._acts.values() if not isinstance(x, dict))
    assert all(len(x2) == len(self._env) for x in self._acts.values() if isinstance(x, dict) for x2 in x.values() )
    acts = {k: v for k, v in self._acts.items() if not k.startswith('log_')}
    obs = self._env.step(acts)
    obs = {k: convert(v) for k, v in obs.items()}
    assert all(len(x) == len(self._env) for x in obs.values()), obs
    acts, self._state = policy(obs, self._state, **self._kwargs)
    if "action" in acts and isinstance(acts["action"], dict):
      for k, v in acts["action"].items():
        acts[k] = v
      del acts["action"]
    if "log_entropy" in acts and isinstance(acts["log_entropy"], dict):
      for k, v in acts["log_entropy"].items():
        acts[f"log_entropy_{k}"] = v
      del acts["log_entropy"]
    acts = {k: convert(v) for k, v in acts.items()}
    if obs['is_last'].any():
      mask = 1 - obs['is_last']
      acts = {k: v * self._expand(mask, len(v.shape)) for k, v in acts.items()}
    acts['reset'] = obs['is_last'].copy()
    self._acts = acts
    trns = {**obs, **acts}
    if obs['is_first'].any():
      for i, first in enumerate(obs['is_first']):
        if first:
          self._eps[i].clear()
    for i in range(len(self._env)):
      # if "action" in trns and isinstance(trns["action"], dict):
      #   out = {}
      #   for k, v in trns.items():
      #     if type(v) is not dict:
      #       out[k] = v[i]
      #     else:
      #       out[k] = {k2: v2[i] for k2, v2 in v.items()}
      #   trn = out
      # else:
      trn = {k: v[i] for k, v in trns.items()}
      [self._eps[i][k].append(v) for k, v in trn.items()]
      [fn(trn, i, **self._kwargs) for fn in self._on_steps]
      step += 1
    if obs['is_last'].any():
      for i, done in enumerate(obs['is_last']):
        if done:
          ep = {k: convert(v) for k, v in self._eps[i].items()}
          # ep = {}
          # for k, v in self._eps[i].items():
          #   if isinstance(v[0], dict):  # Action is a list of dicts
          #     if k not in ep:
          #       ep[k] = []
          #     for act_dict in v:
          #       ep[k].append({k2: convert(v2) for k2, v2 in act_dict.items()})
          #   else:
          #     ep[k] = convert(v)
          # ep = {k: convert(v) for k, v in self._eps[i].items() if not isinstance(v[0], dict)}
          [fn(ep.copy(), i, **self._kwargs) for fn in self._on_episodes]
          episode += 1
    return step, episode

  def _expand(self, value, dims):
    while len(value.shape) < dims:
      value = value[..., None]
    return value
