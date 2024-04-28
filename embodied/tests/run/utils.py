import time

import embodied
import numpy as np


class TestAgent:

  def __init__(self, obs_space, act_space, addr=None):
    self.obs_space = obs_space
    self.act_space = act_space
    if addr:
      self.client = embodied.distr.Client(addr, connect=True)
      self.should_stats = embodied.when.Clock(1)
    else:
      self.client = None
    self._stats = {
        'env_steps': 0, 'replay_steps': 0, 'reports': 0,
        'saves': 0, 'loads': 0, 'created': time.time(),
    }

  def _watcher(self):
    while True:
      if self.queue.empty():
        self.queue.put(self.stats())
      else:
        time.sleep(0.01)

  def stats(self):
    stats = self._stats.copy()
    stats['lifetime'] = time.time() - stats.pop('created')
    return stats

  def init_policy(self, batch_size):
    return (np.zeros(batch_size),)

  def init_train(self, batch_size):
    return (np.zeros(batch_size),)

  def init_report(self, batch_size):
    return ()

  def policy(self, obs, carry, mode='train'):
    B = len(obs['is_first'])
    self._stats['env_steps'] += B
    carry, = carry
    carry = np.asarray(carry)

    assert carry.shape == (B,)
    assert not any(k.startswith('log_') for k in obs.keys())

    target = (carry + 1) * (1 - obs['is_first'])
    assert (obs['step'] == target).all()
    carry = target

    if self.client and self.should_stats():
      self.client.report(self.stats())

    act = {
        k: np.stack([v.sample() for _ in range(B)])
        for k, v in self.act_space.items() if k != 'reset'}
    return act, {}, (carry,)

  def train(self, data, carry):
    B, T = data['step'].shape
    carry, = carry
    assert carry.shape == (B,)
    assert not any(k.startswith('log_') for k in data.keys())
    self._stats['replay_steps'] += B * T
    for t in range(T):
      current = data['step'][:, t]
      reset = data['is_first'][:, t]
      target = (1 - reset) * (carry + 1) + reset * current
      assert (current == target).all()
      carry = current

    outs = {}
    metrics = {}
    return outs, (carry,), metrics

  def report(self, data, carry):
    self._stats['reports'] += 1
    return {
        'scalar': np.float32(0),
        'vector': np.zeros(10),
        'image1': np.zeros((64, 64, 1)),
        'image3': np.zeros((64, 64, 3)),
        'video': np.zeros((10, 64, 64, 3)),
    }, carry

  def dataset(self, generator):
    return generator()

  def save(self):
    self._stats['saves'] += 1
    return self._stats

  def load(self, data):
    self._stats = data
    self._stats['loads'] += 1
