import numpy as np


class RandomAgent:

  def __init__(self, obs_space, act_space):
    self.obs_space = obs_space
    self.act_space = act_space

  def init_policy(self, batch_size):
    return ()

  def init_train(self, batch_size):
    return ()

  def init_report(self, batch_size):
    return ()

  def policy(self, obs, carry=(), mode='train'):
    batch_size = len(obs['is_first'])
    act = {
        k: np.stack([v.sample() for _ in range(batch_size)])
        for k, v in self.act_space.items() if k != 'reset'}
    outs = {}
    return act, outs, carry

  def train(self, data, carry=()):
    outs = {}
    metrics = {}
    return outs, carry, metrics

  def report(self, data, carry=()):
    report = {}
    return report, carry

  def dataset(self, generator):
    return generator()

  def save(self):
    return None

  def load(self, data=None):
    pass
