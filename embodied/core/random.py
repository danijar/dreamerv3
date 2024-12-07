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

  def policy(self, carry, obs, mode='train'):
    batch_size = len(obs['is_first'])
    act = {
        k: np.stack([v.sample() for _ in range(batch_size)])
        for k, v in self.act_space.items() if k != 'reset'}
    return carry, act, {}

  def train(self, carry, data):
    return carry, {}, {}

  def report(self, carry, data):
    return carry, {}

  def stream(self, st):
    return st

  def save(self):
    return None

  def load(self, data=None):
    pass
