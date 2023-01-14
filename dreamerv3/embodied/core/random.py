import numpy as np


class RandomAgent:

  def __init__(self, act_space):
    self.act_space = act_space

  def policy(self, obs, state=None, mode='train'):
    batch_size = len(next(iter(obs.values())))
    act = {
        k: np.stack([v.sample() for _ in range(batch_size)])
        for k, v in self.act_space.items() if k != 'reset'}
    return act, state
