from collections import deque

import numpy as np


class Fifo:

  def __init__(self):
    self.queue = deque()

  def __call__(self):
    return self.queue[0]

  def __setitem__(self, key, steps):
    self.queue.append(key)

  def __delitem__(self, key):
    if self.queue[0] == key:
      self.queue.popleft()
    else:
      # TODO: This branch is unused but very slow.
      self.queue.remove(key)


class Uniform:

  def __init__(self, seed=0):
    self.indices = {}
    self.keys = []
    self.rng = np.random.default_rng(seed)

  def __call__(self):
    index = self.rng.integers(0, len(self.keys)).item()
    return self.keys[index]

  def __setitem__(self, key, steps):
    self.indices[key] = len(self.keys)
    self.keys.append(key)

  def __delitem__(self, key):
    index = self.indices.pop(key)
    last = self.keys.pop()
    if index != len(self.keys):
      self.keys[index] = last
      self.indices[last] = index
