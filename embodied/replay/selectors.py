from collections import defaultdict, deque

import numpy as np

from . import sampletree


class Fifo:

  def __init__(self):
    self.queue = deque()

  def __call__(self):
    return self.queue[0]

  def __setitem__(self, key, stepids):
    self.queue.append(key)

  def __delitem__(self, key):
    if self.queue[0] == key:
      self.queue.popleft()
    else:
      # This is very slow but typically not used.
      self.queue.remove(key)


class Uniform:

  def __init__(self, seed=0):
    self.indices = {}
    self.keys = []
    self.rng = np.random.default_rng(seed)

  def __call__(self):
    index = self.rng.integers(0, len(self.keys)).item()
    return self.keys[index]

  def __setitem__(self, key, stepids):
    self.indices[key] = len(self.keys)
    self.keys.append(key)

  def __delitem__(self, key):
    index = self.indices.pop(key)
    last = self.keys.pop()
    if index != len(self.keys):
      self.keys[index] = last
      self.indices[last] = index


class Recency:

  def __init__(self, uprobs, seed=0):
    assert uprobs[0] >= uprobs[-1], uprobs
    self.uprobs = uprobs
    self.tree = self._build(uprobs)
    self.rng = np.random.default_rng(seed)
    self.step = 0
    self.steps = {}
    self.items = {}

  def __call__(self):
    for retry in range(10):
      try:
        age = self._sample(self.tree, self.rng)
        if len(self.items) < len(self.uprobs):
          age = int(age / len(self.uprobs) * len(self.items))
        return self.items[self.step - 1 - age]
      except KeyError:
        # Item might have been deleted very recently.
        if retry < 9:
          import time
          time.sleep(0.01)
        else:
          raise

  def __setitem__(self, key, stepids):
    self.steps[key] = self.step
    self.items[self.step] = key
    self.step += 1

  def __delitem__(self, key):
    step = self.steps.pop(key)
    del self.items[step]

  def _sample(self, tree, rng, bfactor=16):
    path = []
    for level, prob in enumerate(tree):
      segment = prob[*path]
      path += (rng.choice(len(segment), p=segment),)
    index = sum(
        index * bfactor ** (len(tree) - level - 1)
        for level, index in enumerate(path))
    return index

  def _build(self, uprobs, bfactor=16):
    assert np.isfinite(uprobs).all(), uprobs
    assert (uprobs >= 0).all(), uprobs
    depth = int(np.ceil(np.log(len(uprobs)) / np.log(bfactor)))
    size = bfactor ** depth
    uprobs = np.concatenate([uprobs, np.zeros(size - len(uprobs))])
    tree = [uprobs]
    for level in reversed(range(depth - 1)):
      tree.insert(0, tree[0].reshape((-1, bfactor)).sum(-1))
    for level, prob in enumerate(tree):
      prob = prob.reshape([bfactor] * (1 + level))
      total = prob.sum(-1, keepdims=True)
      with np.errstate(divide='ignore', invalid='ignore'):
        tree[level] = np.where(total, prob / total, prob)
    return tree


class Prioritized:

  def __init__(
      self, exponent=1.0, initial=1.0, zero_on_sample=False,
      maxfrac=0.0, branching=16, seed=0):
    assert 0 <= maxfrac <= 1, maxfrac
    self.exponent = float(exponent)
    self.initial = float(initial)
    self.zero_on_sample = zero_on_sample
    self.maxfrac = maxfrac
    self.tree = sampletree.SampleTree(branching, seed)
    self.prios = defaultdict(lambda: self.initial)
    self.stepitems = defaultdict(list)
    self.items = {}

  def prioritize(self, stepids, priorities):
    if not isinstance(stepids[0], bytes):
      stepids = [x.tobytes() for x in stepids]
    for stepid, priority in zip(stepids, priorities):
      try:
        self.prios[stepid] = priority
      except KeyError:
        print('Ignoring priority update for removed time step.')
        pass
    items = []
    for stepid in stepids:
      items += self.stepitems[stepid]
    for key in list(set(items)):
      try:
        self.tree.update(key, self._aggregate(key))
      except KeyError:
        print('Ignoring tree update for removed time step.')
        pass

  def __call__(self):
    key = self.tree.sample()
    if self.zero_on_sample:
      zeros = [0.0] * len(self.items[key])
      self.prioritize(self.items[key], zeros)
    return key

  def __setitem__(self, key, stepids):
    if not isinstance(stepids[0], bytes):
      stepids = [x.tobytes() for x in stepids]
    self.items[key] = stepids
    [self.stepitems[stepid].append(key) for stepid in stepids]
    self.tree.insert(key, self._aggregate(key))

  def __delitem__(self, key):
    self.tree.remove(key)
    stepids = self.items.pop(key)
    for stepid in stepids:
      stepitems = self.stepitems[stepid]
      stepitems.remove(key)
      if not stepitems:
        del self.stepitems[stepid]
        del self.prios[stepid]

  def _aggregate(self, key):
    # Both list comprehensions in this function are a performance bottleneck
    # because they are called very often.
    prios = [self.prios[stepid] for stepid in self.items[key]]
    if self.exponent != 1.0:
      prios = [x ** self.exponent for x in prios]
    mean = sum(prios) / len(prios)
    if self.maxfrac:
      return self.maxfrac * max(prios) + (1 - self.maxfrac) * mean
    else:
      return mean


class Mixture:

  def __init__(self, selectors, fractions, seed=0):
    assert set(selectors.keys()) == set(fractions.keys())
    assert sum(fractions.values()) == 1, fractions
    for key, frac in list(fractions.items()):
      if not frac:
        selectors.pop(key)
        fractions.pop(key)
    keys = sorted(selectors.keys())
    self.selectors = [selectors[key] for key in keys]
    self.fractions = np.array([fractions[key] for key in keys], np.float32)
    self.rng = np.random.default_rng(seed)

  def __call__(self):
    return self.rng.choice(self.selectors, p=self.fractions)()

  def __setitem__(self, key, stepids):
    for selector in self.selectors:
      selector[key] = stepids

  def __delitem__(self, key):
    for selector in self.selectors:
      del selector[key]

  def prioritize(self, stepids, priorities):
    for selector in self.selectors:
      if hasattr(selector, 'prioritize'):
        selector.prioritize(stepids, priorities)
