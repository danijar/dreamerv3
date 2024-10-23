import math
import operator
from collections import defaultdict
from functools import partial as bind

import numpy as np


class Agg:

  def __init__(self, maxlen=1e6):
    self.reducers = defaultdict(list)
    self.names = {}
    self.maxlen = int(maxlen)

  def add(self, key_or_dict, value=None, agg='default', prefix=None):
    if value is not None:
      self._add_single(key_or_dict, value, agg, prefix)
      return
    for key, value in key_or_dict.items():
      self._add_single(key, value, agg, prefix)

  def result(self, reset=True, prefix=None):
    metrics = {}
    for key, reducers in self.reducers.items():
      if len(reducers) == 1:
        metrics[key] = reducers[0].current()
      else:
        for name, reducer in zip(self.names[key], reducers):
          metrics[f'{key}/{name}'] = reducer.current()
    if prefix:
      metrics = {f'{prefix}/{k}': v for k, v in metrics.items()}
    reset and self.reset()
    return metrics

  def reset(self):
    self.reducers.clear()

  def _add_single(self, key, value, agg, prefix):
    key = f'{prefix}/{key}' if prefix else key
    reducers = self.reducers[key]
    if reducers:
      for reducer in reducers:
        reducer.update(value)
      return
    if agg == 'default':
      agg = 'avg' if np.asarray(value).ndim <= 1 else 'last'
    if isinstance(agg, str):
      aggs = (agg,)
      self.names[key] = None
    else:
      aggs = agg
      self.names[key] = aggs
    for agg in aggs:
      if agg == 'avg':
        reducer = Mean(value)
      elif agg == 'sum':
        reducer = Sum(value)
      elif agg == 'min':
        reducer = Min(value)
      elif agg == 'max':
        reducer = Max(value)
      elif agg == 'stack':
        reducer = Stack(value, self.maxlen)
      elif agg == 'last':
        reducer = Last(value)
      else:
        raise ValueError(agg)
      reducers.append(reducer)


class Reducer:

  def __init__(self, scalar_fn, array_fn, initial):
    self.is_scalar = isinstance(initial, (int, float))
    self.fn = scalar_fn if self.is_scalar else array_fn
    self.interm = self._input(initial)
    self.count = 1

  def update(self, value):
    value = self._input(value)
    if self._isnan(value):
      return
    if self._isnan(self.interm):
      self.interm = value
      return
    self.interm = self.fn(self.interm, value)
    self.count += 1

  def current(self):
    return np.array(self.interm)

  def _input(self, value):
    if self.is_scalar:
      return value
    else:
      return np.asarray(value, np.float64)

  def _isnan(self, value):
    if self.is_scalar:
      return math.isnan(value)
    else:
      return np.isnan(value).any()


class Mean:

  def __init__(self, initial):
    self.reducer = Sum(initial)

  def update(self, value):
    self.reducer.update(value)

  def current(self):
    return self.reducer.current() / self.reducer.count


class Stack:

  def __init__(self, initial, maxlen=1e5):
    self.stack = [initial]
    self.maxlen = int(maxlen)

  def update(self, value):
    if len(self.stack) < self.maxlen:
      self.stack.append(value)

  def current(self):
    return np.stack(self.stack)


class Last:

  def __init__(self, initial):
    self.value = initial

  def update(self, value):
    self.value = value

  def current(self):
    return self.value


Sum = bind(
    Reducer, operator.add, lambda x, y: np.add(x, y, out=x, dtype=np.float64))
Min = bind(
    Reducer, min, lambda x, y: np.minimum(x, y, out=x, dtype=np.float64))
Max = bind(
    Reducer, max, lambda x, y: np.maximum(x, y, out=x, dtype=np.float64))
