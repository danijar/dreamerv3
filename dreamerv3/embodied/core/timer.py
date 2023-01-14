import collections
import contextlib
import time

import numpy as np


class Timer:

  def __init__(self, columns=('frac', 'min', 'avg', 'max', 'count', 'total')):
    available = ('frac', 'avg', 'min', 'max', 'count', 'total')
    assert all(x in available for x in columns), columns
    self._columns = columns
    self._durations = collections.defaultdict(list)
    self._start = time.time()

  def reset(self):
    for timings in self._durations.values():
      timings.clear()
    self._start = time.time()

  @contextlib.contextmanager
  def scope(self, name):
    start = time.time()
    yield
    stop = time.time()
    self._durations[name].append(stop - start)

  def wrap(self, name, obj, methods):
    for method in methods:
      decorator = self.scope(f'{name}.{method}')
      setattr(obj, method, decorator(getattr(obj, method)))

  def stats(self, reset=True, log=False):
    metrics = {}
    metrics['duration'] = time.time() - self._start
    for name, durs in self._durations.items():
      available = {}
      available['count'] = len(durs)
      available['total'] = np.sum(durs)
      available['frac'] = np.sum(durs) / metrics['duration']
      if len(durs):
        available['avg'] = np.mean(durs)
        available['min'] = np.min(durs)
        available['max'] = np.max(durs)
      for key, value in available.items():
        if key in self._columns:
          metrics[f'{name}_{key}'] = value
    if log:
      self._log(metrics)
    if reset:
      self.reset()
    return metrics

  def _log(self, metrics):
    names = self._durations.keys()
    names = sorted(names, key=lambda k: -metrics[f'{k}_frac'])
    print('Timer:'.ljust(20), ' '.join(x.rjust(8) for x in self._columns))
    for name in names:
      values = [metrics[f'{name}_{col}'] for col in self._columns]
      print(f'{name.ljust(20)}', ' '.join((f'{x:8.4f}' for x in values)))
