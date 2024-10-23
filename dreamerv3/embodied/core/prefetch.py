import queue as queuelib

import numpy as np

from . import timer
from .. import distr


class Prefetch:

  def __init__(self, source, transform=None, amount=1):
    self.source = source
    self.transform = transform
    self.queue = queuelib.Queue(amount)
    self.worker = distr.StoppableThread(self._worker, start=True)

  def close(self):
    self.worker.stop()

  def check(self):
    self.worker.check()

  def __iter__(self):
    return self

  def __call__(self):
    return self.__iter__()

  def __next__(self):
    return self.queue.get()

  def _worker(self, context):
    # it = iter(self.source)
    it = self.source()
    while context.running:
      with timer.section('prefetch_source'):
        data = next(it)
      if self.transform:
        with timer.section('prefetch_transform'):
          data = self.transform(data)
      self.queue.put(data)


class Batch:

  def __init__(self, sources, amount=1):
    self.sources = sources
    self.queue = queuelib.Queue(amount)
    self.worker = distr.StoppableThread(self._worker, start=True)

  def close(self):
    self.worker.stop()

  def __iter__(self):
    return self

  def __call__(self):
    return self.__iter__()

  def __next__(self):
    return self.queue.get()

  def _worker(self, context):
    its = [source() for source in self.sources]
    while context.running:
      with timer.section('batch_source'):
        data = [next(it) for it in its]
      with timer.section('batch_stack'):
        data = {k: np.stack([x[k] for x in data]) for k in data[0]}
      self.queue.put(data)
