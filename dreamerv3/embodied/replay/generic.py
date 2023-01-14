import time
from collections import defaultdict, deque
from functools import partial as bind

import embodied
import numpy as np

from . import saver


class Generic:

  def __init__(
      self, length, capacity, remover, sampler, limiter, directory,
      overlap=None, online=False, chunks=1024):
    assert capacity is None or 1 <= capacity
    self.length = length
    self.capacity = capacity
    self.remover = remover
    self.sampler = sampler
    self.limiter = limiter
    self.stride = 1 if overlap is None else length - overlap
    self.streams = defaultdict(bind(deque, maxlen=length))
    self.counters = defaultdict(int)
    self.table = {}
    self.online = online
    if self.online:
      self.online_queue = deque()
      self.online_stride = length
      self.online_counters = defaultdict(int)
    self.saver = directory and saver.Saver(directory, chunks)
    self.load()

  def __len__(self):
    return len(self.table)

  @property
  def stats(self):
    return {'size': len(self)}

  def add(self, step, worker=0):
    step = {k: v for k, v in step.items() if not k.startswith('log_')}
    step['id'] = np.asarray(embodied.uuid(step.get('id')))
    stream = self.streams[worker]
    stream.append(step)
    self.saver and self.saver.add(step, worker)
    self.counters[worker] += 1
    if self.online:
      self.online_counters[worker] += 1
      if len(stream) >= self.length and (
          self.online_counters[worker] >= self.online_stride):
        self.online_queue.append(tuple(stream))
        self.online_counters[worker] = 0
    if len(stream) < self.length or self.counters[worker] < self.stride:
      return
    self.counters[worker] = 0
    key = embodied.uuid()
    seq = tuple(stream)
    wait(self.limiter.want_insert, lambda: 'Replay insert is waiting')
    self.table[key] = seq
    self.remover[key] = seq
    self.sampler[key] = seq
    while self.capacity and len(self) > self.capacity:
      self._remove(self.remover())

  def _sample(self):
    wait(self.limiter.want_sample, lambda: 'Replay sample is waiting')
    if self.online:
      try:
        seq = self.online_queue.popleft()
      except IndexError:
        seq = self.table[self.sampler()]
    else:
      seq = self.table[self.sampler()]
    seq = {k: [step[k] for step in seq] for k in seq[0]}
    seq = {k: embodied.convert(v) for k, v in seq.items()}
    if 'is_first' in seq:
      seq['is_first'][0] = True
    return seq

  def _remove(self, key):
    wait(self.limiter.want_remove, lambda: 'Replay remove is waiting')
    del self.table[key]
    del self.remover[key]
    del self.sampler[key]

  def dataset(self):
    while True:
      yield self._sample()

  def prioritize(self, ids, prios):
    if hasattr(self.sampler, 'prioritize'):
      self.sampler.prioritize(ids, prios)

  def save(self, wait=False):
    if not self.saver:
      return
    self.saver.save(wait)

  def load(self, data=None):
    if not self.saver:
      return
    workers = set()
    for step, worker in self.saver.load(self.capacity, self.length):
      workers.add(worker)
      self.add(step, worker)
    for worker in workers:
      del self.streams[worker]
      del self.counters[worker]


def wait(predicate, message):
  counter = 0
  while not predicate():
    if counter % 100 == 0:
      print(message())
    time.sleep(0.1)
    counter += 1
