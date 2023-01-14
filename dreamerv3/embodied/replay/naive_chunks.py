import concurrent.futures
import threading
import time
import uuid
from collections import deque, defaultdict
from functools import partial as bind

import numpy as np
import embodied

from . import chunk as chunklib


class NaiveChunks(embodied.Replay):

  def __init__(self, length, capacity=None, directory=None, chunks=1024, seed=0):
    assert 1 <= length <= chunks
    self.length = length
    self.capacity = capacity
    self.directory = directory and embodied.Path(directory)
    self.chunks = chunks
    self.buffers = {}
    self.rng = np.random.default_rng(seed)
    self.ongoing = defaultdict(bind(chunklib.Chunk, chunks))
    if directory:
      self.directory.mkdirs()
      self.workers = concurrent.futures.ThreadPoolExecutor(16)
      self.promises = deque()

  def __len__(self):
    return len(self.buffers) * self.length

  @property
  def stats(self):
    return {'size': len(self), 'chunks': len(self.buffers)}

  def add(self, step, worker=0):
    chunk = self.ongoing[worker]
    chunk.append(step)
    if len(chunk) >= self.chunks:
      self.buffers[chunk.uuid] = self.ongoing.pop(worker)
      self.promises.append(self.workers.submit(chunk.save, self.directory))
      for promise in [x for x in self.promises if x.done()]:
        promise.result()
        self.promises.remove(promise)
    while len(self) > self.capacity:
      del self.buffers[next(iter(self.buffers.keys()))]

  def _sample(self):
    counter = 0
    while not self.buffers:
      if counter % 100 == 0:
        print('Replay sample is waiting')
      time.sleep(0.1)
      counter += 1
    keys = tuple(self.buffers.keys())
    chunk = self.buffers[keys[self.rng.integers(0, len(keys))]]
    idx = self.rng.integers(0, len(chunk) - self.length + 1)
    seq = {k: chunk.data[k][idx: idx + self.length] for k in chunk.data.keys()}
    seq['is_first'][0] = True
    return seq

  def dataset(self):
    while True:
      yield self._sample()

  def save(self, wait=False):
    for chunk in self.ongoing.values():
      if chunk.length:
        self.promises.append(self.workers.submit(chunk.save, self.directory))
    if wait:
      [x.result() for x in self.promises]
      self.promises.clear()

  def load(self, data=None):
    filenames = chunklib.Chunk.scan(self.directory, capacity)
    if not filenames:
      return
    threads = min(len(filenames), 32)
    with concurrent.futures.ThreadPoolExecutor(threads) as executor:
      chunks = list(executor.map(chunklib.Chunk.load, filenames))
    self.buffers = {chunk.uuid: chunk for chunk in chunks}
