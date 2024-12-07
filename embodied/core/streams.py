import functools
import queue
import threading

import elements
import numpy as np
import portal

from . import base


class Stateless(base.Stream):

  def __init__(self, nextfn, *args, **kwargs):
    if not callable(nextfn) and hasattr(nextfn, '__next__'):
      nextfn = nextfn.__next__
    self.nextfn = functools.partial(nextfn, *args, **kwargs)

  def __iter__(self):
    return self

  def __next__(self):
    return self.nextfn()

  def save(self):
    return None

  def load(self, data):
    pass


class Prefetch(base.Stream):

  def __init__(self, source, transform=None, amount=1):
    self.source = iter(source) if hasattr(source, '__iter__') else source()
    self.transform = transform or (lambda x: x)
    self.state = self._getstate()
    self.requests = threading.Semaphore(amount)
    self.amount = amount
    self.queue = queue.Queue()
    self.worker = portal.Thread(self._worker)
    self.started = False

  def __iter__(self):
    assert not self.started
    self.worker.start()
    self.started = True
    return self

  def __next__(self):
    assert self.started
    result = self.queue.get()
    self.requests.release()
    if isinstance(result, str):
      raise RuntimeError(result)
    data, self.state = result
    return data

  def save(self):
    return self.state

  def load(self, state):
    if self.started:
      for _ in range(self.amount):
        self.queue.get()
    self.source.load(state)
    if self.started:
      self.requests.release(self.amount)

  def _worker(self):
    try:
      while True:
        self.requests.acquire()
        data = next(self.source)
        data = self.transform(data)
        state = self._getstate()
        self.queue.put((data, state))
    except Exception as e:
      self.queue.put(str(e))
      raise

  def _getstate(self):
    if hasattr(self.source, 'save'):
      return self.source.save()
    else:
      return None


class Consec(base.Stream):

  """
  Example:

  length = 3
  consec = 3
  prefix = 2

  source:   0 1 2 3 4 5 6 7 8 9 10
  chunk 1:  p-p-#-#-#
  chunk 2:        p-p-#-#-#
  chunk 3:              p-p-#-#-#
  """

  def __init__(
      self, source, length, consec, prefix=0, strict=True, contiguous=False):
    self.source = source
    self.length = length
    self.consec = consec
    self.prefix = prefix
    self.strict = strict
    self.contiguous = contiguous
    self.index = 0
    self.current = None
    self.it = None

  def __iter__(self):
    self.it = iter(self.source)
    return self

  def __next__(self):
    if self.index >= self.consec:
      self.index = 0
    if self.index == 0:
      self.current = next(self.it)
      available = self.current['is_first'].shape[-1]
      assert self.length * self.consec + self.prefix <= available, (
          self.length, self.consec, self.prefix, available)
      if self.strict:
        assert self.consec * self.length + self.prefix == available, (
            self.consec, self.length, self.prefix, available)
    start = self.index * self.length
    stop = start + (self.length + self.prefix)
    chunk = {k: v[:, start: stop] for k, v in self.current.items()}
    chunk['consec'] = np.full(chunk['is_first'].shape, self.index, np.int32)
    if self.contiguous:
      # This is expensive but can speed up following operations, such as
      # sending arrays via networking.
      chunk = {k: np.ascontiguousarray(v) for k, v in chunk.items()}
    self.index += 1
    return chunk

  def save(self):
    return {
        'source': self.source.save(),
        'index': self.index,
    }

  def load(self, data):
    self.source.load(data['source'])
    self.index = data['index']


class Zip(base.Stream):

  def __init__(self, sources):
    assert len(sources) > 1, len(sources)
    self.sources = sources
    self.iterators = None
    self.started = False

  def __iter__(self):
    assert not self.started
    self.started = True
    self.iterators = [iter(x) for x in self.sources]
    return self

  def __next__(self):
    parts = [next(x) for x in self.iterators]
    result = elements.tree.map(lambda *el: np.concatenate(el), *parts)
    return result

  def save(self):
    return [x.save() for x in self.iterators]

  def load(self, data):
    assert len(data) == len(self.iterators)
    [it.load(d) for it, d in zip(self.iterators, data)]


class Map(base.Stream):

  def __init__(self, source, fn, *args, **kwargs):
    self.source = source
    self.fn = lambda x: fn(x, *args, **kwargs)
    self.iterator = None
    self.started = False

  def __iter__(self):
    assert not self.started
    self.started = True
    self.iterator = iter(self.source)
    return self

  def __next__(self):
    assert self.started
    return self.fn(next(self.iterator))

  def save(self):
    return self.iterator.save()

  def load(self, data):
    self.iterator.load(data)


class Mixer(base.Stream):

  def __init__(self, sources, weights, seed=0):
    assert sources.keys() == weights.keys(), (sources, weights)
    self.keys = sorted(sources.keys())
    self.iterators = [iter(sources[k]) for k in self.keys]
    weights = np.array([weights[k] for k in self.keys], np.float32)
    self.probs = weights / weights.sum()
    self.seed = seed
    self.started = False
    self.step = 0

  def __iter__(self):
    assert not self.started
    return self

  def __next__(self):
    assert self.started
    rng = np.ranodm.default_rng(seed=[self.seed, self.step])
    self.step += 1
    index = rng.choice(len(self.keys), p=self.probs)
    return next(self.iterators[index])

  def save(self):
    return {
        'step': self.step,
        'seed': self.seed,
        'sources': {k: it.save() for k, it in zip(self.keys, self.iterators)},
    }

  def load(self, data):
    self.step = data['step']
    self.seed = data['seed']
    assert sorted(data['sources'].keys()) == self.keys, (
        data['sources'], self.keys)
    for key in self.keys:
      self.iterators[key].load(data['sources'][key])
