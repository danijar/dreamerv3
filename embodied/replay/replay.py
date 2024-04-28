import threading
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from functools import partial as bind

import embodied
import numpy as np

from . import chunk as chunklib
from . import limiters
from . import selectors


class Replay:

  def __init__(
      self, length, capacity=None, directory=None, chunksize=1024, min_size=1,
      samples_per_insert=None, tolerance=1e4, online=False, selector=None,
      save_wait=False, seed=0):
    assert not capacity or min_size <= capacity

    self.length = length
    self.capacity = capacity
    self.chunksize = chunksize

    self.sampler = selector or selectors.Uniform(seed)

    if samples_per_insert:
      self.limiter = limiters.SamplesPerInsert(
          samples_per_insert, tolerance, min_size)
    else:
      self.limiter = limiters.MinSize(min_size)

    self.chunks = {}
    self.refs = {}
    self.refs_lock = threading.RLock()

    self.items = {}
    self.fifo = deque()
    self.itemid = 0

    self.current = {}
    self.streams = defaultdict(deque)
    self.rwlock = embodied.RWLock()

    self.online = online
    if online:
      self.lengths = defaultdict(int)
      self.queue = deque()

    if directory:
      self.directory = embodied.Path(directory)
      self.directory.mkdir()
      self.workers = ThreadPoolExecutor(16, 'replay_saver')
      self.saved = set()
    else:
      self.directory = None

    self.save_wait = save_wait

    self.metrics = {
        'samples': 0,
        'sample_wait_dur': 0,
        'sample_wait_count': 0,
        'inserts': 0,
        'insert_wait_dur': 0,
        'insert_wait_count': 0,
        'updates': 0,
    }

  def __len__(self):
    return len(self.items)

  def stats(self):
    ratio = lambda x, y: x / y if y else np.nan
    m = self.metrics
    chunk_nbytes = sum(x.nbytes for x in list(self.chunks.values()))
    stats = {
        'items': len(self.items),
        'chunks': len(self.chunks),
        'streams': len(self.streams),
        'ram_gb': chunk_nbytes / (1024 ** 3),
        'inserts': m['inserts'],
        'samples': m['samples'],
        'updates': m['updates'],
        'replay_ratio': ratio(self.length * m['samples'], m['inserts']),
        'insert_wait_avg': ratio(m['insert_wait_dur'], m['inserts']),
        'insert_wait_frac': ratio(m['insert_wait_count'], m['inserts']),
        'sample_wait_avg': ratio(m['sample_wait_dur'], m['samples']),
        'sample_wait_frac': ratio(m['sample_wait_count'], m['samples']),
    }
    for key in self.metrics:
      self.metrics[key] = 0
    return stats

  @embodied.timer.section('replay_add')
  def add(self, step, worker=0):
    with self.rwlock.reading:

      step = {k: v for k, v in step.items() if not k.startswith('log_')}
      step = {k: np.asarray(v) for k, v in step.items()}

      if worker not in self.current:
        chunk = chunklib.Chunk(self.chunksize)
        with self.refs_lock:
          self.refs[chunk.uuid] = 1
        self.chunks[chunk.uuid] = chunk
        self.current[worker] = (chunk.uuid, 0)

      chunkid, index = self.current[worker]
      step['stepid'] = np.frombuffer(
          bytes(chunkid) + index.to_bytes(4, 'big'), np.uint8)
      stream = self.streams[worker]
      chunk = self.chunks[chunkid]
      assert chunk.length == index, (chunk.length, index)
      chunk.append(step)
      assert chunk.length == index + 1, (chunk.length, index + 1)
      stream.append((chunkid, index))
      with self.refs_lock:
        self.refs[chunkid] += 1

      index += 1
      if index < chunk.size:
        self.current[worker] = (chunkid, index)
      else:
        self._complete(chunk, worker)
      assert len(self.streams) == len(self.current)

      if len(stream) >= self.length:
        dur = self._wait(self.limiter.want_insert, 'Replay insert is waiting')
        # These increments are not thread safe, so the metrics will be slightly
        # wrong and it's faster than introducing a lock.
        self.metrics['inserts'] += 1
        self.metrics['insert_wait_dur'] += dur
        self.metrics['insert_wait_count'] += int(dur >= 0.001)
        chunkid, index = stream.popleft()
        self._insert(chunkid, index)

        if self.online and self.lengths[worker] % self.length == 0:
          self.queue.append((chunkid, index))

      if self.online:
        self.lengths[worker] += 1

  @embodied.timer.section('replay_update')
  def update(self, data):
    data = data.copy()
    stepid = data.pop('stepid')
    priority = data.pop('priority', None)
    assert stepid.ndim == 3, stepid.shape
    self.metrics['updates'] += int(np.prod(stepid.shape[:-1]))
    if priority is not None:
      assert priority.ndim == 2, priority.shape
      self.sampler.prioritize(
          stepid.reshape((-1, stepid.shape[-1])),
          priority.flatten())
    if data:
      for i, stepid in enumerate(stepid):
        stepid = stepid[0].tobytes()
        chunkid = embodied.uuid(stepid[:-4])
        index = int.from_bytes(stepid[-4:], 'big')
        values = {k: v[i] for k, v in data.items()}
        try:
          self._setseq(chunkid, index, values)
        except KeyError:
          pass

  @embodied.timer.section('replay_sample')
  def _sample(self):
    dur = self._wait(self.limiter.want_sample, 'Replay sample is waiting')
    self.limiter.sample()
    # These increments are not thread safe, so the metrics can be slightly
    # wrong and it's faster than introducing a lock.
    self.metrics['samples'] += 1
    self.metrics['sample_wait_dur'] += dur
    self.metrics['sample_wait_count'] += int(dur >= 0.001)
    while True:
      try:
        if self.online and self.queue:
          chunkid, index = self.queue.popleft()
          is_online = True
        else:
          with embodied.timer.section('sample'):
            itemid = self.sampler()
          chunkid, index = self.items[itemid]
          is_online = False
        seq = self._getseq(chunkid, index, concat=False)
        return seq, is_online
      except KeyError:
        continue

  def _insert(self, chunkid, index, loading=False):
    while self.capacity and len(self.items) >= self.capacity:
      self._remove(loading=loading)
    itemid = self.itemid
    self.itemid += 1
    self.items[itemid] = (chunkid, index)
    stepids = self._getseq(chunkid, index, ['stepid'])['stepid']
    self.sampler[itemid] = stepids
    self.fifo.append(itemid)
    if not loading:
      self.limiter.insert()

  def _remove(self, loading=False):
    if not loading:
      self.limiter.remove()
    itemid = self.fifo.popleft()
    del self.sampler[itemid]
    chunkid, index = self.items.pop(itemid)
    with self.refs_lock:
      self.refs[chunkid] -= 1
      if self.refs[chunkid] < 1:
        del self.refs[chunkid]
        chunk = self.chunks.pop(chunkid)
        if chunk.succ in self.refs:
          self.refs[chunk.succ] -= 1

  def _getseq(self, chunkid, index, keys=None, concat=True):
    chunk = self.chunks[chunkid]
    available = chunk.length - index
    if available >= self.length:
      with embodied.timer.section('get_slice'):
        seq = chunk.slice(index, self.length)
        if not concat:
          seq = {k: [v] for k, v in seq.items()}
        return seq
    else:
      with embodied.timer.section('get_compose'):
        parts = [chunk.slice(index, available)]
        remaining = self.length - available
        while remaining > 0:
          chunk = self.chunks[chunk.succ]
          used = min(remaining, chunk.length)
          parts.append(chunk.slice(0, used))
          remaining -= used
        seq = {k: [p[k] for p in parts] for k in keys or parts[0].keys()}
        if concat:
          seq = {k: np.concatenate(v, 0) for k, v in seq.items()}
        return seq

  def _setseq(self, chunkid, index, values):
    length = len(next(iter(values.values())))
    chunk = self.chunks[chunkid]
    available = chunk.length - index
    if available >= length:
      with embodied.timer.section('set_slice'):
        return chunk.update(index, length, values)
    else:
      with embodied.timer.section('set_compose'):
        part = {k: v[:available] for k, v in values.items()}
        values = {k: v[available:] for k, v in values.items()}
        chunk.update(index, available, part)
        remaining = length - available
        while remaining > 0:
          chunk = self.chunks[chunk.succ]
          used = min(remaining, chunk.length)
          part = {k: v[:used] for k, v in values.items()}
          values = {k: v[used:] for k, v in values.items()}
          chunk.update(0, used, part)
          remaining -= used

  def dataset(self, batch, length=None):
    # For performance, each batch should be consecutive in memory, rather than
    # a non-consecutive view into a longer batch. For example, this allows
    # near-instant serialization when sending over the network.
    while True:
      seqs, is_online = zip(*[self._sample() for _ in range(batch)])
      if not length or length == self.length:
        data = self._assemble_batch(seqs, 0, self.length)
        data = self._annotate_batch(data, is_online, is_first=True)
        yield data
      else:
        assert length <= self.length, (length, self.length)
        for t in range(0, self.length - self.length % length, length):
          data = self._assemble_batch(seqs, t, t + length)
          data = self._annotate_batch(data, is_online, is_first=(t == 0))
          yield data

  @embodied.timer.section('assemble_batch')
  def _assemble_batch(self, seqs, start, stop):
    shape = (len(seqs), stop - start)
    data = {
        key: np.empty((*shape, *parts[0].shape[1:]), parts[0].dtype)
        for key, parts in seqs[0].items()}
    for n, seq in enumerate(seqs):
      st, dt = 0, 0  # Source and destination time index.
      for p in range(len(seq['stepid'])):
        partlen = len(seq['stepid'][p])
        if start < st + partlen:
          part_start = max(0, start - st)
          part_stop = min(stop - st, partlen)
          num = part_stop - part_start
          for k in data.keys():
            data[k][n, dt: dt + num] = seq[k][p][part_start: part_stop]
          dt += num
        st += partlen
        if st >= stop:
          break
    return data

  @embodied.timer.section('annotate_batch')
  def _annotate_batch(self, data, is_online, is_first):
    data = data.copy()
    if self.online:
      broadcasted = [[x] for x in is_online]
      data['is_online'] = np.full(data['is_first'].shape, broadcasted, bool)
    if 'is_first' in data:
      if is_first:
        data['is_first'] = data['is_first'].copy()
        data['is_first'][:, 0] = True
      if 'is_last' in data:
        # Make sure that abandoned episodes have is_last set.
        next_is_first = np.roll(data['is_first'], shift=-1, axis=1)
        next_is_first[:, -1] = False
        data['is_last'] = data['is_last'] | next_is_first
    return data

  @embodied.timer.section('replay_save')
  def save(self):
    if self.directory:
      with self.rwlock.writing:
        for worker, (chunkid, _) in self.current.items():
          chunk = self.chunks[chunkid]
          if chunk.length > 0:
            self._complete(chunk, worker)
        promises = []
        for chunk in self.chunks.values():
          if chunk.length > 0 and chunk.uuid not in self.saved:
            self.saved.add(chunk.uuid)
            promises.append(self.workers.submit(chunk.save, self.directory))
        if self.save_wait:
          [promise.result() for promise in promises]
    return {'limiter': self.limiter.save()}

  @embodied.timer.section('replay_load')
  def load(self, data=None, directory=None, amount=None):

    directory = directory or self.directory
    amount = amount or self.capacity or np.inf
    if not directory:
      return
    revsorted = lambda x: list(reversed(sorted(list(x))))
    directory = embodied.Path(directory)
    names_loaded = revsorted(x.filename for x in list(self.chunks.values()))
    names_ondisk = revsorted(x.name for x in directory.glob('*.npz'))
    names_ondisk = [x for x in names_ondisk if x not in names_loaded]
    if not names_ondisk:
      return

    numitems = self._numitems(names_loaded + names_ondisk)
    uuids = [embodied.uuid(x.split('-')[1]) for x in names_ondisk]
    total = 0
    numchunks = 0
    for uuid in uuids:
      numchunks += 1
      total += numitems[uuid]
      if total >= amount:
        break

    load = bind(chunklib.Chunk.load, error='none')
    filenames = [directory / x for x in names_ondisk[:numchunks]]

    with ThreadPoolExecutor(16, 'replay_loader') as pool:
      chunks = [x for x in pool.map(load, filenames) if x]

    # We need to recompute the number of items per chunk now because some
    # chunks may be corrupted and thus not available.
    # numitems = self._numitems(chunks + list(self.chunks.values()))
    numitems = self._numitems(chunks)

    with self.rwlock.writing:
      self.saved.update([chunk.uuid for chunk in chunks])
      with self.refs_lock:
        for chunk in chunks:
          self.chunks[chunk.uuid] = chunk
          self.refs[chunk.uuid] = 0
        for chunk in reversed(chunks):
          amount = numitems[chunk.uuid]
          self.refs[chunk.uuid] += amount
          if chunk.succ in self.refs:
            self.refs[chunk.succ] += 1
          for index in range(amount):
            self._insert(chunk.uuid, index, loading=True)

    if data and 'limiter' in data:
      self.limiter.load(data.pop('limiter'))

  @embodied.timer.section('complete_chunk')
  def _complete(self, chunk, worker):
    succ = chunklib.Chunk(self.chunksize)
    with self.refs_lock:
      self.refs[chunk.uuid] -= 1
      self.refs[succ.uuid] = 2
    self.chunks[succ.uuid] = succ
    self.current[worker] = (succ.uuid, 0)
    chunk.succ = succ.uuid
    return succ

  def _numitems(self, chunks):
    chunks = [x.filename if hasattr(x, 'filename') else x for x in chunks]
    chunks = list(reversed(sorted([embodied.Path(x).stem for x in chunks])))
    times, uuids, succs, lengths = zip(*[x.split('-') for x in chunks])
    uuids = [embodied.uuid(x) for x in uuids]
    succs = [embodied.uuid(x) for x in succs]
    lengths = {k: int(v) for k, v in zip(uuids, lengths)}
    future = {}
    for uuid, succ in zip(uuids, succs):
      future[uuid] = lengths[uuid] + future.get(succ, 0)
    numitems = {}
    for uuid, succ in zip(uuids, succs):
      numitems[uuid] = lengths[uuid] + 1 - self.length + future.get(succ, 0)
    numitems = {k: np.clip(v, 0, lengths[k]) for k, v in numitems.items()}
    return numitems

  def _wait(self, predicate, message, sleep=0.01, notify=60):
    if predicate(reason=False):
      return 0
    start = last_notify = time.time()
    while not predicate(reason=False):
      if time.time() - last_notify > notify:
        allowed, reason = predicate(reason=True)
        if allowed:
          break
        dur = time.time() - start
        embodied.print(f'{message} {dur:.1f}s ({reason})')
        last_notify = time.time()
      time.sleep(sleep)
    return time.time() - start
