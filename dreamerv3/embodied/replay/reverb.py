import pickle
from collections import defaultdict
from functools import partial as bind

import embodied
import numpy as np


class Reverb:

  def __init__(
      self, length, capacity=None, directory=None, chunks=None, flush=100):
    del chunks
    import reverb
    self.length = length
    self.capacity = capacity
    self.directory = directory and embodied.Path(directory)
    self.checkpointer = None
    self.server = None
    self.client = None
    self.writers = None
    self.counters = None
    self.signature = None
    self.flush = flush
    if self.directory:
      self.directory.mkdirs()
      path = str(self.directory)
      try:
        self.checkpointer = reverb.checkpointers.DefaultCheckpointer(path)
      except AttributeError:
        self.checkpointer = reverb.checkpointers.RecordIOCheckpointer(path)
      self.sigpath = self.directory.parent / (self.directory.name + '_sig.pkl')
    if self.directory and self.sigpath.exists():
      with self.sigpath.open('rb') as file:
        self.signature = pickle.load(file)
      self._create_server()

  def _create_server(self):
    import reverb
    import tensorflow as tf
    self.server = reverb.Server(tables=[reverb.Table(
        name='table',
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        max_size=int(self.capacity),
        rate_limiter=reverb.rate_limiters.MinSize(1),
        signature={
            key: tf.TensorSpec(shape, dtype)
            for key, (shape, dtype) in self.signature.items()},
    )], port=None, checkpointer=self.checkpointer)
    self.client = reverb.Client(f'localhost:{self.server.port}')
    self.writers = defaultdict(bind(
        self.client.trajectory_writer, self.length))
    self.counters = defaultdict(int)

  def __len__(self):
    if not self.client:
      return 0
    return self.client.server_info()['table'].current_size

  @property
  def stats(self):
    return {'size': len(self)}

  def add(self, step, worker=0):
    step = {k: v for k, v in step.items() if not k.startswith('log_')}
    step = {k: embodied.convert(v) for k, v in step.items()}
    step['id'] = np.asarray(embodied.uuid(step.get('id')))
    if not self.server:
      self.signature = {
          k: ((self.length, *v.shape), v.dtype) for k, v in step.items()}
      self._create_server()
    step = {k: v for k, v in step.items() if not k.startswith('log_')}
    writer = self.writers[worker]
    writer.append(step)
    if len(next(iter(writer.history.values()))) >= self.length:
      seq = {k: v[-self.length:] for k, v in writer.history.items()}
      writer.create_item('table', priority=1.0, trajectory=seq)
      self.counters[worker] += 1
      if self.counters[worker] > self.flush:
        self.counters[worker] = 0
        writer.flush()

  def dataset(self):
    import reverb
    dataset = reverb.TrajectoryDataset.from_table_signature(
        server_address=f'localhost:{self.server.port}',
        table='table',
        max_in_flight_samples_per_worker=10)
    for sample in dataset:
      seq = sample.data
      seq = {k: embodied.convert(v) for k, v in seq.items()}
      # seq['key'] = sample.info.key  # uint64
      # seq['prob'] = sample.info.probability
      if 'is_first' in seq:
        seq['is_first'] = np.array(seq['is_first'])
        seq['is_first'][0] = True
      yield seq

  def prioritize(self, ids, prios):
    raise NotImplementedError

  def save(self, wait=False):
    for writer in self.writers.values():
      writer.flush()
    with self.sigpath.open('wb') as file:
      file.write(pickle.dumps(self.signature))
    if self.directory:
      self.client.checkpoint()

  def load(self, data=None):
    pass
