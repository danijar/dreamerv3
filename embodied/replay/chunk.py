import io

import embodied
import numpy as np


class Chunk:

  __slots__ = ('time', 'uuid', 'succ', 'length', 'size', 'data', 'saved')

  def __init__(self, size=1024):
    self.time = embodied.timestamp(millis=True)
    self.uuid = embodied.uuid()
    self.succ = embodied.uuid(0)
    # self.uuid = int(np.random.randint(1, 2 * 63))
    # self.succ = 0
    self.length = 0
    self.size = size
    self.data = None
    self.saved = False

  def __repr__(self):
    return f'Chunk({self.filename})'

  def __lt__(self, other):
    return self.time < other.time

  @property
  def filename(self):
    succ = self.succ.uuid if isinstance(self.succ, type(self)) else self.succ
    return f'{self.time}-{str(self.uuid)}-{str(succ)}-{self.length}.npz'

  @property
  def nbytes(self):
    if not self.data:
      return 0
    return sum(x.nbytes for x in self.data.values())

  def append(self, step):
    assert self.length < self.size
    if not self.data:
      example = step
      self.data = {
          k: np.empty((self.size, *v.shape), v.dtype)
          for k, v in example.items()}
    for key, value in step.items():
      self.data[key][self.length] = value
    self.length += 1
    # if self.length == self.size:
    #   [x.setflags(write=False) for x in self.data.values()]

  def update(self, index, length, mapping):
    assert 0 <= index <= self.length, (index, self.length)
    assert 0 <= index + length <= self.length, (index, length, self.length)
    for key, value in mapping.items():
      self.data[key][index: index + length] = value

  def slice(self, index, length):
    assert 0 <= index and index + length <= self.length
    return {k: v[index: index + length] for k, v in self.data.items()}

  @embodied.timer.section('chunk_save')
  def save(self, directory, log=False):
    assert not self.saved
    self.saved = True
    filename = embodied.Path(directory) / self.filename
    data = {k: v[:self.length] for k, v in self.data.items()}
    with io.BytesIO() as stream:
      np.savez_compressed(stream, **data)
      stream.seek(0)
      filename.write(stream.read(), mode='wb')
    log and print(f'Saved chunk: {filename.name}')

  @classmethod
  def load(cls, filename, error='raise'):
    assert error in ('raise', 'none')
    time, uuid, succ, length = filename.stem.split('-')
    length = int(length)
    try:
      with embodied.Path(filename).open('rb') as f:
        data = np.load(f)
        data = {k: data[k] for k in data.keys()}
    except Exception as e:
      print(f'Error loading chunk {filename}: {e}')
      if error == 'raise':
        raise
      else:
        return None
    chunk = cls(length)
    chunk.time = time
    chunk.uuid = embodied.uuid(uuid)
    chunk.succ = embodied.uuid(succ)
    # chunk.uuid = int(uuid)
    # chunk.succ = int(succ)
    chunk.length = length
    chunk.data = data
    chunk.saved = True
    return chunk
