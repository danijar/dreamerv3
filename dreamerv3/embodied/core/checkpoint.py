import concurrent.futures
import pickle
import time

from . import path


class Checkpoint:

  def __init__(self, filename, log=True, pickle=pickle, parallel=True):
    self._filename = path.Path(filename)
    self._log = log
    self._pickle = pickle
    self._values = {}
    self._parallel = parallel
    if self._parallel:
      self._worker = concurrent.futures.ThreadPoolExecutor(1)
      self._promise = None

  def __setattr__(self, name, value):
    if name in ('exists', 'save', 'load'):
      return super().__setattr__(name, value)
    if name.startswith('_'):
      return super().__setattr__(name, value)
    has_load = hasattr(value, 'load') and callable(value.load)
    has_save = hasattr(value, 'save') and callable(value.save)
    if not (has_load and has_save):
      message = f"Checkpoint entry '{name}' must implement save() and load()."
      raise ValueError(message)
    self._values[name] = value

  def __getattr__(self, name):
    if name.startswith('_'):
      raise AttributeError(name)
    try:
      return getattr(self._values, name)
    except AttributeError:
      raise ValueError(name)

  def exists(self):
    exists = self._filename.exists()
    self._log and exists and print('Existing checkpoint found.')
    self._log and not exists and print('Existing checkpoint not found.')
    return exists

  def save(self):
    self._log and print(f'Saving checkpoint: {self._filename}')
    if self._parallel:
      self._promise and self._promise.result()
      self._promise = self._worker.submit(self._save)
    else:
      self._save()

  def _save(self):
    data = {k: v.save() for k, v in self._values.items()}
    assert all([not k.startswith('_') for k in data]), list(data.keys())
    data['_timestamp'] = time.time()
    tmp = self._filename.parent / (self._filename.stem + '.tmp')
    with tmp.open('wb') as f:
      f.write(self._pickle.dumps(data))
    tmp.move(self._filename)
    print('Wrote checkpoint:', self._filename)

  def load(self):
    self._log and print(f'Loading checkpoint: {self._filename}')
    with self._filename.open('rb') as f:
      data = self._pickle.load(f)
    for key, value in data.items():
      if key.startswith('_'):
        continue
      try:
        self._values[key].load(value)
      except Exception:
        print(f'Error loading {key} from checkpoint.')
        raise
    if self._log and '_timestamp' in data:
      age = time.time() - data['_timestamp']
      print(f'Loaded checkpoint from {age:.0f} seconds ago.')

  def load_or_save(self):
    if self.exists():
      self.load()
    else:
      self.save()
