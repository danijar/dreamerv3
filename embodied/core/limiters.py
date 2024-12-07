import threading
import time


def wait(predicate, message, info=None, sleep=0.01, notify=60):
  if predicate():
    return 0
  start = last_notify = time.time()
  while not predicate():
    now = time.time()
    if now - last_notify > notify:
      dur = now - start
      print(f'{message} {dur:.1f}s: {info}')
      last_notify = time.time()
    time.sleep(sleep)
  return time.time() - start


class SamplesPerInsert:

  def __init__(self, samples_per_insert, tolerance, minsize):
    assert 1 <= minsize
    self.samples_per_insert = samples_per_insert
    self.minsize = minsize
    self.avail = -minsize
    self.min_avail = -tolerance
    self.max_avail = tolerance * samples_per_insert
    self.size = 0
    self.lock = threading.Lock()

  def save(self):
    return {'size': self.size, 'avail': self.avail}

  def load(self, data):
    self.size = data['size']
    self.avail = data['avail']

  def want_insert(self):
    # if self.samples_per_insert <= 0 or self.size < self.minsize:
    #   return True, 'ok'
    # if self.avail >= self.max_avail:
    #   return False, f'rate limited: {self.avail:.3f} >= {self.max_avail:.3f}'
    # return True, 'ok'

    if self.size < self.minsize:
      return True
    if self.samples_per_insert <= 0:
      return True
    if self.avail < self.max_avail:
      return True
    return False

  def want_sample(self):
    # if self.size < self.minsize:
    #   return False, f'too empty: {self.size} < {self.minsize}'
    # if self.samples_per_insert > 0 and self.avail <= self.min_avail:
    #   return False, f'rate limited: {self.avail:.3f} <= {self.min_avail:.3f}'
    # return True, 'ok'

    if self.size < self.minsize:
      return False
    if self.samples_per_insert <= 0:
      return True
    if self.min_avail < self.avail:
      return True
    return False

  def insert(self):
    with self.lock:
      self.size += 1
      if self.size >= self.minsize:
        self.avail += self.samples_per_insert

  # def remove(self):
  #   with self.lock:
  #     self.size -= 1

  def sample(self):
    with self.lock:
      self.avail -= 1
