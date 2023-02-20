import threading


class MinSize:

  def __init__(self, minimum):
    assert 1 <= minimum, minimum
    self.minimum = minimum
    self.size = 0
    self.lock = threading.Lock()

  def want_load(self):
    with self.lock:
      self.size += 1
    return True, 'ok'

  def want_insert(self):
    with self.lock:
      self.size += 1
    return True, 'ok'

  def want_remove(self):
    with self.lock:
      if self.size < 1:
        return False, 'is empty'
      self.size -= 1
    return True, 'ok'

  def want_sample(self):
    if self.size < self.minimum:
      return False, f'too empty: {self.size} < {self.minimum}'
    return True, 'ok'


class SamplesPerInsert:

  def __init__(self, samples_per_insert, tolerance, minimum=1):
    assert 1 <= minimum
    self.samples_per_insert = samples_per_insert
    self.minimum = minimum
    self.avail = -minimum
    self.min_avail = -tolerance
    self.max_avail = tolerance * samples_per_insert
    self.size = 0
    self.lock = threading.Lock()

  def want_load(self):
    with self.lock:
      self.size += 1
    return True, 'ok'

  def want_insert(self):
    with self.lock:
      if self.avail >= self.max_avail:
        return False, f'rate limited: {self.avail:.3f} >= {self.max_avail:.3f}'
      self.avail += self.samples_per_insert
      self.size += 1
    return True, 'ok'

  def want_remove(self):
    with self.lock:
      if self.size < 1:
        return False, 'is empty'
      self.size -= 1
    return True, 'ok'

  def want_sample(self):
    with self.lock:
      if self.size < self.minimum:
        return False, f'too empty: {self.size} < {self.minimum}'
      if self.avail <= self.min_avail:
        return False, f'rate limited: {self.avail:.3f} <= {self.min_avail:.3f}'
      self.avail -= 1
    return True, 'ok'


class Queue:

  def __init__(self, capacity):
    assert 1 <= capacity
    self.capacity = capacity
    self.size = 0
    self.lock = threading.Lock()

  def want_load(self):
    with self.lock:
      self.size += 1
    return True, 'ok'

  def want_insert(self):
    with self.lock:
      if self.size >= self.capacity:
        return False, f'is full: {self.size} >= {self.capacity}'
      self.size += 1
    return True, 'ok'

  def want_remove(self):
    with self.lock:
      if self.size < 1:
        return False, 'is empty'
      self.size -= 1
    return True, 'ok'

  def want_sample(self):
    if self.size < 1:
      return False, 'is empty'
    else:
      return True, 'ok'
