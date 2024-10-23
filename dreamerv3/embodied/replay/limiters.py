import threading


class MinSize:

  def __init__(self, minimum):
    assert 1 <= minimum, minimum
    self.minimum = minimum
    self.size = 0
    self.lock = threading.Lock()

  def save(self):
    return {'size': self.size}

  def load(self, data):
    self.size = data['size']

  def want_insert(self, reason=False):
    if reason:
      return True, 'ok'
    else:
      return True

  def want_sample(self, reason=False):
    if reason:
      if self.size < self.minimum:
        return False, f'too empty: {self.size} < {self.minimum}'
      return True, 'ok'
    else:
      if self.size < self.minimum:
        return False
      return True

  def insert(self):
    with self.lock:
      self.size += 1

  def remove(self):
    with self.lock:
      self.size -= 1

  def sample(self):
    pass


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

  def save(self):
    return {'size': self.size, 'avail': self.avail}

  def load(self, data):
    self.size = data['size']
    self.avail = data['avail']

  def want_insert(self, reason=False):
    if reason:
      if self.size < self.minimum:
        return True, 'ok'
      if self.avail >= self.max_avail:
        return False, f'rate limited: {self.avail:.3f} >= {self.max_avail:.3f}'
      return True, 'ok'
    else:
      if self.size < self.minimum:
        return True
      if self.avail >= self.max_avail:
        return False
      return True

  def want_sample(self, reason=False):
    if reason:
      if self.size < self.minimum:
        return False, f'too empty: {self.size} < {self.minimum}'
      if self.avail <= self.min_avail:
        return False, f'rate limited: {self.avail:.3f} <= {self.min_avail:.3f}'
      return True, 'ok'
    else:
      if self.size < self.minimum:
        return False
      if self.avail <= self.min_avail:
        return False
      return True

  def insert(self):
    with self.lock:
      self.avail += self.samples_per_insert
      self.size += 1

  def remove(self):
    with self.lock:
      self.size -= 1

  def sample(self):
    with self.lock:
      self.avail -= 1
