class MinSize:

  def __init__(self, minimum):
    assert 1 <= minimum, minimum
    self.minimum = minimum
    self.size = 0

  def want_insert(self):
    self.size += 1
    return True

  def want_remove(self):
    assert self.size > 0
    self.size -= 1
    return True

  def want_sample(self):
    return self.size >= self.minimum


class SamplesPerInsert:

  def __init__(self, samples_per_insert, tolerance):
    # TODO: Make thread-safe.
    self.samples_per_insert = samples_per_insert
    self.tolerance = tolerance
    self.available_samples = 0

  def want_insert(self):
    if self.available_samples >= self.tolerance:
      return False
    self.available_samples += self.samples_per_insert
    return True

  def want_remove(self):
    return True

  def want_sample(self):
    if self.available_samples <= -self.tolerance:
      return False
    self.available_samples -= 1
    return True


class Queue:

  def __init__(self, capacity):
    # TODO: Make thread-safe.
    assert 1 <= capacity
    self.capacity = capacity
    self.size = 0

  def want_insert(self):
    if self.size >= self.capacity:
      return False
    self.size += 1
    return True

  def want_remove(self):
    assert self.size > 0
    self.size -= 1
    return True

  def want_sample(self):
    return 0 < self.size <= self.capacity
