import functools
import threading


@functools.total_ordering
class Counter:

  def __init__(self, initial=0):
    self.value = initial
    self.lock = threading.Lock()

  def __repr__(self):
    return f'Counter({self.value})'

  def __int__(self):
    return int(self.value)

  def __eq__(self, other):
    return int(self) == other

  def __ne__(self, other):
    return int(self) != other

  def __lt__(self, other):
    return int(self) < other

  def __add__(self, other):
    return int(self) + other

  def __radd__(self, other):
    return other - int(self)

  def __sub__(self, other):
    return int(self) - other

  def __rsub__(self, other):
    return other - int(self)

  def increment(self, amount=1):
    with self.lock:
      self.value += amount

  def reset(self):
    with self.lock:
      self.value = 0

  def save(self):
    return self.value

  def load(self, value):
    self.value = value
