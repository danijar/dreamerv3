import time


class Every:
  """Returns True at most every X steps."""

  def __init__(self, every, initial=True):
    """Creates a new Every instance

    Args:
      every: The minimum number of steps between returning True.
      initial: The value to be returned when called for the first time.
    """
    self._every = every
    self._initial = initial
    self._prev = None

  def __call__(self, step):
    step = int(step)
    if self._every < 0:
      return True
    if self._every == 0:
      return False
    if self._prev is None:
      self._prev = (step // self._every) * self._every
      return self._initial
    if step >= self._prev + self._every:
      self._prev += self._every
      return True
    return False


class Ratio:
  """Returns True a fixed fraction of the time."""

  def __init__(self, ratio):
    assert ratio >= 0, ratio
    self._ratio = ratio
    self._prev = None

  def __call__(self, step):
    step = int(step)
    if self._ratio == 0:
      return 0
    if self._prev is None:
      self._prev = step
      return 1
    repeats = int((step - self._prev) * self._ratio)
    self._prev += repeats / self._ratio
    return repeats


class Once:

  def __init__(self):
    self._once = True

  def __call__(self):
    if self._once:
      self._once = False
      return True
    return False


class Until:

  def __init__(self, until):
    self._until = until

  def __call__(self, step):
    step = int(step)
    if not self._until:
      return True
    return step < self._until


class Clock:
  """Returns True at least X seconds apart"""

  def __init__(self, every):
    """Creates a new Clock instance

    Args:
      every: The time (in seconds) between returning True.
    """
    self._every = every
    self._prev = None

  def __call__(self, step=None):
    if self._every < 0:
      return True
    if self._every == 0:
      return False
    now = time.time()
    if self._prev is None:
      self._prev = now
      return True
    if now >= self._prev + self._every:
      # self._prev += self._every
      self._prev = now
      return True
    return False
