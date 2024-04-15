import string
import uuid as uuidlib

import numpy as np


class uuid:
  """UUID that is stored as 16 byte string and can be converted to and from
  int, string, and array types."""

  __slots__ = ('value', '_hash')

  DEBUG_ID = None
  BASE62 = string.digits + string.ascii_letters
  BASE62REV = {x: i for i, x in enumerate(BASE62)}

  # def __new__(cls, val=None):
  #   return val or np.random.randint(1, 2 ** 63)

  @classmethod
  def reset(cls, *, debug):
    cls.DEBUG_ID = 0 if debug else None

  def __init__(self, value=None):
    if value is None:
      if self.DEBUG_ID is None:
        self.value = uuidlib.uuid4().bytes
      else:
        type(self).DEBUG_ID += 1
        self.value = self.DEBUG_ID.to_bytes(16, 'big')
    elif isinstance(value, uuid):
      self.value = value.value
    elif isinstance(value, int):
      self.value = value.to_bytes(16, 'big')
    elif isinstance(value, bytes):
      assert len(value) == 16, value
      self.value = value
    elif isinstance(value, str):
      if self.DEBUG_ID is None:
        integer = 0
        for index, char in enumerate(value[::-1]):
          integer += (62 ** index) * self.BASE62REV[char]
        self.value = integer.to_bytes(16, 'big')
      else:
        self.value = int(value).to_bytes(16, 'big')
    elif isinstance(value, np.ndarray):
      self.value = value.tobytes()
    else:
      raise ValueError(value)
    assert type(self.value) == bytes, type(self.value)  # noqa
    assert len(self.value) == 16, len(self.value)
    self._hash = hash(self.value)

  def __int__(self):
    return int.from_bytes(self.value, 'big')

  def __str__(self):
    if self.DEBUG_ID is not None:
      return str(int(self))
    chars = []
    integer = int(self)
    while integer != 0:
      chars.append(self.BASE62[integer % 62])
      integer //= 62
    while len(chars) < 22:
      chars.append('0')
    return ''.join(chars[::-1])

  def __array__(self):
    return np.frombuffer(self.value, np.uint8)

  def __getitem__(self, index):
    return self.__array__()[index]

  def __repr__(self):
    return str(self)

  def __eq__(self, other):
    return self.value == other.value

  def __hash__(self):
    return self._hash
