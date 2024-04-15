class IndexDict:

  def __init__(self):
    self._indices = {}
    self._items = []

  def keys(self):
    return self._indices.keys()

  def items(self):
    return tuple(self._items)

  def __repr__(self):
    return repr(dict(self._items))

  def __len__(self):
    return len(self._items)

  def __setitem__(self, key, value):
    if key in self._indices:
      return
    self._indices[key] = len(self._items)
    self._items.append((key, value))

  def __getitem__(self, index_or_key):
    if isinstance(index_or_key, int):
      index = index_or_key
    else:
      index = self._indices[index_or_key]
    return self._items[index][1]

  def __delitem__(self, index):
    self.pop(index)

  def pop(self, index_or_key):
    if isinstance(index_or_key, int):
      index = index_or_key
      del self._indices[self._items[index][0]]
    else:
      index = self._indices.pop(index_or_key)
    value = self._items[index][1]
    last = self._items.pop()
    if index != len(self._items):
      self._items[index] = last
      self._indices[last[0]] = index
    return value
