import numpy as np


class SampleTree:

  def __init__(self, branching=16, seed=0):
    assert 2 <= branching
    self.branching = branching
    self.root = Node()
    self.last = None
    self.entries = {}
    self.rng = np.random.default_rng(seed)

  def __len__(self):
    return len(self.entries)

  def insert(self, key, uprob):
    if not self.last:
      node = self.root
    else:
      ups = 0
      node = self.last.parent
      while node and len(node) >= self.branching:
        node = node.parent
        ups += 1
      if not node:
        node = Node()
        node.append(self.root)
        self.root = node
      for _ in range(ups):
        below = Node()
        node.append(below)
        node = below
    entry = Entry(key, uprob)
    node.append(entry)
    self.entries[key] = entry
    self.last = entry

  def remove(self, key):
    entry = self.entries.pop(key)
    entry_parent = entry.parent
    last_parent = self.last.parent
    entry.parent.remove(entry)
    if entry is not self.last:
      entry_parent.append(self.last)
    node = last_parent
    ups = 0
    while node.parent and not len(node):
      above = node.parent
      above.remove(node)
      node = above
      ups += 1
    if not len(node):
      self.last = None
      return
    while isinstance(node, Node):
      node = node.children[-1]
    self.last = node

  def update(self, key, uprob):
    entry = self.entries[key]
    entry.uprob = uprob
    entry.parent.recompute()

  def sample(self):
    node = self.root
    while isinstance(node, Node):
      uprobs = np.array([x.uprob for x in node.children])
      total = uprobs.sum()
      if not np.isfinite(total):
        finite = np.isinf(uprobs)
        probs = finite / finite.sum()
      elif total == 0:
        probs = np.ones(len(uprobs)) / len(uprobs)
      else:
        probs = uprobs / total
      choice = self.rng.choice(np.arange(len(uprobs)), p=probs)
      node = node.children[choice.item()]
    return node.key


class Node:

  __slots__ = ('parent', 'children', 'uprob')

  def __init__(self, parent=None):
    self.parent = parent
    self.children = []
    self.uprob = 0

  def __repr__(self):
    return (
        f'Node(uprob={self.uprob}, '
        f'children={[x.uprob for x in self.children]})'
    )

  def __len__(self):
    return len(self.children)

  def __bool__(self):
    return True

  def append(self, child):
    if child.parent:
      child.parent.remove(child)
    child.parent = self
    self.children.append(child)
    self.recompute()

  def remove(self, child):
    child.parent = None
    self.children.remove(child)
    self.recompute()

  def recompute(self):
    self.uprob = sum(x.uprob for x in self.children)
    self.parent and self.parent.recompute()


class Entry:

  __slots__ = ('parent', 'key', 'uprob')

  def __init__(self, key=None, uprob=None):
    self.parent = None
    self.key = key
    self.uprob = uprob
