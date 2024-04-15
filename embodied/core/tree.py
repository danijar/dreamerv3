from . import printing


def map_(fn, *trees, isleaf=None):
  assert trees, 'Provide one or more nested Python structures'
  kw = dict(isleaf=isleaf)
  first = trees[0]
  assert all(isinstance(x, type(first)) for x in trees)
  if isleaf and isleaf(first):
    return fn(*trees)
  if isinstance(first, list):
    assert all(len(x) == len(first) for x in trees), printing.format_(trees)
    return [map_(
        fn, *[t[i] for t in trees], **kw) for i in range(len(first))]
  if isinstance(first, tuple):
    assert all(len(x) == len(first) for x in trees), printing.format_(trees)
    return tuple([map_(
        fn, *[t[i] for t in trees], **kw) for i in range(len(first))])
  if isinstance(first, dict):
    assert all(set(x.keys()) == set(first.keys()) for x in trees), (
        printing.format_(trees))
    return {k: map_(fn, *[t[k] for t in trees], **kw) for k in first}
  if hasattr(first, 'keys') and hasattr(first, 'get'):
    assert all(set(x.keys()) == set(first.keys()) for x in trees), (
        printing.format_(trees))
    return type(first)(
        {k: map_(fn, *[t[k] for t in trees], **kw) for k in first})
  return fn(*trees)


map = map_
