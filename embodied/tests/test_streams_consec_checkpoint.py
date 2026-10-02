"""Checkpoint consecutive windows without dropping their buffered source batch."""
import elements
import numpy as np
import pytest

from embodied.core import base, streams


class BatchSource(base.Stream):
  def __init__(self, width):
    self.width = width
    self.position = 0

  def __next__(self):
    values = np.tile(np.arange(self.width) + 100 * self.position, (2, 1))
    self.position += 1
    return {'value': values, 'is_first': np.zeros(values.shape, bool)}

  def save(self):
    return self.position

  def load(self, position):
    self.position = position


def create(prefix=2, contiguous=False):
  source = BatchSource(9 + prefix)
  return iter(streams.Consec(source, length=3, consec=3, prefix=prefix, contiguous=contiguous))


def assert_batches_equal(first, second):
  assert first.keys() == second.keys()
  for key in first:
    np.testing.assert_array_equal(first[key], second[key])


@pytest.mark.parametrize('consumed', [1, 2])
@pytest.mark.parametrize('prefix', [0, 2])
@pytest.mark.parametrize('contiguous', [False, True])
def test_partial_batch_restores_exact_remaining_windows(consumed, prefix, contiguous):
  original = create(prefix, contiguous)
  for _ in range(consumed):
    next(original)
  state = original.save()
  restored = create(prefix, contiguous)
  restored.load(state)
  for _ in range(7):
    assert_batches_equal(next(restored), next(original))


def test_restore_overwrites_a_different_buffered_batch():
  original = create()
  next(original)
  state = original.save()
  restored = create()
  for _ in range(4):
    next(restored)
  restored.load(state)
  assert_batches_equal(next(restored), next(original))


def test_save_snapshot_does_not_alias_delivered_window():
  original = create()
  delivered = next(original)
  state = original.save()
  delivered['value'][:] = -100
  restored = create()
  restored.load(state)
  np.testing.assert_array_equal(next(restored)['value'], np.tile(np.arange(3, 8), (2, 1)))


def test_load_snapshot_is_reusable_without_aliasing():
  original = create()
  next(original)
  state = original.save()
  first = create()
  first.load(state)
  next(first)['value'][:] = -1
  second = create()
  second.load(state)
  np.testing.assert_array_equal(next(second)['value'], np.tile(np.arange(3, 8), (2, 1)))


@pytest.mark.parametrize('consumed', [0, 3])
def test_legacy_checkpoint_at_batch_boundary_still_restores(consumed):
  original = create()
  for _ in range(consumed):
    next(original)
  state = original.save()
  state.pop('current', None)
  restored = create()
  restored.load(state)
  assert_batches_equal(next(restored), next(original))


def test_legacy_partial_checkpoint_reports_missing_batch():
  original = create()
  next(original)
  state = original.save()
  state.pop('current', None)
  with pytest.raises(ValueError, match='missing its batch'):
    create().load(state)


def test_native_elements_checkpoint_roundtrip(tmp_path):
  original = create()
  next(original)
  checkpoint = elements.Checkpoint(tmp_path / 'checkpoint')
  checkpoint.stream = original
  checkpoint.save()
  restored = create()
  checkpoint = elements.Checkpoint(tmp_path / 'checkpoint')
  checkpoint.stream = restored
  checkpoint.load()
  for _ in range(5):
    assert_batches_equal(next(restored), next(original))
