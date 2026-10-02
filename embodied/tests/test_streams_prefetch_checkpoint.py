"""Restoring a prefetch stream immediately restores its consumer checkpoint."""
import elements
import pytest

from embodied.core import base, streams


class CounterSource(base.Stream):
  def __init__(self):
    self.position = 0
    self.closed = False

  def __iter__(self):
    return self

  def __next__(self):
    if self.closed:
      raise SystemExit
    result = self.position
    self.position += 1
    return result

  def save(self):
    return self.position

  def load(self, position):
    self.position = position


def stop(stream):
  stream.source.closed = True
  stream.requests.release()
  stream.worker.join(timeout=2)
  assert not stream.worker.running


@pytest.mark.parametrize('amount', [1, 4])
def test_load_before_iteration_updates_save_position(amount):
  stream = streams.Prefetch(CounterSource(), amount=amount)
  stream.load(17)
  assert stream.save() == 17
  iter(stream)
  try:
    assert next(stream) == 17
    assert stream.save() == 18
  finally:
    stop(stream)


@pytest.mark.parametrize('amount', [1, 4])
def test_started_stream_load_and_immediate_save_use_restored_position(amount):
  stream = iter(streams.Prefetch(CounterSource(), amount=amount))
  try:
    assert [next(stream) for _ in range(5)] == list(range(5))
    stream.load(23)
    assert stream.save() == 23
    assert [next(stream) for _ in range(7)] == list(range(23, 30))
    assert stream.save() == 30
  finally:
    stop(stream)


def test_multiple_restores_before_consuming_track_the_last_checkpoint():
  stream = iter(streams.Prefetch(CounterSource(), amount=3))
  try:
    stream.load(10)
    stream.load(20)
    assert stream.save() == 20
    assert next(stream) == 20
  finally:
    stop(stream)


def test_transform_preserves_source_checkpoint_position():
  stream = iter(streams.Prefetch(CounterSource(), transform=lambda value: value * 3, amount=2))
  try:
    next(stream)
    stream.load(13)
    assert stream.save() == 13
    assert next(stream) == 39
    assert stream.save() == 14
  finally:
    stop(stream)


def test_restored_stream_can_be_checkpointed_again_before_consuming(tmp_path):
  original = iter(streams.Prefetch(CounterSource(), amount=3))
  checkpoint = elements.Checkpoint(str(tmp_path / 'checkpoint.pkl'))
  try:
    assert [next(original) for _ in range(17)] == list(range(17))
    checkpoint.stream = original
    checkpoint.save()
  finally:
    stop(original)

  restored = streams.Prefetch(CounterSource())
  checkpoint.stream = restored
  checkpoint.load()
  checkpoint.save()

  restarted = streams.Prefetch(CounterSource())
  checkpoint.stream = restarted
  checkpoint.load()
  iter(restarted)
  try:
    assert next(restarted) == 17
  finally:
    stop(restarted)


def test_save_tracks_consumed_position_instead_of_prefetched_position():
  stream = iter(streams.Prefetch(CounterSource(), amount=4))
  try:
    assert stream.save() == 0
    for position in range(8):
      assert next(stream) == position
      assert stream.save() == position + 1
  finally:
    stop(stream)
