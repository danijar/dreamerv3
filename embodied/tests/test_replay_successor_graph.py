"""Replay sequence availability follows chunk links, not timestamp tie order."""
import elements
import numpy as np
import pytest

from embodied.core import chunk as chunklib
from embodied.core import replay as replaylib
from embodied.core import selectors


def make_chunks(lengths, uuids=None, same_time=True):
  uuids = uuids or list(range(len(lengths), 0, -1))
  chunks = []
  offset = 0
  for position, (length, uuid) in enumerate(zip(lengths, uuids)):
    chunk = chunklib.Chunk(length)
    chunk.uuid = elements.UUID(uuid)
    chunk.time = '20260101T000000F000' if same_time else f'20260101T000000F{position:03d}'
    for index in range(length):
      stepid = np.frombuffer(bytes(chunk.uuid) + index.to_bytes(4, 'big'), np.uint8)
      chunk.append({'value': np.array(offset + index), 'stepid': stepid})
    offset += length
    chunks.append(chunk)
  for chunk, successor in zip(chunks, chunks[1:]):
    chunk.succ = successor.uuid
  return chunks


@pytest.mark.parametrize('length', [4, 5, 6])
def test_same_timestamp_chunk_counts_follow_successor_links(length):
  chunks = make_chunks([2, 2, 2], uuids=[30, 20, 10])
  replay = replaylib.Replay(length=length)
  expected = {
      chunks[0].uuid: min(2, 7 - length),
      chunks[1].uuid: max(0, min(2, 5 - length)),
      chunks[2].uuid: 0,
  }
  assert replay._numitems(chunks) == expected
  assert replay._numitems(list(reversed(chunks))) == expected


def test_same_timestamp_chunks_roundtrip_actual_disk_replay(tmp_path):
  chunks = make_chunks([2, 2, 2], uuids=[30, 20, 10])
  for chunk in chunks:
    chunk.save(tmp_path)
  replay = replaylib.Replay(length=5, directory=tmp_path)
  try:
    replay.load()
    assert len(replay) == 2
    sequences = {tuple(replay._getseq(uuid, index)['value'])
                 for uuid, index in replay.items.values()}
    assert sequences == {tuple(range(5)), tuple(range(1, 6))}
    sampled = replay.sample(20)['value']
    assert all(tuple(sequence) in sequences for sequence in sampled)
  finally:
    replay.workers.shutdown()


def test_missing_successor_does_not_count_unavailable_sequence_steps():
  chunks = make_chunks([2, 2, 2], uuids=[30, 20, 10])
  replay = replaylib.Replay(length=5)
  assert replay._numitems(chunks[:2]) == {chunks[0].uuid: 0, chunks[1].uuid: 0}


def test_independent_worker_chains_are_not_concatenated():
  first = make_chunks([2, 2, 2], uuids=[30, 20, 10])
  second = make_chunks([1, 1], uuids=[50, 40])
  replay = replaylib.Replay(length=5)
  counts = replay._numitems(first + second)
  assert counts[first[0].uuid] == 2
  assert all(counts[chunk.uuid] == 0 for chunk in first[1:] + second)


def test_long_chain_uses_iterative_traversal():
  chunks = make_chunks([1] * 2000)
  counts = replaylib.Replay(length=1500)._numitems(chunks)
  assert sum(counts.values()) == 501
  assert counts[chunks[500].uuid] == 1
  assert counts[chunks[501].uuid] == 0


def test_successor_cycle_is_reported_instead_of_counting_repeated_steps():
  chunks = make_chunks([2, 2])
  chunks[-1].succ = chunks[0].uuid
  with pytest.raises(ValueError, match='successor cycle'):
    replaylib.Replay(length=5)._numitems(chunks)


def test_distinct_timestamps_keep_existing_counts():
  chunks = make_chunks([3, 4, 2], same_time=False)
  counts = replaylib.Replay(length=5)._numitems(chunks)
  assert counts == {chunks[0].uuid: 3, chunks[1].uuid: 2, chunks[2].uuid: 0}


@pytest.mark.parametrize('num_chunks', [1, 3, 17, 64, 501])
def test_randomized_chain_matches_independent_remaining_steps_oracle(num_chunks):
  rng = np.random.default_rng(num_chunks)
  lengths = rng.integers(1, 6, size=num_chunks).tolist()
  uuids = rng.permutation(np.arange(1, num_chunks + 1)).tolist()
  chunks = make_chunks(lengths, uuids)
  sequence_length = int(rng.integers(1, sum(lengths) + 2))
  remaining_steps = np.cumsum(lengths[::-1])[::-1]
  expected = {chunk.uuid: min(length, max(0, int(remaining) - sequence_length + 1))
              for chunk, length, remaining in zip(chunks, lengths, remaining_steps)}
  shuffled = [chunks[index] for index in rng.permutation(num_chunks)]
  assert replaylib.Replay(length=sequence_length)._numitems(shuffled) == expected


def test_corrupt_successor_archive_excludes_cross_chunk_sequences(tmp_path):
  chunks = make_chunks([2, 2, 2], uuids=[30, 20, 10])
  for chunk in chunks:
    chunk.save(tmp_path)
  (tmp_path / chunks[1].filename).write_bytes(b'invalid compressed archive')
  replay = replaylib.Replay(length=5, directory=tmp_path)
  try:
    replay.load()
    assert len(replay) == 0
  finally:
    replay.workers.shutdown()


def test_incremental_load_counts_already_loaded_successor_steps(tmp_path):
  chunks = make_chunks([2] * 6, same_time=False)
  for chunk in chunks[3:]:
    chunk.save(tmp_path)
  replay = replaylib.Replay(length=5, directory=tmp_path)
  try:
    replay.load()
    assert len(replay) == 2
    for chunk in chunks[:3]:
      chunk.save(tmp_path)
    replay.load()
    assert len(replay) == 8
    sequences = {tuple(replay._getseq(uuid, index)['value'])
                 for uuid, index in replay.items.values()}
    assert sequences == {tuple(range(start, start + 5)) for start in range(8)}
    sampled = replay.sample(30)['value']
    assert all(tuple(sequence) in sequences for sequence in sampled)
    replay.load()
    assert len(replay) == 8
  finally:
    replay.workers.shutdown()


@pytest.mark.parametrize('capacity,amount', [(1, None), (2, None), (None, 1)])
def test_bounded_load_keeps_successors_required_by_selected_starts(tmp_path, capacity, amount):
  chunks = make_chunks([2, 2, 2], uuids=[30, 20, 10])
  for chunk in chunks:
    chunk.save(tmp_path)
  replay = replaylib.Replay(length=5, capacity=capacity, directory=tmp_path)
  if capacity == 1:
    # Uniform's pre-existing deletion assertion requires at least two items.
    # Attach Fifo explicitly to isolate successor retention from that defect
    # and from the upstream constructor's falsey-empty-selector fallback.
    replay.sampler = selectors.Fifo()
  try:
    replay.load(amount=amount)
    assert len(replay) == (capacity or 2)
    sequences = {tuple(replay._getseq(uuid, index)['value'])
                 for uuid, index in replay.items.values()}
    assert sequences <= {tuple(range(5)), tuple(range(1, 6))}
    assert sequences
    sampled = replay.sample(20)['value']
    assert all(tuple(sequence) in sequences for sequence in sampled)
    for chunk in chunks:
      assert chunk.uuid in replay.chunks
      assert replay.refs[chunk.uuid] > 0
  finally:
    replay.workers.shutdown()


def test_dependency_loading_stops_after_the_last_selected_window(tmp_path):
  chunks = make_chunks([2] * 10, uuids=list(range(30, 20, -1)))
  for chunk in chunks:
    chunk.save(tmp_path)
  replay = replaylib.Replay(length=5, capacity=2, directory=tmp_path)
  try:
    replay.load()
    assert len(replay) == 2
    assert set(replay.chunks) == {chunk.uuid for chunk in chunks[:3]}
    sequences = {tuple(replay._getseq(uuid, index)['value'])
                 for uuid, index in replay.items.values()}
    assert sequences == {tuple(range(5)), tuple(range(1, 6))}
  finally:
    replay.workers.shutdown()


@pytest.mark.parametrize('seed', [2, 7, 13, 20, 31])
def test_bounded_load_randomized_chain_retains_exact_valid_windows(tmp_path, seed):
  rng = np.random.default_rng(seed)
  lengths = rng.integers(1, 6, size=12).tolist()
  uuids = rng.permutation(np.arange(1, 13)).tolist()
  chunks = make_chunks(lengths, uuids)
  length = int(rng.integers(2, sum(lengths) + 1))
  capacity = int(rng.integers(2, 10))
  for chunk in chunks:
    chunk.save(tmp_path)
  replay = replaylib.Replay(length=length, capacity=capacity, directory=tmp_path)
  try:
    replay.load()
    total_windows = sum(lengths) - length + 1
    assert len(replay) == min(capacity, total_windows)
    expected = {tuple(range(start, start + length)) for start in range(total_windows)}
    restored = {tuple(replay._getseq(uuid, index)['value'])
                for uuid, index in replay.items.values()}
    assert restored <= expected
    assert len(restored) == len(replay)
    assert all(tuple(sequence) in restored for sequence in replay.sample(20)['value'])
  finally:
    replay.workers.shutdown()


def test_dependency_archive_registers_its_valid_starts_without_duplicate_reload(tmp_path):
  chunks = make_chunks([2, 10], uuids=[30, 20])
  for chunk in chunks:
    chunk.save(tmp_path)
  replay = replaylib.Replay(length=5, directory=tmp_path)
  try:
    # Loading works in chunks, so the archive dependency may exceed amount.
    replay.load(amount=1)
    assert len(replay) == 8
    replay.load()
    assert len(replay) == 8
    sequences = {tuple(replay._getseq(uuid, index)['value'])
                 for uuid, index in replay.items.values()}
    assert sequences == {tuple(range(start, start + 5)) for start in range(8)}
  finally:
    replay.workers.shutdown()


def test_all_successor_refs_are_protected_before_capacity_eviction(tmp_path):
  chunks = make_chunks([2] * 5, uuids=[20, 30, 10, 9, 8])
  for chunk in chunks:
    chunk.save(tmp_path)
  replay = replaylib.Replay(length=5, capacity=2, directory=tmp_path)
  try:
    replay.load(amount=6)
    assert len(replay) == 2
    expected = {tuple(range(start, start + 5)) for start in range(6)}
    assert all(tuple(sequence) in expected for sequence in replay.sample(20)['value'])
    for uuid, index in replay.items.values():
      assert tuple(replay._getseq(uuid, index)['value']) in expected
  finally:
    replay.workers.shutdown()
