import collections
import pathlib
import sys
import threading
import time

sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

import embodied
import numpy as np
import pytest


REPLAYS_UNLIMITED = [
    embodied.replay.Replay,
    # embodied.replay.Reverb,
]

REPLAYS_SAVECHUNKS = [
    embodied.replay.Replay,
]

REPLAYS_UNIFORM = [
    embodied.replay.Replay,
]


@pytest.mark.filterwarnings('ignore:.*Pillow.*')
@pytest.mark.filterwarnings('ignore:.*the imp module.*')
@pytest.mark.filterwarnings('ignore:.*distutils.*')
class TestReplay:

  @pytest.mark.parametrize('Replay', REPLAYS_UNLIMITED)
  def test_multiple_keys(self, Replay):
    replay = Replay(length=5, capacity=10)
    for step in range(30):
      replay.add({'image': np.zeros((64, 64, 3)), 'action': np.zeros(12)})
    seq = next(iter(replay.dataset()))
    # assert set(seq.keys()) == {'id', 'image', 'action'}
    assert set(seq.keys()) == {'image', 'action'}
    # assert seq['id'].shape == (5, 16)
    assert seq['image'].shape == (5, 64, 64, 3)
    assert seq['action'].shape == (5, 12)

  @pytest.mark.parametrize('Replay', REPLAYS_UNLIMITED)
  @pytest.mark.parametrize(
      'length,workers,capacity',
      [(1, 1, 1), (2, 1, 2), (5, 1, 10), (1, 2, 2), (5, 3, 15), (2, 7, 20)])
  def test_capacity_exact(self, Replay, length, workers, capacity):
    replay = Replay(length, capacity)
    for step in range(30):
      for worker in range(workers):
        replay.add({'step': step}, worker)
      target = min(workers * max(0, (step + 1) - length + 1), capacity)
      assert len(replay) == target

  @pytest.mark.parametrize('Replay', REPLAYS_UNLIMITED)
  @pytest.mark.parametrize(
      'length,workers,capacity,chunksize',
      [(1, 1, 1, 128), (2, 1, 2, 128), (5, 1, 10, 128), (1, 2, 2, 128),
       (5, 3, 15, 128), (2, 7, 20, 128), (7, 2, 27, 4)])
  def test_sample_sequences(
      self, Replay, length, workers, capacity, chunksize):
    replay = Replay(length, capacity, chunksize=chunksize)
    for step in range(30):
      for worker in range(workers):
        replay.add({'step': step, 'worker': worker}, worker)
    dataset = iter(replay.dataset())
    for _ in range(10):
      seq = next(dataset)
      assert (seq['step'] - seq['step'][0] == np.arange(length)).all()
      assert (seq['worker'] == seq['worker'][0]).all()

  @pytest.mark.parametrize('Replay', REPLAYS_UNLIMITED)
  @pytest.mark.parametrize(
      'length,capacity', [(1, 1), (2, 2), (5, 10), (1, 2), (5, 15), (2, 20)])
  def test_sample_single(self, Replay, length, capacity):
    replay = Replay(length, capacity)
    for step in range(length):
      replay.add({'step': step})
    dataset = iter(replay.dataset())
    for _ in range(10):
      seq = next(dataset)
      assert (seq['step'] == np.arange(length)).all()

  @pytest.mark.parametrize('Replay', REPLAYS_UNIFORM)
  def test_sample_uniform(self, Replay):
    replay = Replay(capacity=20, length=5, seed=0)
    for step in range(7):
      replay.add({'step': step})
    assert len(replay) == 3
    histogram = collections.defaultdict(int)
    dataset = iter(replay.dataset())
    for _ in range(100):
      seq = next(dataset)
      histogram[seq['step'][0]] += 1
    assert len(histogram) == 3, histogram
    histogram = tuple(histogram.values())
    assert histogram[0] > 20
    assert histogram[1] > 20
    assert histogram[2] > 20

  @pytest.mark.parametrize('Replay', REPLAYS_UNLIMITED)
  def test_workers_simple(self, Replay):
    replay = Replay(length=2, capacity=20)
    replay.add({'step': 0}, worker=0)
    replay.add({'step': 1}, worker=1)
    replay.add({'step': 2}, worker=0)
    replay.add({'step': 3}, worker=1)
    dataset = iter(replay.dataset())
    for _ in range(10):
      seq = next(dataset)
      assert tuple(seq['step']) in ((0, 2), (1, 3))

  @pytest.mark.parametrize('Replay', REPLAYS_UNLIMITED)
  def test_workers_random(self, Replay, length=4, capacity=30):
    rng = np.random.default_rng(seed=0)
    replay = Replay(length, capacity)
    streams = {i: iter(range(10)) for i in range(3)}
    for _ in range(40):
      worker = int(rng.integers(0, 3, ()))
      try:
        step = {'step': next(streams[worker]), 'stream': worker}
        replay.add(step, worker=worker)
      except StopIteration:
        pass
    histogram = collections.defaultdict(int)
    dataset = iter(replay.dataset())
    for _ in range(10):
      seq = next(dataset)
      assert (seq['step'] - seq['step'][0] == np.arange(length)).all()
      assert (seq['stream'] == seq['stream'][0]).all()
      histogram[int(seq['stream'][0])] += 1
    assert all(count > 0 for count in histogram.values())

  @pytest.mark.parametrize('Replay', REPLAYS_UNLIMITED)
  @pytest.mark.parametrize(
      'length,workers,capacity',
      [(1, 1, 1), (2, 1, 2), (5, 1, 10), (1, 2, 2), (5, 3, 15), (2, 7, 20)])
  def test_worker_delay(self, Replay, length, workers, capacity):
    # embodied.uuid.reset(debug=True)
    replay = Replay(length, capacity)
    rng = np.random.default_rng(seed=0)
    streams = [iter(range(10)) for _ in range(workers)]
    while streams:
      try:
        worker = rng.integers(0, len(streams))
        replay.add({'step': next(streams[worker])}, worker)
      except StopIteration:
        del streams[worker]

  @pytest.mark.parametrize('Replay', REPLAYS_UNLIMITED)
  @pytest.mark.parametrize(
      'length,capacity,chunksize',
      [(1, 1, 128), (3, 10, 128), (5, 100, 128), (5, 25, 2)])
  def test_restore_exact(self, tmpdir, Replay, length, capacity, chunksize):
    embodied.uuid.reset(debug=True)
    replay = Replay(length, capacity, directory=tmpdir, chunksize=chunksize)
    for step in range(30):
      replay.add({'step': step})
    num_items = np.clip(30 - length + 1, 0, capacity)
    assert len(replay) == num_items
    replay.save(wait=True)
    replay = Replay(length, capacity, directory=tmpdir)
    replay.load()
    assert len(replay) == num_items
    dataset = iter(replay.dataset())
    for _ in range(len(replay)):
      assert len(next(dataset)['step']) == length

  @pytest.mark.parametrize('Replay', REPLAYS_UNLIMITED)
  @pytest.mark.parametrize(
      'length,capacity,chunksize',
      [(1, 1, 128), (3, 10, 128), (5, 100, 128), (5, 25, 2)])
  def test_restore_noclear(self, tmpdir, Replay, length, capacity, chunksize):
    embodied.uuid.reset(debug=True)
    replay = Replay(length, capacity, directory=tmpdir, chunksize=chunksize)
    for _ in range(30):
      replay.add({'foo': 13})
    num_items = np.clip(30 - length + 1, 0, capacity)
    assert len(replay) == num_items
    replay.save(wait=True)
    for _ in range(30):
      replay.add({'foo': 42})
    replay.load()
    dataset = iter(replay.dataset())
    if capacity < num_items:
      for _ in range(len(replay)):
        assert next(dataset)['foo'] == 13

  @pytest.mark.parametrize('Replay', REPLAYS_UNLIMITED)
  @pytest.mark.parametrize('workers', [1, 2, 5])
  @pytest.mark.parametrize('length,capacity', [(1, 1), (3, 10), (5, 100)])
  def test_restore_workers(self, tmpdir, Replay, workers, length, capacity):
    capacity *= workers
    replay = Replay(length, capacity, directory=tmpdir)
    for step in range(50):
      for worker in range(workers):
        replay.add({'step': step}, worker)
    num_items = np.clip((50 - length + 1) * workers, 0, capacity)
    assert len(replay) == num_items
    replay.save(wait=True)
    replay = Replay(length, capacity, directory=tmpdir)
    replay.load()
    assert len(replay) == num_items
    dataset = iter(replay.dataset())
    for _ in range(len(replay)):
      assert len(next(dataset)['step']) == length

  @pytest.mark.parametrize('Replay', REPLAYS_SAVECHUNKS)
  @pytest.mark.parametrize(
      'length,capacity,chunksize', [(1, 1, 1), (3, 10, 5), (5, 100, 12)])
  def test_restore_chunks_exact(
      self, tmpdir, Replay, length, capacity, chunksize):
    embodied.uuid.reset(debug=True)
    assert len(list(embodied.Path(tmpdir).glob('*.npz'))) == 0
    replay = Replay(length, capacity, directory=tmpdir, chunksize=chunksize)
    for step in range(30):
      replay.add({'step': step})
    num_items = np.clip(30 - length + 1, 0, capacity)
    assert len(replay) == num_items
    replay.save(wait=True)
    filenames = list(embodied.Path(tmpdir).glob('*.npz'))
    lengths = [int(x.stem.split('-')[3]) for x in filenames]
    assert len(filenames) == (int(np.ceil(30 / chunksize)))
    assert sum(lengths) == 30
    assert all(1 <= x <= chunksize for x in lengths)
    replay = Replay(length, capacity, directory=tmpdir, chunksize=chunksize)
    replay.load()
    assert sorted(embodied.Path(tmpdir).glob('*.npz')) == sorted(filenames)
    assert len(replay) == num_items
    dataset = iter(replay.dataset())
    for _ in range(len(replay)):
      assert len(next(dataset)['step']) == length

  @pytest.mark.parametrize('Replay', REPLAYS_SAVECHUNKS)
  @pytest.mark.parametrize('workers', [1, 2, 5])
  @pytest.mark.parametrize(
      'length,capacity,chunksize', [(1, 1, 1), (3, 10, 5), (5, 100, 12)])
  def test_restore_chunks_workers(
      self, tmpdir, Replay, workers, length, capacity, chunksize):
    capacity *= workers
    replay = Replay(length, capacity, directory=tmpdir, chunksize=chunksize)
    for step in range(50):
      for worker in range(workers):
        replay.add({'step': step}, worker)
    num_items = np.clip((50 - length + 1) * workers, 0, capacity)
    assert len(replay) == num_items
    replay.save(wait=True)
    filenames = list(embodied.Path(tmpdir).glob('*.npz'))
    lengths = [int(x.stem.split('-')[3]) for x in filenames]
    assert sum(lengths) == 50 * workers
    replay = Replay(length, capacity, directory=tmpdir, chunksize=chunksize)
    replay.load()
    assert len(replay) == num_items
    dataset = iter(replay.dataset())
    for _ in range(len(replay)):
      assert len(next(dataset)['step']) == length

  @pytest.mark.parametrize('Replay', REPLAYS_UNLIMITED)
  @pytest.mark.parametrize(
      'length,capacity,chunksize',
      [(1, 1, 128), (3, 10, 128), (5, 100, 128), (5, 25, 2)])
  def test_restore_insert(self, tmpdir, Replay, length, capacity, chunksize):
    embodied.uuid.reset(debug=True)
    replay = Replay(length, capacity, directory=tmpdir, chunksize=chunksize)
    inserts = int(1.5 * chunksize)
    for step in range(inserts):
      replay.add({'step': step})
    num_items = np.clip(inserts - length + 1, 0, capacity)
    assert len(replay) == num_items
    replay.save(wait=True)
    replay = Replay(length, capacity, directory=tmpdir)
    replay.load()
    assert len(replay) == num_items
    dataset = iter(replay.dataset())
    for _ in range(len(replay)):
      assert len(next(dataset)['step']) == length
    for step in range(inserts):
      replay.add({'step': step})
    num_items = np.clip(2 * (inserts - length + 1), 0, capacity)
    assert len(replay) == num_items

  @pytest.mark.parametrize('Replay', REPLAYS_UNLIMITED)
  def test_threading(
      self, tmpdir, Replay, length=5, capacity=128, chunksize=32,
      adders=8, samplers=4):
    embodied.uuid.reset(debug=True)
    replay = Replay(length, capacity, directory=tmpdir, chunksize=chunksize)
    running = [True]

    def adder():
      ident = threading.get_ident()
      step = 0
      while running[0]:
        replay.add({'step': step}, worker=ident)
        step += 1
        time.sleep(0.001)

    def sampler():
      ident = threading.get_ident()
      dataset = iter(replay.dataset())
      while running[0]:
        seq = next(dataset)
        assert (seq['step'] - seq['step'][0] == np.arange(length)).all()
        time.sleep(0.001)

    workers = []
    for _ in range(adders):
      workers.append(threading.Thread(target=adder))
    for _ in range(samplers):
      workers.append(threading.Thread(target=sampler))

    try:
      [worker.start() for worker in workers]
      for _ in range(4):

        time.sleep(0.1)
        stats = replay.stats()
        assert stats['inserts'] > 0
        assert stats['samples'] > 0

        print('SAVING')
        replay.save(wait=True)
        time.sleep(0.1)

        print('LOADING')
        # replay = Replay(length, capacity, directory=tmpdir, chunksize=chunksize)
        # replay.clear()
        replay.load()

    finally:
      running[0] = False
      [worker.join() for worker in workers]

    assert len(replay) == capacity


  # @pytest.mark.parametrize('Replay', REPLAYS_UNLIMITED)
  # @pytest.mark.parametrize(
  #     'length,capacity,chunksize,workers',
  #     [(3, 100, 16, 4)])
  # def test_restore_capacity(
  #     self, tmpdir, Replay, length, capacity, chunksize, workers):
  #   embodied.uuid.reset(debug=True)
  #   replay = Replay(length, capacity, directory=tmpdir, chunksize=chunksize)

  #   for step in range(500):
  #     for worker in range(workers):
  #       replay.add({'step': step}, worker=worker)

  #   num_items = np.clip(workers * (500 - length + 1), 0, capacity)
  #   assert len(replay) == num_items
  #   replay.save(wait=True)
  #   replay = Replay(length, capacity, directory=tmpdir)

  #   rng = np.random.default_rng(seed=0)
  #   filenames = sorted(embodied.Path(tmpdir).glob('*.npz'))
  #   # for filename in rng.choice(filenames, min(100, len(filenames)), replace=False):
  #   for filename in filenames[:120]:
  #     filename.remove()

  #   # for step in range(10):
  #   #   for worker in range(workers):
  #   #     replay.add({'step': step}, worker=worker)

  #   replay.load()
  #   assert len(replay) == num_items

  #   # dataset = iter(replay.dataset())
  #   # for _ in range(len(replay)):
  #   #   assert len(next(dataset)['step']) == length
  #   # for step in range(inserts):
  #   #   replay.add({'step': step})
  #   # num_items = np.clip(2 * (inserts - length + 1), 0, capacity)
  #   # assert len(replay) == num_items


  # @pytest.mark.parametrize('Replay', REPLAYS_QUEUES)
  # @pytest.mark.parametrize(
  #     'length,capacity,overlap',
  #     [(1, 1, 0), (5, 10, 3), (10, 5, 2)])
  # def test_queue_single(self, Replay, length, capacity, overlap):
  #   replay = Replay(length, capacity, overlap=overlap)
  #   for step in range(length):
  #     replay.add({'step': step})
  #   dataset = iter(replay.dataset())
  #   seq = next(dataset)
  #   assert (seq['step'] == np.arange(length)).all()

  # @pytest.mark.parametrize('Replay', REPLAYS_QUEUES)
  # @pytest.mark.parametrize(
  #     'length,capacity,overlap',
  #     [(1, 5, 0), (2, 5, 1), (5, 10, 3), (10, 5, 0), (10, 5, 2)])
  # def test_queue_order(self, Replay, length, capacity, overlap):
  #   assert overlap < length
  #   assert 5 <= capacity
  #   replay = Replay(length, capacity, overlap=overlap)
  #   inserts = length + 4 * (length - overlap)
  #   for step in range(inserts):
  #     replay.add({'step': step})
  #   dataset = iter(replay.dataset())
  #   for index in range(len(replay)):
  #     seq = next(dataset)
  #     start = index * (length - overlap)
  #     assert seq['step'][0] == start
  #     assert (seq['step'] - start == np.arange(length)).all()

  # @pytest.mark.parametrize('Replay', REPLAYS_QUEUES)
  # @pytest.mark.parametrize(
  #     'length,capacity,overlap,workers',
  #     [(1, 10, 0, 2), (2, 10, 1, 2), (5, 30, 3, 4)])
  # def test_queue_workers(self, Replay, length, capacity, overlap, workers):
  #   assert overlap < length
  #   assert 5 * workers <= capacity
  #   replay = Replay(length, capacity, overlap=overlap)
  #   inserts = length + 4 * (length - overlap)
  #   for step in range(inserts):
  #     for worker in range(workers):
  #       replay.add({'step': step, 'worker': worker}, worker)
  #   dataset = iter(replay.dataset())
  #   assert len(replay) == 5 * workers
  #   for index in range(5):
  #     for worker in range(workers):
  #       seq = next(dataset)
  #       start = index * (length - overlap)
  #       assert seq['step'][0] == start
  #       assert (seq['worker'] == worker).all()
  #       assert (seq['step'] - start == np.arange(length)).all()
