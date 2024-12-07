import pathlib
import sys
import threading
import time
from collections import defaultdict

sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

import embodied
import numpy as np
import pytest


REPLAYS = [
    ('Replay', embodied.replay.Replay),
]

STEP = {
    'image': np.zeros((64, 64, 3), np.uint8),
    'vector': np.zeros(1024, np.float32),
    'action': np.zeros(12, np.float32),
    'is_first': np.array(False),
    'is_last': np.array(False),
    'is_terminal': np.array(False),
}


class TestReplay:

  @pytest.mark.parametrize('name,Replay', REPLAYS)
  def test_speed(self, name, Replay, inserts=2e5, workers=8, samples=1e5):
    print('')
    initial = time.time()
    replay = Replay(length=32, capacity=1e5, chunksize=1024)
    start = time.time()
    for step in range(int(inserts / workers)):
      for worker in range(workers):
        replay.add(STEP, worker)
    duration = time.time() - start
    print(name, 'inserts/sec:', int(inserts / duration))
    start = time.time()
    dataset = iter(replay.dataset(1))
    for _ in range(int(samples)):
      next(dataset)
    duration = time.time() - start
    print(name, 'samples/sec:', int(samples / duration))
    print(name, 'total duration:', time.time() - initial)

  @pytest.mark.parametrize('chunksize', [64, 128, 256, 512, 1024, 2048, 4096])
  def test_chunk_size(self, chunksize, inserts=2e5, workers=8, samples=2e5):
    print('')
    initial = time.time()
    replay = embodied.replay.Replay(length=64, chunksize=chunksize)
    start = time.time()
    for step in range(int(inserts / workers)):
      for worker in range(workers):
        replay.add(STEP, worker)
    duration = time.time() - start
    print('chunksize', chunksize, 'inserts/sec:', int(inserts / duration))
    start = time.time()
    dataset = iter(replay.dataset(1))
    for _ in range(int(samples)):
      next(dataset)
    duration = time.time() - start
    print('chunksize', chunksize, 'samples/sec:', int(samples / duration))
    print('chunksize', chunksize, 'total duration:', time.time() - initial)

  @pytest.mark.parametrize('name,Replay', REPLAYS)
  def test_removal(self, name, Replay, inserts=1e6, workers=1):
    print('')
    replay = Replay(length=32, capacity=1e5, chunksize=1024)
    start = time.time()
    for step in range(int(inserts)):
        replay.add(STEP)
    duration = time.time() - start
    print(name, 'inserts/sec:', int(inserts / duration))

  @pytest.mark.parametrize('name,Replay', REPLAYS)
  def test_parallel(self, tmpdir, name, Replay, duration=5):
    print('')
    replay = Replay(length=16, capacity=1e4, chunksize=32, directory=tmpdir)

    running = True
    adds = defaultdict(int)
    samples = defaultdict(int)
    saves = defaultdict(int)
    errors = []

    def adder():
      try:
        ident = threading.get_ident()
        step = {'foo': np.zeros((64, 64, 3))}
        while running:
          replay.add(step, threading.get_ident())
          adds[ident] += 1
      except Exception as e:
        errors.append(e)
        raise

    def sampler():
      try:
        ident = threading.get_ident()
        dataset = iter(replay.dataset(1))
        while running:
          next(dataset)
          samples[ident] += 1
      except Exception as e:
        errors.append(e)
        raise

    def saver():
      try:
        ident = threading.get_ident()
        while running:
          data = replay.save()
          time.sleep(0.1)
          replay.load(data)
          time.sleep(0.1)
          saves[ident] += 1
      except Exception as e:
        errors.append(e)
        raise

    workers = [threading.Thread(target=saver)]
    for _ in range(32):
      workers.append(threading.Thread(target=adder))
    for _ in range(8):
      workers.append(threading.Thread(target=sampler))

    print(f'Starting {len(workers)} threads')
    [x.start() for x in workers]
    time.sleep(duration)
    running = False
    [x.join() for x in workers]
    if errors:
      print(f'Found {len(errors)} errors: {errors}')
      raise errors[0]
    print('adds/sec:', sum(adds.values()) / duration)
    print('samples/sec:', sum(samples.values()) / duration)
    print('save_load/sec:', sum(saves.values()) / duration)
