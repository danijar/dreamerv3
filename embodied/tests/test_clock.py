from concurrent.futures import ThreadPoolExecutor

import pytest

from embodied.core import clock


@pytest.fixture
def clock_server(monkeypatch):
  bindings = {}
  now = [100.0]

  class Server:

    def __init__(self, port, name):
      assert name == 'ClockServer'

    def bind(self, name, fn, workers):
      assert workers == 2
      bindings[name] = fn

    def start(self, block):
      assert block is False

  monkeypatch.setattr(clock.portal, 'Server', Server)
  monkeypatch.setattr(clock.time, 'time', lambda: now[0])
  clock._start_server(1234, replicas=2)

  def call(name, *args):
    with ThreadPoolExecutor(2) as pool:
      futures = [
          pool.submit(bindings[name], replica, *arg)
          for replica, arg in enumerate(args)]
      return [future.result(timeout=5) for future in futures]

  return now, call


@pytest.mark.parametrize('skips', [(True, False), (False, True), (True, True)])
def test_skipped_deadline_remains_due(clock_server, skips):
  now, call = clock_server
  assert call('create', (10,), (10,)) == [0, 0]
  now[0] = 110.0
  assert call('should', (0, skips[0]), (0, skips[1])) == [False, False]
  now[0] = 110.1
  assert call('should', (0, False), (0, False)) == [True, True]
  assert call('should', (0, False), (0, False)) == [False, False]


def test_skipping_before_deadline_does_not_delay_it(clock_server):
  now, call = clock_server
  assert call('create', (10,), (10,)) == [0, 0]
  now[0] = 105.0
  assert call('should', (0, True), (0, False)) == [False, False]
  now[0] = 110.0
  assert call('should', (0, False), (0, False)) == [True, True]
  now[0] = 119.9
  assert call('should', (0, False), (0, False)) == [False, False]
  now[0] = 120.0
  assert call('should', (0, False), (0, False)) == [True, True]


@pytest.mark.parametrize('every, expected', [(0, False), (-1, True)])
def test_disabled_and_always_clocks(clock_server, every, expected):
  now, call = clock_server
  assert call('create', (every,), (every,)) == [0, 0]
  now[0] = 1000.0
  assert call('should', (0, True), (0, False)) == [False, False]
  assert call('should', (0, False), (0, False)) == [expected, expected]


def test_independent_clock_deadlines(clock_server):
  now, call = clock_server
  assert call('create', (10,), (10,)) == [0, 0]
  assert call('create', (20,), (20,)) == [1, 1]
  now[0] = 120.0
  assert call('should', (0, True), (0, False)) == [False, False]
  assert call('should', (1, False), (1, False)) == [True, True]
  assert call('should', (0, False), (0, False)) == [True, True]
  assert call('should', (1, False), (1, False)) == [False, False]
