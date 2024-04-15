import pathlib
import queue
import sys
import time
import traceback

sys.path.append(str(pathlib.Path(__file__).parent.parent.parent.parent))

import embodied
import pytest


class TestThread:

  def test_kill(self):
    def fn():
      while True:
        time.sleep(0.01)
    worker = embodied.distr.Thread(fn, start=True)
    worker.kill()
    assert not worker.running
    worker.join()
    assert not worker.running
    worker.join()  # Noop

  def test_stop(self):
    def fn(context, q):
      q.put('start')
      while context.running:
        time.sleep(0.01)
      q.put('stop')
    q = queue.SimpleQueue()
    worker = embodied.distr.StoppableThread(fn, q)
    worker.start()
    worker.stop()
    assert q.get() == 'start'
    assert q.get() == 'stop'

  def test_exitcode(self):
    worker = embodied.distr.Thread(lambda: None)
    assert worker.exitcode is None
    worker.start()
    worker.join()
    assert worker.exitcode == 0

  def test_exception(self):
    def fn1234(q):
      q.put(42)
      raise KeyError('foo')
    q = queue.SimpleQueue()
    worker = embodied.distr.Thread(fn1234, q, start=True)
    q.get()
    time.sleep(0.01)
    assert not worker.running
    assert worker.exitcode == 1
    with pytest.raises(KeyError) as info:
      worker.check()
    worker.kill()  # Shoud not hang or reraise.
    with pytest.raises(KeyError) as info:
      worker.check()  # Can reraise multiple times.
    assert repr(info.value) == "KeyError('foo')"
    e = info.value
    typ, tb = type(e), e.__traceback__
    tb = ''.join(traceback.format_exception(typ, e, tb))
    assert 'Traceback' in tb
    assert ' File ' in tb
    assert 'fn1234' in tb
    assert "KeyError: 'foo'" in tb

  def test_nested_exception(self):
    threads = []
    def inner():
      raise KeyError('foo')
    def outer():
      child = embodied.distr.Thread(inner, start=True)
      threads.append(child)
      while True:
        child.check()
        time.sleep(0.01)
    parent = embodied.distr.Thread(outer)
    threads.append(parent)
    parent.start()
    time.sleep(0.1)
    with pytest.raises(KeyError) as info:
      parent.check()
    assert repr(info.value) == "KeyError('foo')"
    assert not threads[0].running
    assert not threads[1].running
