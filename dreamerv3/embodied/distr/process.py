import multiprocessing as mp
import os
import queue
import signal
import sys

import cloudpickle

from . import utils


class Process:

  initializers = []
  current_name = None

  def __init__(self, fn, *args, name=None, start=False, pass_running=False):
    name = name or fn.__name__
    fn = cloudpickle.dumps(fn)
    inits = cloudpickle.dumps(self.initializers)
    context = mp.get_context()
    self.errqueue = context.Queue()
    self.exception = None
    self.process = context.Process(target=self._wrapper, name=name, args=(
        fn, name, args, utils.get_print_lock(), self.errqueue, inits))
    self.started = False
    start and self.start()

  @property
  def name(self):
    return self.process.name

  @property
  def pid(self):
    return self.process.pid

  @property
  def running(self):
    running = self.process.is_alive()
    if running:
      assert self.exitcode is None, (self.name, self.exitcode)
    return running

  @property
  def exitcode(self):
    return self.process.exitcode

  def start(self):
    assert not self.started
    self.started = True
    self.process.start()

  def check(self):
    if self.process.exitcode not in (None, 0):
      if self.exception is None:
        try:
          self.exception = self.errqueue.get(timeout=0.1)
        except queue.Empty:
          if self.exitcode in (-15, 15):
            msg = 'Process was terminated.'
          else:
            msg = f'Process excited with code {self.exitcode}'
          self.exception = RuntimeError(msg)
      self.kill()
      raise self.exception

  def join(self, timeout=None):
    if self.exitcode in (-15, 15):
      assert not self.running
      return
    self.check()
    if self.running:
      self.process.join(timeout)

  def kill(self):
    utils.kill_subprocs(self.pid)
    if self.running:
      self.process.terminate()
      self.process.join(timeout=0.1)
    if self.running:
      try:
        os.kill(self.pid, signal.SIGKILL)
        self.process.join(timeout=0.1)
      except ProcessLookupError:
        pass
    if self.running:
      print(f'Process {self.name} did not shut down yet.')

  def __repr__(self):
    attrs = ('name', 'pid', 'running', 'exitcode')
    attrs = [f'{k}={getattr(self, k)}' for k in attrs]
    return f'{type(self).__name__}(' + ', '.join(attrs) + ')'

  @staticmethod
  def _wrapper(fn, name, args, lock, errqueue, inits):
    Process.current_name = name
    try:
      for init in cloudpickle.loads(inits):
        init()
      fn = cloudpickle.loads(fn)
      fn(*args)
      sys.exit(0)
    except Exception as e:
      utils.warn_remote_error(e, name, lock)
      errqueue.put(e)
      sys.exit(1)
    finally:
      pid = mp.current_process().pid
      utils.kill_subprocs(pid)


class StoppableProcess(Process):

  def __init__(self, fn, *args, name=None, start=False):
    self.runflag = mp.get_context().Event()
    def fn2(runflag, *args):
      assert runflag is not None
      context = utils.Context(runflag.is_set)
      fn(context, *args)
    super().__init__(fn2, self.runflag, *args, name=name, start=start)

  def start(self):
    self.runflag.set()
    super().start()

  def stop(self, wait=1):
    self.check()
    if not self.running:
      return
    self.runflag.clear()
    if wait is True:
      self.join()
    elif wait:
      self.join(wait)
      self.kill()
