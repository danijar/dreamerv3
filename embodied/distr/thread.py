import threading

from . import utils


class Thread:

  def __init__(self, fn, *args, name=None, start=False):
    self.fn = fn
    self._exitcode = None
    self.exception = None
    name = name or fn.__name__
    self.old_name = name[:]
    self.thread = threading.Thread(
        target=self._wrapper, args=args, name=name, daemon=True)
    self.started = False
    start and self.start()

  @property
  def name(self):
    return self.thread.name

  @property
  def ident(self):
    return self.thread.ident

  @property
  def running(self):
    running = self.thread.is_alive()
    if running:
      assert self.exitcode is None, (self.name, self.exitcode)
    return running

  @property
  def exitcode(self):
    return self._exitcode

  def start(self):
    assert not self.started
    self.started = True
    self.thread.start()

  def check(self):
    assert self.started
    if self.exception is not None:
      raise self.exception

  def join(self, timeout=None):
    self.check()
    self.thread.join(timeout)

  def kill(self):
    if not self.running:
      return
    utils.kill_thread(self.thread)
    self.thread.join(0.1)
    if self.running:
      print(f'Thread {self.name} did not shut down yet.')

  def __repr__(self):
    attrs = ('name', 'ident', 'running', 'exitcode')
    attrs = [f'{k}={getattr(self, k)}' for k in attrs]
    return f'{type(self).__name__}(' + ', '.join(attrs) + ')'

  def _wrapper(self, *args):
    try:
      self.fn(*args)
    except SystemExit:
      return
    except Exception as e:
      utils.warn_remote_error(e, self.name)
      self._exitcode = 1
      self.exception = e
    finally:
      if self._exitcode is None:
        self._exitcode = 0


class StoppableThread(Thread):

  def __init__(self, fn, *args, name=None, start=False):
    self.runflag = None
    def fn2(*args):
      assert self.runflag is not None
      context = utils.Context(lambda: self.runflag)
      fn(context, *args)
    super().__init__(fn2, *args, name=name, start=start)

  def start(self):
    self.runflag = True
    super().start()

  def stop(self, wait=1):
    self.runflag = False
    self.check()
    if not self.running:
      return
    if wait is True:
      self.join()
    elif wait:
      self.join(wait)
      self.kill()
