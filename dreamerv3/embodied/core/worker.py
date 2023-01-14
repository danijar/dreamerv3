import atexit
import concurrent.futures
import enum
import os
import sys
import time
import traceback
from functools import partial as bind


class Worker:

  initializers = []

  def __init__(self, fn, strategy='thread', state=False):
    if not state:
      fn = lambda s, *args, fn=fn, **kwargs: (s, fn(*args, **kwargs))
    inits = self.initializers
    self.impl = {
        'blocking': BlockingWorker,
        'thread': ThreadWorker,
        'process': bind(ProcessPipeWorker, initializers=inits),
        'daemon': bind(ProcessPipeWorker, initializers=inits, daemon=True),
        'process_slow': bind(ProcessWorker, initializers=inits),
    }[strategy](fn)
    self.promise = None

  def __call__(self, *args, **kwargs):
    self.promise and self.promise()  # Raise previous exception if any.
    self.promise = self.impl(*args, **kwargs)
    return self.promise

  def wait(self):
    return self.impl.wait()

  def close(self):
    self.impl.close()


class BlockingWorker:

  def __init__(self, fn):
    self.fn = fn
    self.state = None

  def __call__(self, *args, **kwargs):
    self.state, result = self.fn(self.state, *args, **kwargs)
    # return lambda: result
    return lambda result=result: result

  def wait(self):
    pass

  def close(self):
    pass


class ThreadWorker:

  def __init__(self, fn):
    self.fn = fn
    self.state = None
    self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    self.futures = []

  def __call__(self, *args, **kwargs):
    future = self.executor.submit(self._worker, *args, **kwargs)
    self.futures.append(future)
    future.add_done_callback(lambda f: self.futures.remove(f))
    return future.result

  def wait(self):
    concurrent.futures.wait(self.futures)

  def close(self):
    self.executor.shutdown(wait=False, cancel_futures=True)

  def _worker(self, *args, **kwargs):
    self.state, output = self.fn(self.state, *args, **kwargs)
    return output


class ProcessWorker:

  def __init__(self, fn, initializers=()):
    import cloudpickle
    import multiprocessing
    fn = cloudpickle.dumps(fn)
    initializers = cloudpickle.dumps(initializers)
    self.executor = concurrent.futures.ProcessPoolExecutor(
        max_workers=1, mp_context=multiprocessing.get_context('spawn'),
        initializer=self._initializer, initargs=(fn, initializers))
    self.futures = []

  def __call__(self, *args, **kwargs):
    future = self.executor.submit(self._worker, *args, **kwargs)
    self.futures.append(future)
    future.add_done_callback(lambda f: self.futures.remove(f))
    return future.result

  def wait(self):
    concurrent.futures.wait(self.futures)

  def close(self):
    self.executor.shutdown(wait=False, cancel_futures=True)

  @staticmethod
  def _initializer(fn, initializers):
    global _FN, _STATE
    import cloudpickle
    _FN = cloudpickle.loads(fn)
    _STATE = None
    for initializer in cloudpickle.loads(initializers):
      initializers()

  @staticmethod
  def _worker(*args, **kwargs):
    global _FN, _STATE
    _STATE, output = _FN(_STATE, *args, **kwargs)
    return output


class ProcessPipeWorker:

  def __init__(self, fn, initializers=(), daemon=False):
    import multiprocessing
    import cloudpickle
    self._context = multiprocessing.get_context('spawn')
    self._pipe, pipe = self._context.Pipe()
    fn = cloudpickle.dumps(fn)
    initializers = cloudpickle.dumps(initializers)
    self._process = self._context.Process(
        target=self._loop,
        args=(pipe, fn, initializers),
        daemon=daemon)
    self._process.start()
    self._nextid = 0
    self._results = {}
    assert self._submit(Message.OK)()
    atexit.register(self.close)

  def __call__(self, *args, **kwargs):
    return self._submit(Message.RUN, (args, kwargs))

  def wait(self):
    pass

  def close(self):
    try:
      self._pipe.send((Message.STOP, self._nextid, None))
      self._pipe.close()
    except (AttributeError, IOError):
      pass  # The connection was already closed.
    try:
      self._process.join(0.1)
      if self._process.exitcode is None:
        try:
          os.kill(self._process.pid, 9)
          time.sleep(0.1)
        except Exception:
          pass
    except (AttributeError, AssertionError):
      pass

  def _submit(self, message, payload=None):
    callid = self._nextid
    self._nextid += 1
    self._pipe.send((message, callid, payload))
    return Future(self._receive, callid)

  def _receive(self, callid):
    while callid not in self._results:
      try:
        message, callid, payload = self._pipe.recv()
      except (OSError, EOFError):
        raise RuntimeError('Lost connection to worker.')
      if message == Message.ERROR:
        raise Exception(payload)
      assert message == Message.RESULT, message
      self._results[callid] = payload
    return self._results.pop(callid)

  @staticmethod
  def _loop(pipe, function, initializers):
    try:
      callid = None
      state = None
      import cloudpickle
      initializers = cloudpickle.loads(initializers)
      function = cloudpickle.loads(function)
      [fn() for fn in initializers]
      while True:
        if not pipe.poll(0.1):
          continue  # Wake up for keyboard interrupts.
        message, callid, payload = pipe.recv()
        if message == Message.OK:
          pipe.send((Message.RESULT, callid, True))
        elif message == Message.STOP:
          return
        elif message == Message.RUN:
          args, kwargs = payload
          state, result = function(state, *args, **kwargs)
          pipe.send((Message.RESULT, callid, result))
        else:
          raise KeyError(f'Invalid message: {message}')
    except (EOFError, KeyboardInterrupt):
      return
    except Exception:
      stacktrace = ''.join(traceback.format_exception(*sys.exc_info()))
      print(f'Error inside process worker: {stacktrace}.', flush=True)
      pipe.send((Message.ERROR, callid, stacktrace))
      return
    finally:
      try:
        pipe.close()
      except Exception:
        pass


class Future:

  def __init__(self, receive, callid):
    self._receive = receive
    self._callid = callid
    self._result = None
    self._complete = False

  def __call__(self):
    if not self._complete:
      self._result = self._receive(self._callid)
      self._complete = True
    return self._result


class Message(enum.Enum):

  OK = 1
  RUN = 2
  RESULT = 3
  STOP = 4
  ERROR = 5
