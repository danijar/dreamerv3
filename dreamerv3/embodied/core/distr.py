import ctypes
import multiprocessing
import sys
import threading
import time
import traceback
import uuid

import numpy as np

from . import basics


class Client:

  def __init__(self, address, timeout_ms=-1, ipv6=False):
    import zmq
    addresses = [address] if isinstance(address, str) else address
    context = zmq.Context.instance()
    self.socket = context.socket(zmq.REQ)
    self.socket.setsockopt(zmq.IDENTITY, uuid.uuid4().bytes)
    self.socket.RCVTIMEO = timeout_ms
    for address in addresses:
      basics.print(f'Client connecting to {address}', color='green')
      ipv6 and self.socket.setsockopt(zmq.IPV6, 1)
      self.socket.connect(address)
    self.result = True

  def __call__(self, data):
    assert isinstance(data, dict), type(data)
    if self.result is None:
      self._receive()
    self.result = None
    self.socket.send(pack(data))
    return self._receive

  def _receive(self):
    try:
      recieved = self.socket.recv()
    except Exception as e:
      raise RuntimeError(f'Failed to receive data from server: {e}')
    self.result = unpack(recieved)
    if self.result.get('type', 'data') == 'error':
      msg = self.result.get('message', None)
      raise RuntimeError(f'Server responded with an error: {msg}')
    return self.result


class Server:

  def __init__(self, address, function, ipv6=False):
    import zmq
    context = zmq.Context.instance()
    self.socket = context.socket(zmq.REP)
    basics.print(f'Server listening at {address}', color='green')
    ipv6 and self.socket.setsockopt(zmq.IPV6, 1)
    self.socket.bind(address)
    self.function = function

  def run(self):
    while True:
      payload = self.socket.recv()
      inputs = unpack(payload)
      assert isinstance(inputs, dict), type(inputs)
      try:
        result = self.function(inputs)
        assert isinstance(result, dict), type(result)
      except Exception as e:
        result = {'type': 'error', 'message': str(e)}
        self.socket.send(pack(payload))
        raise
      payload = pack(result)
      self.socket.send(payload)


class BatchServer:

  def __init__(self, address, batch, function, ipv6=False):
    import zmq
    context = zmq.Context.instance()
    self.socket = context.socket(zmq.ROUTER)
    basics.print(f'BatchServer listening at {address}', color='green')
    ipv6 and self.socket.setsockopt(zmq.IPV6, 1)
    self.socket.bind(address)
    self.function = function
    self.batch = batch

  def run(self):
    inputs = None
    while True:
      addresses = []
      for i in range(self.batch):
        address, empty, payload = self.socket.recv_multipart()
        data = unpack(payload)
        assert isinstance(data, dict), type(data)
        if inputs is None:
          inputs = {
              k: np.empty((self.batch, *v.shape), v.dtype)
              for k, v in data.items() if not isinstance(v, str)}
        for key, value in data.items():
          inputs[key][i] = value
        addresses.append(address)
      try:
        results = self.function(inputs, [x.hex() for x in addresses])
        assert isinstance(results, dict), type(results)
        for key, value in results.items():
          if not isinstance(value, str):
            assert len(value) == self.batch, (key, value.shape)
      except Exception as e:
        results = {'type': 'error', 'message': str(e)}
        self._respond(addresses, results)
        raise
      self._respond(addresses, results)

  def _respond(self, addresses, results):
    for i, address in enumerate(addresses):
      payload = pack({
          k: v if isinstance(v, str) else v[i]
          for k, v in results.items()})
      self.socket.send_multipart([address, b'', payload])


def pack(data):
  import msgpack
  return msgpack.packb({
      k: v if isinstance(v, str) else (v.shape, v.dtype.name, v.tobytes())
      for k, v in data.items()})


def unpack(buffer):
  import msgpack
  tostr = lambda x: x.decode('utf-8') if isinstance(x, bytes) else x
  return {
      tostr(k): (
          tostr(v) if isinstance(v, (str, bytes))
          else np.frombuffer(v[2], tostr(v[1])).reshape(v[0]))
      for k, v in msgpack.unpackb(buffer).items()}


class Thread(threading.Thread):

  lock = threading.Lock()

  def __init__(self, fn, *args, name=None):
    self.fn = fn
    self.exitcode = None
    name = name or fn.__name__
    super().__init__(target=self._wrapper, args=args, name=name, daemon=True)

  def _wrapper(self, *args):
    try:
      self.fn(*args)
    except Exception:
      with self.lock:
        print('-' * 79)
        print(f'Exception in worker: {self.name}')
        print('-' * 79)
        print(''.join(traceback.format_exception(*sys.exc_info())))
        self.exitcode = 1
      raise
    self.exitcode = 0

  def terminate(self):
    if not self.is_alive():
      return
    if hasattr(self, '_thread_id'):
      thread_id = self._thread_id
    else:
      thread_id = [k for k, v in threading._active.items() if v is self][0]
    result = ctypes.pythonapi.PyThreadState_SetAsyncExc(
        ctypes.c_long(thread_id), ctypes.py_object(SystemExit))
    if result > 1:
      ctypes.pythonapi.PyThreadState_SetAsyncExc(
          ctypes.c_long(thread_id), None)
    print('Shut down worker:', self.name)


class Process:

  lock = None

  def __init__(self, fn, *args, name=None):
    mp = multiprocessing.get_context('spawn')
    if Process.lock is None:
      Process.lock = mp.Lock()
    name = name or fn.__name__
    self._process = mp.Process(
        target=self._wrapper, args=(Process.lock, fn, *args),
        name=name)

  def start(self):
    self._process.start()

  @property
  def name(self):
    return self._process.name

  @property
  def exitcode(self):
    return self._process.exitcode

  def terminate(self):
    self._process.terminate()
    print('Shut down worker:', self.name)

  def _wrapper(self, lock, fn, *args):
    try:
      fn(*args)
    except Exception:
      with lock:
        print('-' * 79)
        print(f'Exception in worker: {self.name}')
        print('-' * 79)
        print(''.join(traceback.format_exception(*sys.exc_info())))
      raise


def run(workers):
  [x.start() for x in workers]
  while True:
    for worker in workers:
      if worker.exitcode not in (None, 0):
        # Wait for everybody who wants to print their error messages.
        time.sleep(1)
        [x.terminate() for x in workers if x is not worker]
        print(f'Stopped workers due to crash in {worker.name}.')
        return
    time.sleep(0.1)
