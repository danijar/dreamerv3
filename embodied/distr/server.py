import concurrent.futures
import time
import traceback
from collections import deque, namedtuple

import numpy as np

from ..core import agg
from ..core import printing
from . import sockets
from . import pool as poollib
from . import thread


Method = namedtuple('Method', (
    'name,workfn,donefn,pool,workers,batched,insize,inqueue,inprog'))


class Server:

  def __init__(
      self, address, name='Server', ipv6=False, workers=1, errors=True):
    self.address = address
    self.workers = workers
    self.name = name
    self.errors = errors
    self.ipv6 = ipv6
    self.methods = {}
    self.default_pool = poollib.ThreadPool(workers, 'work')
    self.other_pools = []
    self.done_pool = poollib.ThreadPool(1, 'log')
    self.result_set = set()
    self.done_queue = deque()
    self.done_proms = deque()
    self.agg = agg.Agg()
    self.loop = thread.StoppableThread(self._loop, name=f'{name}_loop')
    self.exception = None

  def bind(self, name, workfn, donefn=None, workers=0, batch=0):
    if workers:
      pool = poollib.ThreadPool(workers, name)
      self.other_pools.append(pool)
    else:
      workers = self.workers
      pool = self.default_pool
    batched = (batch > 0)
    insize = max(1, batch)
    self.methods[name] = Method(
        name, workfn, donefn, pool, workers, batched, insize, deque(), [0])

  def start(self):
    self.loop.start()

  def check(self):
    self.loop.check()
    [not x.done() or x.result() for x in self.result_set.copy()]
    [not x.done() or x.result() for x in self.done_proms.copy()]
    if self.exception:
      exception = self.exception
      self.exception = None
      raise exception

  def close(self):
    self._print('Shutting down')
    self.loop.stop()
    self.default_pool.close()
    self.done_pool.close()
    for pool in self.other_pools:
      pool.close()

  def run(self):
    try:
      self.start()
      while True:
        self.check()
        time.sleep(1)
    finally:
      self.close()

  def __enter__(self):
    self.start()
    return self

  def __exit__(self, type, value, traceback):
    self.close()

  def stats(self):
    return {
        **self.agg.result(),
        'result_set': len(self.result_set),
        'done_queue': len(self.done_queue),
        'done_proms': len(self.done_proms),
    }

  def _loop(self, context):
    socket = sockets.ServerSocket(self.address, self.ipv6)
    self._print(f'Listening at {self.address}')
    while context.running:
      now = time.time()
      result = socket.receive()
      self._handle_request(socket, result, now)
      for method in self.methods.values():
        self._handle_input(method, now)
      self._handle_results(socket, now)
      self._handle_dones()
      time.sleep(0.0001)
    socket.close()

  def _handle_request(self, socket, result, now):
    if result is None:
      return
    addr, rid, name, payload = result
    method = self.methods.get(name, None)
    if not method:
      socket.send_error(addr, rid, f'Unknown method {name}.')
      return
    method.inqueue.append((addr, rid, payload, now))
    self._handle_input(method, now)

  def _handle_input(self, method, now):
    if len(method.inqueue) < method.insize:
      return
    if method.inprog[0] >= 2 * method.workers:
      return
    method.inprog[0] += 1
    if method.batched:
      inputs = [method.inqueue.popleft() for _ in range(method.insize)]
      addr, rid, payload, recvd = zip(*inputs)
    else:
      addr, rid, payload, recvd = method.inqueue.popleft()
    future = method.pool.submit(self._work, method, addr, rid, payload, recvd)
    future.method = method
    future.addr = addr
    future.rid = rid
    self.result_set.add(future)
    if method.donefn:
      self.done_queue.append(future)

  def _handle_results(self, socket, now):
    completed, self.result_set = concurrent.futures.wait(
        self.result_set, 0, concurrent.futures.FIRST_COMPLETED)
    for future in completed:
      method = future.method
      try:
        result = future.result()
        addr, rid, payload, logs, recvd = result
        if method.batched:
          for addr, rid, payload in zip(addr, rid, payload):
            socket.send_result(addr, rid, payload)
          for recvd in recvd:
            self.agg.add(method.name, now - recvd, ('min', 'avg', 'max'))
        else:
          socket.send_result(addr, rid, payload)
          self.agg.add(method.name, now - recvd, ('min', 'avg', 'max'))
      except Exception as e:
        print(f'Exception in server {self.name}:')
        typ, tb = type(e), e.__traceback__
        full = ''.join(traceback.format_exception(typ, e, tb)).strip('\n')
        print(full)
        if method.batched:
          for addr, rid in zip(future.addr, future.rid):
            socket.send_error(addr, rid, repr(e))
        else:
          socket.send_error(future.addr, future.rid, repr(e))
        if self.errors:
          self.exception = e
      finally:
        if not method.donefn:
          method.inprog[0] -= 1

  def _handle_dones(self):
    while self.done_queue and self.done_queue[0].done():
      future = self.done_queue.popleft()
      if future.exception():
        continue
      addr, rid, payload, logs, recvd = future.result()
      future2 = self.done_pool.submit(future.method.donefn, logs)
      future2.method = future.method
      self.done_proms.append(future2)
    while self.done_proms and self.done_proms[0].done():
      future = self.done_proms.popleft()
      future.result()
      future.method.inprog[0] -= 1

  def _work(self, method, addr, rid, payload, recvd):
    if method.batched:
      data = [sockets.unpack(x) for x in payload]
      data = {
          k: np.stack([data[i][k] for i in range(method.insize)])
          for k, v in data[0].items()}
    else:
      data = sockets.unpack(payload)
    if method.donefn:
      result, logs = method.workfn(data)
    else:
      result = method.workfn(data)
      result = result or {}
      logs = None
    if method.batched:
      results = [
          {k: v[i] for k, v in result.items()} for i in range(method.insize)]
      payload = [sockets.pack(x) for x in results]
    else:
      payload = sockets.pack(result)
    return addr, rid, payload, logs, recvd

  def _print(self, text):
    printing.print_(f'[{self.name}] {text}')
