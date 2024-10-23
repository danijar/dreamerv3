import collections
import time

import numpy as np

from ..core import printing

from . import process
from . import server
from . import sockets


class ProcServer:

  def __init__(
      self, address, name='Server', ipv6=False, workers=1, errors=True):
    self.address = address
    self.inner = f'ipc:///tmp/inner{np.random.randint(2 ** 32)}'
    self.name = name
    self.ipv6 = ipv6
    self.server = server.Server(self.inner, name, ipv6, workers, errors)
    self.batches = {}
    self.batcher = None

  def bind(self, name, workfn, logfn=None, workers=0, batch=0):
    self.batches[name] = batch
    self.server.bind(name, workfn, logfn, workers, batch=0)

  def start(self):
    self.batcher = process.StoppableProcess(
        self._batcher, self.address, self.inner,
        self.batches, self.name, self.ipv6, name='batcher', start=True)
    self.server.start()

  def check(self):
    self.batcher.check()
    self.server.check()

  def close(self):
    self.server.close()
    self.batcher.stop()
    assert not self.batcher.running

  def run(self):
    try:
      self.start()
      while True:
        self.check()
        time.sleep(1)
    finally:
      self.close()

  def stats(self):
    return self.server.stats()

  def __enter__(self):
    self.start()
    return self

  def __exit__(self, type, value, traceback):
    self.close()

  @staticmethod
  def _batcher(context, address, inner, batches, name, ipv6):

    socket = sockets.ServerSocket(address, ipv6)
    inbound = sockets.ClientSocket(identity=0, pings=0, maxage=0)
    inbound.connect(inner, timeout=120)
    queues = collections.defaultdict(list)
    buffers = collections.defaultdict(dict)
    pending = {}
    printing.print_(f'[{name}] Listening at {address}')

    while context.running:

      result = socket.receive()
      if result:
        addr, rid, name, payload = result
        batch = batches.get(name, None)
        if batch is not None:
          if batch:
            queue = queues[name]
            queue.append((addr, rid, payload))
            if len(queue) == batch:
              addrs, rids, payloads = zip(*queue)
              queue.clear()
              datas = [sockets.unpack(x) for x in payloads]
              idx = range(batch)
              bufs = buffers[name]
              for key, value in datas[0].items():
                bufs[key] = np.stack(
                    [datas[i][key] for i in idx], out=bufs.get(key, None))
              payload = sockets.pack(bufs)
              rid = inbound.send_call(name, payload)
              pending[rid] = (name, addrs, rids)
          else:
            inner_rid = inbound.send_call(name, payload)
            pending[inner_rid] = (name, addr, rid)
        else:
          socket.send_error(addr, rid, f'Unknown method {name}.')

      try:
        result = inbound.receive()
        if result:
          inner_rid, payload = result
          name, addr, rid = pending.pop(inner_rid)
          if batches[name]:
            addrs, rids = addr, rid
            result = sockets.unpack(payload)
            results = [
                {k: v[i] for k, v in result.items()}
                for i in range(batches[name])]
            payloads = [sockets.pack(x) for x in results]
            for addr, rid, payload in zip(addrs, rids, payloads):
              socket.send_result(addr, rid, payload)
          else:
            socket.send_result(addr, rid, payload)
      except sockets.RemoteError as e:
        inner_rid, msg = e.args[:2]
        name, addr, rid = pending.pop(inner_rid)
        if batches[name]:
          addrs, rids = addr, rid
          for addr, rid in zip(addrs, rids):
            socket.send_error(addr, rid, msg)
        else:
          socket.send_error(addr, rid, msg)

    socket.close()
    inbound.close()
