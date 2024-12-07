import os
import pathlib
import sys
import time
from collections import defaultdict

sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

import zerofun
import numpy as np


class TestDistr:

  def test_batched_throughput(self, clients=32, batch=16, workers=4):
    assert int(os.popen('ulimit -n').read()) > 1024

    addr = f'tcp://localhost:{zerofun.get_free_port()}'
    stats = defaultdict(int)
    barrier = zerofun.mp.Barrier(1 + clients)

    def client(context, addr, barrier):
      data = {
          'foo': np.zeros((64, 64, 3,), np.uint8),
          'bar': np.zeros((1024,), np.float32),
          'baz': np.zeros((), bool),
      }
      client = zerofun.Client(addr)
      client.connect()
      barrier.wait()
      while context.running:
        client.function(data).result()

    def workfn(data):
      time.sleep(0.002)
      return data, data

    def donefn(data):
      stats['batches'] += 1
      stats['frames'] += len(data['foo'])
      stats['nbytes'] += sum(x.nbytes for x in data.values())

    procs = [
        zerofun.StoppableProcess(client, addr, barrier, start=True)
        for _ in range(clients)]

    server = zerofun.Server(addr)
    # server = zerofun.Server2(addr)

    server.bind('function', workfn, donefn, batch=batch, workers=workers)
    with server:
      barrier.wait()
      start = time.time()
      while True:
        server.check()
        now = time.time()
        dur = now - start
        print(
            f'{stats["batches"] / dur:.2f} bat/s ' +
            f'{stats["frames"] / dur:.2f} frm/s ' +
            f'{stats["nbytes"] / dur / (1024 ** 3):.2f} gib/s')
        stats.clear()
        start = now
        time.sleep(1)
    [x.stop() for x in procs]

  #############################################################################

  def test_proxy_throughput(self, clients=32, batch=16, workers=4):
    assert int(os.popen('ulimit -n').read()) > 1024

    def client(context, outer_addr, barrier):
      data = {
          'foo': np.zeros((64, 64, 3,), np.uint8),
          'bar': np.zeros((1024,), np.float32),
          'baz': np.zeros((), bool),
      }
      client = zerofun.Client(outer_addr)
      client.connect()
      barrier.wait()
      while context.running:
        client.function(data).result()

    def proxy(context, outer_addr, inner_addr, barrier):
      client = zerofun.Client(
          inner_addr, pings=0, maxage=0, name='ProxyInner')
      client.connect()
      server = zerofun.Server(
          outer_addr, errors=True, name='ProxyOuter')
      def function(data):
        return client.function(data).result()
      server.bind('function', function, batch=batch, workers=workers)
      with server:
        barrier.wait()
        while context.running:
          server.check()
          time.sleep(0.1)

    def backend(context, inner_addr, barrier):
      stats = defaultdict(int)
      def workfn(data):
        time.sleep(0.002)
        return data, data
      def donefn(data):
        stats['batches'] += 1
        stats['frames'] += len(data['foo'])
        stats['nbytes'] += sum(x.nbytes for x in data.values())
      server = zerofun.Server(
          inner_addr, errors=True, name='Backend')
      server.bind('function', workfn, donefn, workers=workers)
      with server:
        barrier.wait()
        start = time.time()
        while context.running:
          server.check()
          now = time.time()
          dur = now - start
          print(
              f'{stats["batches"] / dur:.2f} bat/s ' +
              f'{stats["frames"] / dur:.2f} frm/s ' +
              f'{stats["nbytes"] / dur / (1024**3):.2f} gib/s')
          stats.clear()
          start = now
          time.sleep(1)

    inner_addr = 'ipc:///tmp/test-inner'
    outer_addr = 'ipc:///tmp/test-outer'
    barrier = zerofun.mp.Barrier(2 + clients)
    procs = [
        zerofun.StoppableProcess(client, outer_addr, barrier)
        for _ in range(clients)]
    procs.append(zerofun.StoppableProcess(
        proxy, outer_addr, inner_addr, barrier))
    procs.append(zerofun.StoppableProcess(
        backend, inner_addr, barrier))
    zerofun.run(procs)


if __name__ == '__main__':
  TestDistr().test_batched_throughput()  # 4100 frm/s Server
  # TestDistr().test_batched_throughput()  # 4200 frm/s Server2
  TestDistr().test_proxy_throughput()  # 3000 frm/s
