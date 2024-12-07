import threading
import time

import portal


CLIENT = None
REPLICA = None


def setup(is_server, replica, replicas, port, addr):
  global CLIENT, REPLICA
  assert CLIENT is None
  if replicas <= 1:
    return
  print('CLOCK PORT:', port)
  print('CLOCK ADDR:', addr)
  if is_server:
    _start_server(port, replicas)
  client = portal.Client(addr, 'ClockClient')
  client.connect()
  CLIENT = client
  REPLICA = replica


def _start_server(port, replicas):

  clocks = []
  requests = []
  result = [None]
  receive = threading.Barrier(replicas)
  respond = threading.Barrier(replicas)

  def create(replica, every):
    requests.append(every)
    receive.wait()
    if replica == 0:
      assert len(requests) == replicas, (len(requests), replicas)
      assert all(x == every for x in requests)
      clockid = len(clocks)
      clocks.append([float(every), time.time()])
      result[0] = clockid
      requests.clear()
    respond.wait()
    return result[0]

  def should(replica, clockid, skip):
    requests.append((clockid, skip))
    receive.wait()
    if replica == 0:
      assert len(requests) == replicas, (len(requests), replicas)
      clockids, skips = zip(*requests)
      assert all(x == clockid for x in clockids)
      every, prev = clocks[clockid]
      now = time.time()
      if every == 0:
        decision = False
      elif every < 0:
        decision = True
      elif now >= prev + every:
        clocks[clockid][1] = now
        decision = True
      else:
        decision = False
      decision = decision and not any(skips)
      result[0] = decision
      requests.clear()
    respond.wait()
    return result[0]

  server = portal.Server(port, 'ClockServer')
  server.bind('create', create, workers=replicas)
  server.bind('should', should, workers=replicas)
  server.start(block=False)


class GlobalClock:

  def __init__(self, every, first=False):
    self.multihost = bool(CLIENT)
    if self.multihost:
      self.clockid = CLIENT.create(REPLICA, every).result()
      self.skip_next = (not first)
    else:
      self.clock = LocalClock(every, first)

  def __call__(self, step=None, skip=None):
    if self.multihost:
      if self.skip_next:
        self.skip_next = False
        skip = True
      return CLIENT.should(REPLICA, self.clockid, bool(skip)).result()
    else:
      return self.clock(step, skip)


class LocalClock:

  def __init__(self, every, first=False):
    self.every = every
    self.prev = None
    self.first = first

  def __call__(self, step=None, skip=None):
    if skip:
      return False
    if self.every == 0:  # Zero means off
      return False
    if self.every < 0:  # Negative means always
      return True
    now = time.time()
    if self.prev is None:
      self.prev = now
      return self.first
    if now >= self.prev + self.every:
      self.prev = now
      return True
    return False
