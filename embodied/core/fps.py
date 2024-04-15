import time


class FPS:

  def __init__(self):
    self.start = time.time()
    self.count = 0

  def step(self, amount=1):
    self.count += amount

  def result(self, reset=True):
    now = time.time()
    fps = self.count / (now - self.start)
    if reset:
      self.start = now
      self.count = 0
    return fps
