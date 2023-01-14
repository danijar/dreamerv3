from . import generic
from . import selectors
from . import limiters


class Uniform(generic.Generic):

  def __init__(
      self, length, capacity=None, directory=None, online=False,
      chunks=1024, seed=0):
    super().__init__(
        length=length,
        capacity=capacity,
        remover=selectors.Fifo(),
        sampler=selectors.Uniform(seed),
        limiter=limiters.MinSize(1),
        directory=directory,
        online=online,
        chunks=chunks,
    )
