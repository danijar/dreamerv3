import enum
from functools import partial as bind

from . import worker


class Parallel:

  def __init__(self, ctor, strategy):
    self.worker = worker.Worker(
        bind(self._respond, ctor), strategy, state=True)
    self.callables = {}

  def __getattr__(self, name):
    if name.startswith('_'):
      raise AttributeError(name)
    try:
      if name not in self.callables:
        self.callables[name] = self.worker(Message.CALLABLE, name)()
      if self.callables[name]:
        return bind(self.worker, Message.CALL, name)
      else:
        return self.worker(Message.READ, name)()
    except AttributeError:
      raise ValueError(name)

  def __len__(self):
    return self.worker(Message.CALL, '__len__')()

  def close(self):
    self.worker.close()

  @staticmethod
  def _respond(ctor, state, message, name, *args, **kwargs):
    state = state or ctor()
    if message == Message.CALLABLE:
      assert not args and not kwargs, (args, kwargs)
      result = callable(getattr(state, name))
    elif message == Message.CALL:
      result = getattr(state, name)(*args, **kwargs)
    elif message == Message.READ:
      assert not args and not kwargs, (args, kwargs)
      result = getattr(state, name)
    return state, result


class Message(enum.Enum):

  CALLABLE = 2
  CALL = 3
  READ = 4
