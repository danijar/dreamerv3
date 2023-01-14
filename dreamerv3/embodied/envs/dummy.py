import embodied
import numpy as np


class Dummy(embodied.Env):

  def __init__(self, task, size=(64, 64), length=100):
    assert task in ('cont', 'disc')
    self._task = task
    self._size = size
    self._length = length
    self._step = 0
    self._done = False

  @property
  def obs_space(self):
    return {
        'image': embodied.Space(np.uint8, self._size + (3,)),
        'vector': embodied.Space(np.float32, (7,)),
        'step': embodied.Space(np.int32, (), 0, self._length),
        'reward': embodied.Space(np.float32),
        'is_first': embodied.Space(bool),
        'is_last': embodied.Space(bool),
        'is_terminal': embodied.Space(bool),
    }

  @property
  def act_space(self):
    if self._task == 'cont':
      space = embodied.Space(np.float32, (6,))
    else:
      space = embodied.Space(np.int32, (), 0, 5)
    return {'action': space, 'reset': embodied.Space(bool)}

  def step(self, action):
    if action['reset'] or self._done:
      self._step = 0
      self._done = False
      return self._obs(0.0, is_first=True)
    action = action['action']
    self._step += 1
    self._done = (self._step >= self._length)
    return self._obs(1.0, is_last=self._done, is_terminal=self._done)

  def _obs(self, reward, is_first=False, is_last=False, is_terminal=False):
    return dict(
        image=np.zeros(self._size + (3,), np.uint8),
        vector=np.zeros(7, np.float32),
        step=self._step,
        reward=reward,
        is_first=is_first,
        is_last=is_last,
        is_terminal=is_terminal,
    )
