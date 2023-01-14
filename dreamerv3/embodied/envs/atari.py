import embodied
import numpy as np


class Atari(embodied.Env):

  LOCK = None

  def __init__(
      self, name, repeat=4, size=(84, 84), gray=True, noops=0, lives='unused',
      sticky=True, actions='all', length=108000, resize='opencv', seed=None):
    assert size[0] == size[1]
    assert lives in ('unused', 'discount', 'reset'), lives
    assert actions in ('all', 'needed'), actions
    assert resize in ('opencv', 'pillow'), resize
    if self.LOCK is None:
      import multiprocessing as mp
      mp = mp.get_context('spawn')
      self.LOCK = mp.Lock()
    self._resize = resize
    if self._resize == 'opencv':
      import cv2
      self._cv2 = cv2
    if self._resize == 'pillow':
      from PIL import Image
      self._image = Image
    import gym.envs.atari
    if name == 'james_bond':
      name = 'jamesbond'
    self._repeat = repeat
    self._size = size
    self._gray = gray
    self._noops = noops
    self._lives = lives
    self._sticky = sticky
    self._length = length
    self._random = np.random.RandomState(seed)
    with self.LOCK:
      self._env = gym.envs.atari.AtariEnv(
          game=name,
          obs_type='image',
          frameskip=1, repeat_action_probability=0.25 if sticky else 0.0,
          full_action_space=(actions == 'all'))
    assert self._env.unwrapped.get_action_meanings()[0] == 'NOOP'
    shape = self._env.observation_space.shape
    self._buffer = [np.zeros(shape, np.uint8) for _ in range(2)]
    self._ale = self._env.unwrapped.ale
    self._last_lives = None
    self._done = True
    self._step = 0

  @property
  def obs_space(self):
    shape = self._size + (1 if self._gray else 3,)
    return {
        'image': embodied.Space(np.uint8, shape),
        'reward': embodied.Space(np.float32),
        'is_first': embodied.Space(bool),
        'is_last': embodied.Space(bool),
        'is_terminal': embodied.Space(bool),
    }

  @property
  def act_space(self):
    return {
        'action': embodied.Space(np.int32, (), 0, self._env.action_space.n),
        'reset': embodied.Space(bool),
    }

  def step(self, action):
    if action['reset'] or self._done:
      with self.LOCK:
        self._reset()
      self._done = False
      self._step = 0
      return self._obs(0.0, is_first=True)
    total = 0.0
    dead = False
    for repeat in range(self._repeat):
      _, reward, over, info = self._env.step(action['action'])
      self._step += 1
      total += reward
      if repeat == self._repeat - 2:
        self._screen(self._buffer[1])
      if over:
        break
      if self._lives != 'unused':
        current = self._ale.lives()
        if current < self._last_lives:
          dead = True
          self._last_lives = current
          break
    if not self._repeat:
      self._buffer[1][:] = self._buffer[0][:]
    self._screen(self._buffer[0])
    self._done = over or (self._length and self._step >= self._length)
    return self._obs(
        total,
        is_last=self._done or (dead and self._lives == 'reset'),
        is_terminal=dead or over)

  def _reset(self):
    self._env.reset()
    if self._noops:
      for _ in range(self._random.randint(self._noops)):
         _, _, dead, _ = self._env.step(0)
         if dead:
           self._env.reset()
    self._last_lives = self._ale.lives()
    self._screen(self._buffer[0])
    self._buffer[1].fill(0)

  def _obs(self, reward, is_first=False, is_last=False, is_terminal=False):
    np.maximum(self._buffer[0], self._buffer[1], out=self._buffer[0])
    image = self._buffer[0]
    if image.shape[:2] != self._size:
      if self._resize == 'opencv':
        image = self._cv2.resize(
            image, self._size, interpolation=self._cv2.INTER_AREA)
      if self._resize == 'pillow':
        image = self._image.fromarray(image)
        image = image.resize(self._size, self._image.NEAREST)
        image = np.array(image)
    if self._gray:
      weights = [0.299, 0.587, 1 - (0.299 + 0.587)]
      image = np.tensordot(image, weights, (-1, 0)).astype(image.dtype)
      image = image[:, :, None]
    return dict(
        image=image,
        reward=reward,
        is_first=is_first,
        is_last=is_last,
        is_terminal=is_last,
    )

  def _screen(self, array):
    self._ale.getScreenRGB2(array)

  def close(self):
    return self._env.close()
