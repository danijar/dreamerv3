import embodied
import numpy as np


class DMLab(embodied.Env):

  # Small action set used by IMPALA.
  IMPALA_ACTION_SET = (
      (  0, 0,  0,  1, 0, 0, 0),  # Forward
      (  0, 0,  0, -1, 0, 0, 0),  # Backward
      (  0, 0, -1,  0, 0, 0, 0),  # Strafe Left
      (  0, 0,  1,  0, 0, 0, 0),  # Strafe Right
      (-20, 0,  0,  0, 0, 0, 0),  # Look Left
      ( 20, 0,  0,  0, 0, 0, 0),  # Look Right
      (-20, 0,  0,  1, 0, 0, 0),  # Look Left + Forward
      ( 20, 0,  0,  1, 0, 0, 0),  # Look Right + Forward
      (  0, 0,  0,  0, 1, 0, 0),  # Fire
  )

  # Large action set used by PopArt and R2D2.
  POPART_ACTION_SET = [
      (  0,   0,  0,  1, 0, 0, 0),  # FW
      (  0,   0,  0, -1, 0, 0, 0),  # BW
      (  0,   0, -1,  0, 0, 0, 0),  # Strafe Left
      (  0,   0,  1,  0, 0, 0, 0),  # Strafe Right
      (-10,   0,  0,  0, 0, 0, 0),  # Small LL
      ( 10,   0,  0,  0, 0, 0, 0),  # Small LR
      (-60,   0,  0,  0, 0, 0, 0),  # Large LL
      ( 60,   0,  0,  0, 0, 0, 0),  # Large LR
      (  0,  10,  0,  0, 0, 0, 0),  # Look Down
      (  0, -10,  0,  0, 0, 0, 0),  # Look Up
      (-10,   0,  0,  1, 0, 0, 0),  # FW + Small LL
      ( 10,   0,  0,  1, 0, 0, 0),  # FW + Small LR
      (-60,   0,  0,  1, 0, 0, 0),  # FW + Large LL
      ( 60,   0,  0,  1, 0, 0, 0),  # FW + Large LR
      (  0,   0,  0,  0, 1, 0, 0),  # Fire
  ]

  def __init__(
      self, level, repeat=4, size=(64, 64), mode='train',
      action_set=IMPALA_ACTION_SET, episodic=True, seed=None):
    import deepmind_lab
    cache = None
    # path = os.environ.get('DMLAB_CACHE', None)
    # if path:
    #   cache = Cache(path)
    self._size = size
    self._repeat = repeat
    self._action_set = action_set
    self._episodic = episodic
    self._random = np.random.RandomState(seed)
    config = dict(height=size[0], width=size[1], logLevel='WARN')
    if mode == 'train':
      if level.endswith('_test'):
        level = level.replace('_test', '_train')
    elif mode == 'eval':
      config.update(allowHoldOutLevels='true', mixerSeed=0x600D5EED)
    else:
      raise NotImplementedError(mode)
    config = {k: str(v) for k, v in config.items()}
    self._env = deepmind_lab.Lab(
        level='contributed/dmlab30/' + level,
        observations=['RGB_INTERLEAVED'],
        level_cache=cache, config=config)
    self._prev_image = None
    self._done = True

  @property
  def obs_space(self):
    return {
        'image': embodied.Space(np.uint8, self._size + (3,)),
        'reward': embodied.Space(np.float32),
        'is_first': embodied.Space(bool),
        'is_last': embodied.Space(bool),
        'is_terminal': embodied.Space(bool),
    }

  @property
  def act_space(self):
    return {
        'action': embodied.Space(np.int32, (), 0, len(self._action_set)),
        'reset': embodied.Space(bool),
    }

  def step(self, action):
    if action['reset'] or self._done:
      self._env.reset(seed=self._random.randint(0, 2 ** 31 - 1))
      self._done = False
      return self._obs(0.0, is_first=True)
    raw_action = np.array(self._action_set[action['action']], np.intc)
    reward = self._env.step(raw_action, num_steps=self._repeat)
    self._done = not self._env.is_running()
    return self._obs(reward, is_last=self._done)

  def _obs(self, reward, is_first=False, is_last=False):
    return dict(
        image=self.render(),
        reward=reward,
        is_first=is_first,
        is_last=is_last,
        is_terminal=is_last if self._episodic else False,
    )

  def render(self):
    if not self._done:
      self._prev_image = self._env.observations()['RGB_INTERLEAVED']
    return self._prev_image

  def close(self):
    self._env.close()


class Cache:

  def __init__(self, cache_dir):
    self._cache_dir = cache_dir

  def get_path(self, key):
    import hashlib, os
    key = hashlib.md5(key.encode('utf-8')).hexdigest()
    dir_, filename = key[:3], key[3:]
    return os.path.join(self._cache_dir, dir_, filename)

  def fetch(self, key, pk3_path):
    import tensorflow as tf
    path = self.get_path(key)
    try:
      tf.io.gfile.copy(path, pk3_path, overwrite=True)
      return True
    except tf.errors.OpError:
      return False

  def write(self, key, pk3_path):
    import os
    import tensorflow as tf
    path = self.get_path(key)
    try:
      if not tf.io.gfile.exists(path):
        tf.io.gfile.makedirs(os.path.dirname(path))
        tf.io.gfile.copy(pk3_path, path)
    except Exception as e:
      print(f'Could to store level: {e}')
