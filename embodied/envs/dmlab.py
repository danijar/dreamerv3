import functools
import re
import zlib

import deepmind_lab
import elements
import embodied
import numpy as np


class DMLab(embodied.Env):

  TOKENIZER = re.compile(r'([A-Za-z_]+|[^A-Za-z_ ]+)')

  def __init__(
      self, level, repeat=4, size=(64, 64), mode='train',
      actions='popart', episodic=True, text=None, seed=None):
    if level == 'goals':  # Shortcut for convenience
      level = 'dmlab_explore_goal_locations_small'
    self._size = size
    self._repeat = repeat
    self._actions = {
        'impala': IMPALA_ACTION_SET,
        'popart': POPART_ACTION_SET,
    }[actions]
    if text is None:
      text = bool(level.startswith('language'))
    self._episodic = episodic
    self._text = text
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
    obs = ['RGB_INTERLEAVED', 'INSTR'] if text else ['RGB_INTERLEAVED']
    self._env = deepmind_lab.Lab(
        level='contributed/dmlab30/' + level,
        observations=obs, config=config)
    self._current_image = None
    if self._text:
      self._current_instr = None
      self._instr_length = 32
      self._embed_size = 32
      self._vocab_buckets = 64 * 1024
      self._embeddings = np.random.default_rng(seed=0).normal(
          0.0, 1.0, (self._vocab_buckets, self._embed_size)).astype(np.float32)
    self._done = True

  @property
  def obs_space(self):
    spaces = {
        'image': elements.Space(np.uint8, self._size + (3,)),
        'reward': elements.Space(np.float32),
        'is_first': elements.Space(bool),
        'is_last': elements.Space(bool),
        'is_terminal': elements.Space(bool),
    }
    if self._text:
      spaces['instr'] = elements.Space(
          np.float32, self._instr_length * self._embed_size)
    return spaces

  @property
  def act_space(self):
    return {
        'action': elements.Space(np.int32, (), 0, len(self._actions)),
        'reset': elements.Space(bool),
    }

  def step(self, action):
    if action['reset'] or self._done:
      self._env.reset(seed=self._random.randint(0, 2 ** 31 - 1))
      self._done = False
      return self._obs(0.0, is_first=True)
    raw_action = np.array(self._actions[action['action']], np.intc)
    reward = self._env.step(raw_action, num_steps=self._repeat)
    self._done = not self._env.is_running()
    return self._obs(reward, is_last=self._done)

  def _obs(self, reward, is_first=False, is_last=False):
    if not self._done:
      self._current_image = self._env.observations()['RGB_INTERLEAVED']
      if self._text:
        self._current_instr = self._embed(self._env.observations()['INSTR'])
    obs = dict(
        image=self._current_image,
        reward=np.float32(reward),
        is_first=is_first,
        is_last=is_last,
        is_terminal=is_last if self._episodic else False,
    )
    if self._text:
      obs['instr'] = self._current_instr
    return obs

  def _embed(self, text):
    tokens = self.TOKENIZER.findall(text.lower())
    indices = [self._hash(token) for token in tokens]
    # print('EMBED', text, '->', tokens, '->', indices)
    indices = indices + [0] * (self._instr_length - len(indices))
    embeddings = [self._embeddings[i] for i in indices]
    return np.concatenate(embeddings)

  @functools.cache
  def _hash(self, token):
    return zlib.crc32(token.encode('utf-8')) % self._vocab_buckets

  def close(self):
    self._env.close()


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
