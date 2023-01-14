import logging
import threading

import embodied
import numpy as np


class MinecraftBase(embodied.Env):

  _LOCK = threading.Lock()

  def __init__(
      self, actions,
      repeat=1,
      size=(64, 64),
      break_speed=100.0,
      gamma=10.0,
      sticky_attack=30,
      sticky_jump=10,
      pitch_limit=(-60, 60),
      logs=True,  # TODO
  ):
    if logs:
      logging.basicConfig(level=logging.DEBUG)
    self._repeat = repeat
    self._size = size
    if break_speed != 1.0:
      sticky_attack = 0

    # Make env
    with self._LOCK:
      from .import minecraft_minerl
      self._gymenv = minecraft_minerl.MineRLEnv(size, break_speed, gamma).make()
    from . import from_gym
    self._env = from_gym.FromGym(self._gymenv)
    self._inventory = {}

    # Observations
    self._inv_keys = [
        k for k in self._env.obs_space if k.startswith('inventory/')
        if k != 'inventory/log2']
    self._step = 0
    self._max_inventory = None
    self._equip_enum = self._gymenv.observation_space[
        'equipped_items']['mainhand']['type'].values.tolist()
    self._obs_space = self.obs_space

    # Actions
    self._noop_action = minecraft_minerl.NOOP_ACTION
    actions = self._insert_defaults(actions)
    self._action_names = tuple(actions.keys())
    self._action_values = tuple(actions.values())
    message = f'Minecraft action space ({len(self._action_values)}):'
    print(message, ', '.join(self._action_names))
    self._sticky_attack_length = sticky_attack
    self._sticky_attack_counter = 0
    self._sticky_jump_length = sticky_jump
    self._sticky_jump_counter = 0
    self._pitch_limit = pitch_limit
    self._pitch = 0

  @property
  def obs_space(self):
    return {
        'image': embodied.Space(np.uint8, self._size + (3,)),
        'inventory': embodied.Space(np.float32, len(self._inv_keys), 0),
        'inventory_max': embodied.Space(np.float32, len(self._inv_keys), 0),
        'equipped': embodied.Space(np.float32, len(self._equip_enum), 0, 1),
        'reward': embodied.Space(np.float32),
        'health': embodied.Space(np.float32),
        'hunger': embodied.Space(np.float32),
        'breath': embodied.Space(np.float32),
        'is_first': embodied.Space(bool),
        'is_last': embodied.Space(bool),
        'is_terminal': embodied.Space(bool),
        **{f'log_{k}': embodied.Space(np.int64) for k in self._inv_keys},
        'log_player_pos': embodied.Space(np.float32, 3),
    }

  @property
  def act_space(self):
    return {
        'action': embodied.Space(np.int64, (), 0, len(self._action_values)),
        'reset': embodied.Space(bool),
    }

  def step(self, action):
    action = action.copy()
    index = action.pop('action')
    action.update(self._action_values[index])
    action = self._action(action)
    if action['reset']:
      obs = self._reset()
    else:
      following = self._noop_action.copy()
      for key in ('attack', 'forward', 'back', 'left', 'right'):
        following[key] = action[key]
      for act in [action] + ([following] * (self._repeat - 1)):
        obs = self._env.step(act)
        if 'error' in self._env.info:
          obs = self._reset()
          break
    obs = self._obs(obs)
    self._step += 1
    assert 'pov' not in obs, list(obs.keys())
    return obs

  @property
  def inventory(self):
    return self._inventory

  def _reset(self):
    with self._LOCK:
      obs = self._env.step({'reset': True})
    self._step = 0
    self._max_inventory = None
    self._sticky_attack_counter = 0
    self._sticky_jump_counter = 0
    self._pitch = 0
    self._inventory = {}
    return obs

  def _obs(self, obs):
    obs['inventory/log'] += obs.pop('inventory/log2')
    self._inventory = {
        k.split('/', 1)[1]: obs[k] for k in self._inv_keys
        if k != 'inventory/air'}
    inventory = np.array([obs[k] for k in self._inv_keys], np.float32)
    if self._max_inventory is None:
      self._max_inventory = inventory
    else:
      self._max_inventory = np.maximum(self._max_inventory, inventory)
    index = self._equip_enum.index(obs['equipped_items/mainhand/type'])
    equipped = np.zeros(len(self._equip_enum), np.float32)
    equipped[index] = 1.0
    player_x = obs['location_stats/xpos']
    player_y = obs['location_stats/ypos']
    player_z = obs['location_stats/zpos']
    obs = {
        'image': obs['pov'],
        'inventory': inventory,
        'inventory_max': self._max_inventory.copy(),
        'equipped': equipped,
        'health': np.float32(obs['life_stats/life'] / 20),
        'hunger': np.float32(obs['life_stats/food'] / 20),
        'breath': np.float32(obs['life_stats/air'] / 300),
        'reward': 0.0,
        'is_first': obs['is_first'],
        'is_last': obs['is_last'],
        'is_terminal': obs['is_terminal'],
        **{f'log_{k}': np.int64(obs[k]) for k in self._inv_keys},
        'log_player_pos': np.array([player_x, player_y, player_z], np.float32),
    }
    for key, value in obs.items():
      space = self._obs_space[key]
      if not isinstance(value, np.ndarray):
        value = np.array(value)
      assert value in space, (key, value, value.dtype, value.shape, space)
    return obs

  def _action(self, action):
    if self._sticky_attack_length:
      if action['attack']:
        self._sticky_attack_counter = self._sticky_attack_length
      if self._sticky_attack_counter > 0:
        action['attack'] = 1
        action['jump'] = 0
        self._sticky_attack_counter -= 1
    if self._sticky_jump_length:
      if action['jump']:
        self._sticky_jump_counter = self._sticky_jump_length
      if self._sticky_jump_counter > 0:
        action['jump'] = 1
        action['forward'] = 1
        self._sticky_jump_counter -= 1
    if self._pitch_limit and action['camera'][0]:
      lo, hi = self._pitch_limit
      if not (lo <= self._pitch + action['camera'][0] <= hi):
        action['camera'] = (0, action['camera'][1])
      self._pitch += action['camera'][0]
    return action

  def _insert_defaults(self, actions):
    actions = {name: action.copy() for name, action in actions.items()}
    for key, default in self._noop_action.items():
      for action in actions.values():
        if key not in action:
          action[key] = default
    return actions
