import unittest

import gym
import numpy as np

from embodied.envs.from_gym import FromGym


class GymEnv:

  def __init__(self, modern=False, truncated=False, terminated=False, dictionary=False):
    self.modern = modern
    self.truncated = truncated
    self.terminated = terminated
    self.dictionary = dictionary
    self.resets = 0
    self.observation_space = gym.spaces.Box(-10, 10, (1,), dtype=np.float32)
    space = gym.spaces.Box(-1, 1, (1,), dtype=np.float32)
    self.action_space = gym.spaces.Dict({'motor': space}) if dictionary else space

  def reset(self):
    self.resets += 1
    obs = np.array([0], np.float32)
    return (obs, {'reset_count': self.resets}) if self.modern else obs

  def step(self, action):
    if self.dictionary:
      assert self.action_space.contains(action), action
    obs = np.array([1], np.float32)
    info = {'TimeLimit.truncated': self.truncated}
    if self.modern:
      return obs, 2.0, self.terminated, self.truncated, info
    return obs, 2.0, self.terminated or self.truncated, info


class TestFromGymTermination(unittest.TestCase):

  def test_legacy_time_limit_remains_bootstrappable(self):
    env = FromGym(GymEnv(truncated=True))
    env.step({'reset': True, 'action': np.zeros(1, np.float32)})
    obs = env.step({'reset': False, 'action': np.zeros(1, np.float32)})
    self.assertTrue(obs['is_last'])
    self.assertFalse(obs['is_terminal'])

  def test_modern_reset_and_step_preserve_terminal_flags(self):
    for terminated, truncated in ((False, False), (True, False), (False, True), (True, True)):
      with self.subTest(terminated=terminated, truncated=truncated):
        base = GymEnv(modern=True, terminated=terminated, truncated=truncated)
        env = FromGym(base)
        initial = env.step({'reset': True, 'action': np.zeros(1, np.float32)})
        self.assertTrue(initial['is_first'])
        np.testing.assert_array_equal(initial['image'], [0])
        self.assertEqual(env.info['reset_count'], 1)
        obs = env.step({'reset': False, 'action': np.zeros(1, np.float32)})
        self.assertEqual(obs['is_last'], terminated or truncated)
        self.assertEqual(obs['is_terminal'], terminated)
        self.assertEqual(obs['reward'], 2.0)
        if terminated or truncated:
          self.assertTrue(env.step({'reset': False, 'action': np.zeros(1, np.float32)})['is_first'])

  def test_dict_action_does_not_forward_wrapper_reset(self):
    env = FromGym(GymEnv(modern=True, dictionary=True))
    env.step({'reset': True, 'motor': np.zeros(1, np.float32)})
    obs = env.step({'reset': False, 'motor': np.zeros(1, np.float32)})
    self.assertFalse(obs['is_last'])


if __name__ == '__main__':
  unittest.main()
