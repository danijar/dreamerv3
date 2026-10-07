"""Atari's observation formatter preserves independent episode flags.

These tests exercise the actual formatter and image processing. They require
the real Atari module's dependencies, but do not load a ROM or claim to test an
ALE rollout.
"""
import collections
import unittest

import numpy as np

from embodied.envs.atari import Atari


class AtariTerminalObservationTest(unittest.TestCase):

  def environment(self, gray=False, aggregate='max', clip_reward=False):
    env = Atari.__new__(Atari)
    env.size = (4, 3)
    env.gray = gray
    env.aggregate = aggregate
    env.resize = 'pillow'
    env.clip_reward = clip_reward
    first = np.full((3, 4, 3), [20, 40, 60], np.uint8)
    second = np.full((3, 4, 3), [10, 50, 30], np.uint8)
    env.buffers = collections.deque([first, second], maxlen=2)
    return env

  def test_life_discount_is_terminal_without_ending_episode(self):
    obs = self.environment()._obs(2., is_last=False, is_terminal=True)
    self.assertTrue(obs['is_terminal'])
    self.assertFalse(obs['is_last'])
    self.assertFalse(obs['is_first'])

  def test_time_limit_ends_episode_without_marking_terminal(self):
    obs = self.environment()._obs(2., is_last=True, is_terminal=False)
    self.assertFalse(obs['is_terminal'])
    self.assertTrue(obs['is_last'])
    self.assertFalse(obs['is_first'])

  def test_true_game_over_preserves_both_flags(self):
    obs = self.environment()._obs(2., is_last=True, is_terminal=True)
    self.assertTrue(obs['is_terminal'])
    self.assertTrue(obs['is_last'])

  def test_reset_and_ordinary_observation_flags(self):
    env = self.environment()
    first = env._obs(0., is_first=True)
    self.assertTrue(first['is_first'])
    self.assertFalse(first['is_last'])
    self.assertFalse(first['is_terminal'])
    normal = env._obs(1.)
    self.assertFalse(normal['is_first'])
    self.assertFalse(normal['is_last'])
    self.assertFalse(normal['is_terminal'])

  def test_flags_do_not_change_pixel_pooling_or_reward(self):
    for aggregate in ('max', 'mean'):
      for gray in (False, True):
        for last in (False, True):
          for terminal in (False, True):
            with self.subTest(aggregate=aggregate, gray=gray, last=last,
                             terminal=terminal):
              env = self.environment(gray, aggregate, clip_reward=True)
              original = [x.copy() for x in env.buffers]
              obs = env._obs(-3., is_last=last, is_terminal=terminal)
              pixel = np.array([20, 50, 60] if aggregate == 'max'
                               else [15, 45, 45], dtype=np.uint8)
              if gray:
                pixel = np.array([int(sum(float(x) * weight for x, weight
                                         in zip(pixel, Atari.WEIGHTS)))],
                                 dtype=np.uint8)
              expected = np.broadcast_to(pixel, (3, 4, len(pixel)))
              np.testing.assert_array_equal(obs['image'], expected)
              self.assertEqual(obs['image'].dtype, np.uint8)
              self.assertEqual(obs['reward'], np.float32(-1.))
              self.assertEqual(obs['is_last'], last)
              self.assertEqual(obs['is_terminal'], terminal)
              for before, after in zip(original, env.buffers):
                np.testing.assert_array_equal(before, after)


if __name__ == '__main__':
  unittest.main()
