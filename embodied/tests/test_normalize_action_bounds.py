import unittest

import elements
import numpy as np

from embodied.core import wrappers


class ActionEnv:

  def __init__(self):
    self.act_space = {
        'action': elements.Space(
            np.float32, (4,),
            np.array([-2, -np.inf, 0, -np.inf], np.float32),
            np.array([6, np.inf, np.inf, 4], np.float32)),
        'reset': elements.Space(bool),
    }

  def step(self, action):
    return action


class TestNormalizeActionBounds(unittest.TestCase):

  def test_only_finite_intervals_are_normalized(self):
    wrapped = wrappers.NormalizeAction(ActionEnv())
    space = wrapped.act_space['action']
    np.testing.assert_array_equal(space.low, [-1, -np.inf, 0, -np.inf])
    np.testing.assert_array_equal(space.high, [1, np.inf, np.inf, 4])

  def test_unbounded_coordinates_keep_valid_values(self):
    env = ActionEnv()
    wrapped = wrappers.NormalizeAction(env)
    for bounded in (-1, 0, 1):
      action = np.array([bounded, 300, 20, -500], np.float32)
      self.assertTrue(action in wrapped.act_space['action'])
      result = wrapped.step({'action': action, 'reset': False})['action']
      expected = action.copy()
      expected[0] = 2 + 4 * bounded
      np.testing.assert_array_equal(result, expected)
      self.assertEqual(result.dtype, np.float32)
      self.assertTrue(result in env.act_space['action'])


if __name__ == '__main__':
  unittest.main()
