import unittest

import numpy as np

from embodied.core import replay, selectors


class TestUniformSingleItem(unittest.TestCase):

  def test_delete_last_item_and_repopulate(self):
    selector = selectors.Uniform(seed=3)
    for key in range(5):
      selector[key] = []
      self.assertEqual(selector(), key)
      del selector[key]
      self.assertEqual(len(selector), 0)
      self.assertEqual(selector.keys, [])
      self.assertEqual(selector.indices, {})

  def test_capacity_one_replay_evicts_and_samples(self):
    buffer = replay.Replay(length=2, capacity=1, chunksize=3)
    for value in range(10):
      buffer.add({'value': np.int32(value), 'is_first': False})
      if value >= 1:
        self.assertEqual(len(buffer), 1)
        batch = buffer.sample(1)
        np.testing.assert_array_equal(batch['value'], [[value - 1, value]])


if __name__ == '__main__':
  unittest.main()
