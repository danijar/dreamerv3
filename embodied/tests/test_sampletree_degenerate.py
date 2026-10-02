"""Regression coverage for equal leaf mass in degenerate replay trees."""
import unittest

import numpy as np
from embodied.core import replay as replaylib
from embodied.core import selectors


def leaf_probabilities(tree):
  """Obtain exact leaf probabilities by forcing each path through sample()."""
  paths = []

  def visit(node, path):
    if isinstance(node, selectors.SampleTreeNode):
      for index, child in enumerate(node.children):
        visit(child, path + [index])
    else:
      paths.append((node.key, path))

  visit(tree.root, [])
  probabilities = {}
  rng = tree.rng
  for key, path in paths:
    class Route:
      def __init__(self):
        self.depth = 0
        self.prob = 1.

      def choice(self, choices, p):
        index = path[self.depth]
        self.depth += 1
        self.prob *= p[index]
        return np.int64(index)

    route = Route()
    tree.rng = route
    assert tree.sample() == key
    probabilities[key] = route.prob
  tree.rng = rng
  return probabilities


class DegenerateTreeTests(unittest.TestCase):
  def assert_distribution(self, tree, expected):
    actual = leaf_probabilities(tree)
    self.assertEqual(actual.keys(), expected.keys())
    np.testing.assert_allclose(list(actual.values()), [expected[k] for k in actual], atol=1e-15)
    self.assertAlmostEqual(sum(actual.values()), 1.)

  def test_zero_mass_is_uniform_over_leaves(self):
    for branching, count in [(2, 3), (3, 10), (5, 17), (16, 37)]:
      with self.subTest(branching=branching, count=count):
        tree = selectors.SampleTree(branching)
        for key in range(count):
          tree.insert(key, 0.)
        self.assert_distribution(tree, {k: 1 / count for k in range(count)})

  def test_infinite_mass_is_uniform_over_infinite_leaves(self):
    for branching, count in [(2, 3), (3, 10), (5, 17), (16, 37)]:
      with self.subTest(branching=branching, count=count):
        tree = selectors.SampleTree(branching)
        for key in range(count):
          tree.insert(key, np.inf)
        self.assert_distribution(tree, {k: 1 / count for k in range(count)})

  def test_finite_leaves_have_no_mass_when_infinite_leaves_exist(self):
    tree = selectors.SampleTree(2)
    weights = [np.inf, 1., np.inf, np.inf, 100.]
    for key, weight in enumerate(weights):
      tree.insert(key, weight)
    self.assert_distribution(tree, {k: 1 / 3 if w == np.inf else 0. for k, w in enumerate(weights)})

  def test_zero_mass_distribution_survives_removal(self):
    tree = selectors.SampleTree(3)
    for key in range(17):
      tree.insert(key, 0.)
    for key in [0, 3, 9, 15, 16]:
      tree.remove(key)
    self.assert_distribution(tree, {k: 1 / 12 for k in tree.entries})

  def test_infinite_count_tracks_updates_and_removals(self):
    tree = selectors.SampleTree(2)
    for key in range(9):
      tree.insert(key, np.inf)
    tree.update(0, 1.)
    tree.update(3, 0.)
    tree.remove(5)
    tree.remove(8)
    self.assert_distribution(tree, {k: 0. if k in [0, 3] else 1 / 5 for k in tree.entries})

  def test_zero_on_sample_does_not_bias_remaining_initial_priorities(self):
    sampler = selectors.Prioritized(initial=np.inf, zero_on_sample=True, branching=2)
    for key in range(7):
      sampler[key] = [bytes([key])]
    sampled = sampler()
    self.assert_distribution(sampler.tree, {k: 0. if k == sampled else 1 / 6 for k in range(7)})

  def test_finite_weighted_distribution_is_unchanged(self):
    tree = selectors.SampleTree(3)
    weights = [0., 3., 1., 1., 2., 2., 10.]
    for key, weight in enumerate(weights):
      tree.insert(key, weight)
    self.assert_distribution(tree, {k: w / sum(weights) for k, w in enumerate(weights)})

  def test_clear_and_reinsert_keeps_degenerate_counts_consistent(self):
    tree = selectors.SampleTree(2)
    for key in range(5):
      tree.insert(key, np.inf)
    for key in [2, 4, 0, 1, 3]:
      tree.remove(key)
    for key in range(3):
      tree.insert(key, 0.)
    self.assert_distribution(tree, {k: 1 / 3 for k in range(3)})

  def test_counts_track_randomized_mutations(self):
    tree = selectors.SampleTree(3)
    rng = np.random.default_rng(47)
    weights = {}
    next_key = 0
    for _ in range(1000):
      operation = rng.integers(3) if weights else 0
      if operation == 0:
        weight = rng.choice([0., 1., 7., np.inf])
        tree.insert(next_key, weight)
        weights[next_key] = weight
        next_key += 1
      elif operation == 1:
        key = rng.choice(list(weights)).item()
        weight = rng.choice([0., 1., 7., np.inf])
        tree.update(key, weight)
        weights[key] = weight
      else:
        key = rng.choice(list(weights)).item()
        tree.remove(key)
        del weights[key]
      self.assertEqual(tree.root.num_entries, len(weights))
      self.assertEqual(tree.root.num_infinite, sum(w == np.inf for w in weights.values()))

  def test_replay_sample_consumes_infinite_initial_priorities(self):
    sampler = selectors.Prioritized(initial=np.inf, zero_on_sample=True, branching=2)
    replay = replaylib.Replay(length=2, capacity=7, chunksize=3)
    # Attach the empty selector explicitly: Replay currently treats empty
    # selectors passed to its constructor as false and chooses Uniform.
    replay.sampler = sampler
    for step in range(8):
      replay.add({'value': step})
    self.assertEqual(sampler.tree.root.num_infinite, 7)
    batch = replay.sample(1)
    np.testing.assert_array_equal(np.diff(batch['value'], axis=1), [[1]])
    self.assertLess(sampler.tree.root.num_infinite, 7)
    expected = {k: int(entry.uprob == np.inf) / sampler.tree.root.num_infinite
                for k, entry in sampler.tree.entries.items()}
    self.assert_distribution(sampler.tree, expected)


if __name__ == '__main__':
  unittest.main()
