import collections
import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

import numpy as np
import pytest
from embodied.replay import sampletree


class TestSampleTree:

  @pytest.mark.parametrize('branching', [2, 3, 5, 10])
  def test_root_sum(self, branching):
    tree = sampletree.SampleTree(branching)
    entries = range(50)
    for index, uprob in enumerate(entries):
      assert tree.root.uprob == sum(entries[:index])
      tree.insert(index, uprob)

  @pytest.mark.parametrize('inserts', [1, 2, 10, 100])
  @pytest.mark.parametrize('branching', [2, 3, 5, 10])
  def test_depth_inserts(self, inserts, branching):
    tree = sampletree.SampleTree(branching)
    for index in range(inserts):
      tree.insert(index, 1)
    assert len(tree) == inserts
    depths = self._find_leave_depths(tree)
    target = max(1, int(np.ceil(np.log(inserts) / np.log(branching))))
    assert all(x == target for x in depths)

  @pytest.mark.parametrize('inserts', [2, 10, 100])
  @pytest.mark.parametrize('remove_every', [2, 3, 4])
  @pytest.mark.parametrize('branching', [2, 3, 5, 10])
  def test_depth_removals(self, inserts, remove_every, branching):
    tree = sampletree.SampleTree(branching)
    for index in range(0, inserts, 1):
      tree.insert(index, 1)
    removals = list(range(0, inserts, remove_every))
    for index in removals:
      tree.remove(index)
    assert len(tree) == inserts - len(removals)
    depths = self._find_leave_depths(tree)
    target = max(1, int(np.ceil(np.log(inserts) / np.log(branching))))
    assert all(x == target for x in depths)

  @pytest.mark.parametrize('inserts', [2, 10, 100])
  @pytest.mark.parametrize('branching', [2, 3, 5, 10])
  def test_removal_num_nodes(self, inserts, branching):
    tree = sampletree.SampleTree(branching)
    assert len(self._get_flat_nodes(tree)) == 1
    rng = np.random.default_rng(seed=0)
    for key in rng.permutation(np.arange(inserts)):
      tree.insert(key, 1)
    num_nodes = len(self._get_flat_nodes(tree))
    for key in rng.permutation(np.arange(inserts)):
      tree.remove(key)
    assert len(self._get_flat_nodes(tree)) == 1
    for key in rng.permutation(np.arange(inserts)):
      tree.insert(key, 1)
    assert len(self._get_flat_nodes(tree)) == num_nodes

  @pytest.mark.parametrize('branching', [2, 3, 5, 10])
  def test_sample_single(self, branching):
    tree = sampletree.SampleTree(branching)
    tree.insert(12, 1.0)
    tree.insert(123, 1.0)
    tree.insert(42, 1.0)
    tree.remove(12)
    tree.remove(42)
    for _ in range(10):
      assert tree.sample() == 123

  @pytest.mark.parametrize('inserts', [2, 10])
  @pytest.mark.parametrize('branching', [2, 3, 5, 10])
  @pytest.mark.parametrize('uprob', [1e-5, 1.0, 1e5])
  def test_sample_uniform(self, inserts, branching, uprob):
    tree = sampletree.SampleTree(branching, seed=0)
    keys = list(range(inserts))
    for key in keys:
      tree.insert(key, 1.0)
    for key in keys[::3]:
      tree.remove(key)
      keys.remove(key)
    histogram = collections.defaultdict(int)
    for _ in range(100 * len(keys)):
      key = tree.sample()
      histogram[key] += 1
    assert len(histogram) > 0
    assert len(histogram) == len(keys)
    assert all(k in histogram for k in keys)
    for key, count in histogram.items():
      prob = count / (100 * len(keys))
      assert prob > 0.5 * (1 / len(keys))

  @pytest.mark.parametrize('scale', [1e-5, 1, 1e5])
  @pytest.mark.parametrize('branching', [2, 3, 5, 10])
  def test_sample_frequencies(self, scale, branching):
    tree = sampletree.SampleTree(branching, seed=0)
    keys = [0, 1, 2, 3, 4, 5]
    uprobs = [0, 3, 1, 1, 2, 2]
    entries = dict(zip(keys, uprobs))
    for key, uprob in entries.items():
      tree.insert(key, scale * uprob)
    histogram = collections.defaultdict(int)
    for _ in range(100 * len(entries)):
      key = tree.sample()
      histogram[key] += 1
    assert len(histogram) > 0
    total = sum(entries.values())
    for key, uprob in entries.items():
      if uprob == 0:
        assert key not in histogram
    for key, count in histogram.items():
      prob = count / (100 * len(entries))
      target = entries[key] / total
      assert 0.7 * target < prob < 1.3 * target

  @pytest.mark.parametrize('branching', [2, 3, 5, 10])
  def test_update_frequencies(self, branching):
    tree = sampletree.SampleTree(branching, seed=0)
    keys = [0, 1, 2, 3, 4, 5]
    uprobs = [0, 3, 1, 1, 2, 2]
    entries = dict(zip(keys, uprobs))
    for key in entries.keys():
      tree.insert(key, 100)
    for key, uprob in entries.items():
      tree.update(key, uprob)
    histogram = collections.defaultdict(int)
    for _ in range(100 * len(entries)):
      key = tree.sample()
      histogram[key] += 1
    assert len(histogram) > 0
    total = sum(entries.values())
    for key, uprob in entries.items():
      if uprob == 0:
        assert key not in histogram
    for key, count in histogram.items():
      prob = count / (100 * len(entries))
      target = entries[key] / total
      assert 0.7 * target < prob < 1.3 * target

  @pytest.mark.parametrize('branching', [2, 3, 5, 10])
  def test_zero_probs_mixed(self, branching):
    tree = sampletree.SampleTree(branching, seed=0)
    impossible = []
    for index in range(100):
      if index % 3 == 0:
        tree.insert(index, 1.0)
      else:
        tree.insert(index, 0.0)
        impossible.append(index)
    for _ in range(1000):
      assert tree.sample() not in impossible

  @pytest.mark.parametrize('branching', [2, 3, 5, 10])
  def test_zero_probs_only(self, branching):
    tree = sampletree.SampleTree(branching, seed=0)
    for index in range(100):
      tree.insert(index, 0.0)
    for _ in range(1000):
      assert tree.sample() in range(100)

  @pytest.mark.parametrize('branching', [2, 3, 5, 10])
  def test_infinity_probs(self, branching):
    tree = sampletree.SampleTree(branching, seed=0)
    possible = []
    for index in range(100):
      if index % 3 == 0:
        tree.insert(index, np.inf)
        possible.append(index)
      else:
        tree.insert(index, 1.0)
    for _ in range(1000):
      assert tree.sample() in possible

  def _find_leave_depths(self, tree):
    depths = []
    queue = [(tree.root, 0)]
    while queue:
      node, depth = queue.pop()
      if hasattr(node, 'children'):
        for child in node.children:
          queue.append((child, depth + 1))
      else:
        depths.append(depth)
    assert len(depths) > 0
    return depths

  def _get_flat_nodes(self, tree):
    nodes = []
    queue = [tree.root]
    while queue:
      node = queue.pop()
      nodes.append(node)
      if hasattr(node, 'children'):
        queue += node.children
    return nodes
