"""Regression coverage for deterministic mixed stream iteration and restore."""
import unittest

from embodied.core import base, streams


class CounterSource(base.Stream):
  def __init__(self, name):
    self.name = name
    self.position = 0

  def __iter__(self):
    return self

  def __next__(self):
    item = (self.name, self.position)
    self.position += 1
    return item

  def save(self):
    return self.position

  def load(self, position):
    self.position = position


def make_mixer(seed=12, reverse=False, weights=None):
  names = ['alpha', 'zeta'][:: -1 if reverse else 1]
  sources = {name: CounterSource(name) for name in names}
  return streams.Mixer(sources, weights or {'alpha': 1., 'zeta': 3.}, seed=seed)


class MixerTests(unittest.TestCase):
  def test_iteration_starts_stream(self):
    mixer = make_mixer()
    self.assertIs(iter(mixer), mixer)
    self.assertIn(next(mixer)[0], ['alpha', 'zeta'])

  def test_rng_path_after_started_flag_is_set(self):
    mixer = make_mixer()
    mixer.started = True
    self.assertIn(next(mixer)[0], ['alpha', 'zeta'])

  def test_load_maps_named_sources_to_list_positions(self):
    mixer = make_mixer()
    mixer.load({'step': 5, 'seed': 9, 'sources': {'alpha': 7, 'zeta': 11}})
    self.assertEqual([source.position for source in mixer.iterators], [7, 11])
    self.assertEqual(mixer.step, 5)
    self.assertEqual(mixer.seed, 9)

  def test_same_seed_and_source_keys_give_same_sequence(self):
    first, second = iter(make_mixer()), iter(make_mixer(reverse=True))
    self.assertEqual([next(first) for _ in range(100)], [next(second) for _ in range(100)])

  def test_zero_weight_source_is_never_selected(self):
    mixer = iter(make_mixer(weights={'alpha': 1., 'zeta': 0.}))
    self.assertEqual([next(mixer) for _ in range(100)], [('alpha', i) for i in range(100)])

  def test_weighted_sampling_frequency(self):
    mixer = iter(make_mixer())
    selected = [next(mixer)[0] for _ in range(2000)]
    self.assertLess(abs(selected.count('alpha') / 2000 - .25), .035)

  def test_checkpoint_restores_exact_continuation(self):
    original = iter(make_mixer())
    for _ in range(37):
      next(original)
    state = original.save()
    expected = [next(original) for _ in range(100)]
    restored = iter(make_mixer(seed=999, reverse=True))
    restored.load(state)
    self.assertEqual([next(restored) for _ in range(100)], expected)

  def test_checkpoint_can_load_before_iteration(self):
    original = iter(make_mixer())
    for _ in range(23):
      next(original)
    restored = make_mixer(seed=987)
    restored.load(original.save())
    iter(restored)
    self.assertEqual(next(restored), next(original))

  def test_next_requires_iteration(self):
    with self.assertRaises(AssertionError):
      next(make_mixer())

  def test_second_iteration_is_rejected(self):
    mixer = iter(make_mixer())
    with self.assertRaises(AssertionError):
      iter(mixer)

  def test_checkpoint_restores_native_map_streams(self):
    def create():
      sources = {name: streams.Map(CounterSource(name), lambda x: (x[0], x[1] * 2))
                 for name in ['alpha', 'zeta']}
      return iter(streams.Mixer(sources, {'alpha': 1., 'zeta': 2.}, seed=72))

    original = create()
    for _ in range(21):
      next(original)
    state = original.save()
    restored = create()
    restored.load(state)
    self.assertEqual([next(original) for _ in range(100)], [next(restored) for _ in range(100)])


if __name__ == '__main__':
  unittest.main()
