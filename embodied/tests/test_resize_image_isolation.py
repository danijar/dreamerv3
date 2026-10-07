import functools
import unittest

import elements
import embodied
import numpy as np
from embodied.core.wrappers import CheckSpaces, ResizeImage


class CachedImageEnv(embodied.Env):
  """A real Env fixture with the cached space contract used by FromGym."""

  def __init__(self):
    self.image = np.arange(16 * 16 * 3, dtype=np.uint16).reshape(16, 16, 3).astype(np.uint8)
    self.observation = {'image': self.image, 'reward': np.float32(0.),
                        'is_first': np.bool_(True), 'is_last': np.bool_(False),
                        'is_terminal': np.bool_(False)}

  @functools.cached_property
  def obs_space(self):
    return {'image': elements.Space(np.uint8, (16, 16, 3)),
            'reward': elements.Space(np.float32), 'is_first': elements.Space(bool),
            'is_last': elements.Space(bool), 'is_terminal': elements.Space(bool)}

  @functools.cached_property
  def act_space(self):
    return {'action': elements.Space(np.int32, (), 0, 2), 'reset': elements.Space(bool)}

  def step(self, action):
    return self.observation


class ResizeIsolationTest(unittest.TestCase):
  def test_reading_wrapper_metadata_preserves_the_base_environment_space(self):
    env = CachedImageEnv()
    original = env.obs_space.copy()
    wrapper = ResizeImage(env, (4, 4))
    spaces = wrapper.obs_space
    self.assertIsNot(spaces, env.obs_space)
    self.assertEqual(spaces['image'].shape, (4, 4, 3))
    for key, space in original.items():
      self.assertIs(env.obs_space[key], space)

  def test_sibling_wrappers_keep_their_own_resize_contract_after_metadata_access(self):
    env = CachedImageEnv()
    first = ResizeImage(env, (4, 4))
    self.assertEqual(first.obs_space['image'].shape, (4, 4, 3))
    second = ResizeImage(env, (4, 4))
    actual = second.step({'action': np.int32(0), 'reset': True})
    self.assertEqual(actual['image'].shape, second.obs_space['image'].shape)
    indices = [2, 6, 10, 14]
    expected = env.image[np.ix_(indices, indices)]
    np.testing.assert_array_equal(actual['image'], expected)

  def test_resized_observation_does_not_replace_a_cached_source_dictionary_entry(self):
    env = CachedImageEnv()
    wrapper = ResizeImage(env, (4, 4))
    raw = env.observation
    original_image = raw['image']
    actual = wrapper.step({'action': np.int32(0), 'reset': True})
    self.assertIsNot(actual, raw)
    self.assertIs(raw['image'], original_image)
    self.assertEqual(raw['image'].shape, (16, 16, 3))
    self.assertEqual(actual['image'].shape, (4, 4, 3))
    np.testing.assert_array_equal(raw['image'], env.image)

  def test_base_and_wrapped_checkspaces_continue_to_validate_their_actual_images(self):
    env = CachedImageEnv()
    wrapper = ResizeImage(env, (4, 4))
    _ = wrapper.obs_space
    action = {'action': np.int32(0), 'reset': True}
    base = CheckSpaces(env).step(action)
    wrapped = CheckSpaces(wrapper).step(action)
    self.assertEqual(base['image'].shape, (16, 16, 3))
    self.assertEqual(wrapped['image'].shape, (4, 4, 3))

  def test_identity_resize_returns_equivalent_contents_in_a_separate_mapping(self):
    env = CachedImageEnv()
    wrapper = ResizeImage(env, (16, 16))
    self.assertEqual(wrapper._keys, [])
    self.assertIsNot(wrapper.obs_space, env.obs_space)
    actual = wrapper.step({'action': np.int32(0), 'reset': True})
    self.assertIsNot(actual, env.observation)
    self.assertIs(actual['image'], env.image)


if __name__ == '__main__':
  unittest.main()
