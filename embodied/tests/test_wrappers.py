import elements
import numpy as np

from embodied.core import wrappers


class TestResizeImage:

  def test_nonsquare_size_uses_height_width_order(self):

    class Env:

      @property
      def obs_space(self):
        return {'image': elements.Space(np.uint8, (10, 20, 3))}

      @property
      def act_space(self):
        return {}

      def step(self, action):
        del action
        return {'image': np.zeros((10, 20, 3), np.uint8)}

    env = wrappers.ResizeImage(Env(), size=(32, 64))
    assert env.obs_space['image'].shape == (32, 64, 3)
    assert env.step({})['image'].shape == (32, 64, 3)
