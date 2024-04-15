import embodied
import numpy as np


class ProcGen(embodied.Env):

  def __init__(self, task, size=(64, 64), resize='pillow', **kwargs):
    assert resize in ('opencv', 'pillow'), resize
    import procgen  # noqa
    from . import from_gym
    self.size = size
    self.resize = resize
    if self.size == (64, 64):
      self.source = 'step'
    else:
      self.source = 'info'

    if self.source == 'info':
      kwargs['render_mode'] = 'rgb_array'
    try:
      self.env = from_gym.FromGym(f'procgen:procgen-{task}-v0', **kwargs)
    except Exception:
      self.env = from_gym.FromGym(f'procgen-{task}-v0', **kwargs)
    if self.source == 'info':
      self.inner = self.env
      while not hasattr(self.inner, 'get_info'):
        self.inner = self.inner.env

  @property
  def obs_space(self):
    spaces = self.env.obs_space.copy()
    if self.source != 'step':
      spaces['image'] = embodied.Space(np.uint8, (*self.size, 3))
    return spaces

  @property
  def act_space(self):
    return self.env.act_space

  def step(self, action):
    obs = self.env.step(action)
    if self.source == 'step':
      pass
    elif self.source == 'info':
      info = self.inner.get_info()
      assert len(info) == 1
      obs['image'] = self._resize(info[0]['rgb'], self.size, self.resize)
    else:
      raise NotImplementedError(self.source)
    return obs

  def _resize(self, image, size, method):
    if method == 'opencv':
      import cv2
      image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
      return image
    elif method == 'pillow':
      from PIL import Image
      image = Image.fromarray(image)
      image = image.resize((size[1], size[0]), Image.BILINEAR)
      image = np.array(image)
      return image
    else:
      raise NotImplementedError(method)
