import os

import embodied
import jax
import jax.numpy as jnp
import numpy as np

from . import jaxutils
from . import ninjax as nj

tree_map = jax.tree_util.tree_map
tree_flatten = jax.tree_util.tree_flatten


def Wrapper(agent_cls):
  class Agent(JAXAgent):
    configs = agent_cls.configs
    inner = agent_cls
    def __init__(self, *args, **kwargs):
      super().__init__(agent_cls, *args, **kwargs)
  return Agent


class JAXAgent(embodied.Agent):

  def __init__(self, agent_cls, obs_space, act_space, step, config):
    self.config = config.jax
    self.setup()
    self.agent = agent_cls(obs_space, act_space, step, config, name='agent')
    self.rng = jaxutils.RNG(config.seed)
    self._init_policy = nj.pure(lambda x: self.agent.policy_initial(len(x)))
    self._init_train = nj.pure(lambda x: self.agent.train_initial(len(x)))
    self._policy = nj.pure(self.agent.policy)
    self._train = nj.pure(self.agent.train)
    self._report = nj.pure(self.agent.report)
    available = jax.devices(self.config.platform)
    self.policy_device = available[self.config.policy_device]
    self.train_devices = [available[i] for i in self.config.train_devices]
    self.single_device = len(self.config.train_devices) == 1 and (
        self.config.policy_device == self.config.train_devices[0])
    print(f'JAX devices ({jax.local_device_count()}):', jax.devices())
    print('Policy device:', str(self.policy_device))
    print('Train devices:', ', '.join([str(x) for x in self.train_devices]))
    if self.config.parallel:
      pkw = dict(devices=[self.policy_device])
      tkw = dict(devices=self.train_devices)
      self._init_train = nj.pmap(self._init_train, 'i', **tkw)
      self._init_policy = nj.pmap(self._init_policy, 'i', **pkw)
      self._train = nj.pmap(self._train, 'i', **tkw)
      self._policy = nj.pmap(self._policy, 'i', static=['mode'], **pkw)
      self._report = nj.pmap(self._report, 'i', **tkw)
    else:
      pkw = dict(device=self.policy_device)
      tkw = dict(device=self.train_devices[0])
      self._init_train = nj.jit(self._init_train, **tkw)
      self._init_policy = nj.jit(self._init_policy, **pkw)
      self._train = nj.jit(self._train, **tkw)
      self._policy = nj.jit(self._policy, static=['mode'], **pkw)
      self._report = nj.jit(self._report, **tkw)
    self._once = True
    self.varibs = self._init_varibs(config, obs_space, act_space)
    self.sync()

  def setup(self):
    try:
      import tensorflow as tf
      tf.config.set_visible_devices([], 'GPU')
      tf.config.set_visible_devices([], 'TPU')
    except Exception as e:
      print('Could not disable TensorFlow devices:', e)
    if not self.config.prealloc:
      os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.8'
    xla_flags = []
    if self.config.logical_cpus:
      count = self.config.logical_cpus
      xla_flags.append(f'--xla_force_host_platform_device_count={count}')
    if xla_flags:
      os.environ['XLA_FLAGS'] = ' '.join(xla_flags)
    jax.config.update('jax_platform_name', self.config.platform)
    jax.config.update('jax_disable_jit', not self.config.jit)
    jax.config.update('jax_debug_nans', self.config.debug_nans)
    if self.config.platform == 'cpu':
      jax.config.update('jax_disable_most_optimizations', self.config.debug)
    jaxutils.COMPUTE_DTYPE = getattr(jnp, self.config.precision)

  def train(self, data, state=None):
    data = self._convert_inps(data)
    rng = self._next_rngs(mirror=False)
    if state is None:
      state, self.varibs = self._init_train(self.varibs, rng, data['is_first'])
    (outs, state, mets), self.varibs = self._train(
        self.varibs, rng, data, state)
    outs = self._convert_outs(outs)
    mets = self._convert_mets(mets)
    if self._once:
      self._once = False
      assert jaxutils.Optimizer.PARAM_COUNTS
      for name, count in jaxutils.Optimizer.PARAM_COUNTS.items():
        mets[f'params_{name}'] = float(count)
    return outs, state, mets

  def policy(self, obs, state=None, mode='train'):
    obs = obs.copy()
    obs = self._convert_inps(obs)
    rng = self._next_rngs()
    varibs = self.varibs if self.single_device else self.policy_varibs
    if state is None:
      state, _ = self._init_policy(varibs, rng, obs['is_first'])
    else:
      state = tree_map(
          jnp.asarray, state, is_leaf=lambda x: isinstance(x, list))
    (outs, state), _ = self._policy(varibs, rng, obs, state, mode=mode)
    outs = self._convert_outs(outs)
    return outs, state

  def report(self, data):
    data = self._convert_inps(data)
    rng = self._next_rngs()
    mets, _ = self._report(self.varibs, rng, data)
    mets = self._convert_mets(mets)
    return mets

  def dataset(self, generator):
    return self.agent.dataset(generator)

  def save(self):
    data = tree_flatten(tree_map(jnp.asarray, self.varibs))[0]
    data = [np.asarray(x) for x in data]
    return data

  def load(self, state):
    self.varibs = tree_flatten(self.varibs)[1].unflatten(state)
    self.sync()

  def sync(self):
    if not self.single_device:
      self.policy_varibs = jax.device_put(self.varibs, self.policy_device)

  def _convert_inps(self, value, replicas=None):
    if self.config.parallel:
      replicas = replicas or jax.local_device_count()
      check = tree_map(lambda x: len(x) % replicas == 0, value)
      if not all(jax.tree_util.tree_leaves(check)):
        shapes = tree_map(lambda x: x.shape, value)
        raise ValueError(
            f'Batch must by divisible by {replicas} replicas: {shapes}')
      value = tree_map(
          lambda x: x.reshape((replicas, -1) + x.shape[1:]), value)
    return value

  def _convert_outs(self, value):
    value = tree_map(np.asarray, jax.device_get(value))
    if self.config.parallel:
      value = tree_map(lambda x: x.reshape((-1,) + x.shape[2:]), value)
    return value

  def _convert_mets(self, value):
    value = jax.device_get(value)
    value = tree_map(np.asarray, value)
    if self.config.parallel:
      value = tree_map(lambda x: x[0], value)
    return value

  def _next_rngs(self, replicas=None, mirror=False):
    if not self.config.parallel:
      return self.rng.next()
    elif mirror:
      replicas = replicas or jax.local_device_count()
      return jnp.repeat(self.rng.next()[None], replicas, axis=0)
    else:
      replicas = replicas or jax.local_device_count()
      return jnp.stack(self.rng.next(replicas))

  def _init_varibs(self, config, obs_space, act_space):
    varibs = {}
    rng = self._next_rngs(mirror=True)
    dims = (config.batch_length, config.batch_size)
    data = self._dummy_batch({**obs_space, **act_space}, dims)
    state, varibs = self._init_train(varibs, rng, data['is_first'])
    varibs = self._train(varibs, rng, data, state, init_only=True)
    # obs = self._dummy_batch(obs_space, (1,))
    # state, varibs = self._init_policy(varibs, rng, obs['is_first'])
    # varibs = self._policy(
    #     varibs, rng, obs, state, mode='train', init_only=True)
    return varibs

  def _dummy_batch(self, spaces, batch_dims):
    spaces = list(spaces.items())
    data = {k: np.zeros(v.shape, v.dtype) for k, v in spaces}
    for dim in reversed(batch_dims):
      data = {k: np.repeat(v[None], dim, axis=0) for k, v in data.items()}
    return data
