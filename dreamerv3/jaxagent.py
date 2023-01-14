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
    def __init__(self, obs_space, act_space, step, config):
      super().__init__(agent_cls, obs_space, act_space, step, config)
  return Agent


class JAXAgent(embodied.Agent):

  def __init__(self, agent_cls, obs_space, act_space, step, config):
    self.config = config.jax
    self.setup()
    self.agent = agent_cls(obs_space, act_space, step, config, name='agent')
    self.rng = jaxutils.RNG(config.seed)
    self.varibs = {}
    self._init_policy = nj.pure(lambda x: self.agent.policy_initial(len(x)))
    self._init_train = nj.pure(lambda x: self.agent.train_initial(len(x)))
    self._policy = nj.pure(self.agent.policy)
    self._train = nj.pure(self.agent.train)
    self._report = nj.pure(self.agent.report)
    if self.config.parallel:
      self._init_train = nj.pmap(self._init_train, 'i')
      self._init_policy = nj.pmap(self._init_policy, 'i')
      self._train = nj.pmap(self._train, 'i')
      self._policy = nj.pmap(self._policy, 'i', static=['mode'])
      self._report = nj.pmap(self._report, 'i')
    else:
      self._init_train = nj.jit(self._init_train)
      self._init_policy = nj.jit(self._init_policy)
      self._train = nj.jit(self._train)
      self._policy = nj.jit(self._policy, static=['mode'])
      self._report = nj.jit(self._report)
    self._once = True

  def setup(self):
    try:
      import tensorflow as tf
      tf.config.set_visible_devices([], 'GPU')
      tf.config.set_visible_devices([], 'TPU')
    except Exception as e:
      print('Could not disable TensorFlow devices:', e)
    if not self.config.prealloc:
      os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
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
    print(f'JAX DEVICES ({jax.local_device_count()}):', jax.devices())

  def train(self, data, state=None):
    data = self._convert_inps(data)
    rng = self._next_rngs(mirror=not self.varibs)
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
    padding = jax.local_device_count() - len(obs['is_first'])
    if padding > 0:
      obs = {
          k: np.concatenate([v, np.zeros((padding,) + v.shape[1:], v.dtype)])
          for k, v in obs.items()}
    obs = self._convert_inps(obs)
    rng = self._next_rngs()
    if state is None:
      state, _ = self._init_policy(self.varibs, rng, obs['is_first'])
    (outs, state), _ = self._policy(self.varibs, rng, obs, state, mode=mode)
    outs = self._convert_outs(outs)
    if padding > 0:
      outs = {k: v[:-padding] for k, v in outs.items()}
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
