import os
import re
import threading

import chex
import embodied
import jax
import jax.numpy as jnp
import numpy as np

from . import jaxutils
from . import ninjax as nj


def Wrapper(agent_cls):
  class Agent(JAXAgent):
    configs = agent_cls.configs
    inner = agent_cls
    def __init__(self, *args, **kwargs):
      super().__init__(agent_cls, *args, **kwargs)
  return Agent


class JAXAgent(embodied.Agent):

  def __init__(self, agent_cls, obs_space, act_space, config):
    print('Observation space')
    [embodied.print(f'  {k:<16} {v}') for k, v in obs_space.items()]
    print('Action space')
    [embodied.print(f'  {k:<16} {v}') for k, v in act_space.items()]

    self.obs_space = obs_space
    self.act_space = act_space
    self.config = config
    self.jaxcfg = config.jax
    self.logdir = embodied.Path(config.logdir)
    self._setup()
    self.agent = agent_cls(obs_space, act_space, config, name='agent')
    self.rng = np.random.default_rng(config.seed)
    self.spaces = {**obs_space, **act_space, **self.agent.aux_spaces}
    self.keys = [k for k in self.spaces if (
        not k.startswith('_') and not k.startswith('log_') and k != 'reset')]

    available = jax.devices(self.jaxcfg.platform)
    embodied.print(f'JAX devices ({jax.local_device_count()}):', available)
    if self.jaxcfg.assert_num_devices > 0:
      assert len(available) == self.jaxcfg.assert_num_devices, (
          available, len(available), self.jaxcfg.assert_num_devices)

    policy_devices = [available[i] for i in self.jaxcfg.policy_devices]
    train_devices = [available[i] for i in self.jaxcfg.train_devices]
    print('Policy devices:', ', '.join([str(x) for x in policy_devices]))
    print('Train devices: ', ', '.join([str(x) for x in train_devices]))

    self.policy_mesh = jax.sharding.Mesh(policy_devices, 'i')
    self.policy_sharded = jax.sharding.NamedSharding(
        self.policy_mesh, jax.sharding.PartitionSpec('i'))
    self.policy_mirrored = jax.sharding.NamedSharding(
        self.policy_mesh, jax.sharding.PartitionSpec())

    self.train_mesh = jax.sharding.Mesh(train_devices, 'i')
    self.train_sharded = jax.sharding.NamedSharding(
        self.train_mesh, jax.sharding.PartitionSpec('i'))
    self.train_mirrored = jax.sharding.NamedSharding(
        self.train_mesh, jax.sharding.PartitionSpec())

    self.pending_outs = None
    self.pending_mets = None
    self.pending_sync = None

    self._transform()
    self.policy_lock = threading.Lock()
    self.train_lock = threading.Lock()
    self.params = self._init_params(obs_space, act_space)
    self.updates = embodied.Counter()

    pattern = re.compile(self.agent.policy_keys)
    self.policy_keys = [k for k in self.params.keys() if pattern.search(k)]
    assert self.policy_keys, (list(self.params.keys()), self.agent.policy_keys)
    self.should_sync = embodied.when.Every(self.jaxcfg.sync_every)
    self.policy_params = jax.device_put(
        {k: self.params[k].copy() for k in self.policy_keys},
        self.policy_mirrored)

    self._lower_train()
    self._lower_report()
    self._train = self._train.compile()
    self._report = self._report.compile()
    self._stack = jax.jit(lambda xs: jax.tree.map(
        jnp.stack, xs, is_leaf=lambda x: isinstance(x, list)))
    self._split = jax.jit(lambda xs: jax.tree.map(
        lambda x: [y[0] for y in jnp.split(x, len(x))], xs))
    print('Done compiling train and report!')

  def init_policy(self, batch_size):
    seed = self._next_seeds(self.policy_sharded)
    batch_size //= len(self.policy_mesh.devices)
    carry = self._init_policy(self.policy_params, seed, batch_size)
    if self.jaxcfg.fetch_policy_carry:
      carry = self._take_outs(fetch_async(carry))
    else:
      carry = self._split(carry)
    return carry

  def init_train(self, batch_size):
    seed = self._next_seeds(self.train_sharded)
    batch_size //= len(self.train_mesh.devices)
    carry = self._init_train(self.params, seed, batch_size)
    return carry

  def init_report(self, batch_size):
    seed = self._next_seeds(self.train_sharded)
    batch_size //= len(self.train_mesh.devices)
    carry = self._init_report(self.params, seed, batch_size)
    return carry

  @embodied.timer.section('jaxagent_policy')
  def policy(self, obs, carry, mode='train'):
    obs = self._filter_data(obs)

    with embodied.timer.section('prepare_carry'):
      if self.jaxcfg.fetch_policy_carry:
        carry = jax.tree.map(
            np.stack, carry, is_leaf=lambda x: isinstance(x, list))
      else:
        with self.policy_lock:
          carry = self._stack(carry)

    with embodied.timer.section('check_inputs'):
      for key, space in self.obs_space.items():
        if key in self.keys:
          assert np.isfinite(obs[key]).all(), (obs[key], key, space)
      if self.jaxcfg.fetch_policy_carry:
        for keypath, value in jax.tree_util.tree_leaves_with_path(carry):
          assert np.isfinite(value).all(), (value, keypath)

    with embodied.timer.section('upload_inputs'):
      with self.policy_lock:
        obs, carry = jax.device_put((obs, carry), self.policy_sharded)
        seed = self._next_seeds(self.policy_sharded)

    with embodied.timer.section('jit_policy'):
      with self.policy_lock:
        acts, outs, carry = self._policy(
            self.policy_params, obs, carry, seed, mode)

    with embodied.timer.section('swap_params'):
      with self.policy_lock:
        if self.pending_sync:
          old = self.policy_params
          self.policy_params = self.pending_sync
          jax.tree.map(lambda x: x.delete(), old)
          self.pending_sync = None

    with embodied.timer.section('fetch_outputs'):
      if self.jaxcfg.fetch_policy_carry:
        acts, outs, carry = self._take_outs(fetch_async((acts, outs, carry)))
      else:
        carry = self._split(carry)
        acts, outs = self._take_outs(fetch_async((acts, outs)))

    with embodied.timer.section('check_outputs'):
      finite = outs.pop('finite', {})
      for key, (isfinite, _, _) in finite.items():
        assert isfinite.all(), str(finite)
      for key, space in self.act_space.items():
        if key == 'reset':
          continue
        elif space.discrete:
          assert (acts[key] >= 0).all(), (acts[key], key, space)
        else:
          assert np.isfinite(acts[key]).all(), (acts[key], key, space)

    return acts, outs, carry

  @embodied.timer.section('jaxagent_train')
  def train(self, data, carry):
    seed = data['seed']
    data = self._filter_data(data)
    allo = {k: v for k, v in self.params.items() if k in self.policy_keys}
    dona = {k: v for k, v in self.params.items() if k not in self.policy_keys}
    with embodied.timer.section('jit_train'):
      with self.train_lock:
        self.params, outs, carry, mets = self._train(
            allo, dona, data, carry, seed)
    self.updates.increment()

    if self.should_sync(self.updates) and not self.pending_sync:
      self.pending_sync = jax.device_put(
          {k: allo[k] for k in self.policy_keys}, self.policy_mirrored)
    else:
      jax.tree.map(lambda x: x.delete(), allo)

    return_outs = {}
    if self.pending_outs:
      return_outs = self._take_outs(self.pending_outs)
    self.pending_outs = fetch_async(outs)

    return_mets = {}
    if self.pending_mets:
      return_mets = self._take_mets(self.pending_mets)
    self.pending_mets = fetch_async(mets)

    if self.jaxcfg.profiler:
      outdir, copyto = self.logdir, None
      if str(outdir).startswith(('gs://', '/gcs/')):
        copyto = outdir
        outdir = embodied.Path('/tmp/profiler')
        outdir.mkdir()
      if self.updates == 100:
        embodied.print(f'Start JAX profiler: {str(outdir)}', color='yellow')
        jax.profiler.start_trace(str(outdir))
      if self.updates == 120:
        from embodied.core import path as pathlib
        embodied.print('Stop JAX profiler', color='yellow')
        jax.profiler.stop_trace()
        if copyto:
          pathlib.GFilePath(outdir).copy(copyto)
          print(f'Copied profiler result {outdir} to {copyto}')

    return return_outs, carry, return_mets

  @embodied.timer.section('jaxagent_report')
  def report(self, data, carry):
    seed = data['seed']
    data = self._filter_data(data)
    with embodied.timer.section('jit_report'):
      with self.train_lock:
        mets, carry = self._report(self.params, data, carry, seed)
        mets = self._take_mets(fetch_async(mets))
    return mets, carry

  def dataset(self, generator):
    def transform(data):
      return {
          **jax.device_put(data, self.train_sharded),
          'seed': self._next_seeds(self.train_sharded)}
    return embodied.Prefetch(generator, transform)

  @embodied.timer.section('jaxagent_save')
  def save(self):
    with self.train_lock:
      return jax.device_get(self.params)

  @embodied.timer.section('jaxagent_load')
  def load(self, state):
    with self.train_lock:
      with self.policy_lock:
        chex.assert_trees_all_equal_shapes(self.params, state)
        jax.tree.map(lambda x: x.delete(), self.params)
        jax.tree.map(lambda x: x.delete(), self.policy_params)
        self.params = jax.device_put(state, self.train_mirrored)
        self.policy_params = jax.device_put(
            {k: self.params[k].copy() for k in self.policy_keys},
            self.policy_mirrored)

  def _setup(self):
    try:
      import tensorflow as tf
      tf.config.set_visible_devices([], 'GPU')
      tf.config.set_visible_devices([], 'TPU')
    except Exception as e:
      print('Could not disable TensorFlow devices:', e)
    if not self.jaxcfg.prealloc:
      os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    xla_flags = []
    if self.jaxcfg.logical_cpus:
      count = self.jaxcfg.logical_cpus
      xla_flags.append(f'--xla_force_host_platform_device_count={count}')
    if self.jaxcfg.nvidia_flags:
      xla_flags.append('--xla_gpu_enable_latency_hiding_scheduler=true')
      xla_flags.append('--xla_gpu_enable_async_all_gather=true')
      xla_flags.append('--xla_gpu_enable_async_reduce_scatter=true')
      xla_flags.append('--xla_gpu_enable_triton_gemm=false')
      os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'
      os.environ['NCCL_IB_SL'] = '1'
      os.environ['NCCL_NVLS_ENABLE'] = '0'
      os.environ['CUDA_MODULE_LOADING'] = 'EAGER'
    if self.jaxcfg.xla_dump:
      outdir = embodied.Path(self.config.logdir) / 'xla_dump'
      outdir.mkdir()
      xla_flags.append(f'--xla_dump_to={outdir}')
      xla_flags.append('--xla_dump_hlo_as_long_text')
    if xla_flags:
      os.environ['XLA_FLAGS'] = ' '.join(xla_flags)
    jax.config.update('jax_platform_name', self.jaxcfg.platform)
    jax.config.update('jax_disable_jit', not self.jaxcfg.jit)
    if self.jaxcfg.transfer_guard:
      jax.config.update('jax_transfer_guard', 'disallow')
    if self.jaxcfg.platform == 'cpu':
      jax.config.update('jax_disable_most_optimizations', self.jaxcfg.debug)
    jaxutils.COMPUTE_DTYPE = getattr(jnp, self.jaxcfg.compute_dtype)
    jaxutils.PARAM_DTYPE = getattr(jnp, self.jaxcfg.param_dtype)

  def _transform(self):

    def init_policy(params, seed, batch_size):
      pure = nj.pure(self.agent.init_policy)
      return pure(params, batch_size, seed=seed)[1]

    def policy(params, obs, carry, seed, mode):
      pure = nj.pure(self.agent.policy)
      return pure(params, obs, carry, mode, seed=seed)[1]

    def init_train(params, seed, batch_size):
      pure = nj.pure(self.agent.init_train)
      return pure(params, batch_size, seed=seed)[1]

    def train(alloc, donated, data, carry, seed):
      pure = nj.pure(self.agent.train)
      combined = {**alloc, **donated}
      params, (outs, carry, mets) = pure(combined, data, carry, seed=seed)
      mets = {k: v[None] for k, v in mets.items()}
      return params, outs, carry, mets

    def init_report(params, seed, batch_size):
      pure = nj.pure(self.agent.init_report)
      return pure(params, batch_size, seed=seed)[1]

    def report(params, data, carry, seed):
      pure = nj.pure(self.agent.report)
      _, (mets, carry) = pure(params, data, carry, seed=seed)
      mets = {k: v[None] for k, v in mets.items()}
      return mets, carry

    from jax.experimental.shard_map import shard_map
    s = jax.sharding.PartitionSpec('i')  # sharded
    m = jax.sharding.PartitionSpec()     # mirrored
    if len(self.policy_mesh.devices) > 1:
      init_policy = lambda params, seed, batch_size, fn=init_policy: shard_map(
          lambda params, seed: fn(params, seed, batch_size),
          self.policy_mesh, (m, s), s, check_rep=False)(params, seed)
      policy = lambda params, obs, carry, seed, mode, fn=policy: shard_map(
          lambda params, obs, carry, seed: fn(params, obs, carry, seed, mode),
          self.policy_mesh, (m, s, s, s), s, check_rep=False)(
              params, obs, carry, seed)
    if len(self.train_mesh.devices) > 1:
      init_train = lambda params, seed, batch_size, fn=init_train: shard_map(
          lambda params, seed: fn(params, seed, batch_size),
          self.train_mesh, (m, s), s, check_rep=False)(params, seed)
      train = shard_map(
          train, self.train_mesh,
          (m, m, s, s, s), (m, s, s, m), check_rep=False)
      init_report = lambda params, seed, batch_size, fn=init_report: shard_map(
          lambda params, seed: fn(params, seed, batch_size),
          self.train_mesh, (m, s), s, check_rep=False)(params, seed)
      report = shard_map(
          report, self.train_mesh,
          (m, s, s, s), (m, s), check_rep=False)

    ps, pm = self.policy_sharded, self.policy_mirrored
    self._init_policy = jax.jit(
        init_policy, (pm, ps), ps, static_argnames=['batch_size'])
    self._policy = jax.jit(
        policy, (pm, ps, ps, ps), ps, static_argnames=['mode'])

    ts, tm = self.train_sharded, self.train_mirrored
    self._init_train = jax.jit(
        init_train, (tm, ts), ts, static_argnames=['batch_size'])
    self._train = jax.jit(
        train, (tm, tm, ts, ts, ts), (tm, ts, ts, tm), donate_argnums=[1])
    self._init_report = jax.jit(
        init_report, (tm, ts), ts, static_argnames=['batch_size'])
    self._report = jax.jit(
        report, (tm, ts, ts, ts), (tm, ts))

  def _take_mets(self, mets):
    mets = jax.tree.map(lambda x: x.__array__(), mets)
    mets = {k: v[0] for k, v in mets.items()}
    mets = jax.tree.map(
        lambda x: np.float32(x) if x.dtype == jnp.bfloat16 else x, mets)
    return mets

  def _take_outs(self, outs):
    outs = jax.tree.map(lambda x: x.__array__(), outs)
    outs = jax.tree.map(
        lambda x: np.float32(x) if x.dtype == jnp.bfloat16 else x, outs)
    return outs

  def _init_params(self, obs_space, act_space):
    B, T = self.config.batch_size, self.config.batch_length
    seed = jax.device_put(np.array([self.config.seed, 0], np.uint32))
    data = jax.device_put(self._dummy_batch(self.spaces, (B, T)))
    params = nj.init(self.agent.init_train, static_argnums=[1])(
        {}, B, seed=seed)
    _, carry = jax.jit(nj.pure(self.agent.init_train), static_argnums=[1])(
        params, B, seed=seed)
    params = nj.init(self.agent.train)(params, data, carry, seed=seed)
    return jax.device_put(params, self.train_mirrored)

  def _next_seeds(self, sharding):
    shape = [2 * x for x in sharding.mesh.devices.shape]
    seeds = self.rng.integers(0, np.iinfo(np.uint32).max, shape, np.uint32)
    return jax.device_put(seeds, sharding)

  def _filter_data(self, data):
    return {k: v for k, v in data.items() if k in self.keys}

  def _dummy_batch(self, spaces, batch_dims):
    spaces = [(k, v) for k, v in spaces.items()]
    data = {k: np.zeros(v.shape, v.dtype) for k, v in spaces}
    data = self._filter_data(data)
    for dim in reversed(batch_dims):
      data = {k: np.repeat(v[None], dim, axis=0) for k, v in data.items()}
    return data

  def _lower_train(self):
    B = self.config.batch_size
    T = self.config.batch_length
    data = self._dummy_batch(self.spaces, (B, T))
    data = jax.device_put(data, self.train_sharded)
    seed = self._next_seeds(self.train_sharded)
    carry = self.init_train(self.config.batch_size)
    allo = {k: v for k, v in self.params.items() if k in self.policy_keys}
    dona = {k: v for k, v in self.params.items() if k not in self.policy_keys}
    self._train = self._train.lower(allo, dona, data, carry, seed)

  def _lower_report(self):
    B = self.config.batch_size
    T = self.config.batch_length_eval
    data = self._dummy_batch(self.spaces, (B, T))
    data = jax.device_put(data, self.train_sharded)
    seed = self._next_seeds(self.train_sharded)
    carry = self.init_report(self.config.batch_size)
    self._report = self._report.lower(self.params, data, carry, seed)


def fetch_async(value):
  with jax._src.config.explicit_device_get_scope():
    [x.copy_to_host_async() for x in jax.tree_util.tree_leaves(value)]
  return value
