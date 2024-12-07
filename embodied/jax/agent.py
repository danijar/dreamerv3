import contextlib
import dataclasses
import re
import threading
import time

import chex
import elements
import embodied
import jax
import jax.experimental.multihost_utils
import jax.numpy as jnp
import ninjax as nj
import numpy as np
P = jax.sharding.PartitionSpec

from . import internal
from . import transform


@dataclasses.dataclass
class Options:

  policy_devices: tuple = (0,)
  train_devices: tuple = (0,)
  policy_mesh: str = '-1,1,1'
  train_mesh: str = '-1,1,1'
  profiler: bool = True
  expect_devices: int = 0
  use_shardmap: bool = False
  enable_policy: bool = True
  ckpt_chunksize: int = -1
  precompile: bool = True


class Agent(embodied.Agent):

  def __new__(subcls, obs_space, act_space, config):
    keys = Options.__dataclass_fields__
    options = {k: v for k, v in config.jax.items() if k in keys}
    setup = {k: v for k, v in config.jax.items() if k not in keys}
    jaxcfg = Options(**options)
    internal.setup(**setup)
    model = super().__new__(subcls)
    model.__init__(obs_space, act_space, config)
    outer = super().__new__(Agent)
    outer.__init__(model, obs_space, act_space, config, jaxcfg)
    return outer

  def __init__(self, model, obs_space, act_space, config, jaxcfg):
    assert not any(k.startswith('log/') for k in obs_space)
    assert 'reset' not in act_space

    self.model = model
    self.obs_space = obs_space
    self.act_space = act_space
    self.config = config
    self.jaxcfg = jaxcfg
    self.logdir = elements.Path(config.logdir)

    ext_space = self.model.ext_space  # Extra inputs to train and report.
    elements.print('Observations', color='cyan')
    [elements.print(f'  {k:<16} {v}') for k, v in obs_space.items()]
    elements.print('Actions', color='cyan')
    [elements.print(f'  {k:<16} {v}') for k, v in act_space.items()]
    elements.print('Extras', color='cyan')
    [elements.print(f'  {k:<16} {v}') for k, v in ext_space.items()]
    self.spaces = dict(**obs_space, **act_space, **ext_space)
    assert not (obs_space.keys() & ext_space.keys()), (obs_space, ext_space)
    assert not (act_space.keys() & ext_space.keys()), (act_space, ext_space)

    available = jax.devices()
    elements.print(f'JAX devices ({jax.device_count()}):', available)
    if self.jaxcfg.expect_devices > 0:
      if len(available) != self.jaxcfg.expect_devices:
        print('ALERT: Wrong number of devices')
        while True:
          time.sleep(1)
    assert len(available) == jax.process_count() * jax.local_device_count()
    flatten = lambda x: x.reshape(-1).tolist()
    devices = np.array(available).reshape(
        jax.process_count(), jax.local_device_count())
    self.policy_devices = flatten(devices[:, self.jaxcfg.policy_devices])
    self.train_devices = flatten(devices[:, self.jaxcfg.train_devices])
    print('Policy devices:', ', '.join([str(x) for x in self.policy_devices]))
    print('Train devices: ', ', '.join([str(x) for x in self.train_devices]))

    # d = DP, f = FSDP, t = TP
    self.policy_mesh = internal.mesh(
        self.policy_devices, self.jaxcfg.policy_mesh, ('d', 'f', 't'))
    self.policy_sharded = jax.sharding.NamedSharding(
        self.policy_mesh, P(('d', 'f')))
    self.policy_mirrored = jax.sharding.NamedSharding(self.policy_mesh, P())
    self.train_mesh = internal.mesh(
        self.train_devices, self.jaxcfg.train_mesh, ('d', 'f', 't'))
    self.train_sharded = jax.sharding.NamedSharding(
        self.train_mesh, P(('d', 'f')))
    self.train_mirrored = jax.sharding.NamedSharding(self.train_mesh, P())
    if self.train_mesh.shape['t'] > len(self.jaxcfg.train_devices) or (
        self.policy_mesh.shape['t'] > len(self.jaxcfg.policy_devices)):
          raise NotImplementedError('Inter-node TP is not supported!')
    if self.jaxcfg.use_shardmap:
      assert self.train_mesh.shape['d'] == self.train_mesh.size
      assert self.policy_mesh.shape['d'] == self.policy_mesh.size

    # self.train_node_mesh = internal.node_mesh(self.train_mesh, mp_dims=('t',))
    # print('Train Node mesh:',self.train_node_mesh)

    self.partition_rules = getattr(
        self.model, 'partition_rules', ([('.*', P())], []))
    elements.print('Initializing parameters...', color='yellow')
    with self.train_mesh:
      self.params, self.train_params_sharding = self._init_params()
    elements.print('Done initializing!', color='yellow')
    pattern = re.compile(self.model.policy_keys)
    self.policy_keys = [k for k in self.params.keys() if pattern.search(k)]
    assert self.policy_keys, (list(self.params.keys()), self.model.policy_keys)

    self.policy_params_sharding = {
        k: jax.sharding.NamedSharding(self.policy_mesh, v.spec)
        for k, v in self.train_params_sharding.items()
        if k in self.policy_keys}

    shared_kwargs = {'use_shardmap': jaxcfg.use_shardmap}
    tm, ts = self.train_mirrored, self.train_sharded
    pm, ps = self.policy_mirrored, self.policy_sharded
    tp, pp = self.train_params_sharding, self.policy_params_sharding
    _, ar = self.partition_rules
    self._init_train = transform.apply(
        nj.pure(self.model.init_train), self.train_mesh,
        (tp, tm), (ts,), ar, single_output=True, static_argnums=(2,),
        **shared_kwargs)
    self._init_report = transform.apply(
        nj.pure(self.model.init_report), self.train_mesh,
        (tp, tm), (ts,), ar, single_output=True, static_argnums=(2,),
        **shared_kwargs)
    self._init_policy = transform.apply(
        nj.pure(self.model.init_policy), self.policy_mesh,
        (pp, pm), (ps,), ar, single_output=True, static_argnums=(2,),
        **shared_kwargs)
    allo_sharding = {k: v for k, v in tp.items() if k in self.policy_keys}
    dona_sharding = {k: v for k, v in tp.items() if k not in self.policy_keys}
    self._train = transform.apply(
        nj.pure(self.model.train), self.train_mesh,
        (dona_sharding, allo_sharding, tm, ts, ts), (tp, ts, ts, tm), ar,
        return_params=True, donate_params=True, first_outnums=(3,),
        **shared_kwargs)
    self._report = transform.apply(
        nj.pure(self.model.report), self.train_mesh,
        (tp, tm, ts, ts), (ts, tm), ar,
        first_outnums=(1,), **shared_kwargs)
    self._policy = transform.apply(
        nj.pure(self.model.policy), self.policy_mesh,
        (pp, pm, ps, ps), (ps, ps, ps), ar,
        static_argnums=(4,), **shared_kwargs)

    self.policy_lock = threading.Lock()
    self.train_lock = threading.Lock()
    self.n_updates = elements.Counter()
    self.n_batches = elements.Counter()
    self.n_actions = elements.Counter()

    self.pending_outs = None
    self.pending_mets = None
    self.pending_sync = None

    if self.jaxcfg.enable_policy:
      policy_params = {
          k: self.params[k].copy() for k in self.policy_keys}
      self.policy_params = internal.move(
        policy_params, self.policy_params_sharding)

    self._split = jax.jit(
        lambda xs: jax.tree.map(lambda x: list(x), xs),
        internal.local_sharding(self.policy_sharded),
        internal.local_sharding(self.policy_mirrored))
    self._stack = jax.jit(
        lambda xs: jax.tree.map(
            jnp.stack, xs, is_leaf=lambda x: isinstance(x, list)),
        internal.local_sharding(self.policy_mirrored),
        internal.local_sharding(self.policy_sharded))

    self._ckpt_groups = internal.grouped_ckpt_fns(
        self.params, self.jaxcfg.ckpt_chunksize)
    if self.jaxcfg.precompile:
      elements.print('Compiling train and report...', color='yellow')
      with self.train_mesh:
        self._compile_train()
        print('Train cost analysis:')
        print(self._format_jit_stats(self._train))
        self._compile_report()
        print('Report cost analysis:')
        print(self._format_jit_stats(self._report))
      elements.print('Done compiling!', color='yellow')

  def init_policy(self, batch_size):
    if not self.jaxcfg.enable_policy:
      raise Exception('Policy not available when enable_policy=False')
    batch_size = batch_size * jax.process_count()
    if self.jaxcfg.use_shardmap:
      batch_size = batch_size // self.policy_mesh.size
    return self._split(internal.to_local(self._init_policy(
        self.policy_params, self._seeds(0, self.policy_mirrored), batch_size)))

  def init_train(self, batch_size):
    batch_size = batch_size * jax.process_count()
    if self.jaxcfg.use_shardmap:
      batch_size  = batch_size // self.train_mesh.size
    return self._init_train(
        self.params, self._seeds(0, self.train_mirrored), batch_size)

  def init_report(self, batch_size):
    batch_size = batch_size * jax.process_count()
    if self.jaxcfg.use_shardmap:
      batch_size  = batch_size // self.train_mesh.size
    return self._init_report(
        self.params, self._seeds(0, self.train_mirrored), batch_size)

  @elements.timer.section('jaxagent_policy')
  def policy(self, carry, obs, mode='train'):
    if not self.jaxcfg.enable_policy:
      raise Exception('Policy not available when enable_policy=False')
    assert not any(k.startswith('log/') for k in obs), obs.keys()
    assert sorted(obs.keys()) == sorted(self.obs_space.keys()), (
        sorted(obs.keys()), sorted(self.obs_space.keys()))
    for key, space in self.obs_space.items():
      assert np.isfinite(obs[key]).all(), (obs[key], key, space)

    with self.policy_lock:
      obs = internal.device_put(obs, self.policy_sharded)
      with self.n_actions.lock:
        counter = self.n_actions.value
        self.n_actions.value += 1
      seed = self._seeds(counter, self.policy_mirrored)
      carry = internal.to_global(self._stack(carry), self.policy_sharded)

    with self.policy_lock:
      carry, acts, outs = self._policy(
          self.policy_params, seed, carry, obs, mode)

    if self.jaxcfg.enable_policy:
      with self.policy_lock:
        if self.pending_sync:
          old = self.policy_params
          self.policy_params = self.pending_sync
          jax.tree.map(lambda x: x.delete(), old)
          self.pending_sync = None

    acts, outs = self._take_outs(internal.fetch_async((acts, outs)))
    carry = self._split(internal.to_local(carry))

    finite = outs.pop('finite', {})
    for key, fin in finite.items():
      assert all(x.all() for x in jax.tree.leaves(fin)), str(finite)
    for key, space in self.act_space.items():
      if space.discrete:
        assert (acts[key] >= 0).all(), (acts[key], key, space)
      else:
        assert np.isfinite(acts[key]).all(), (acts[key], key, space)

    return carry, acts, outs

  @elements.timer.section('jaxagent_train')
  def train(self, carry, data):
    seed = data.pop('seed')
    assert sorted(data.keys()) == sorted(self.spaces.keys()), (
        sorted(data.keys()), sorted(self.spaces.keys()))
    allo = {k: v for k, v in self.params.items() if k in self.policy_keys}
    dona = {k: v for k, v in self.params.items() if k not in self.policy_keys}
    with self.train_lock:
      with elements.timer.section('jit_train'):
        with jax.profiler.StepTraceAnnotation(
            'train', step_num=int(self.n_updates)):
          self.params, carry, outs, mets = self._train(
              dona, allo, seed, carry, data)
    self.n_updates.increment()

    if self.jaxcfg.enable_policy:
      if not self.pending_sync:
        self.pending_sync = internal.move(
            {k: allo[k] for k in self.policy_keys},
            self.policy_params_sharding)
      else:
        jax.tree.map(lambda x: x.delete(), allo)

    return_outs = {}
    if self.pending_outs:
      return_outs = self._take_outs(self.pending_outs)
    self.pending_outs = internal.fetch_async(outs)

    return_mets = {}
    if self.pending_mets:
      return_mets = self._take_outs(self.pending_mets)
    self.pending_mets = internal.fetch_async(mets)

    if self.jaxcfg.profiler:
      outdir, copyto = self.logdir, None
      if str(outdir).startswith(('gs://', '/gcs/', '/cns/')):
        copyto = outdir
        outdir = elements.Path('/tmp/profiler')
        outdir.mkdir()
      if self.n_updates == 100:
        elements.print(f'Start JAX profiler: {str(outdir)}', color='yellow')
        jax.profiler.start_trace(str(outdir))
      if self.n_updates == 120:
        elements.print('Stop JAX profiler', color='yellow')
        jax.profiler.stop_trace()
        if copyto:
          for subdir in elements.Path(outdir).glob('*'):
            subdir.copy(copyto, recursive=True)
          print(f'Copied profiler result {outdir} to {copyto}')

    return carry, return_outs, return_mets

  @elements.timer.section('jaxagent_report')
  def report(self, carry, data):
    seed = data.pop('seed')
    assert sorted(data.keys()) == sorted(self.spaces.keys()), (
        sorted(data.keys()), sorted(self.spaces.keys()))
    with self.train_lock:
      carry, mets = self._report(self.params, seed, carry, data)
      mets = self._take_outs(internal.fetch_async(mets))
    mets['params/summary'] = self._summary()
    return carry, mets

  def stream(self, st):
    def fn(data):
      for key, value in data.items():
        if np.issubdtype(value.dtype, np.floating):
          assert not np.isnan(value).any(), (key, value)
      data = internal.device_put(data, self.train_sharded)
      with self.n_batches.lock:
        counter = self.n_batches.value
        self.n_batches.value += 1
      seed = self._seeds(counter, self.train_mirrored)
      return {**data, 'seed': seed}
    return embodied.streams.Prefetch(st, fn)

  @elements.timer.section('jaxagent_save')
  def save(self):
    with self.train_lock:
      params = {}
      for keys, gather_fn, _ in self._ckpt_groups:
        group = {k: self.params[k] for k in keys}
        params.update(jax.device_get(gather_fn(group)))
    assert params
    counters = {
        'updates': int(self.n_updates),
        'batches': int(self.n_batches),
        'actions': int(self.n_actions),
    }
    data = {'params': params, 'counters': counters}
    return data

  @elements.timer.section('jaxagent_load')
  def load(self, data, regex=None):
    params = data['params']
    assert params

    with contextlib.ExitStack() as stack:
      stack.enter_context(self.train_lock)
      stack.enter_context(self.policy_lock)

      with self.n_updates.lock:
        self.n_updates.value = int(data['counters']['updates'])
      with self.n_batches.lock:
        # We restore n_batches to the checkpointed update counter, so the
        # prefetched batches that were not trained on get repeated.
        self.n_batches.value = int(data['counters']['updates'])
      with self.n_actions.lock:
        self.n_actions.value = int(data['counters']['actions'])

      if regex:
        params = {k: v for k, v in params.items() if re.match(regex, k)}
        keys = params.keys()
        jax.tree.map(lambda x: x.delete(), [self.params[k] for k in keys])
        params = internal.ckpt_fn({k: self.params[k] for k in keys})[1](
            internal.device_put(params, self.train_mirrored))
        print('Loaded pretrained checkpoint with keys:', list(params.keys()))
        self.params.update(params)
      else:
        chex.assert_trees_all_equal_shapes(self.params, params)
        jax.tree.map(lambda x: x.delete(), self.params)

        loaded = {}
        for keys, _, shard_fn in self._ckpt_groups:
          group = {k: params[k] for k in keys}
          group = shard_fn(internal.device_put(group, self.train_mirrored))
          loaded.update(group)
        self.params = loaded

      if self.jaxcfg.enable_policy:
        jax.tree.map(lambda x: x.delete(), self.policy_params)
        policy_params = {
            k: self.params[k].copy() for k in self.policy_keys}
        self.policy_params = internal.move(
            policy_params, self.policy_params_sharding)

  def _take_outs(self, outs):
    outs = jax.tree.map(lambda x: x.__array__(), outs)
    outs = jax.tree.map(
        lambda x: np.float32(x) if x.dtype == jnp.bfloat16 else x, outs)
    return outs

  def _seeds(self, counter, sharding):
    rng = np.random.default_rng(seed=[self.config.seed, int(counter)])
    seeds = rng.integers(0, np.iinfo(np.uint32).max, (2,), np.uint32)
    return internal.device_put(seeds, sharding)

  def _init_params(self):
    B = min(self.config.batch_size, len(self.jaxcfg.train_devices))
    GB = B * jax.process_count()
    T = self.config.batch_length
    C = self.config.replay_context
    tm, ts = self.train_mirrored, self.train_sharded
    us = self.jaxcfg.use_shardmap

    with jax._src.config.explicit_device_get_scope():
      seed = jax.device_put(np.array([self.config.seed, 0], np.uint32), tm)
    data = internal.device_put(self._zeros(self.spaces, (B, T + C)), ts)
    pr, ar = self.partition_rules

    params, params_sharding = transform.init(
        self.model.init_train, self.train_mesh,
        ({}, self.train_mirrored),
        param_partition_rules=pr,
        act_partition_rules=ar,
        static_argnums=(2,),
        dummy_inputs=({}, seed, GB),
        print_partition=(len(pr) >= 2),
    )
    carry = transform.apply(
        nj.pure(self.model.init_train), self.train_mesh,
        (params_sharding, tm), (ts,), single_output=True,
        static_argnums=(2,), use_shardmap=us)(
            params, seed, GB // self.train_mesh.size if us else GB)
    params, params_sharding = transform.init(
        self.model.train, self.train_mesh,
        (params_sharding, tm, ts, ts),
        param_partition_rules=pr,
        act_partition_rules=ar,
        dummy_inputs=(params, seed, carry, data),
        print_partition=(len(pr) >= 2),
    )
    return params, params_sharding

  def _compile_train(self):
    B = self.config.batch_size
    T = self.config.batch_length
    C = self.config.replay_context
    data = self._zeros(self.spaces, (B, T + C))
    data = internal.device_put(data, self.train_sharded)
    seed = self._seeds(0, self.train_mirrored)
    carry = self.init_train(B)
    allo = {k: v for k, v in self.params.items() if k in self.policy_keys}
    dona = {k: v for k, v in self.params.items() if k not in self.policy_keys}
    self._train = self._train.lower(dona, allo, seed, carry, data).compile()

  def _compile_report(self):
    B = self.config.batch_size
    T = self.config.report_length
    C = self.config.replay_context
    data = self._zeros(self.spaces, (B, T + C))
    data = internal.device_put(data, self.train_sharded)
    seed = self._seeds(0, self.train_mirrored)
    carry = self.init_report(B)
    self._report = self._report.lower(self.params, seed, carry, data).compile()

  def _summary(self):
    lines = []
    for k, v in self.params.items():
      lines.append(f'{k:<40} {v.dtype} {v.size} {v.shape}')
    return '\n'.join(lines)

  def _zeros(self, spaces, batch_shape):
    data = {k: np.zeros(v.shape, v.dtype) for k, v in spaces.items()}
    for dim in reversed(batch_shape):
      data = {k: np.repeat(v[None], dim, axis=0) for k, v in data.items()}
    return data

  def _format_jit_stats(self, compiled):
    try:
      cost = compiled.cost_analysis()
      mem = compiled.memory_analysis()
      lines = []
      lines.append(f"FLOPS:            {cost[0]['flops']:.1e}")
      lines.append(f"Memory (temp):    {mem.temp_size_in_bytes:.1e}")
      lines.append(f"Memory (inputs):  {mem.argument_size_in_bytes:.1e}")
      lines.append(f"Memory (outputs): {mem.output_size_in_bytes:.1e}")
      lines.append(f"Memory (code):    {mem.generated_code_size_in_bytes:.1e}")
      return ''.join(f'  {line}\n' for line in lines)
    except (TypeError, AttributeError, KeyError):
      return 'No available'

def init(fun, **jit_kwargs):
  if not getattr(fun, '_is_pure', False):
    fun = nj.pure(fun)
  def wrapper(*args, **kwargs):
    state, out = fun(*args, create=True, modify=True, ignore=True, **kwargs)
    del out
    return state, ()
  return wrapper
