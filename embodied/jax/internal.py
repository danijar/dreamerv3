import concurrent.futures
import math
import os
import string

import elements
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec as P

from . import nets


def setup(
    platform=None,
    compute_dtype=jnp.bfloat16,
    debug=False,
    jit=True,
    prealloc=False,
    mock_devices=0,
    transfer_guard=True,
    deterministic=True,
    autotune=1,
    gpuflags=True,
    tpuflags=False,
    xladump=None,
    debug_nans=False,
    process_id=-1,
    num_processes=1,
    coordinator_address=None,
    compilation_cache=True,
):
  platform and jax.config.update('jax_platforms', platform)
  jax.config.update('jax_disable_most_optimizations', debug)
  jax.config.update('jax_disable_jit', not jit)
  if transfer_guard and jit and not debug_nans:
    jax.config.update('jax_transfer_guard', 'disallow')
  os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = str(bool(prealloc)).lower()
  jax.config.update('jax_debug_nans', debug_nans)
  jax.config.update('jax_enable_compilation_cache', compilation_cache)

  xlaflags = []
  xlaflags.append(f'--xla_gpu_autotune_level={autotune}')
  if deterministic:
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    xlaflags.append('--xla_gpu_deterministic_ops=true')
  if mock_devices:
    xlaflags.append(f'--xla_force_host_platform_device_count={mock_devices}')
  if xladump:
    elements.Path(xladump).mkdir()
    xlaflags.append(f'--xla_dump_to={xladump}')
    xlaflags.append('--xla_dump_hlo_as_long_text')
  if gpuflags and platform == 'gpu':
    # xla_flags.append('--xla_gpu_enable_latency_hiding_scheduler=true')
    # xla_flags.append('--xla_gpu_enable_async_all_gather=true')
    # xla_flags.append('--xla_gpu_enable_async_reduce_scatter=true')
    # xla_flags.append('--xla_gpu_enable_triton_gemm=false')
    # os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'
    # os.environ['NCCL_IB_SL'] = '1'
    # os.environ['NCCL_NVLS_ENABLE'] = '0'
    # os.environ['CUDA_MODULE_LOADING'] = 'EAGER'
    xlaflags += [
        '--xla_disable_hlo_passes=rematerialization',
        '--xla_gpu_all_gather_combine_threshold_bytes=134217728',
        '--xla_gpu_all_reduce_combine_threshold_bytes=134217728',
        '--xla_gpu_enable_all_gather_combine_by_dim=false',
        '--xla_gpu_enable_highest_priority_async_stream=true',
        '--xla_gpu_enable_latency_hiding_scheduler=true',
        '--xla_gpu_enable_pipelined_all_gather=true',
        '--xla_gpu_enable_pipelined_all_reduce=true',
        '--xla_gpu_enable_pipelined_reduce_scatter=true',
        '--xla_gpu_enable_reduce_scatter_combine_by_dim=false',
        '--xla_gpu_enable_triton_gemm=false',
        '--xla_gpu_enable_triton_softmax_fusion=false',
        '--xla_gpu_enable_while_loop_double_buffering=true',
        '--xla_gpu_graph_level=0',
        '--xla_gpu_reduce_scatter_combine_threshold_bytes=67108864',
    ]
  if tpuflags and platform == 'tpu':
    xlaflags += [
        '--xla_disable_hlo_passes=rematerialization',
        '--xla_tpu_megacore_fusion_allow_ags=false',
        '--xla_enable_async_collective_permute=true',
        '--xla_tpu_enable_ag_backward_pipelining=true',
        '--xla_tpu_enable_data_parallel_all_reduce_opt=true',
        '--xla_tpu_data_parallel_opt_different_sized_ops=true',
        '--xla_tpu_enable_async_collective_fusion=true',
        '--xla_tpu_enable_async_collective_fusion_multiple_steps=true',
        '--xla_tpu_overlap_compute_collective_tc=true',
        '--xla_enable_async_all_gather=true',
    ]
  if xlaflags:
    os.environ['XLA_FLAGS'] = ' '.join(xlaflags)

  if num_processes > 1 and platform != 'tpu':
    # Note that the process_id is unrelated to the jax.process_index() that JAX
    # will assign later. It is only used to establish initial communication and
    # for error handling, whereas jax.process_index() depends on the underlying
    # hardware mesh.
    assert process_id >= 0
    assert coordinator_address
    jax.distributed.initialize(coordinator_address, num_processes, process_id)
    index, count = jax.process_index(), jax.process_count()
    print(f'JAX multi-host initialized: ({process_id}) {index} / {count}')

  if isinstance(compute_dtype, str):
    compute_dtype = getattr(jnp, compute_dtype)
  nets.COMPUTE_DTYPE = compute_dtype


def get_named_axes():
  axes = []
  for x in string.ascii_lowercase:
    try:
      jax.lax.axis_index(x)
    except NameError:
      continue
    axes.append(x)
  return axes


def get_data_axes():
  axes = ('d', 'f')
  for x in axes:
    try:
      jax.lax.axis_index(x)
    except NameError:
      return ()
  return axes


def fetch_async(value):
  if is_multihost():
    value = to_local(value)
  with jax._src.config.explicit_device_get_scope():
    [x.copy_to_host_async() for x in jax.tree.leaves(value)]
  return value


def is_multihost():
  return jax.process_count() > 1


def device_put(value, sharding):
  if is_multihost():
    with jax._src.config.explicit_device_put_scope():
      value = jax.tree.map(
          lambda x: jax.make_array_from_process_local_data(sharding, x), value)
  else:
    value = jax.device_put(value, sharding)
  return value


def local_sharding(sharding):
  return jax.tree.map(lambda s: jax.sharding.NamedSharding(
    s.mesh.local_mesh, s.spec), sharding)


def to_local(x):
  return jax.tree.map(_to_local, x)


def _to_local(x):
  shape, sharding = x.shape, x.sharding
  spec, mesh = sharding.spec, sharding.mesh
  fullspec = [*spec, *([None] * (len(shape) - len(spec)))]
  assert len(shape) == len(fullspec)
  shard_shape = []
  for d, s in zip(shape, fullspec):
    if s is None:
      ms, lms = 1, 1
    else:
      if not isinstance(s, tuple):
        s = (s,)
      ms  = math.prod(mesh.shape[si] for si in s)
      lms = math.prod(mesh.local_mesh.shape[si] for si in s)
    shard_shape.append(d // ms * lms)
  shard_shape = tuple(shard_shape)
  arrs = [arr.data for arr in x.addressable_shards]
  sharding_local = jax.sharding.NamedSharding(mesh.local_mesh, spec)
  x = jax.make_array_from_single_device_arrays(
      shard_shape, sharding_local, arrs)
  return x


def to_global(x, global_sharding):
  if isinstance(global_sharding, jax.sharding.NamedSharding):
    return jax.tree.map(lambda xi: _to_global(xi, global_sharding), x)
  else:
    return jax.tree.map(lambda xi, gs: _to_global(xi, gs), x, global_sharding)


def _to_global(x, global_sharding):
  shape, sharding = x.shape, x.sharding
  spec = sharding.spec
  fullspec = [*spec, *([None] * (len(shape) - len(spec)))]
  assert len(shape) == len(fullspec)
  shard_shape = []
  for d, s in zip(shape, fullspec):
    if s is None:
      ms, lms = 1, 1
    else:
      if not isinstance(s, tuple):
        s = (s,)
      ms = math.prod(global_sharding.mesh.shape[si] for si in s)
      lms = math.prod(sharding.mesh.shape[si] for si in s)
    shard_shape.append(d // lms * ms)
  shard_shape = tuple(shard_shape)
  arrs = [arr.data for arr in x.addressable_shards]
  x = jax.make_array_from_single_device_arrays(
      shard_shape, global_sharding, arrs)
  return x


def move(xs, dst_sharding):
  if is_multihost():
    xs = to_local(xs)
    xs = jax.device_put(xs, local_sharding(dst_sharding))
    xs = to_global(xs, dst_sharding)
  else:
    xs = jax.device_put(xs, dst_sharding)
  return xs


def mesh(devices, shape, names):
  shape = list(map(int, shape.split(',')))
  # At most a single -1 is allowed
  assert sum(i == -1 for i in shape) <= 1
  n = len(devices)
  prod = math.prod(i for i in shape if i != -1)
  assert n % prod == 0
  shape = [i if i != -1 else n // prod for i in shape]
  assert math.prod(shape) == n
  devices = np.array(devices).reshape(shape)
  return jax.sharding.Mesh(devices, names)


def grouped_ckpt_fns(params, chunksize):
  if chunksize <= 0:
    groups = [list(params.keys())]
  else:
    groups = []
    keys, size = [], 0
    for k, v in params.items():
      if size + v.nbytes <= chunksize:
        keys.append(k)
        size += v.nbytes
      else:
        groups.append(keys)
        keys, size = [k], v.nbytes
    keys and groups.append(keys)
  assert sum(len(keys) for keys in groups) == len(params)
  assert all(len(keys) for keys in groups)
  msg = f'Compiling {len(groups)} checkpoint groups...'
  elements.print(msg, color='yellow')
  maxsize = max(sum(params[k].nbytes for k in g) for g in groups)
  print(f'Largest checkpoint group: {maxsize / (1024 ** 3):.0f} GB')

  gather_fns, shard_fns = [], []
  with concurrent.futures.ThreadPoolExecutor(64) as pool:
    for keys in groups:
      gather_fn, shard_fn = ckpt_fn(
          {k: params[k] for k in keys}, compile=False)
      gather_fns.append(pool.submit(gather_fn.compile))
      shard_fns.append(pool.submit(shard_fn.compile))
  gather_fns = [future.result() for future in gather_fns]
  shard_fns = [future.result() for future in shard_fns]

  return list(zip(groups, gather_fns, shard_fns))


def ckpt_fn(params, compile=True):
  mesh = params[list(params.keys())[0]].sharding.mesh
  mirrored = jax.sharding.NamedSharding(mesh, P())
  struct = lambda x, s: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=s)
  keys = params.keys()
  original = {k: params[k].sharding for k in keys}
  inspec = {k: struct(params[k], original[k]) for k in keys}
  gather_fn = jax.jit(lambda x: x, (original,), mirrored).lower(inspec)
  inspec = {k: struct(params[k], mirrored) for k in keys}
  shard_fn = jax.jit(lambda x: x, (mirrored,), original).lower(inspec)
  if compile:
    gather_fn = gather_fn.compile()
    shard_fn = shard_fn.compile()
  return gather_fn, shard_fn


# def node_mesh(mesh, mp_dims=('t',)):
#   n_mp = math.prod(mesh.shape[d] for d in mp_dims)
#   n_local = mesh.local_mesh.size
#   n_mp_nodes = max(1, n_mp // n_local)
#   total_nodes = mesh.size // n_local
#   n_data_nodes = total_nodes // n_mp_nodes
#   assert n_data_nodes * n_mp_nodes == total_nodes
#   data_node_rank, model_node_rank = divmod(jax.process_index(), n_mp_nodes)
#   data_node_size, model_node_size = n_data_nodes, n_mp_nodes
#   return {
#       'data_node_rank': data_node_rank,
#       'data_node_size': data_node_size,
#       'model_node_rank': model_node_rank,
#       'model_node_size': model_node_size,
#   }

