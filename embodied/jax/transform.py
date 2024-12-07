import threading
import re
from collections import Counter

import jax
from jax.sharding import PartitionSpec as P
import ninjax as nj

from . import nets as nn


LOCK = threading.Lock()


# Add tracer_sharding attribute to abstract values. This allows us to use
# shard_map based on layer callback shardings, even though JAX does not
# currently expose the shardings of tracer objects.
TRACER_SHARDINGS = {}


def init(
    fn, mesh, arg_shardings,
    param_partition_rules=(),
    act_partition_rules=(),
    static_argnums=(),
    dummy_inputs=(),
    print_partition=False,
):

  def init(fun, **jit_kwargs):
    if not getattr(fun, '_is_pure', False):
      fun = nj.pure(fun)
    def wrapper(*args, **kwargs):
      state, out = fun(*args, create=True, modify=True, ignore=True, **kwargs)
      del out
      return state, ()
    return wrapper
  fn = init(fn)

  def fn(*args, inner=fn):
    params, seed, *args = args
    old = nn.LAYER_CALLBACK
    nn.LAYER_CALLBACK = create_layer_callback(mesh, act_partition_rules)
    params, _ = inner(params, *args, seed=seed)
    nn.LAYER_CALLBACK = old
    return params

  fn = jax.jit(fn, static_argnums=static_argnums)

  params_shapes = fn.eval_shape(*dummy_inputs)
  params_sharding, grouping = resolve_rules(
      params_shapes, param_partition_rules, mesh)
  if print_partition:
    print_grouping(grouping)

  fn = jax.jit(fn, arg_shardings, params_sharding, static_argnums, None)
  params = fn(*dummy_inputs)

  return params, params_sharding


def apply(
    fn, mesh, in_shardings, out_shardings,
    partition_rules=(),
    static_argnums=(),
    single_output=False,
    return_params=False,
    donate_params=False,
    # shard_map specific
    split_rng=True,
    use_shardmap=False,
    first_outnums=(),
):

  if single_output:
    assert len(out_shardings) == 1

  def fn(*args, inner=fn):
    if donate_params:
      donated, allocated, seed, *args = args
      params = {**donated, **allocated}
    else:
      params, seed, *args = args
    if use_shardmap and len(mesh.devices) > 1 and split_rng:
      seed = jax.random.fold_in(seed, jax.lax.axis_index('d'))
    params, outs = inner(params, *args, seed=seed)
    outs = (outs,) if single_output else outs
    assert isinstance(outs, tuple)
    return (params, *outs) if return_params else outs

  if use_shardmap and len(mesh.devices) > 1:

    def fn(*args, inner=fn):
      outs = list(inner(*args))
      for i in first_outnums:
        outs[i] = jax.tree.map(lambda x: x[None], outs[i])
      return tuple(outs)

    from jax.experimental.shard_map import shard_map
    ispecs = list(jax.tree.map(lambda s: s.spec, in_shardings))
    for i in sorted(static_argnums):
      ispecs.insert(i, None)
    ispecs = tuple(ispecs)
    ospecs = jax.tree.map(lambda s: s.spec, out_shardings)
    fn = shard_map(fn, mesh, ispecs, ospecs, check_rep=False)

    def fn(*args, inner=fn):
      outs = list(inner(*args))
      for i in first_outnums:
        outs[i] = jax.tree.map(lambda x: x[0], outs[i])
      return tuple(outs)

  if single_output:
    def fn(*args, inner=fn):
      outs = inner(*args)
      assert len(outs) == 1
      return outs[0]

  if single_output:
    out_shardings = out_shardings[0]
  donate = [0] if donate_params else []

  if not use_shardmap:
    def fn(*args, inner=fn):
      with LOCK:
        old = nn.LAYER_CALLBACK
        nn.LAYER_CALLBACK = create_layer_callback(mesh, partition_rules)
        outs = inner(*args)
        nn.LAYER_CALLBACK = old
      return outs

  fn = jax.jit(fn, in_shardings, out_shardings, static_argnums, None, donate)

  return fn


def create_layer_callback(mesh, partition_rules):
  def layer_callback(y, name):
    name = f'{nj.ninjax.SCOPE}/{name}'
    for rule, spec in partition_rules:
      if re.search(rule, name):
        sharding = jax.sharding.NamedSharding(mesh, spec)
        def apply(y):
          y = jax.lax.with_sharding_constraint(y, sharding)
          if not hasattr(type(y), 'tracer_shardings'):
            type(y).tracer_sharding = property(
                lambda self: TRACER_SHARDINGS[id(self)])
          TRACER_SHARDINGS[id(y)] = sharding
          return y
        return jax.tree.map(apply, y)
    else:
      raise Exception(f'No matching rule found for activation key: {name}')
  return layer_callback


def resolve_rules(params, partition_rules, mesh):
  if len(partition_rules) == 0:
    partition_rules = [('.*', P())]
  params_spec, grouping = dict(), dict()
  for k in params.keys():
    for rule, spec in partition_rules:
      if re.search(rule, k):
        params_spec[k] = spec
        if rule not in grouping:
          grouping[rule] = []
        grouping[rule].append(k)
        break
    else:
      raise Exception(f'No matching rule found for param key: {k}')
  assert set(params.keys()) == set(params_spec.keys())
  sharding = jax.tree.map(
      lambda spec: jax.sharding.NamedSharding(mesh, spec), params_spec)
  return sharding, grouping


def print_grouping(grouping):
  for rule, ps in grouping.items():
    if len(ps) == 0:
      continue
    print(f'Partition rule "{rule}" matches {len(ps)} param tensors')
    ks = ['/'.join(p.split('/')[-2:]) for p in ps]
    ks = Counter(ks)
    ks = ks.most_common(len(ks))
    ks = [f'- .../{k}: {v}' for k, v in ks]
    print('\n'.join(ks))
