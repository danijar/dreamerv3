import functools

import jax
import jax.numpy as jnp
import ninjax as nj

from . import internal

sg = jax.lax.stop_gradient
f32 = jnp.float32
i32 = jnp.int32

COMPUTE_DTYPE = jnp.bfloat16


class Normalize(nj.Module):

  rate: float = 0.01
  limit: float = 1e-8
  perclo: float = 5.0
  perchi: float = 95.0
  debias: bool = True

  def __init__(self, impl):
    self.impl = impl
    if self.debias and self.impl != 'none':
      self.corr = nj.Variable(jnp.zeros, (), f32, name='corr')
    if self.impl == 'none':
      pass
    elif self.impl == 'meanstd':
      self.mean = nj.Variable(jnp.zeros, (), f32, name='mean')
      self.sqrs = nj.Variable(jnp.zeros, (), f32, name='sqrs')
    elif self.impl == 'perc':
      self.lo = nj.Variable(jnp.zeros, (), f32, name='lo')
      self.hi = nj.Variable(jnp.zeros, (), f32, name='hi')
    else:
      raise NotImplementedError(self.impl)

  def __call__(self, x, update):
    if update:
      self.update(x)
    return self.stats()

  def update(self, x):
    x = sg(f32(x))
    if self.impl == 'none':
      pass
    elif self.impl == 'meanstd':
      self._update(self.mean, self._mean(x))
      self._update(self.sqrs, self._mean(jnp.square(x)))
    elif self.impl == 'perc':
      self._update(self.lo, self._perc(x, self.perclo))
      self._update(self.hi, self._perc(x, self.perchi))
    else:
      raise NotImplementedError(self.impl)
    if self.debias and self.impl != 'none':
      self._update(self.corr, 1.0)

  def stats(self):
    corr = 1.0
    if self.debias and self.impl != 'none':
      corr /= jnp.maximum(self.rate, self.corr.read())
    if self.impl == 'none':
      return 0.0, 1.0
    elif self.impl == 'meanstd':
      mean = self.mean.read() * corr
      std = jnp.sqrt(jax.nn.relu(self.sqrs.read() * corr - mean ** 2))
      std = jnp.maximum(self.limit, std)
      return mean, std
    elif self.impl == 'perc':
      lo, hi = self.lo.read() * corr, self.hi.read() * corr
      return sg(lo), sg(jnp.maximum(self.limit, hi - lo))
    else:
      raise NotImplementedError(self.impl)

  def _mean(self, x):
    x = x.mean()
    axes = internal.get_data_axes()
    if axes:
      x = jax.lax.pmean(x, axes)
    return x

  def _perc(self, x, q):
    axes = internal.get_data_axes()
    if axes:
      x = jax.lax.all_gather(x, axes)
    x = jnp.percentile(x, q)
    return x

  def _update(self, var, x):
    var.write((1 - self.rate) * var.read() + self.rate * sg(x))


class SlowModel:

  def __init__(self, model, *, source, rate=1.0, every=1):
    assert rate == 1 or rate < 0.5, rate
    self.source = source
    self.model = model
    self.rate = rate
    self.every = every
    name = self.model.path + '_count'
    self.count = nj.Variable(jnp.zeros, (), i32, name=name)

  def __getattr__(self, name):
    self._initonce()
    return getattr(self.model, name)

  def __call__(self, *args, **kwargs):
    self._initonce()
    return self.model(*args, **kwargs)

  def update(self):
    self._initonce()
    mix = jnp.where(self.count.read() % self.every == 0, self.rate, 0)
    fn = lambda src, dst: mix * src + (1 - mix) * dst
    values = jax.tree.map(fn, self.source.values, self.model.values)
    [self.model.write(k, v) for k, v in values.items()]
    self.count.write(self.count.read() + 1)

  def _initonce(self, *args, method=None, **kwargs):
    assert self.source.values, 'no parameters to track'
    if not self.model.values:
      p = self.model.path + '/'
      nj.context().update({p + k: v for k, v in self.source.values.items()})
    assert self.model.values.keys() == self.source.values.keys(), (
        self.model.values.keys(), self.source.values.keys())


class LayerScan:

  def __init__(self, module, count, names=('__call__',)):
    self.module = module
    self.count = count
    self.names = names

  def __call__(self, *args, **kwargs):
    # Magic methods need to be forwarded explicitly.
    return self.__getattr__('__call__')(*args, **kwargs)

  def __getattr__(self, name):
    value = getattr(self.module, name)
    if name in self.names:
      assert callable(value)
      value = nj.pure(value, nested=True)
      value = functools.partial(
          layer_scan, value, self.module.path, self.count)
    return value


def layer_scan(fn, scope, count, inp, *args, **kwargs):
  isinner = lambda k: k.startswith(scope + '/')

  args_ = jax.tree.map(lambda x: x[0], args)  # Copy structure
  kwargs_ = jax.tree.map(lambda x: x, kwargs)  # Copy structure
  state_ = {k: v[0] if isinner(k) else v for k, v in nj.context().items()}
  state, _, accessed, modified, created = fn(
      state_, inp, *args_, ignore=True, track=True,
      seed=nj.seed(None, True), **kwargs_)

  # print('-' * 79)
  # print('accessed:', accessed)
  # print('modified:', modified)
  # print('created:', created)

  inner = lambda xs: {k: v for k, v in xs.items() if isinner(k)}
  outer = lambda xs: {k: v for k, v in xs.items() if not isinner(k)}

  unchanging = {
      k: v for k, v in nj.context().items()
      if k in accessed and k not in modified and k not in created}
  unchanging_inner = inner(unchanging)
  unchanging_outer = outer(unchanging)

  creations = {k: v for k, v in state.items() if k in created}
  creations_inner = inner(creations)
  creations_outer = outer(creations)
  nj.context().update(creations_outer)
  del creations_inner  # Will be created inside the scan.

  # Inner values do not exist yet, so we only keep them in the creations. This
  # is fine, because inner values cannot change across scan iterations anyways.
  # Outer values can change over iterations, so we need to thread them even
  # during creation.
  changing_inner = inner({
      # k: v for k, v in state.items()
      k: v for k, v in nj.context().items()
      if k in modified and k not in created})
  changing_outer = outer({
      k: v for k, v in state.items()
      if k in modified})

  # f = lambda x: {k: v.shape for k, v in x.items()}
  # print('-' * 79)
  # print('unchanging_inner', f(unchanging_inner))
  # print('unchanging_outer', f(unchanging_outer))
  # print('creations_inner', f(inner(creations)))
  # print('creations_outer', f(creations_outer))
  # print('changing_inner', f(changing_inner))
  # print('changing_outer', f(changing_outer))

  def body(carry, x):
    inp, changing_outer = carry
    arg, seed, unchanging_inner, changing_inner = x
    state = {
        **unchanging_inner, **unchanging_outer,
        **changing_inner, **changing_outer}
    state, out = fn(state, inp, *arg, **kwargs, seed=seed)
    out, *other = out if isinstance(out, tuple) else (out,)
    changing = {k: v for k, v in state.items() if k in modified}
    changing_inner = inner(changing)
    changing_outer = outer(changing)
    creations = {k: v for k, v in state.items() if k in created}
    creations_inner = inner(creations)
    carry = (out, changing_outer)
    y = (other, creations_inner, changing_inner)
    return carry, y

  seeds = nj.seed(count, True)
  carry, ys = jax.lax.scan(
      f=body,
      init=(inp, changing_outer),
      xs=(args, seeds, unchanging_inner, changing_inner),
      length=count)
  out, changing_outer = carry
  other, creations_inner, changing_inner = ys

  if nj.context().modify:
    nj.context().update(creations_inner)
    nj.context().update(changing_inner)
    nj.context().update(changing_outer)

  return (out, *other) if len(other) else out
