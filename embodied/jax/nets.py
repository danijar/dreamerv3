import functools
import math
from typing import Callable

import einops
import jax
import jax.ad_checkpoint as adc
import jax.numpy as jnp
import ninjax as nj
import numpy as np

COMPUTE_DTYPE = jnp.bfloat16
LAYER_CALLBACK = lambda tensor, name: tensor

f32 = jnp.float32


def cast(xs, force=False):
  if force:
    should = lambda x: True
  else:
    should = lambda x: jnp.issubdtype(x.dtype, jnp.floating)
  return jax.tree.map(lambda x: COMPUTE_DTYPE(x) if should(x) else x, xs)


def act(name):
  if name == 'none':
    return lambda x: x
  elif name == 'mish':
    return lambda x: x * jnp.tanh(jax.nn.softplus(x))
  elif name == 'relu2':
    return lambda x: jnp.square(jax.nn.relu(x))
  elif name == 'swiglu':
    def fn(x):
      x, y = jnp.split(x, 2, -1)
      return jax.nn.silu(x) * y
    return fn
  else:
    return getattr(jax.nn, name)


def init(name):
  if callable(name):
    return name
  elif name.endswith(('_in', '_out', '_avg')):
    dist, fan = name.rsplit('_', 1)
  else:
    dist, fan = name, 'in'
  return Initializer(dist, fan, 1.0)


def dropout(x, prob, training):
  if not prob or not training:
    return x
  keep = jax.random.bernoulli(nj.seed(), 1.0 - prob, x.shape)
  return x * keep / (1.0 - prob)


def symlog(x):
  return jnp.sign(x) * jnp.log1p(jnp.abs(x))


def symexp(x):
  return jnp.sign(x) * jnp.expm1(jnp.abs(x))


def where(condition, xs, ys):
  assert condition.dtype == bool, condition.dtype
  def fn(x, y):
    assert x.shape == y.shape, (x.shape, y.shape)
    expanded = jnp.expand_dims(condition, list(range(condition.ndim, x.ndim)))
    return jnp.where(expanded, x, y)
  return jax.tree.map(fn, xs, ys)


def mask(xs, mask):
  return where(mask, xs, jax.tree.map(jnp.zeros_like, xs))


def available(*trees, bdims=None):
  def fn(*xs):
    masks = []
    for x in xs:
      if jnp.issubdtype(x.dtype, jnp.floating):
        mask = (x != -jnp.inf)
      elif jnp.issubdtype(x.dtype, jnp.signedinteger):
        mask = (x != -1)
      elif (
          jnp.issubdtype(x.dtype, jnp.unsignedinteger) or
          jnp.issubdtype(x.dtype, bool)):
        shape = x.shape if bdims is None else x.shape[:bdims]
        mask = jnp.full(shape, True, bool)
      else:
        raise NotImplementedError(x.dtype)
      if bdims is not None:
        mask = mask.all(tuple(range(bdims, mask.ndim)))
      masks.append(mask)
    return jnp.stack(masks, 0).all(0)
  return jax.tree.map(fn, *trees)


@functools.partial(jax.custom_vjp, nondiff_argnums=[1, 2])
def ensure_dtypes(x, fwd=None, bwd=None):
  fwd = fwd or COMPUTE_DTYPE
  bwd = bwd or COMPUTE_DTYPE
  assert x.dtype == fwd, (x.dtype, fwd)
  return x
def ensure_dtypes_fwd(x, fwd=None, bwd=None):
  fwd = fwd or COMPUTE_DTYPE
  bwd = bwd or COMPUTE_DTYPE
  return ensure_dtypes(x, fwd, bwd), ()
def ensure_dtypes_bwd(fwd, bwd, cache, dx):
  fwd = fwd or COMPUTE_DTYPE
  bwd = bwd or COMPUTE_DTYPE
  assert dx.dtype == bwd, (dx.dtype, bwd)
  return (dx,)
ensure_dtypes.defvjp(ensure_dtypes_fwd, ensure_dtypes_bwd)


def rms(xs):
  xs = jax.tree.leaves(xs)
  count = sum(x.size for x in xs)
  sumsq = jnp.stack([f32(jnp.square(x).sum()) for x in xs]).sum()
  return jnp.sqrt(sumsq / f32(count))


def rope(x, ts=None, inverse=False, maxlen=4096):
  B, T, _, D = x.shape
  if ts is None:
    ts = jnp.ones(B, jnp.int32)[:, None] * jnp.arange(T)[None, :]  # [B, T]
  assert ts.shape == (B, T), (ts.shape, (B, T))
  if inverse:
    ts = -ts
  freq_exponents = (2.0 / D) * jnp.arange(D // 2)  # [D/2]
  timescale = maxlen ** freq_exponents
  radians = ts[:, :, None] / timescale[None, None, :]  # [B, T, D/2]
  radians = radians[..., None, :].astype(x.dtype)  # [B, T, 1, D/2]
  sin, cos = jnp.sin(radians), jnp.cos(radians)
  x1, x2 = jnp.split(x, 2, axis=-1)  # [B, T, H, D/2]
  res = jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)
  return res


class Initializer:

  def __init__(self, dist='trunc_normal', fan='in', scale=1.0):
    self.dist = dist
    self.fan = fan
    self.scale = scale

  def __call__(self, shape, dtype=jnp.float32, fshape=None):
    shape = (shape,) if isinstance(shape, int) else tuple(shape)
    assert all(isinstance(x, int) for x in shape), (
        shape, [type(x) for x in shape])
    assert all(x > 0 for x in shape), shape
    fanin, fanout = self.compute_fans(shape if fshape is None else fshape)
    fan = {
        'avg': (fanin + fanout) / 2, 'in': fanin, 'out': fanout, 'none': 1,
    }[self.fan]
    if self.dist == 'zeros':
      x = jnp.zeros(shape, dtype)
    elif self.dist == 'uniform':
      limit = np.sqrt(1 / fan)
      x = jax.random.uniform(nj.seed(), shape, dtype, -limit, limit)
    elif self.dist == 'normal':
      x = jax.random.normal(nj.seed(), shape)
      x *= np.sqrt(1 / fan)
    elif self.dist == 'trunc_normal':
      x = jax.random.truncated_normal(nj.seed(), -2, 2, shape)
      x *= 1.1368 * np.sqrt(1 / fan)
    elif self.dist == 'normed':
      x = jax.random.uniform(nj.seed(), shape, dtype, -1, 1)
      x *= (1 / jnp.linalg.norm(x.reshape((-1, shape[-1])), 2, 0))
    else:
      raise NotImplementedError(self.dist)
    x *= self.scale
    x = x.astype(dtype)
    return x

  def __repr__(self):
    return f'Initializer({self.dist}, {self.fan}, {self.scale})'

  def __eq__(self, other):
    attributes = ('dist', 'fan', 'scale')
    return all(getattr(self, k) == getattr(other, k) for k in attributes)

  @staticmethod
  def compute_fans(shape):
    if len(shape) == 0:
      return (1, 1)
    elif len(shape) == 1:
      return (1, shape[0])
    elif len(shape) == 2:
      return shape
    else:
      space = math.prod(shape[:-2])
      return (shape[-2] * space, shape[-1] * space)


class Embed(nj.Module):

  einit: str | Callable = Initializer('trunc_normal', 'out')
  combine: bool = False

  def __init__(self, classes, units, shape=()):
    self.classes = classes
    self.units = units
    self.shape = shape

  def __call__(self, x):
    batch_shape = x.shape[:x.ndim - len(self.shape)]
    event_shape = x.shape[x.ndim - len(self.shape):]
    assert event_shape == self.shape, (self.shape, x.shape)
    N = math.prod(self.shape)
    K = self.classes
    D = self.units
    shape = (*self.shape, self.classes, self.units)
    table = self.value('table', init(self.einit), shape)
    table = table.reshape(N, K, D)
    table = table.astype(COMPUTE_DTYPE)
    index = x.reshape(-1, N)
    embed = table[jnp.arange(N), index]
    if self.combine:
      embed = embed.sum(-2).reshape(*batch_shape, self.units)
    else:
      embed = embed.reshape(*batch_shape, *self.shape, self.units)
    return embed


class Linear(nj.Module):

  bias: bool = True
  winit: str | Callable = Initializer('trunc_normal')
  binit: str | Callable = Initializer('zeros')
  outscale: float = 1.0

  def __init__(self, units):
    self.units = (units,) if isinstance(units, int) else tuple(units)

  def __call__(self, x):
    ensure_dtypes(x)
    size = math.prod(self.units)
    shape = (x.shape[-1], size)
    x = x @ self.value('kernel', self._scaled_winit, shape).astype(x.dtype)
    if self.bias:
      x += self.value('bias', init(self.binit), size).astype(x.dtype)
    x = x.reshape((*x.shape[:-1], *self.units))
    return x

  def _scaled_winit(self, *args, **kwargs):
    return init(self.winit)(*args, **kwargs) * self.outscale


class BlockLinear(nj.Module):

  bias: bool = True
  winit: str | Callable = Initializer('trunc_normal')
  binit: str | Callable = Initializer('zeros')
  outscale: float = 1.0

  def __init__(self, units, blocks):
    assert isinstance(units, int), (units, type(units))
    assert blocks <= units and units % blocks == 0, (blocks, units)
    self.units = units
    self.blocks = blocks

  def __call__(self, x):
    ensure_dtypes(x)
    assert x.shape[-1] % self.blocks == 0, (x.shape, self.blocks)
    insize = x.shape[-1]
    shape = (self.blocks, insize // self.blocks, self.units // self.blocks)
    kernel = self.value('kernel', self._scaled_winit, shape).astype(x.dtype)
    x = x.reshape((*x.shape[:-1], self.blocks, insize // self.blocks))
    x = jnp.einsum('...ki,kio->...ko', x, kernel)
    x = x.reshape((*x.shape[:-2], self.units))
    if self.bias:
      x += self.value('bias', init(self.binit), self.units).astype(x.dtype)
    return x

  def _scaled_winit(self, *args, **kwargs):
    return init(self.winit)(*args, **kwargs) * self.outscale


class Conv2D(nj.Module):

  transp: bool = False
  groups: int = 1
  pad: str = 'same'
  bias: bool = True
  winit: str | Callable = Initializer('trunc_normal')
  binit: str | Callable = Initializer('zeros')
  outscale: float = 1.0

  def __init__(self, depth, kernel, stride=1):
    self.depth = depth
    self.kernel = (kernel,) * 2 if isinstance(kernel, int) else kernel
    self.stride = stride

  def __call__(self, x):
    ensure_dtypes(x)
    shape = (*self.kernel, x.shape[-1] // self.groups, self.depth)
    kernel = self.value('kernel', self._scaled_winit, shape).astype(x.dtype)
    if self.transp:
      assert self.pad == 'same', self.pad
      # Manual implementation of fractionally strided convolution because the
      # cuDNN implementation used by XLA has bugs and performance issues.
      x = x.repeat(self.stride, -2).repeat(self.stride, -3)
      maskh = ((jnp.arange(x.shape[-3]) - 1) % self.stride == 0)[:, None]
      maskw = ((jnp.arange(x.shape[-2]) - 1) % self.stride == 0)[None, :]
      x *= (maskh * maskw)[:, :, None]
      stride = (1, 1)
    else:
      stride = (self.stride, self.stride)
    x = jax.lax.conv_general_dilated(
        x, kernel, stride, self.pad.upper(),
        feature_group_count=self.groups,
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
    if self.bias:
      x += self.value('bias', init(self.binit), self.depth).astype(x.dtype)
    return x

  def _scaled_winit(self, *args, **kwargs):
    return init(self.winit)(*args, **kwargs) * self.outscale


class Conv3D(nj.Module):

  transp: bool = False
  groups: int = 1
  pad: str = 'same'
  bias: bool = True
  winit: str | Callable = Initializer('trunc_normal')
  binit: str | Callable = Initializer('zeros')

  def __init__(self, depth, kernel, stride=1):
    self.depth = depth
    self.kernel = (kernel,) * 3 if isinstance(kernel, int) else kernel
    self.stride = (stride,) * 3 if isinstance(stride, int) else stride

  def __call__(self, x):
    ensure_dtypes(x)
    if self.transp:
      assert self.groups == 1, self.groups
      shape = (*self.kernel, x.shape[-1], self.depth)
      kernel = self.value('kernel', init(self.winit), shape).astype(x.dtype)
      x = jax.lax.conv_transpose(
          x, kernel, self.stride, self.pad.upper(),
          dimension_numbers=('NTHWC', 'THWIO', 'NTHWC'))
    else:
      shape = (*self.kernel, x.shape[-1] // self.groups, self.depth)
      kernel = self.value('kernel', init(self.winit), shape).astype(x.dtype)
      x = jax.lax.conv_general_dilated(
          x, kernel, self.stride, self.pad.upper(),
          feature_group_count=self.groups,
          dimension_numbers=('NTHWC', 'THWIO', 'NTHWC'))
    if self.bias:
      x += self.value('bias', init(self.binit), self.depth).astype(x.dtype)
    return x


class Norm(nj.Module):

  axis: tuple = (-1,)
  eps: float = 1e-4
  scale: bool = True
  shift: bool = True

  def __init__(self, impl):
    if '1em' in impl:
      impl, exp = impl.split('1em')
      self._fields['eps'] = 10 ** -int(exp)
    self.impl = impl

  def __call__(self, x):
    ensure_dtypes(x)
    dtype = x.dtype
    x = f32(x)
    axis = [a % x.ndim for a in self.axis]
    shape = [x.shape[i] if i in axis else 1 for i in range(min(axis), x.ndim)]
    if self.impl == 'none':
      pass
    elif self.impl == 'rms':
      mean2 = jnp.square(x).mean(axis, keepdims=True)
      mean2 = adc.checkpoint_name(mean2, 'small')
      scale = self._scale(shape, x.dtype)
      x = x * (jax.lax.rsqrt(mean2 + self.eps) * scale)
    elif self.impl == 'layer':
      mean = x.mean(axis, keepdims=True)
      mean2 = jnp.square(x).mean(axis, keepdims=True)
      mean2 = adc.checkpoint_name(mean2, 'small')
      var = jnp.maximum(0, mean2 - jnp.square(mean))
      var = adc.checkpoint_name(var, 'small')
      scale = self._scale(shape, x.dtype)
      shift = self._shift(shape, x.dtype)
      x = (x - mean) * (jax.lax.rsqrt(var + self.eps) * scale) + shift
    else:
      raise NotImplementedError(self.impl)
    x = x.astype(dtype)
    return x

  def _scale(self, shape, dtype):
    if not self.scale:
      return jnp.ones(shape, dtype)
    return self.value('scale', jnp.ones, shape, f32).astype(dtype)

  def _shift(self, shape, dtype):
    if not self.shift:
      return jnp.zeros(shape, dtype)
    return self.value('shift', jnp.zeros, shape, f32).astype(dtype)


class Attention(nj.Module):

  heads: int = 8
  kv_heads: int = 0
  dropout: float = 0.0
  rope: bool = True
  qknorm: str = 'none'
  bias: bool = True
  winit: str | Callable = Initializer('trunc_normal')
  binit: str | Callable = Initializer('zeros')
  outscale: float = 1.0

  def __call__(self, x, mask=None, ts=None, training=True):
    kw = dict(bias=self.bias, winit=self.winit, binit=self.binit)
    B, T, D = x.shape
    kv_heads = self.kv_heads or self.heads
    assert self.heads % kv_heads == 0
    head_ratio = self.heads // kv_heads
    if head_ratio == 1:
      qkv = self.sub('qkv', Linear, 3 * D, **kw)(x)
      q, k, v = jnp.split(qkv, 3, -1)
    else:
      q = self.sub('q', Linear, D, **kw)(x)
      k = self.sub('k', Linear, D // head_ratio, **kw)(x)
      v = self.sub('v', Linear, D // head_ratio, **kw)(x)
    q = einops.rearrange(q, 'b t (h d) -> b t h d', h=self.heads)
    k = einops.rearrange(k, 'b t (h d) -> b t h d', h=kv_heads)
    v = einops.rearrange(v, 'b t (h d) -> b t h d', h=kv_heads)

    if self.qknorm != 'none':
      q = self.sub('normq', Norm, self.qknorm)(q)
      k = self.sub('normk', Norm, self.qknorm)(k)

    if self.rope:
      q = rope(q, ts)
      k = rope(k, ts)

    q = einops.rearrange(q, 'b t (h g) d -> b t h g d', h=kv_heads)
    logits = einops.einsum(q, k, 'b tq h g d, b tk h d -> b h g tq tk')
    logits = logits * (1.0 / np.sqrt(k.shape[-1]))
    logits = f32(logits)
    if mask is not None:
      Tq, Tk = q.shape[1], k.shape[1]
      assert mask.shape == (B, Tq, Tk), (mask.shape, (B, Tq, Tk))
      mask = einops.rearrange(mask, 'b tq tk -> b 1 1 tq tk')
      logits = jnp.where(mask, logits, -1e30)
    weights = jax.nn.softmax(logits)
    weights = weights.astype(x.dtype)
    weights = dropout(weights, self.dropout, training)
    x = einops.einsum(weights, v, 'b h g tq tk, b tk h d -> b tq h g d')
    x = einops.rearrange(x, 'b t h g d -> b t (h g d)')
    x = self.sub('proj', Linear, D, **kw, outscale=self.outscale)(x)
    return x


class DictConcat:

  def __init__(self, spaces, fdims, squish=lambda x: x):
    assert 1 <= fdims, fdims
    self.keys = sorted(spaces.keys())
    self.spaces = spaces
    self.fdims = fdims
    self.squish = squish

  def __call__(self, xs):
    assert all(k in xs for k in self.spaces), (self.spaces, xs.keys())
    bdims = xs[self.keys[0]].ndim - len(self.spaces[self.keys[0]].shape)
    ys = []
    for key in self.keys:
      space = self.spaces[key]
      x = xs[key]
      m = available(x, bdims=bdims)
      x = mask(x, m)
      assert x.shape[bdims:] == space.shape, (key, bdims, space.shape, x.shape)
      if space.dtype == jnp.uint8 and len(space.shape) in (2, 3):
        raise NotImplementedError('Images are not supported.')
      elif space.discrete:
        classes = np.asarray(space.classes).flatten()
        assert (classes == classes[0]).all(), classes
        classes = classes[0].item()
        x = x.astype(jnp.int32)
        x = jax.nn.one_hot(x, classes, dtype=COMPUTE_DTYPE)
      else:
        x = self.squish(x)
        x = x.astype(COMPUTE_DTYPE)
      x = mask(x, m)
      x = x.reshape((*x.shape[:bdims + self.fdims - 1], -1))
      ys.append(x)
    return jnp.concatenate(ys, -1)


class DictEmbed(nj.Module):

  squish: Callable = lambda x: x
  padone: bool = True
  bias: bool = True
  einit: str | Callable = Initializer('trunc_normal', 'out')
  winit: str | Callable = Initializer('trunc_normal')
  binit: str | Callable = Initializer('zeros')
  impl: str = 'onehot'

  def __init__(self, spaces, units):
    self.keys = sorted(spaces.keys())
    self.spaces = spaces
    self.units = units
    self.ekw = dict(einit=self.einit)
    self.lkw = dict(bias=self.bias, winit=self.winit, binit=self.binit)

  def __call__(self, xs, bshape):
    assert isinstance(bshape, tuple), bshape
    assert all(k in xs for k in self.spaces), (self.spaces, xs.keys())
    ys = []
    init = self.value('init', self.einit, (self.units,))
    init = jnp.broadcast_to(init, (*bshape, self.units))
    init = COMPUTE_DTYPE(init)
    ys.append(init)
    for key in self.keys:
      try:
        space = self.spaces[key]
        x = xs[key]
        assert x.dtype == space.dtype, (key, space.dtype, x.dtype, x.shape)
        m = available(x, bdims=len(bshape))
        x = mask(x, m)
        if space.discrete:
          if space.dtype == jnp.uint8 and len(space.shape) in (2, 3):
            raise NotImplementedError('Images are not supported.')
          classes = int(np.asarray(space.classes).max())
          assert classes <= 256, (key, space, classes)
          if self.impl == 'lookup':
            x = self.sub(
                key, Embed, classes, self.units, space.shape,
                combine=True, **self.ekw)(x)
            # x = x.reshape((*x.shape[:len(bshape)], -1))
          elif self.impl == 'onehot':
            x = jax.nn.one_hot(x, classes, dtype=COMPUTE_DTYPE)
            x = x.reshape((*x.shape[:len(bshape)], -1))
            x = self.sub(key, Linear, self.units, **self.lkw)(x)
          else:
            raise NotImplementedError(self.impl)
        else:
          x = self.squish(x)
          x = x.astype(COMPUTE_DTYPE)
          x = x.reshape((*x.shape[:len(bshape)], -1))
          x = self.sub(key, Linear, self.units, **self.lkw)(x)
        x = mask(x, m)
        ys.append(x)
      except Exception:
        print(f"Error encoding key '{key}' with space {space}.")
        raise
    x = sum(ys)
    return x


class MLP(nj.Module):

  act: str = 'silu'
  norm: str = 'rms'
  bias: bool = True
  winit: str | Callable = Initializer('trunc_normal')
  binit: str | Callable = Initializer('zeros')

  def __init__(self, layers=5, units=1024):
    self.layers = layers
    self.units = units
    self.kw = dict(bias=self.bias, winit=self.winit, binit=self.binit)

  def __call__(self, x):
    shape = x.shape[:-1]
    x = x.astype(COMPUTE_DTYPE)
    x = x.reshape([-1, x.shape[-1]])
    for i in range(self.layers):
      x = self.sub(f'linear{i}', Linear, self.units, **self.kw)(x)
      x = self.sub(f'norm{i}', Norm, self.norm)(x)
      x = act(self.act)(x)
    x = x.reshape((*shape, x.shape[-1]))
    return x


class Transformer(nj.Module):

  units: int = 1024
  layers: int = 12
  heads: int = 8
  ffup: int = 4
  act: str = 'silu'
  norm: str = 'rms'
  glu: bool = False
  rope: bool = True
  qknorm: str = 'none'
  bias: bool = True
  winit: str | Callable = Initializer('trunc_normal')
  binit: str | Callable = Initializer('zeros')
  outscale: float = 1.0

  def __call__(self, x, mask=None, ts=None, training=True):
    kw = {k: getattr(self, k) for k in ('bias', 'winit', 'binit')}
    ak = {k: getattr(self, k) for k in ('heads', 'rope', 'qknorm', 'outscale')}
    D = x.shape[-1]
    assert D == self.units, (D, self.units)
    for i in range(self.layers):
      with nj.scope(f'layer{i}'):
        skip = x
        x = self.sub('norm1', Norm, self.norm)(x)
        x  = self.sub('mha', Attention, **kw, **ak)(x, mask, ts, training)
        x += skip
        skip = x
        x = self.sub('norm2', Norm, self.norm)(x)
        if self.glu:
          U = max(D, int((D * self.ffup * 2 / 3) // 32 * 32))
          ff1 = self.sub('ff1', Linear, U, **kw)
          ff2 = self.sub('ff2', Linear, U, **kw)
          ff3 = self.sub('ff3', Linear, D, **kw, outscale=self.outscale)
          x = ff3(act(self.act)(ff1(x)) * ff2(x))
        else:
          ff1 = self.sub('ff1', Linear, D * self.ffup, **kw)
          ff2 = self.sub('ff2', Linear, D, **kw, outscale=self.outscale)
          x = ff2(act(self.act)(ff1(x)))
        x += skip
    x = self.sub('outnorm', Norm, self.norm)(x)
    return x


class GRU(nj.Module):

  units: int = 1024
  bias: bool = True
  winit: str | Callable = Initializer('trunc_normal')
  binit: str | Callable = Initializer('zeros')
  norm: str = 'rms'
  update_bias: float = -1.0

  def initial(self, batch_size):
    return jnp.zeros((batch_size, self.units), COMPUTE_DTYPE)

  def __call__(self, carry, inputs, resets, single=False):
    assert carry.dtype == COMPUTE_DTYPE, carry.dtype
    assert inputs.dtype == COMPUTE_DTYPE, inputs.dtype
    assert resets.dtype == bool, resets.dtype
    if single:
      return self.step(carry, inputs, resets)
    carry, outputs = nj.scan(
        lambda carry, args: self.step(carry, *args),
        carry, (inputs, resets), axis=1)
    return carry, outputs

  def step(self, carry, inp, reset):
    # NOTE: When passing previous actions as input, ensure to zero out past
    # actions on is_first and clip actions to bounds if needed.
    kw = dict(bias=self.bias, winit=self.winit, binit=self.binit)
    carry = mask(carry, ~reset)
    x = jnp.concatenate([carry, inp], -1)
    x = self.sub('norm', Norm, self.norm)(x)
    x = self.sub('linear', Linear, 3 * self.units, **kw)(x)
    res, cand, update = jnp.split(x, 3, -1)
    cand = jnp.tanh(jax.nn.sigmoid(res) * cand)
    update = jax.nn.sigmoid(update + self.update_bias)
    carry = output = update * cand + (1 - update) * carry
    return carry, output

