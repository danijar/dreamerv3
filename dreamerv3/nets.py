import einops
import jax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp

from . import jaxutils
from . import ninjax as nj

f32 = jnp.float32
tfd = tfp.distributions
sg = lambda x: jax.tree_util.tree_map(jax.lax.stop_gradient, x)
cast = jaxutils.cast_to_compute


class RSSM(nj.Module):

  deter: int = 4096
  hidden: int = 2048
  stoch: int = 32
  classes: int = 32
  norm: str = 'rms'
  act: str = 'gelu'
  unroll: bool = False
  unimix: float = 0.01
  outscale: float = 1.0
  imglayers: int = 2
  obslayers: int = 1
  dynlayers: int = 1
  absolute: bool = False
  cell: str = 'gru'
  blocks: int = 8
  block_fans: bool = False
  block_norm: bool = False

  def __init__(self, **kw):
    self.kw = kw

  def initial(self, bsize):
    carry = dict(
        deter=jnp.zeros([bsize, self.deter], f32),
        stoch=jnp.zeros([bsize, self.stoch, self.classes], f32))
    if self.cell == 'stack':
      carry['feat'] = jnp.zeros([bsize, self.hidden], f32)
    return cast(carry)

  def outs_to_carry(self, outs):
    keys = ('deter', 'stoch')
    if self.cell == 'stack':
      keys += ('feat',)
    return {k: outs[k][:, -1] for k in keys}

  def observe(self, carry, action, embed, reset, bdims=2):
    kw = dict(**self.kw, norm=self.norm, act=self.act)
    assert bdims in (1, 2)
    if isinstance(action, dict):
      action = jaxutils.concat_dict(action)
    carry, action, embed = cast((carry, action, embed))
    if bdims == 2:
      return jaxutils.scan(
          lambda carry, inputs: self.observe(carry, *inputs, bdims=1),
          carry, (action, embed, reset), self.unroll, axis=1)
    deter, stoch, action = jaxutils.reset(
        (carry['deter'], carry['stoch'], action), reset)
    deter, feat = self._gru(deter, stoch, action)
    x = embed if self.absolute else jnp.concatenate([feat, embed], -1)
    for i in range(self.obslayers):
      x = self.get(f'obs{i}', Linear, self.hidden, **kw)(x)
    logit = self._logit('obslogit', x)
    stoch = cast(self._dist(logit).sample(seed=nj.seed()))
    carry = dict(deter=deter, stoch=stoch)
    outs = dict(deter=deter, stoch=stoch, logit=logit)
    if self.cell == 'stack':
      carry['feat'] = feat
      outs['feat'] = feat
    return cast(carry), cast(outs)

  def imagine(self, carry, action, bdims=2):
    assert bdims in (1, 2)
    if isinstance(action, dict):
      action = jaxutils.concat_dict(action)
    carry, action = cast((carry, action))
    if bdims == 2:
      return jaxutils.scan(
          lambda carry, action: self.imagine(carry, action, bdims=1),
          cast(carry), cast(action), self.unroll, axis=1)
    deter, feat = self._gru(carry['deter'], carry['stoch'], action)
    logit = self._prior(feat)
    stoch = cast(self._dist(logit).sample(seed=nj.seed()))
    carry = dict(deter=deter, stoch=stoch)
    outs = dict(deter=deter, stoch=stoch, logit=logit)
    if self.cell == 'stack':
      carry['feat'] = feat
      outs['feat'] = feat
    return cast(carry), cast(outs)

  def loss(self, outs, free=1.0):
    metrics = {}
    prior = self._prior(outs.get('feat', outs['deter']))
    post = outs['logit']
    dyn = self._dist(sg(post)).kl_divergence(self._dist(prior))
    rep = self._dist(post).kl_divergence(self._dist(sg(prior)))
    if free:
      dyn = jnp.maximum(dyn, free)
      rep = jnp.maximum(rep, free)
    metrics.update(jaxutils.tensorstats(
        self._dist(prior).entropy(), 'prior_ent'))
    metrics.update(jaxutils.tensorstats(
        self._dist(post).entropy(), 'post_ent'))
    return {'dyn': dyn, 'rep': rep}, metrics

  def _prior(self, feat):
    kw = dict(**self.kw, norm=self.norm, act=self.act)
    x = feat
    for i in range(self.imglayers):
      x = self.get(f'img{i}', Linear, self.hidden, **kw)(x)
    return self._logit('imglogit', x)

  def _gru(self, deter, stoch, action):
    kw = dict(**self.kw, norm=self.norm, act=self.act)
    inkw = {**self.kw, 'norm': self.norm, 'binit': False}
    stoch = stoch.reshape((stoch.shape[0], -1))
    action /= sg(jnp.maximum(1, jnp.abs(action)))
    if self.cell == 'gru':
      x0 = self.get('dynnorm', Norm, self.norm)(deter)
      x1 = self.get('dynin1', Linear, self.hidden, **inkw)(stoch)
      x2 = self.get('dynin2', Linear, self.hidden, **inkw)(action)
      x = jnp.concatenate([x0, x1, x2], -1)
      for i in range(self.dynlayers):
        x = self.get(f'dyn{i}', Linear, self.hidden, **kw)(x)
      x = self.get('dyncore', Linear, 3 * self.deter, **self.kw)(x)
      reset, cand, update = jnp.split(x, 3, -1)
      reset = jax.nn.sigmoid(reset)
      cand = jnp.tanh(reset * cand)
      update = jax.nn.sigmoid(update - 1)
      deter = update * cand + (1 - update) * deter
      out = deter
    elif self.cell == 'mgu':
      x0 = self.get('dynnorm', Norm, self.norm)(deter)
      x1 = self.get('dynin1', Linear, self.hidden, **inkw)(stoch)
      x2 = self.get('dynin2', Linear, self.hidden, **inkw)(action)
      x = jnp.concatenate([x0, x1, x2], -1)
      for i in range(self.dynlayers):
        x = self.get(f'dyn{i}', Linear, self.hidden, **kw)(x)
      x = self.get('dyncore', Linear, 2 * self.deter, **self.kw)(x)
      cand, update = jnp.split(x, 2, -1)
      update = jax.nn.sigmoid(update - 1)
      cand = jnp.tanh((1 - update) * cand)
      deter = update * cand + (1 - update) * deter
      out = deter
    elif self.cell == 'blockgru':
      g = self.blocks
      flat2group = lambda x: einops.rearrange(x, '... (g h) -> ... g h', g=g)
      group2flat = lambda x: einops.rearrange(x, '... g h -> ... (g h)', g=g)
      x0 = self.get('dynin0', Linear, self.hidden, **kw)(deter)
      x1 = self.get('dynin1', Linear, self.hidden, **kw)(stoch)
      x2 = self.get('dynin2', Linear, self.hidden, **kw)(action)
      x = jnp.concatenate([x0, x1, x2], -1)[..., None, :].repeat(g, -2)
      x = group2flat(jnp.concatenate([flat2group(deter), x], -1))
      for i in range(self.dynlayers):
        x = self.get(
            f'dyn{i}', BlockLinear, self.deter, g, **kw,
            block_norm=self.block_norm, block_fans=self.block_fans)(x)
      x = self.get(
          'dyncore', BlockLinear, 3 * self.deter, g, **self.kw,
          block_fans=self.block_fans)(x)
      gates = jnp.split(flat2group(x), 3, -1)
      reset, cand, update = [group2flat(x) for x in gates]
      reset = jax.nn.sigmoid(reset)
      cand = jnp.tanh(reset * cand)
      update = jax.nn.sigmoid(update - 1)
      deter = update * cand + (1 - update) * deter
      out = deter
    elif self.cell == 'stack':
      result = []
      deters = jnp.split(deter, self.dynlayers, -1)
      x = jnp.concatenate([stoch, action], -1)
      x = self.get('in', Linear, self.hidden, **kw)(x)
      for i in range(self.dynlayers):
        skip = x
        x = get_act(self.act)(jnp.concatenate([
            self.get(f'dyngru{i}norm1', Norm, self.norm)(deters[i]),
            self.get(f'dyngru{i}norm2', Norm, self.norm)(x)], -1))
        x = self.get(
            f'dyngru{i}core', Linear, 3 * deters[i].shape[-1], **self.kw)(x)
        reset, cand, update = jnp.split(x, 3, -1)
        reset = jax.nn.sigmoid(reset)
        cand = jnp.tanh(reset * cand)
        update = jax.nn.sigmoid(update - 1)
        deter = update * cand + (1 - update) * deters[i]
        result.append(deter)
        x = self.get(f'dyngru{i}proj', Linear, self.hidden, **self.kw)(x)
        x += skip
        skip = x
        x = self.get(f'dynmlp{i}norm', Norm, self.norm)(x)
        x = self.get(
            f'dynmlp{i}up', Linear, deters[i].shape[-1], **self.kw)(x)
        x = get_act(self.act)(x)
        x = self.get(f'dynmlp{i}down', Linear, self.hidden, **self.kw)(x)
        x += skip
      out = self.get('outnorm', Norm, self.norm)(x)
      deter = jnp.concatenate(result, -1)
    else:
      raise NotImplementedError(self.cell)
    return deter, out

  def _logit(self, name, x):
    kw = dict(**self.kw, outscale=self.outscale)
    kw['binit'] = False
    x = self.get(name, Linear, self.stoch * self.classes, **kw)(x)
    logit = x.reshape(x.shape[:-1] + (self.stoch, self.classes))
    if self.unimix:
      probs = jax.nn.softmax(logit, -1)
      uniform = jnp.ones_like(probs) / probs.shape[-1]
      probs = (1 - self.unimix) * probs + self.unimix * uniform
      logit = jnp.log(probs)
    return logit

  def _dist(self, logit):
    return tfd.Independent(jaxutils.OneHotDist(logit.astype(f32)), 1)


class SimpleEncoder(nj.Module):

  depth: int = 128
  mults: tuple = (1, 2, 4, 2)
  layers: int = 5
  units: int = 1024
  symlog: bool = True
  norm: str = 'rms'
  act: str = 'gelu'
  kernel: int = 4
  outer: bool = False
  minres: int = 4

  def __init__(self, spaces, **kw):
    assert all(len(s.shape) <= 3 for s in spaces.values()), spaces
    self.spaces = spaces
    self.veckeys = [k for k, s in spaces.items() if len(s.shape) <= 2]
    self.imgkeys = [k for k, s in spaces.items() if len(s.shape) == 3]
    self.vecinp = Input(self.veckeys, featdims=1)
    self.imginp = Input(self.imgkeys, featdims=3)
    self.depths = tuple(self.depth * mult for mult in self.mults)
    self.kw = kw

  def __call__(self, data, bdims=2):
    kw = dict(**self.kw, norm=self.norm, act=self.act)
    outs = []

    shape = data['is_first'].shape[:bdims]
    data = {k: data[k] for k in self.spaces}
    data = jaxutils.onehot_dict(data, self.spaces)

    if self.veckeys:
      x = self.vecinp(data, bdims, f32)
      x = x.reshape((-1, *x.shape[bdims:]))
      x = jaxutils.symlog(x) if self.symlog else x
      x = jaxutils.cast_to_compute(x)
      for i in range(self.layers):
        x = self.get(f'mlp{i}', Linear, self.units, **kw)(x)
      outs.append(x)

    if self.imgkeys:
      print('ENC')
      x = self.imginp(data, bdims, jaxutils.COMPUTE_DTYPE) - 0.5
      x = x.reshape((-1, *x.shape[bdims:]))
      for i, depth in enumerate(self.depths):
        stride = 1 if self.outer and i == 0 else 2
        x = self.get(f'conv{i}', Conv2D, depth, self.kernel, stride, **kw)(x)
      assert x.shape[-3] == x.shape[-2] == self.minres, x.shape
      x = x.reshape((x.shape[0], -1))
      print(x.shape, 'out')
      outs.append(x)

    x = jnp.concatenate(outs, -1)
    x = x.reshape((*shape, *x.shape[1:]))
    return x


class SimpleDecoder(nj.Module):

  inputs: tuple = ('deter', 'stoch')
  depth: int = 128
  mults: tuple = (1, 2, 4, 3)
  sigmoid: bool = True
  layers: int = 5
  units: int = 1024
  norm: str = 'rms'
  act: str = 'gelu'
  outscale: float = 1.0
  vecdist: str = 'symlog_mse'
  kernel: int = 4
  outer: bool = False
  block_fans: bool = False
  block_norm: bool = False
  block_space: int = 0
  hidden_stoch: bool = False
  space_hidden: int = 0
  minres: int = 4

  def __init__(self, spaces, **kw):
    assert all(len(s.shape) <= 3 for s in spaces.values()), spaces
    self.inp = Input(self.inputs, featdims=1)
    self.veckeys = [k for k, s in spaces.items() if len(s.shape) <= 2]
    self.imgkeys = [k for k, s in spaces.items() if len(s.shape) == 3]
    self.spaces = spaces
    self.depths = tuple([self.depth * mult for mult in self.mults])
    self.imgdep = sum(self.spaces[k].shape[-1] for k in self.imgkeys)
    self.kw = kw

  def __call__(self, lat, bdims=2):
    kw = dict(**self.kw, norm=self.norm, act=self.act)
    outs = {}

    if self.veckeys:
      inp = self.inp(lat, bdims, jaxutils.COMPUTE_DTYPE)
      x = inp.reshape((-1, inp.shape[-1]))
      for i in range(self.layers):
        x = self.get(f'mlp{i}', Linear, self.units, **kw)(x)
      x = x.reshape((*inp.shape[:bdims], *x.shape[1:]))
      for k in self.veckeys:
        dist = (
            dict(dist='softmax', bins=self.spaces[k].classes)
            if self.spaces[k].discrete else dict(dist=self.vecdist))
        k = k.replace('/', '_')
        outs[k] = self.get(f'out_{k}', Dist, self.spaces[k].shape, **dist)(x)

    if self.imgkeys:
      inp = self.inp(lat, bdims, jaxutils.COMPUTE_DTYPE)
      print('DEC')
      shape = (self.minres, self.minres, self.depths[-1])
      x = inp.reshape((-1, inp.shape[-1]))

      if self.space_hidden:
        x = self.get('space0', Linear, self.space_hidden * self.units, **kw)(x)
        x = self.get('space1', Linear, shape, **kw)(x)
      elif self.block_space:
        g = self.block_space
        x0 = einops.rearrange(cast(lat['deter']), 'b t ... -> (b t) ...')
        x1 = einops.rearrange(cast(lat['stoch']), 'b t l c -> (b t) (l c)')
        x0 = self.get(
            'space0', BlockLinear, int(np.prod(shape)), g, **self.kw,
            block_fans=self.block_fans, block_norm=self.block_norm)(x0)
        x0 = einops.rearrange(
            x0, '... (g h w c) -> ... h w (g c)',
            h=self.minres, w=self.minres, g=g)
        if self.hidden_stoch:
          x1 = self.get('space1hid', Linear, 2 * self.units, **kw)(x1)
        x1 = self.get('space1', Linear, shape, **self.kw)(x1)
        x = self.get('spacenorm', Norm, self.norm, act=self.act)(x0 + x1)
      else:
        x = self.get('space', Linear, shape, **kw)(x)

      print(x.shape, 'in')
      for i, depth in reversed(list(enumerate(self.depths[:-1]))):
        x = self.get(
            f'conv{i}', Conv2D, depth, self.kernel, 2, **kw, transp=True)(x)
      outkw = dict(**self.kw, outscale=self.outscale, transp=True)
      stride = 1 if self.outer else 2
      x = self.get(
          'imgout', Conv2D, self.imgdep, self.kernel, stride, **outkw)(x)
      x = jax.nn.sigmoid(x) if self.sigmoid else x + 0.5
      print(x.shape, 'out')
      x = x.reshape((*inp.shape[:bdims], *x.shape[1:]))
      split = np.cumsum([self.spaces[k].shape[-1] for k in self.imgkeys][:-1])
      for k, out in zip(self.imgkeys, jnp.split(x, split, -1)):
        outs[k] = jaxutils.MSEDist(f32(out), 3, 'sum')

    return outs


class MLP(nj.Module):

  layers: int = None
  units: int = None
  block_fans: bool = False
  block_norm: bool = False

  def __init__(self, shape, dist='mse', inputs=['tensor'], **kw):
    shape = (shape,) if isinstance(shape, (int, np.integer)) else shape
    assert isinstance(shape, (tuple, dict, type(None))), shape
    assert isinstance(dist, (str, dict)), dist
    assert isinstance(dist, dict) == isinstance(shape, dict), (dist, shape)
    self.shape = shape
    self.dist = dist
    self.inputs = Input(inputs, featdims=1)
    distonly = ('outscale', 'minstd', 'maxstd', 'unimix', 'bins')
    self.lkw = {k: v for k, v in kw.items() if k not in distonly}
    forbidden = ('binit', 'norm', 'act')
    self.dkw = {k: v for k, v in kw.items() if k not in forbidden}

  def __call__(self, inputs, bdims=2, training=False):
    feat = self.inputs(inputs, bdims, jaxutils.COMPUTE_DTYPE)
    x = feat.reshape([-1, feat.shape[-1]])
    for i in range(self.layers):
      x = self.get(f'h{i}', Linear, self.units, **self.lkw)(x)
    x = x.reshape((*feat.shape[:bdims], -1))
    if self.shape is None:
      return x
    elif isinstance(self.shape, dict):
      return {
          k: self._out(k, v, self.dist[k], x) for k, v in self.shape.items()}
    else:
      return self._out('dist', self.shape, self.dist, x)

  def _out(self, name, shape, dist, x):
    name = name.replace('/', '_').replace('.', '_')
    return self.get(name, Dist, shape, dist, **self.dkw)(x)


class Dist(nj.Module):

  outscale: float = 0.1
  minstd: float = 1.0
  maxstd: float = 1.0
  unimix: float = 0.0
  bins: int = 255

  def __init__(self, shape, dist='mse', **kw):
    assert all(isinstance(dim, (int, np.integer)) for dim in shape), shape
    forbidden = ('binit', 'norm', 'act')
    assert all(k not in kw for k in forbidden), (forbidden, kw)
    self.shape = shape
    self.dist = dist
    self.kw = dict(**kw, outscale=self.outscale)

  def __call__(self, inputs):
    dist = self.inner(inputs)
    assert tuple(dist.batch_shape) == tuple(inputs.shape[:-1]), (
        dist.batch_shape, dist.event_shape, inputs.shape)
    return dist

  def inner(self, inputs):
    shape = self.shape
    padding = 0

    if 'twohot' in self.dist or self.dist == 'softmax':
      padding = int(self.bins % 2)
      shape = (*self.shape, self.bins + padding)

    out = self.get('out', Linear, int(np.prod(shape)), **self.kw)(inputs)
    out = out.reshape(inputs.shape[:-1] + shape).astype(f32)
    out = out[..., :-padding] if padding else out

    if 'normal' in self.dist:
      units = int(np.prod(self.shape))
      std = self.get('std', Linear, units, **self.kw)(inputs)
      std = std.reshape(inputs.shape[:-1] + self.shape).astype(f32)

    if self.dist == 'symlog_mse':
      fwd, bwd = jaxutils.symlog, jaxutils.symexp
      return jaxutils.TransformedMseDist(out, len(self.shape), fwd, bwd)

    if self.dist == 'hyperbolic_mse':
      fwd = lambda x, eps=1e-3: (
          jnp.sign(x) * (jnp.sqrt(jnp.abs(x) + 1) - 1) + eps * x)
      bwd = lambda x, eps=1e-3: jnp.sign(x) * (jnp.square(
          jnp.sqrt(1 + 4 * eps * (eps + 1 + jnp.abs(x))) / 2 / eps -
          1 / 2 / eps) - 1)
      return jaxutils.TransformedMseDist(out, len(self.shape), fwd, bwd)

    if self.dist == 'symlog_and_twohot':
      bins = np.linspace(-20, 20, out.shape[-1])
      return jaxutils.TwoHotDist(
          out, bins, len(self.shape), jaxutils.symlog, jaxutils.symexp)

    if self.dist == 'symexp_twohot':
      if out.shape[-1] % 2 == 1:
        half = jnp.linspace(-20, 0, (out.shape[-1] - 1) // 2 + 1, dtype=f32)
        half = jaxutils.symexp(half)
        bins = jnp.concatenate([half, -half[:-1][::-1]], 0)
      else:
        half = jnp.linspace(-20, 0, out.shape[-1] // 2, dtype=f32)
        half = jaxutils.symexp(half)
        bins = jnp.concatenate([half, -half[::-1]], 0)
      return jaxutils.TwoHotDist(out, bins, len(self.shape))

    if self.dist == 'hyperbolic_twohot':
      eps = 0.001
      f = lambda x: np.sign(x) * (np.square(np.sqrt(
          1 + 4 * eps * (eps + 1 + np.abs(x))) / 2 / eps - 1 / 2 / eps) - 1)
      bins = f(np.linspace(-300, 300, out.shape[-1]))
      return jaxutils.TwoHotDist(out, bins, len(self.shape))

    if self.dist == 'mse':
      return jaxutils.MSEDist(out, len(self.shape), 'sum')

    if self.dist == 'huber':
      return jaxutils.HuberDist(out, len(self.shape), 'sum')

    if self.dist == 'normal':
      lo, hi = self.minstd, self.maxstd
      std = (hi - lo) * jax.nn.sigmoid(std + 2.0) + lo
      dist = tfd.Normal(jnp.tanh(out), std)
      dist = tfd.Independent(dist, len(self.shape))
      dist.minent = np.prod(self.shape) * tfd.Normal(0.0, lo).entropy()
      dist.maxent = np.prod(self.shape) * tfd.Normal(0.0, hi).entropy()
      return dist

    if self.dist == 'trunc_normal':
      lo, hi = self.minstd, self.maxstd
      std = (hi - lo) * jax.nn.sigmoid(std + 2.0) + lo
      dist = tfd.TruncatedNormal(jnp.tanh(out), std, -1, 1)
      dist = tfd.Independent(dist, len(self.shape))
      dist.minent = np.prod(self.shape) * (
          tfd.TruncatedNormal(1.0, lo, -1, 1).entropy())
      dist.maxent = np.prod(self.shape) * (
          tfd.TruncatedNormal(0.0, hi, -1, 1).entropy())
      return dist

    if self.dist == 'binary':
      dist = tfd.Bernoulli(out)
      if self.shape:
        dist = tfd.Independent(dist, len(self.shape))
      return dist

    if self.dist == 'softmax':
      dist = tfd.Categorical(out)
      if len(self.shape) > 1:
        dist = tfd.Independent(dist, len(self.shape) - 1)
      return dist

    if self.dist == 'onehot':
      if self.unimix:
        probs = jax.nn.softmax(out, -1)
        uniform = jnp.ones_like(probs) / probs.shape[-1]
        probs = (1 - self.unimix) * probs + self.unimix * uniform
        out = jnp.log(probs)
      dist = jaxutils.OneHotDist(out)
      if len(self.shape) > 1:
        dist = tfd.Independent(dist, len(self.shape) - 1)
      dist.minent = 0.0
      dist.maxent = np.prod(self.shape[:-1]) * np.log(self.shape[-1])
      return dist

    raise NotImplementedError(self.dist)


class Conv2D(nj.Module):

  groups: int = 1
  transp: bool = False
  act: str = 'none'
  norm: str = 'none'
  pad: str = 'same'
  bias: bool = True
  outscale: float = 1.0
  winit: str = 'normal'
  binit: bool = False
  fan: str = 'in'
  dtype: str = 'default'

  def __init__(self, depth, kernel, stride=1):
    self.depth = depth
    self.kernel = kernel
    self.stride = stride
    self._winit = Initializer(self.winit, self.outscale, self.fan, self.dtype)
    self._binit = Initializer('zeros', 1.0, self.fan, self.dtype)
    self._norm = Norm(self.norm, name='norm')

  def __call__(self, x):
    assert x.dtype == jaxutils.COMPUTE_DTYPE, (x.dtype, x.shape)
    x = self._layer(x)
    x = self._norm(x)
    x = get_act(self.act)(x)
    return x

  def _layer(self, x):
    if self.transp:
      assert self.groups == 1, self.groups
      shape = (self.kernel, self.kernel, x.shape[-1], self.depth)
      kernel = self.get('kernel', self._winit, shape)
      kernel = jaxutils.cast_to_compute(kernel)
      flops = int(np.prod(shape)) * x.shape[-3] * x.shape[-2]
      x = jax.lax.conv_transpose(
          x, kernel, (self.stride, self.stride), self.pad.upper(),
          dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
    else:
      G = self.groups
      shape = (self.kernel, self.kernel, x.shape[-1] // G, self.depth)
      kernel = self.get('kernel', self._winit, shape)
      kernel = jaxutils.cast_to_compute(kernel)
      x = jax.lax.conv_general_dilated(
          x, kernel, (self.stride, self.stride), self.pad.upper(),
          feature_group_count=self.groups,
          dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
      flops = int(np.prod(shape)) * x.shape[-3] * x.shape[-2]
    if self.bias:
      if self.binit:
        args = (self._winit, self.depth, shape)
      else:
        args = (self._binit, self.depth)
      x += self.get('bias', *args).astype(x.dtype)
      flops += int(np.prod(x.shape[-3:]))
    assert x.dtype == jaxutils.COMPUTE_DTYPE, (x.dtype, x.shape)
    return x


class Linear(nj.Module):

  act: str = 'none'
  norm: str = 'none'
  bias: bool = True
  outscale: float = 1.0
  winit: str = 'normal'
  binit: bool = False
  fan: str = 'in'
  dtype: str = 'default'
  fanin: int = 0

  def __init__(self, units):
    self.units = (units,) if isinstance(units, int) else tuple(units)
    self._winit = Initializer(
        self.winit, self.outscale, self.fan, self.dtype)
    self._binit = Initializer('zeros', 1.0, self.fan, self.dtype)
    self._norm = Norm(self.norm, name='norm')

  def __call__(self, x):
    assert x.dtype == jaxutils.COMPUTE_DTYPE, (x.dtype, x.shape)
    x = self._layer(x)
    x = self._norm(x)
    x = get_act(self.act)(x)
    return x

  def _layer(self, x):
    shape = (x.shape[-1], int(np.prod(self.units)))
    fan_shape = (self.fanin, shape[1]) if self.fanin else None
    x = x @ self.get('kernel', self._winit, shape, fan_shape).astype(x.dtype)
    flops = int(np.prod(shape))
    if self.bias:
      if self.binit:
        args = (self._winit, np.prod(self.units), shape)
      else:
        args = (self._binit, np.prod(self.units))
      x += self.get('bias', *args).astype(x.dtype)
      flops += int(np.prod(self.units))
    assert x.dtype == jaxutils.COMPUTE_DTYPE, (x.dtype, x.shape)
    if len(self.units) > 1:
      x = x.reshape(x.shape[:-1] + self.units)
    return x


class BlockLinear(nj.Module):

  act: str = 'none'
  norm: str = 'none'
  bias: bool = True
  outscale: float = 1.0
  winit: str = 'normal'
  binit: bool = False
  fan: str = 'in'
  dtype: str = 'default'
  block_fans: bool = False
  block_norm: bool = False

  def __init__(self, units, groups):
    self.units = (units,) if isinstance(units, int) else tuple(units)
    assert groups <= np.prod(units), (groups, units)
    self.groups = groups
    self._winit = Initializer(
        self.winit, self.outscale, self.fan, self.dtype,
        block_fans=self.block_fans)
    self._binit = Initializer('zeros', 1.0, self.fan, self.dtype)
    if self.block_norm:
      self._norm = [
          Norm(self.norm, name=f'norm{i}') for i in range(self.groups)]
    else:
      self._norm = Norm(self.norm, name='norm')

  def __call__(self, x):
    assert x.dtype == jaxutils.COMPUTE_DTYPE, (x.dtype, x.shape)
    x = self._layer(x)
    if self.block_norm and self._norm != 'none':
      x = jnp.concatenate([
          f(y) for f, y in zip(self._norm, jnp.split(x, self.groups, -1))], -1)
    else:
      x = self._norm(x)
    x = get_act(self.act)(x)
    return x

  def _layer(self, x):
    bdims, indim, outdim = x.shape[:-1], x.shape[-1], np.prod(self.units)
    if indim % self.groups != 0:
      pad = int(np.ceil(indim / self.groups)) * self.groups - indim
      x = jnp.concatenate([x, jnp.zeros((*x.shape[:-1], pad), x.dtype)], -1)
      indim = x.shape[-1]
    assert indim % self.groups == outdim % self.groups == 0, (
        indim, outdim, self.groups, self.units)
    shape = (self.groups, indim // self.groups, outdim // self.groups)
    kernel = self.get('kernel', self._winit, shape, shape).astype(x.dtype)
    flops = int(np.prod(shape))
    x = x.reshape((*bdims, self.groups, indim // self.groups))
    x = jnp.einsum('...ki,kio->...ko', x, kernel)
    x = x.reshape((*bdims, outdim))
    if self.bias:
      if self.binit:
        args = (self._winit, np.prod(self.units), shape)
      else:
        args = (self._binit, np.prod(self.units))
      bias = self.get('bias', *args)
      x += bias.astype(x.dtype)
      flops += int(np.prod(self.units))
    if len(self.units) > 1:
      x = x.reshape(x.shape[:-1] + self.units)
    assert x.dtype == jaxutils.COMPUTE_DTYPE, (x.dtype, x.shape)
    return x


class Embed(nj.Module):

  outscale: float = 1.0
  winit: str = 'normal'
  fan: str = 'in'
  dtype: str = 'default'

  def __init__(self, count, units):
    self.count = count
    self.units = units
    self._winit = Initializer(self.winit, self.outscale, self.fan, self.dtype)

  def __call__(self, x):
    assert x.dtype in (jnp.uint32, jnp.int32), x.dtype
    shape = (self.count, self.units)
    fan_shape = (1, self.units)
    w = self.get('embed', self._winit, shape, fan_shape).astype(x.dtype)
    return jnp.take(w, x, axis=0)


class Norm(nj.Module):

  act: str = 'none'

  def __init__(self, impl, eps=1e-4):
    if '1em' in impl:
      impl, exponent = impl.split('1em')
      eps = 10 ** -int(exponent)
    self._impl = impl
    self._eps = eps

  def __call__(self, x):
    x = self._norm(x)
    x = get_act(self.act)(x)
    return x

  def _norm(self, x):
    if self._impl == 'none':
      return x
    elif self._impl == 'layer':
      x = x.astype(f32)
      mean = x.mean(-1)[..., None]
      mean2 = jnp.square(x).mean(-1)[..., None]
      var = jnp.maximum(0, mean2 - jnp.square(mean))
      scale = self.get('scale', jnp.ones, x.shape[-1], f32)
      offset = self.get('offset', jnp.zeros, x.shape[-1], f32)
      mult = scale * jax.lax.rsqrt(var + self._eps)
      x = (x - mean) * mult + offset
      return cast(x)
    elif self._impl == 'rms':
      dtype = x.dtype
      x = f32(x) if x.dtype == jnp.float16 else x
      scale = self.get('scale', jnp.ones, x.shape[-1], f32).astype(x.dtype)
      mult = jax.lax.rsqrt((x * x).mean(-1)[..., None] + self._eps) * scale
      return (x * mult).astype(dtype)
    elif self._impl == 'rms_instance':
      x = x.astype(f32)
      scale = self.get('scale', jnp.ones, x.shape[-1], f32)
      mult = jax.lax.rsqrt((x * x).mean((-3, -2), keepdims=True) + self._eps)
      mult = mult * scale
      return cast(x * mult)
    elif self._impl == 'grn':
      assert len(x.shape) >= 4, x.shape
      x = x.astype(f32)
      norm = jnp.linalg.norm(x, 2, (-3, -2), keepdims=True)
      norm /= (norm.mean(-1, keepdims=True) + self._eps)
      scale = self.get('scale', jnp.ones, x.shape[-1], f32)
      offset = self.get('offset', jnp.zeros, x.shape[-1], f32)
      x = (norm * scale + 1) * x + offset
      return cast(x)
    elif self._impl == 'instance':
      x = x.astype(f32)
      mean = x.mean(axis=(-3, -2), keepdims=True)
      var = x.var(axis=(-3, -2), keepdims=True)
      scale = self.get('scale', jnp.ones, x.shape[-1], f32)
      offset = self.get('offset', jnp.zeros, x.shape[-1], f32)
      x = (scale * jax.lax.rsqrt(var + self._eps)) * (x - mean) + offset
      return cast(x)
    else:
      raise NotImplementedError(self._impl)


class Input:

  def __init__(self, keys=['tensor'], featdims=1):
    self.keys = tuple(keys)
    self.featdims = featdims

  def __call__(self, inputs, bdims=2, dtype=None):
    if not isinstance(inputs, dict):
      inputs = {'tensor': inputs}
    try:
      xs = []
      for key in self.keys:
        x = inputs[key]
        if jnp.issubdtype(x.dtype, jnp.complexfloating):
          x = jnp.concatenate([x.real, x.imag], -1)
        x = x.astype(dtype or inputs[self.keys[0]].dtype)
        x = x.reshape((*x.shape[:bdims + self.featdims - 1], -1))
        msg = f'Invalid input ({nj.SCOPE}, {key}, {x.shape}, {x.dtype}): {{x}}'
        jaxutils.check(jnp.isfinite(x).all(), msg, x=x)
        xs.append(x)
      xs = jnp.concatenate(xs, -1)
    except (KeyError, ValueError, TypeError) as e:
      shapes = {k: v.shape for k, v in inputs.items()}
      raise ValueError(
          f'Error: {e}\n'
          f'Input shapes: {shapes}\n' +
          f'Requested keys: {self.keys}')
    return xs


class Initializer:

  VARIANCE_FACTOR = 1.0

  def __init__(
      self, dist='normal', scale=1.0, fan='in', dtype='default',
      block_fans=False):
    self.dist = dist
    self.scale = scale
    self.fan = fan
    self.dtype = dtype
    self.block_fans = block_fans

  def __call__(self, shape, fan_shape=None):
    shape = (shape,) if isinstance(shape, (int, np.integer)) else tuple(shape)
    assert all(x > 0 for x in shape), shape
    dtype = jaxutils.PARAM_DTYPE if self.dtype == 'default' else self.dtype
    dtype = getattr(jnp, dtype) if isinstance(dtype, str) else dtype
    fanin, fanout = self._fans(fan_shape or shape)
    fan = {'avg': (fanin + fanout) / 2, 'in': fanin, 'out': fanout}[self.fan]
    if self.dist == 'zeros':
      value = jnp.zeros(shape, dtype)
    elif self.dist == 'uniform':
      limit = np.sqrt(self.VARIANCE_FACTOR / fan)
      value = jax.random.uniform(nj.seed(), shape, dtype, -limit, limit)
    elif self.dist == 'normal':
      value = jax.random.truncated_normal(nj.seed(), -2, 2, shape)
      value *= 1.1368 * np.sqrt(self.VARIANCE_FACTOR / fan)
      value = value.astype(dtype)
    elif self.dist == 'normed':
      value = jax.random.uniform(nj.seed(), shape, dtype, -1, 1)
      value /= jnp.linalg.norm(value.reshape((-1, shape[-1])), 2, 0)
    elif self.dist == 'complex':
      assert jnp.issubdtype(dtype, jnp.complexfloating), dtype
      realdt = jnp.finfo(dtype).dtype
      value = jax.random.truncated_normal(
          nj.seed(), -2, 2, (2, *shape), realdt)
      value = value[0] + 1j * value[1]
      value *= jax.lax.convert_element_type(1.137 * np.sqrt(1 / fan), realdt)
    elif self.dist == 'ortho':
      nrows, ncols = shape[-1], np.prod(shape) // shape[-1]
      matshape = (nrows, ncols) if nrows > ncols else (ncols, nrows)
      mat = jax.random.normal(nj.seed(), matshape, dtype)
      qmat, rmat = jnp.linalg.qr(mat)
      qmat *= jnp.sign(jnp.diag(rmat))
      qmat = qmat.T if nrows < ncols else qmat
      qmat = qmat.reshape(nrows, *shape[:-1])
      value = jnp.moveaxis(qmat, 0, -1)
    else:
      raise NotImplementedError(self.dist)
    value *= self.scale
    return value

  def _fans(self, shape):
    if len(shape) == 0:
      return (1, 1)
    elif len(shape) == 1:
      return (1, shape[0])
    elif len(shape) == 2:
      return shape
    elif len(shape) == 3 and self.block_fans:
      return shape[1:]
    else:
      space = int(np.prod(shape[:-2]))
      return (shape[-2] * space, shape[-1] * space)


def get_act(name):
  if callable(name):
    return name
  elif name == 'none':
    return lambda x: x
  elif name == 'cswiglu':
    def fn(x):
      x, y = jnp.split(x, 2, -1)
      y1, y2 = jnp.split(y, 2, -1)
      pad = jnp.ones_like(y1)
      x = jax.nn.swish(jnp.concatenate([x, -x], -1))
      y = jnp.concatenate([y1, pad, y2, pad], -1)
      return x * y
    return fn
  elif name == 'mish':
    return lambda x: x * jnp.tanh(jax.nn.softplus(x))
  elif hasattr(jax.nn, name):
    return getattr(jax.nn, name)
  else:
    raise NotImplementedError(name)
