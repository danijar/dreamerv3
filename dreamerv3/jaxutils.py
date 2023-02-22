import re

import jax
import jax.numpy as jnp
import numpy as np
import optax
from tensorflow_probability.substrates import jax as tfp

from . import ninjax as nj

tfd = tfp.distributions
tree_map = jax.tree_util.tree_map
sg = lambda x: tree_map(jax.lax.stop_gradient, x)
COMPUTE_DTYPE = jnp.float32


def cast_to_compute(values):
  return tree_map(lambda x: x.astype(COMPUTE_DTYPE), values)


def parallel():
  try:
    jax.lax.axis_index('i')
    return True
  except NameError:
    return False


def tensorstats(tensor, prefix=None):
  metrics = {
      'mean': tensor.mean(),
      'std': tensor.std(),
      'mag': jnp.abs(tensor).max(),
      'min': tensor.min(),
      'max': tensor.max(),
      'dist': subsample(tensor),
  }
  if prefix:
    metrics = {f'{prefix}_{k}': v for k, v in metrics.items()}
  return metrics


def subsample(values, amount=1024):
  values = values.flatten()
  if len(values) > amount:
    values = jax.random.permutation(nj.rng(), values)[:amount]
  return values


def scan(fn, inputs, start, unroll=True, modify=False):
  fn2 = lambda carry, inp: (fn(carry, inp),) * 2
  if not unroll:
    return nj.scan(fn2, start, inputs, modify=modify)[1]
  length = len(jax.tree_util.tree_leaves(inputs)[0])
  carrydef = jax.tree_util.tree_structure(start)
  carry = start
  outs = []
  for index in range(length):
    carry, out = fn2(carry, tree_map(lambda x: x[index], inputs))
    flat, treedef = jax.tree_util.tree_flatten(out)
    assert treedef == carrydef, (treedef, carrydef)
    outs.append(flat)
  outs = [
      jnp.stack([carry[i] for carry in outs], 0)
      for i in range(len(outs[0]))]
  return carrydef.unflatten(outs)


def symlog(x):
  return jnp.sign(x) * jnp.log(1 + jnp.abs(x))


def symexp(x):
  return jnp.sign(x) * (jnp.exp(jnp.abs(x)) - 1)


class OneHotDist(tfd.OneHotCategorical):

  def __init__(self, logits=None, probs=None, dtype=jnp.float32):
    super().__init__(logits, probs, dtype)

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
     return super()._parameter_properties(dtype)

  def sample(self, sample_shape=(), seed=None):
    sample = sg(super().sample(sample_shape, seed))
    probs = self._pad(super().probs_parameter(), sample.shape)
    return sg(sample) + (probs - sg(probs)).astype(sample.dtype)

  def _pad(self, tensor, shape):
    while len(tensor.shape) < len(shape):
      tensor = tensor[None]
    return tensor


class MSEDist:

  def __init__(self, mode, dims, agg='sum'):
    self._mode = mode
    self._dims = tuple([-x for x in range(1, dims + 1)])
    self._agg = agg
    self.batch_shape = mode.shape[:len(mode.shape) - dims]
    self.event_shape = mode.shape[len(mode.shape) - dims:]

  def mode(self):
    return self._mode

  def mean(self):
    return self._mode

  def log_prob(self, value):
    assert self._mode.shape == value.shape, (self._mode.shape, value.shape)
    distance = ((self._mode - value) ** 2)
    if self._agg == 'mean':
      loss = distance.mean(self._dims)
    elif self._agg == 'sum':
      loss = distance.sum(self._dims)
    else:
      raise NotImplementedError(self._agg)
    return -loss


class SymlogDist:

  def __init__(self, mode, dims, dist='mse', agg='sum', tol=1e-8):
    self._mode = mode
    self._dims = tuple([-x for x in range(1, dims + 1)])
    self._dist = dist
    self._agg = agg
    self._tol = tol
    self.batch_shape = mode.shape[:len(mode.shape) - dims]
    self.event_shape = mode.shape[len(mode.shape) - dims:]

  def mode(self):
    return symexp(self._mode)

  def mean(self):
    return symexp(self._mode)

  def log_prob(self, value):
    assert self._mode.shape == value.shape, (self._mode.shape, value.shape)
    if self._dist == 'mse':
      distance = (self._mode - symlog(value)) ** 2
      distance = jnp.where(distance < self._tol, 0, distance)
    elif self._dist == 'abs':
      distance = jnp.abs(self._mode - symlog(value))
      distance = jnp.where(distance < self._tol, 0, distance)
    else:
      raise NotImplementedError(self._dist)
    if self._agg == 'mean':
      loss = distance.mean(self._dims)
    elif self._agg == 'sum':
      loss = distance.sum(self._dims)
    else:
      raise NotImplementedError(self._agg)
    return -loss


class DiscDist:

  def __init__(
      self, logits, dims=0, low=-20, high=20,
      transfwd=symlog, transbwd=symexp):
    self.logits = logits
    self.probs = jax.nn.softmax(logits)
    self.dims = tuple([-x for x in range(1, dims + 1)])
    self.bins = jnp.linspace(low, high, logits.shape[-1])
    self.low = low
    self.high = high
    self.transfwd = transfwd
    self.transbwd = transbwd
    self.batch_shape = logits.shape[:len(logits.shape) - dims - 1]
    self.event_shape = logits.shape[len(logits.shape) - dims: -1]

  def mean(self):
    return self.transbwd((self.probs * self.bins).sum(-1))

  def mode(self):
    return self.transbwd((self.probs * self.bins).sum(-1))

  def log_prob(self, x):
    x = self.transfwd(x)
    below = (self.bins <= x[..., None]).astype(jnp.int32).sum(-1) - 1
    above = len(self.bins) - (
        self.bins > x[..., None]).astype(jnp.int32).sum(-1)
    below = jnp.clip(below, 0, len(self.bins) - 1)
    above = jnp.clip(above, 0, len(self.bins) - 1)
    equal = (below == above)
    dist_to_below = jnp.where(equal, 1, jnp.abs(self.bins[below] - x))
    dist_to_above = jnp.where(equal, 1, jnp.abs(self.bins[above] - x))
    total = dist_to_below + dist_to_above
    weight_below = dist_to_above / total
    weight_above = dist_to_below / total
    target = (
        jax.nn.one_hot(below, len(self.bins)) * weight_below[..., None] +
        jax.nn.one_hot(above, len(self.bins)) * weight_above[..., None])
    log_pred = self.logits - jax.scipy.special.logsumexp(
        self.logits, -1, keepdims=True)
    return (target * log_pred).sum(-1).sum(self.dims)


def video_grid(video):
  B, T, H, W, C = video.shape
  return video.transpose((1, 2, 0, 3, 4)).reshape((T, H, B * W, C))


def balance_stats(dist, target, thres):
  # Values are NaN when there are no positives or negatives in the current
  # batch, which means they will be ignored when aggregating metrics via
  # np.nanmean() later, as they should.
  pos = (target.astype(jnp.float32) > thres).astype(jnp.float32)
  neg = (target.astype(jnp.float32) <= thres).astype(jnp.float32)
  pred = (dist.mean().astype(jnp.float32) > thres).astype(jnp.float32)
  loss = -dist.log_prob(target)
  return dict(
      pos_loss=(loss * pos).sum() / pos.sum(),
      neg_loss=(loss * neg).sum() / neg.sum(),
      pos_acc=(pred * pos).sum() / pos.sum(),
      neg_acc=((1 - pred) * neg).sum() / neg.sum(),
      rate=pos.mean(),
      avg=target.astype(jnp.float32).mean(),
      pred=dist.mean().astype(jnp.float32).mean(),
  )


class Moments(nj.Module):

  def __init__(
      self, impl='mean_std', decay=0.99, max=1e8, eps=0.0, perclo=5,
      perchi=95):
    self.impl = impl
    self.decay = decay
    self.max = max
    self.eps = eps
    self.perclo = perclo
    self.perchi = perchi
    if self.impl == 'off':
      pass
    elif self.impl == 'mean_std':
      self.step = nj.Variable(jnp.zeros, (), jnp.int32, name='step')
      self.mean = nj.Variable(jnp.zeros, (), jnp.float32, name='mean')
      self.sqrs = nj.Variable(jnp.zeros, (), jnp.float32, name='sqrs')
    elif self.impl == 'min_max':
      self.low = nj.Variable(jnp.zeros, (), jnp.float32, name='low')
      self.high = nj.Variable(jnp.zeros, (), jnp.float32, name='high')
    elif self.impl == 'perc_ema':
      self.low = nj.Variable(jnp.zeros, (), jnp.float32, name='low')
      self.high = nj.Variable(jnp.zeros, (), jnp.float32, name='high')
    elif self.impl == 'perc_ema_corr':
      self.step = nj.Variable(jnp.zeros, (), jnp.int32, name='step')
      self.low = nj.Variable(jnp.zeros, (), jnp.float32, name='low')
      self.high = nj.Variable(jnp.zeros, (), jnp.float32, name='high')
    elif self.impl == 'mean_mag':
      self.mag = nj.Variable(jnp.zeros, (), jnp.float32, name='mag')
    elif self.impl == 'max_mag':
      self.mag = nj.Variable(jnp.zeros, (), jnp.float32, name='mag')
    else:
      raise NotImplementedError(self.impl)

  def __call__(self, x):
    self.update(x)
    return self.stats()

  def update(self, x):
    if parallel():
      mean = lambda x: jax.lax.pmean(x.mean(), 'i')
      min_ = lambda x: jax.lax.pmin(x.min(), 'i')
      max_ = lambda x: jax.lax.pmax(x.max(), 'i')
      per = lambda x, q: jnp.percentile(jax.lax.all_gather(x, 'i'), q)
    else:
      mean = jnp.mean
      min_ = jnp.min
      max_ = jnp.max
      per = jnp.percentile
    x = sg(x.astype(jnp.float32))
    m = self.decay
    if self.impl == 'off':
      pass
    elif self.impl == 'mean_std':
      self.step.write(self.step.read() + 1)
      self.mean.write(m * self.mean.read() + (1 - m) * mean(x))
      self.sqrs.write(m * self.sqrs.read() + (1 - m) * mean(x * x))
    elif self.impl == 'min_max':
      low, high = min_(x), max_(x)
      self.low.write(m * jnp.minimum(self.low.read(), low) + (1 - m) * low)
      self.high.write(m * jnp.maximum(self.high.read(), high) + (1 - m) * high)
    elif self.impl == 'perc_ema':
      low, high = per(x, self.perclo), per(x, self.perchi)
      self.low.write(m * self.low.read() + (1 - m) * low)
      self.high.write(m * self.high.read() + (1 - m) * high)
    elif self.impl == 'perc_ema_corr':
      self.step.write(self.step.read() + 1)
      low, high = per(x, self.perclo), per(x, self.perchi)
      self.low.write(m * self.low.read() + (1 - m) * low)
      self.high.write(m * self.high.read() + (1 - m) * high)
    elif self.impl == 'mean_mag':
      curr = mean(jnp.abs(x))
      self.mag.write(m * self.mag.read() + (1 - m) * curr)
    elif self.impl == 'max_mag':
      curr = max_(jnp.abs(x))
      self.mag.write(m * jnp.maximum(self.mag.read(), curr) + (1 - m) * curr)
    else:
      raise NotImplementedError(self.impl)

  def stats(self):
    if self.impl == 'off':
      return 0.0, 1.0
    elif self.impl == 'mean_std':
      corr = 1 - self.decay ** self.step.read().astype(jnp.float32)
      mean = self.mean.read() / corr
      var = (self.sqrs.read() / corr) - self.mean.read() ** 2
      std = jnp.sqrt(jnp.maximum(var, 1 / self.max ** 2) + self.eps)
      return sg(mean), sg(std)
    elif self.impl == 'min_max':
      offset = self.low.read()
      invscale = jnp.maximum(1 / self.max, self.high.read() - self.low.read())
      return sg(offset), sg(invscale)
    elif self.impl == 'perc_ema':
      offset = self.low.read()
      invscale = jnp.maximum(1 / self.max, self.high.read() - self.low.read())
      return sg(offset), sg(invscale)
    elif self.impl == 'perc_ema_corr':
      corr = 1 - self.decay ** self.step.read().astype(jnp.float32)
      lo = self.low.read() / corr
      hi = self.high.read() / corr
      invscale = jnp.maximum(1 / self.max, hi - lo)
      return sg(lo), sg(invscale)
    elif self.impl == 'mean_mag':
      offset = jnp.array(0)
      invscale = jnp.maximum(1 / self.max, self.mag.read())
      return sg(offset), sg(invscale)
    elif self.impl == 'max_mag':
      offset = jnp.array(0)
      invscale = jnp.maximum(1 / self.max, self.mag.read())
      return sg(offset), sg(invscale)
    else:
      raise NotImplementedError(self.impl)


class Optimizer(nj.Module):

  PARAM_COUNTS = {}

  def __init__(
      self, lr, opt='adam', eps=1e-5, clip=100.0, warmup=0, wd=0.0,
      wd_pattern=r'/(w|kernel)$', lateclip=0.0):
    assert opt in ('adam', 'belief', 'yogi')
    assert wd_pattern[0] not in ('0', '1')
    # assert self.path not in self.PARAM_COUNTS
    self.PARAM_COUNTS[self.path] = None
    wd_pattern = re.compile(wd_pattern)
    chain = []
    if clip:
      chain.append(optax.clip_by_global_norm(clip))
    if opt == 'adam':
      chain.append(optax.scale_by_adam(eps=eps))
    else:
      raise NotImplementedError(opt)
    if lateclip:
      chain.append(late_grad_clip(lateclip))
    if wd:
      chain.append(optax.additive_weight_decay(wd, lambda params: (
          tree_map(lambda k: bool(wd_pattern.search(k)), tree_keys(params)))))
    if warmup:
      schedule = optax.linear_schedule(0.0, -lr, warmup)
      chain.append(optax.inject_hyperparams(optax.scale)(schedule))
    else:
      chain.append(optax.scale(-lr))
    self.opt = optax.chain(*chain)
    self.step = nj.Variable(jnp.array, 0, jnp.int32, name='step')
    self.scaling = (COMPUTE_DTYPE == jnp.float16)
    if self.scaling:
      self.opt = optax.apply_if_finite(self.opt, max_consecutive_errors=1000)
      self.grad_scale = nj.Variable(
          jnp.array, 1e4, jnp.float32, name='grad_scale')
      self.good_steps = nj.Variable(
          jnp.array, 0, jnp.int32, name='good_steps')

  def __call__(self, modules, lossfn, *args, has_aux=False, **kwargs):
    def wrapped(*args, **kwargs):
      outs = lossfn(*args, **kwargs)
      loss, aux = outs if has_aux else (outs, None)
      assert loss.dtype == jnp.float32, (self.name, loss.dtype)
      assert loss.shape == (), (self.name, loss.shape)
      if self.scaling:
        loss *= sg(self.grad_scale.read())
      return loss, aux
    metrics = {}
    loss, params, grads, aux = nj.grad(
        wrapped, modules, has_aux=True)(*args, **kwargs)
    if not self.PARAM_COUNTS[self.path]:
      count = sum([np.prod(x.shape) for x in params.values()])
      print(f'Optimizer {self.name} has {count:,} variables.')
      self.PARAM_COUNTS[self.path] = count
    if parallel():
      grads = tree_map(lambda x: jax.lax.pmean(x, 'i'), grads)
    if self.scaling:
      grads = tree_map(lambda x: x / self.grad_scale.read(), grads)
      finite = self._update_scale(grads)
      metrics[f'{self.name}_grad_scale'] = self.grad_scale.read()
      metrics[f'{self.name}_grad_overflow'] = (~finite).astype(jnp.float32)
    optstate = self.get('state', self.opt.init, params)
    updates, optstate = self.opt.update(grads, optstate, params)
    self.put('state', optstate)
    nj.context().update(optax.apply_updates(params, updates))
    norm = optax.global_norm(grads)
    if self.scaling:
      norm = jnp.where(jnp.isfinite(norm), norm, jnp.nan)
    self.step.write(self.step.read() + jnp.isfinite(norm).astype(jnp.int32))
    metrics['loss'] = loss.mean()
    metrics['grad_norm'] = norm
    metrics['grad_steps'] = self.step.read()
    metrics = {f'{self.name}_{k}': v for k, v in metrics.items()}
    return (metrics, aux) if has_aux else metrics

  def _update_scale(self, grads):
    finite = jnp.array([
        jnp.isfinite(x).all() for x in jax.tree_util.tree_leaves(grads)]).all()
    keep = (finite & (self.good_steps.read() < 1000))
    incr = (finite & (self.good_steps.read() >= 1000))
    decr = ~finite
    self.good_steps.write(
        keep.astype(jnp.int32) * (self.good_steps.read() + 1))
    self.grad_scale.write(jnp.clip(
        keep.astype(jnp.float32) * self.grad_scale.read() +
        incr.astype(jnp.float32) * self.grad_scale.read() * 2 +
        decr.astype(jnp.float32) * self.grad_scale.read() / 2,
        1e-4, 1e4))
    return finite


def late_grad_clip(value=1.0):
  def init_fn(params):
    return ()
  def update_fn(updates, state, params):
    updates = tree_map(lambda x: jnp.clip(x, -value, value), updates)
    return updates, ()
  return optax.GradientTransformation(init_fn, update_fn)


def tree_keys(params, prefix=''):
  if hasattr(params, 'items'):
    return type(params)({
        k: tree_keys(v, prefix + '/' + k.lstrip('/'))
        for k, v in params.items()})
  elif isinstance(params, (tuple, list)):
    return [tree_keys(x, prefix) for x in params]
  elif isinstance(params, jnp.ndarray):
    return prefix
  else:
    raise TypeError(type(params))


class SlowUpdater:

  def __init__(self, src, dst, fraction=1.0, period=1):
    self.src = src
    self.dst = dst
    self.fraction = fraction
    self.period = period
    self.updates = nj.Variable(jnp.zeros, (), jnp.int32, name='updates')

  def __call__(self):
    assert self.src.getm()
    updates = self.updates.read()
    need_init = (updates == 0).astype(jnp.float32)
    need_update = (updates % self.period == 0).astype(jnp.float32)
    mix = jnp.clip(1.0 * need_init + self.fraction * need_update, 0, 1)
    source = {
        k.replace(f'/{self.src.name}/', f'/{self.dst.name}/'): v
        for k, v in self.src.getm().items()}
    self.dst.putm(tree_map(
        lambda s, d: mix * s + (1 - mix) * d,
        source, self.dst.getm()))
    self.updates.write(updates + 1)
