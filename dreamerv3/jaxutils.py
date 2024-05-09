import collections
import re

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.experimental import checkify
from tensorflow_probability.substrates import jax as tfp

from . import ninjax as nj

tfd = tfp.distributions
tfb = tfp.bijectors
treemap = jax.tree_util.tree_map
sg = lambda x: treemap(jax.lax.stop_gradient, x)
f32 = jnp.float32
i32 = jnp.int32
COMPUTE_DTYPE = f32
PARAM_DTYPE = f32
ENABLE_CHECKS = False


def cast_to_compute(values):
  return treemap(
      lambda x: x if x.dtype == COMPUTE_DTYPE else x.astype(COMPUTE_DTYPE),
      values)


def get_param_dtype():
  return PARAM_DTYPE


def check(predicate, message, **kwargs):
  if ENABLE_CHECKS:
    checkify.check(predicate, message, **kwargs)


def parallel():
  try:
    jax.lax.axis_index('i')
    return True
  except NameError:
    return False


def scan(fun, carry, xs, unroll=False, axis=0):
  unroll = jax.tree_util.tree_leaves(xs)[0].shape[axis] if unroll else 1
  return nj.scan(fun, carry, xs, False, unroll, axis)


def tensorstats(tensor, prefix=None):
  assert tensor.size > 0, tensor.shape
  assert jnp.issubdtype(tensor.dtype, jnp.floating), tensor.dtype
  tensor = tensor.astype(f32)  # To avoid overflows.
  metrics = {
      'mean': tensor.mean(),
      'std': tensor.std(),
      'mag': jnp.abs(tensor).mean(),
      'min': tensor.min(),
      'max': tensor.max(),
      'dist': subsample(tensor),
  }
  if prefix:
    metrics = {f'{prefix}/{k}': v for k, v in metrics.items()}
  return metrics


def subsample(values, amount=1024):
  values = values.flatten()
  if len(values) > amount:
    values = jax.random.permutation(nj.seed(), values)[:amount]
  return values


def symlog(x):
  return jnp.sign(x) * jnp.log1p(jnp.abs(x))


def symexp(x):
  return jnp.sign(x) * jnp.expm1(jnp.abs(x))


def switch(pred, lhs, rhs):
  def fn(lhs, rhs):
    assert lhs.shape == rhs.shape, (pred.shape, lhs.shape, rhs.shape)
    mask = pred
    while len(mask.shape) < len(lhs.shape):
      mask = mask[..., None]
    return jnp.where(mask, lhs, rhs)
  return treemap(fn, lhs, rhs)


def reset(xs, reset):
  def fn(x):
    mask = reset
    while len(mask.shape) < len(x.shape):
      mask = mask[..., None]
    return x * (1 - mask.astype(x.dtype))
  return treemap(fn, xs)


class OneHotDist(tfd.OneHotCategorical):

  def __init__(self, logits=None, probs=None, dtype=f32):
    super().__init__(logits, probs, dtype)

  @classmethod
  def _parameter_properties(cls, dtype, num_classes=None):
     return super()._parameter_properties(dtype)

  def sample(self, sample_shape=(), seed=None):
    sample = sg(super().sample(sample_shape, seed))
    probs = self._pad(super().probs_parameter(), sample.shape)
    sample = sg(sample) + (probs - sg(probs)).astype(sample.dtype)
    return sample

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


class HuberDist:

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
    distance = jnp.sqrt(1 + distance) - 1
    if self._agg == 'mean':
      loss = distance.mean(self._dims)
    elif self._agg == 'sum':
      loss = distance.sum(self._dims)
    else:
      raise NotImplementedError(self._agg)
    return -loss


class TransformedMseDist:

  def __init__(self, mode, dims, fwd, bwd, agg='sum', tol=1e-8):
    self._mode = mode
    self._dims = tuple([-x for x in range(1, dims + 1)])
    self._fwd = fwd
    self._bwd = bwd
    self._agg = agg
    self._tol = tol
    self.batch_shape = mode.shape[:len(mode.shape) - dims]
    self.event_shape = mode.shape[len(mode.shape) - dims:]

  def mode(self):
    return self._bwd(self._mode)

  def mean(self):
    return self._bwd(self._mode)

  def log_prob(self, value):
    assert self._mode.shape == value.shape, (self._mode.shape, value.shape)
    distance = (self._mode - self._fwd(value)) ** 2
    distance = jnp.where(distance < self._tol, 0, distance)
    if self._agg == 'mean':
      loss = distance.mean(self._dims)
    elif self._agg == 'sum':
      loss = distance.sum(self._dims)
    else:
      raise NotImplementedError(self._agg)
    return -loss


class TwoHotDist:

  def __init__(
      self, logits, bins, dims=0, transfwd=None, transbwd=None):
    assert logits.shape[-1] == len(bins), (logits.shape, len(bins))
    assert logits.dtype == f32, logits.dtype
    assert bins.dtype == f32, bins.dtype
    self.logits = logits
    self.probs = jax.nn.softmax(logits)
    self.dims = tuple([-x for x in range(1, dims + 1)])
    self.bins = jnp.array(bins)
    self.transfwd = transfwd or (lambda x: x)
    self.transbwd = transbwd or (lambda x: x)
    self.batch_shape = logits.shape[:len(logits.shape) - dims - 1]
    self.event_shape = logits.shape[len(logits.shape) - dims: -1]

  def mean(self):
    # The naive implementation results in a non-zero result even if the bins
    # are symmetric and the probabilities uniform, because the sum operation
    # goes left to right, accumulating numerical errors. Instead, we use a
    # symmetric sum to ensure that the predicted rewards and values are
    # actually zero at initialization.
    # return self.transbwd((self.probs * self.bins).sum(-1))
    n = self.logits.shape[-1]
    if n % 2 == 1:
      m = (n - 1) // 2
      p1 = self.probs[..., :m]
      p2 = self.probs[..., m: m + 1]
      p3 = self.probs[..., m + 1:]
      b1 = self.bins[..., :m]
      b2 = self.bins[..., m: m + 1]
      b3 = self.bins[..., m + 1:]
      wavg = (p2 * b2).sum(-1) + ((p1 * b1)[..., ::-1] + (p3 * b3)).sum(-1)
      return self.transbwd(wavg)
    else:
      p1 = self.probs[..., :n // 2]
      p2 = self.probs[..., n // 2:]
      b1 = self.bins[..., :n // 2]
      b2 = self.bins[..., n // 2:]
      wavg = ((p1 * b1)[..., ::-1] + (p2 * b2)).sum(-1)
      return self.transbwd(wavg)

  def mode(self):
    return self.transbwd((self.probs * self.bins).sum(-1))

  def log_prob(self, x):
    assert x.dtype == f32, x.dtype
    x = self.transfwd(x)
    below = (self.bins <= x[..., None]).astype(i32).sum(-1) - 1
    above = len(self.bins) - (
        self.bins > x[..., None]).astype(i32).sum(-1)
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
  pos = (target.astype(f32) > thres).astype(f32)
  neg = (target.astype(f32) <= thres).astype(f32)
  pred = (dist.mean().astype(f32) > thres).astype(f32)
  loss = -dist.log_prob(target)
  return dict(
      pos_loss=(loss * pos).sum() / pos.sum(),
      neg_loss=(loss * neg).sum() / neg.sum(),
      pos_acc=(pred * pos).sum() / pos.sum(),
      neg_acc=((1 - pred) * neg).sum() / neg.sum(),
      rate=pos.mean(),
      avg=target.astype(f32).mean(),
      pred=dist.mean().astype(f32).mean(),
  )


class Moments(nj.Module):

  rate: float = 0.01
  limit: float = 1e-8
  perclo: float = 5.0
  perchi: float = 95.0

  def __init__(self, impl='mean_std'):
    self.impl = impl
    if self.impl == 'off':
      pass
    elif self.impl == 'mean_std':
      self.mean = nj.Variable(jnp.zeros, (), f32, name='mean')
      self.sqrs = nj.Variable(jnp.zeros, (), f32, name='sqrs')
      self.corr = nj.Variable(jnp.zeros, (), f32, name='corr')
    elif self.impl == 'min_max':
      self.low = nj.Variable(jnp.zeros, (), f32, name='low')
      self.high = nj.Variable(jnp.zeros, (), f32, name='high')
    elif self.impl == 'perc':
      self.low = nj.Variable(jnp.zeros, (), f32, name='low')
      self.high = nj.Variable(jnp.zeros, (), f32, name='high')
    elif self.impl == 'perc_corr':
      self.low = nj.Variable(jnp.zeros, (), f32, name='low')
      self.high = nj.Variable(jnp.zeros, (), f32, name='high')
      self.corr = nj.Variable(jnp.zeros, (), f32, name='corr')
    else:
      raise NotImplementedError(self.impl)

  def __call__(self, x, update=True):
    update and self.update(x)
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
    x = sg(x.astype(f32))
    m = self.rate
    if self.impl == 'off':
      pass
    elif self.impl == 'mean_std':
      self.mean.write((1 - m) * self.mean.read() + m * mean(x))
      self.sqrs.write((1 - m) * self.sqrs.read() + m * mean(x * x))
      self.corr.write((1 - m) * self.corr.read() + m * 1.0)
    elif self.impl == 'min_max':
      low, high = min_(x), max_(x)
      self.low.write((1 - m) * jnp.minimum(self.low.read(), low) + m * low)
      self.high.write((1 - m) * jnp.maximum(self.high.read(), high) + m * high)
    elif self.impl == 'perc':
      low, high = per(x, self.perclo), per(x, self.perchi)
      self.low.write((1 - m) * self.low.read() + m * low)
      self.high.write((1 - m) * self.high.read() + m * high)
    elif self.impl == 'perc_corr':
      low, high = per(x, self.perclo), per(x, self.perchi)
      self.low.write((1 - m) * self.low.read() + m * low)
      self.high.write((1 - m) * self.high.read() + m * high)
      self.corr.write((1 - m) * self.corr.read() + m * 1.0)
    else:
      raise NotImplementedError(self.impl)

  def stats(self):
    if self.impl == 'off':
      return 0.0, 1.0
    elif self.impl == 'mean_std':
      corr = jnp.maximum(self.rate, self.corr.read())
      mean = self.mean.read() / corr
      std = jnp.sqrt(jax.nn.relu(self.sqrs.read() / corr - mean ** 2))
      std = jnp.maximum(self.limit, std)
      return sg(mean), sg(std)
    elif self.impl == 'min_max':
      offset = self.low.read()
      span = self.high.read() - self.low.read()
      span = jnp.maximum(self.limit, span)
      return sg(offset), sg(span)
    elif self.impl == 'perc':
      offset = self.low.read()
      span = self.high.read() - self.low.read()
      span = jnp.maximum(self.limit, span)
      return sg(offset), sg(span)
    elif self.impl == 'perc_corr':
      corr = jnp.maximum(self.rate, self.corr.read())
      lo = self.low.read() / corr
      hi = self.high.read() / corr
      span = hi - lo
      span = jnp.maximum(self.limit, span)
      return sg(lo), sg(span)
    else:
      raise NotImplementedError(self.impl)


class Optimizer(nj.Module):

  # Normalization
  scaler: str = 'adam'
  eps: float = 1e-7
  beta1: float = 0.9
  beta2: float = 0.999

  # Learning rate
  warmup: int = 1000
  anneal: int = 0
  schedule: str = 'constant'

  # Regularization
  wd: float = 0.0
  wd_pattern: str = r'/kernel$'

  # Clipping
  pmin: float = 1e-3
  globclip: float = 0.0
  agc: float = 0.0

  # Smoothing
  momentum: bool = False
  nesterov: bool = False

  # Metrics
  details: bool = False

  def __init__(self, lr):
    self.lr = lr
    chain = []

    if self.globclip:
      chain.append(optax.clip_by_global_norm(self.globclip))
    if self.agc:
      chain.append(scale_by_agc(self.agc, self.pmin))

    if self.scaler == 'adam':
      chain.append(optax.scale_by_adam(self.beta1, self.beta2, self.eps))
    elif self.scaler == 'rms':
      chain.append(scale_by_rms(self.beta2, self.eps))
    else:
      raise NotImplementedError(self.scaler)

    if self.momentum:
      chain.append(scale_by_momentum(self.beta1, self.nesterov))

    if self.wd:
      assert not self.wd_pattern[0].isnumeric(), self.wd_pattern
      pattern = re.compile(self.wd_pattern)
      wdmaskfn = lambda params: {k: bool(pattern.search(k)) for k in params}
      chain.append(optax.add_decayed_weights(self.wd, wdmaskfn))

    if isinstance(self.lr, dict):
      chain.append(scale_by_groups({pfx: -lr for pfx, lr in self.lr.items()}))
    else:
      chain.append(optax.scale(-self.lr))

    self.chain = optax.chain(*chain)
    self.step = nj.Variable(jnp.array, 0, i32, name='step')
    self.scaling = (COMPUTE_DTYPE == jnp.float16)
    if self.scaling:
      self.chain = optax.apply_if_finite(
          self.chain, max_consecutive_errors=1000)
      self.grad_scale = nj.Variable(jnp.array, 1e4, f32, name='grad_scale')
      self.good_steps = nj.Variable(jnp.array, 0, i32, name='good_steps')
    self.once = True

  def __call__(self, modules, lossfn, *args, has_aux=False, **kwargs):
    def wrapped(*args, **kwargs):
      outs = lossfn(*args, **kwargs)
      loss, aux = outs if has_aux else (outs, None)
      assert loss.dtype == f32, (self.name, loss.dtype)
      assert loss.shape == (), (self.name, loss.shape)
      if self.scaling:
        loss *= sg(self.grad_scale.read())
      return loss, aux

    metrics = {}
    loss, params, grads, aux = nj.grad(
        wrapped, modules, has_aux=True)(*args, **kwargs)
    if self.scaling:
      loss /= self.grad_scale.read()
    if not isinstance(modules, (list, tuple)):
      modules = [modules]
    counts = {k: int(np.prod(v.shape)) for k, v in params.items()}
    if self.once:
      self.once = False
      prefs = []
      for key in counts:
        parts = key.split('/')
        prefs += ['/'.join(parts[: i + 1]) for i in range(min(len(parts), 2))]
      subcounts = {
          prefix: sum(v for k, v in counts.items() if k.startswith(prefix))
          for prefix in set(prefs)}
      print(f'Optimizer {self.name} has {sum(counts.values()):,} variables:')
      for prefix, count in sorted(subcounts.items(), key=lambda x: -x[1]):
        print(f'{count:>14,} {prefix}')

    if parallel():
      grads = treemap(lambda x: jax.lax.pmean(x, 'i'), grads)
    if self.scaling:
      invscale = 1.0 / self.grad_scale.read()
      grads = treemap(lambda x: x * invscale, grads)
    optstate = self.get('state', self.chain.init, params)
    updates, optstate = self.chain.update(grads, optstate, params)
    self.put('state', optstate)

    if self.details:
      metrics.update(self._detailed_stats(optstate, params, updates, grads))

    scale = 1
    step = self.step.read().astype(f32)
    if self.warmup > 0:
      scale *= jnp.clip(step / self.warmup, 0, 1)
    assert self.schedule == 'constant' or self.anneal > self.warmup
    prog = jnp.clip((step - self.warmup) / (self.anneal - self.warmup), 0, 1)
    if self.schedule == 'constant':
      pass
    elif self.schedule == 'linear':
      scale *= 1 - prog
    elif self.schedule == 'cosine':
      scale *= 0.5 * (1 + jnp.cos(jnp.pi * prog))
    else:
      raise NotImplementedError(self.schedule)
    updates = treemap(lambda x: x * scale, updates)

    nj.context().update(optax.apply_updates(params, updates))
    grad_norm = optax.global_norm(grads)
    update_norm = optax.global_norm(updates)
    param_norm = optax.global_norm([x.find() for x in modules])
    isfin = jnp.isfinite
    if self.scaling:
      self._update_scale(grads, jnp.isfinite(grad_norm))
      metrics['grad_scale'] = self.grad_scale.read()
      metrics['grad_overflow'] = (~jnp.isfinite(grad_norm)).astype(f32)
      grad_norm = jnp.where(jnp.isfinite(grad_norm), grad_norm, jnp.nan)
      self.step.write(self.step.read() + isfin(grad_norm).astype(i32))
    else:
      check(isfin(grad_norm), f'{self.path} grad norm: {{x}}', x=grad_norm)
      self.step.write(self.step.read() + 1)
    check(isfin(update_norm), f'{self.path} updates: {{x}}', x=update_norm)
    check(isfin(param_norm), f'{self.path} params: {{x}}', x=param_norm)

    metrics['loss'] = loss.mean()
    metrics['grad_norm'] = grad_norm
    metrics['update_norm'] = update_norm
    metrics['param_norm'] = param_norm
    metrics['grad_steps'] = self.step.read()
    metrics['param_count'] = jnp.array(sum(counts.values()))
    metrics = {f'{self.name}_{k}': v for k, v in metrics.items()}
    return (metrics, aux) if has_aux else metrics

  def _update_scale(self, grads, finite):
    keep = (finite & (self.good_steps.read() < 1000))
    incr = (finite & (self.good_steps.read() >= 1000))
    decr = ~finite
    self.good_steps.write(
        keep.astype(i32) * (self.good_steps.read() + 1))
    self.grad_scale.write(jnp.clip(
        keep.astype(f32) * self.grad_scale.read() +
        incr.astype(f32) * self.grad_scale.read() * 2 +
        decr.astype(f32) * self.grad_scale.read() / 2,
        1e-4, 1e5))
    return finite

  def _detailed_stats(self, optstate, params, updates, grads):
    groups = {
        'all': r'.*',
        'enc': r'/enc/.*',
        'dec': r'/dec/.*',
        'dyn': r'/dyn/.*',
        'con': r'/con/.*',
        'rew': r'/rew/.*',
        'actor': r'/actor/.*',
        'critic': r'/critic/.*',
        'out': r'/out/kernel$',
        'repr': r'/repr_logit/kernel$',
        'prior': r'/prior_logit/kernel$',
        'offset': r'/offset$',
        'scale': r'/scale$',
    }
    metrics = {}
    stddev = None
    for state in getattr(optstate, 'inner_state', optstate):
      if isinstance(state, optax.ScaleByAdamState):
        corr = 1 / (1 - 0.999 ** state.count)
        stddev = treemap(lambda x: jnp.sqrt(x * corr), state.nu)
    for name, pattern in groups.items():
      keys = [k for k in params if re.search(pattern, k)]
      ps = [params[k] for k in keys]
      us = [updates[k] for k in keys]
      gs = [grads[k] for k in keys]
      if not ps:
        continue
      metrics.update({f'{k}/{name}': v for k, v in dict(
          param_count=jnp.array(np.sum([np.prod(x.shape) for x in ps])),
          param_abs_max=jnp.stack([jnp.abs(x).max() for x in ps]).max(),
          param_abs_mean=jnp.stack([jnp.abs(x).mean() for x in ps]).mean(),
          param_norm=optax.global_norm(ps),
          update_abs_max=jnp.stack([jnp.abs(x).max() for x in us]).max(),
          update_abs_mean=jnp.stack([jnp.abs(x).mean() for x in us]).mean(),
          update_norm=optax.global_norm(us),
          grad_norm=optax.global_norm(gs),
      ).items()})
      if stddev is not None:
        sc = [stddev[k] for k in keys]
        pr = [
            jnp.abs(x) / jnp.maximum(1e-3, jnp.abs(y)) for x, y in zip(us, ps)]
        metrics.update({f'{k}/{name}': v for k, v in dict(
            scale_abs_max=jnp.stack([jnp.abs(x).max() for x in sc]).max(),
            scale_abs_min=jnp.stack([jnp.abs(x).min() for x in sc]).min(),
            scale_abs_mean=jnp.stack([jnp.abs(x).mean() for x in sc]).mean(),
            prop_max=jnp.stack([x.max() for x in pr]).max(),
            prop_min=jnp.stack([x.min() for x in pr]).min(),
            prop_mean=jnp.stack([x.mean() for x in pr]).mean(),
        ).items()})
    return metrics


def expand_groups(groups, keys):
  if isinstance(groups, (float, int)):
    return {key: groups for key in keys}
  groups = {
      group if group.endswith('/') else f'{group}/': value
      for group, value in groups.items()}
  assignment = {}
  groupcount = collections.defaultdict(int)
  for key in keys:
    matches = [prefix for prefix in groups if key.startswith(prefix)]
    if not matches:
      raise ValueError(
          f'Parameter {key} not fall into any of the groups:\n' +
          ''.join(f'- {group}\n' for group in groups.keys()))
    if len(matches) > 1:
      raise ValueError(
          f'Parameter {key} fall into more than one of the groups:\n' +
          ''.join(f'- {group}\n' for group in groups.keys()))
    assignment[key] = matches[0]
    groupcount[matches[0]] += 1
  for group in groups.keys():
    if not groupcount[group]:
      raise ValueError(
          f'Group {group} did not match any of the {len(keys)} keys.')
  expanded = {key: groups[assignment[key]] for key in keys}
  return expanded


def scale_by_groups(groups):

  def init_fn(params):
    return ()

  def update_fn(updates, state, params=None):
    scales = expand_groups(groups, updates.keys())
    updates = treemap(lambda u, s: u * s, updates, scales)
    return updates, state

  return optax.GradientTransformation(init_fn, update_fn)


def scale_by_agc(clip=0.03, pmin=1e-3):

  def init_fn(params):
    return ()

  def update_fn(updates, state, params=None):
    def fn(param, update):
      unorm = jnp.linalg.norm(update.flatten(), 2)
      pnorm = jnp.linalg.norm(param.flatten(), 2)
      upper = clip * jnp.maximum(pmin, pnorm)
      return update * (1 / jnp.maximum(1.0, unorm / upper))
    updates = treemap(fn, params, updates)
    return updates, ()

  return optax.GradientTransformation(init_fn, update_fn)


def scale_by_rms(beta=0.999, eps=1e-8):

  def init_fn(params):
    nu = treemap(lambda t: jnp.zeros_like(t, f32), params)
    step = jnp.zeros((), i32)
    return (step, nu)

  def update_fn(updates, state, params=None):
    step, nu = state
    step = optax.safe_int32_increment(step)
    nu = treemap(lambda v, u: beta * v + (1 - beta) * (u * u), nu, updates)
    nu_hat = optax.bias_correction(nu, beta, step)
    updates = treemap(lambda u, v: u / (jnp.sqrt(v) + eps), updates, nu_hat)
    return updates, (step, nu)

  return optax.GradientTransformation(init_fn, update_fn)


def scale_by_momentum(beta=0.9, nesterov=False):

  def init_fn(params):
    mu = treemap(lambda t: jnp.zeros_like(t, f32), params)
    step = jnp.zeros((), i32)
    return (step, mu)

  def update_fn(updates, state, params=None):
    step, mu = state
    step = optax.safe_int32_increment(step)
    mu = optax.update_moment(updates, mu, beta, 1)
    if nesterov:
      mu_nesterov = optax.update_moment(updates, mu, beta, 1)
      mu_hat = optax.bias_correction(mu_nesterov, beta, step)
    else:
      mu_hat = optax.bias_correction(mu, beta, step)
    return mu_hat, (step, mu)

  return optax.GradientTransformation(init_fn, update_fn)


def concat_dict(mapping, batch_shape=None):
  tensors = [v for _, v in sorted(mapping.items(), key=lambda x: x[0])]
  if batch_shape is not None:
    tensors = [x.reshape((*batch_shape, -1)) for x in tensors]
  return jnp.concatenate(tensors, -1)


def onehot_dict(mapping, spaces, filter=False, limit=256):
  result = {}
  for key, value in mapping.items():
    if key not in spaces and filter:
      continue
    space = spaces[key]
    if space.discrete and space.dtype != jnp.uint8:
      if limit:
        assert space.classes <= limit, (key, space, limit)
      value = jax.nn.one_hot(value, space.classes)
    result[key] = value
  return result


class SlowUpdater(nj.Module):

  def __init__(self, src, dst, fraction=1.0, period=1):
    self.src = src
    self.dst = dst
    self.fraction = fraction
    self.period = period
    self.updates = nj.Variable(jnp.zeros, (), i32, name='updates')

  def __call__(self):
    assert self.src.find()
    updates = self.updates.read()
    need_init = (updates == 0).astype(f32)
    need_update = (updates % self.period == 0).astype(f32)
    mix = jnp.clip(1.0 * need_init + self.fraction * need_update, 0, 1)
    params = {
        k.replace(f'/{self.src.name}/', f'/{self.dst.name}/'): v
        for k, v in self.src.find().items()}
    ema = treemap(
        lambda s, d: mix * s + (1 - mix) * d,
        params, self.dst.find())
    for name, param in ema.items():
      assert param.dtype == jnp.float32, (
          f'EMA of {name} should be float32 not {param.dtype}')
    self.dst.put(ema)
    self.updates.write(updates + 1)
