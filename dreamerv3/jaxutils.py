import re

import jax
import jax.numpy as jnp
import numpy as np
import optax
from tensorflow_probability.substrates import jax as tfp

from . import ninjax as nj

tfd = tfp.distributions
tree_map = jax.tree_util.tree_map # This is a function from the jax.tree_util module that applies a given function to each element in a nested structure (such as lists, tuples, dictionaries, etc.) in a recursive manner
sg = lambda x: tree_map(jax.lax.stop_gradient, x)
COMPUTE_DTYPE = jnp.float32


def cast_to_compute(values):
  """ 
   change the data type of all the elements within a nested structure (values) to a specific data type used for computation,
   here the type is COMPUTE_DTYPE which is jnp.float32
  """
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
  """ Two usages:
  1-iter though the inputs trajectory, and get the posterior and prior dict for all timesteps (containing stacked z_1:T, h_1:T,...) (prior + post)  
  2-iter though the inputs trajectory, and get the prior dict for all timesteps (containing stacked z_1:T, h_1:T,...) (prior only)


  Args:
      fn (function): obs_step() or img_step() 
      inputs (tuple): swap(action), swap(embed), swap(is_first) , each SHAPE:(T,B,.)
      start (tuple): initial state TUPLE(post_dict, prior_dict) in WM stage, here post_dict and prior_dict are the same when init \n or initial state single(prior_dict) in Actor/Critic (imagination) stage \n each item in dict is of shape: (B,.)
      unroll (bool, optional): _description_. Defaults to True.
      modify (bool, optional): _description_. Defaults to False.

  Returns:
      tuple: (post_dict, prior_dict) or (prior_dict) containing stacked states for all timesteps, SHAPE:(T,B,.)
  
  """
  fn2 = lambda carry, inp: (fn(carry, inp),) * 2 # copy the fn output to become a tuple of 2 elements
  if not unroll:
    return nj.scan(fn2, start, inputs, modify=modify)[1]  # the stacked version of traj state dict/tuple dict for all timesteps,same as unroll version
  # a "leaf" is defined as an element that cannot be further broken down in terms of the data structure hierarchy.
  length = len(jax.tree_util.tree_leaves(inputs)[0]) # Trajectory length, inputs[0]: action (T,B,.) after swapped
  carrydef = jax.tree_util.tree_structure(start)   # carrydef: the structure (post_dict, prior_dict) or (prior_dict)
  carry = start
  outs = []
  for index in range(length):
    # carry (out is the copy) is the state, it passed on to the next timestep
    carry, out = fn2(carry, tree_map(lambda x: x[index], inputs)) # extract each timestep of inputs:(action,embed,is_first), pass it to fn:obs_step()
    flat, treedef = jax.tree_util.tree_flatten(out)  # out: (post_dict, prior_dict) or (prior_dict)
    assert treedef == carrydef, (treedef, carrydef)
    outs.append(flat) # outs: [z_1,h_1,z_1_prob,est_z_1...],[z_2...
  # stack all z_t , all h_t, all z_t_prob, all est_z_t..., and put the stacked results into a list
  # after stacking z_1:T ,SHAPE:(T,B,.)
  outs = [
      jnp.stack([carry[i] for carry in outs], 0)
      for i in range(len(outs[0]))]  # len(outs[0])---6 for logit /8 for mean+std (post+prior:[z_1,h_1,z_1_prob,est_z_1...]) or 3/4 only prior
  return carrydef.unflatten(outs)    # return the form back to (post_dict, prior_dict) or (prior_dict)


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
    """sample from the distribution, and add gradient info of the distribution parameters (categorical probs) to the sample as sampling itself is not differentiable.
    The gradient trick is at the bottom of page 4 in the paper

    Args:
        sample_shape (tuple, optional): sample shape. Defaults to ().
        seed (_type_, optional): random seed for sampling. Defaults to None.

    Returns:
        array: sample result with gradient info
    """
    sample = sg(super().sample(sample_shape, seed))
    probs = self._pad(super().probs_parameter(), sample.shape) # probs_parameter() is the probs of the distribution, same shape as the softmax(logits)
    return sg(sample) + (probs - sg(probs)).astype(sample.dtype)

  def _pad(self, tensor, shape):
    """padd the dimensions of tensor to match the dimension number of shape
        not like 0-padding in CNN, just add dimensions to the tensor
    Args:
        tensor (array): to be padded
        shape (tuple): target shape

    Returns:
        array: padded tensor
    """
    while len(tensor.shape) < len(shape):
      tensor = tensor[None]
    return tensor


class MSEDist:

  def __init__(self, mode, dims, agg='sum'):
    """Mean Squared Error (MSE) distance calculation

    Args:
        mode (array): the one to be compared with, target array
        dims (int): the number of event/feat dimensions, if dims=1, then the event space is the last dim, it will calculate the distance between arrays along the last dim
        agg (str, optional): use mean or sum to aggregate the distance elements in the last dim. Defaults to 'sum'.
    """
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
    """calculate the distance between the mode and the value after symlogged, and return the distance * (-1) as a return or log_prob 

    Args:
        mode (array): the one to be compared with
        dims (int): the number of event/feat dimensions, if dims=1, then the event space is the last dim, it will calculate the distance between arrays along the last dim
        dist (str, optional): distance calculation method. Defaults to 'mse'.
        agg (str, optional): use mean or sum to aggregate the distance elements in the last dim. Defaults to 'sum'.
        tol (_type_, optional): distance elements set to zero below this threshold. Defaults to 1e-8.
    """
    self._mode = mode
    self._dims = tuple([-x for x in range(1, dims + 1)])  # if dims=1, then _dims=(-1,); if dims=2, then _dims=(-1,-2)
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
    """distance calculation in discrete settings, here we have the symlog + two-hot encodings transformation

    Args:
        logits (array): the one to be compared with
        dims (int, optional): number of event/feat dims. Defaults to 0.
        low (int, optional): bin lower bound. Defaults to -20.
        high (int, optional): bin higher bound. Defaults to 20.
        transfwd (fn, optional): forward transformation method. Defaults to symlog.
        transbwd (fn, optional): backward transformation method. Defaults to symexp.
    """
    self.logits = logits
    self.probs = jax.nn.softmax(logits)
    self.dims = tuple([-x for x in range(1, dims + 1)])
    self.bins = jnp.linspace(low, high, logits.shape[-1]) # logits.shape[-1] is the number of bins
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
    """calculate log_prob (-1*cross entropy loss), Equation (8) + (9) + (10)

    Args:
        x (array): input array, should be 1-dim smaller than logits (x should not have the bin dim in the end)

    Returns:
        array: cross entropy loss, (1-dim smaller than input x) sum over all classes and the last dim of x (event/feat dim)
    """
    x = self.transfwd(x) # apply symlog to x
    # calculate the two-hot encoding for x
    below = (self.bins <= x[..., None]).astype(jnp.int32).sum(-1) - 1  # find the nearest bin index below each element in last dim of x
    above = len(self.bins) - (
        self.bins > x[..., None]).astype(jnp.int32).sum(-1) # find the nearest bin index above each element in last dim of x
    below = jnp.clip(below, 0, len(self.bins) - 1)
    above = jnp.clip(above, 0, len(self.bins) - 1)
    # below and above has same shape as x
    equal = (below == above) # identify the out-of-range elements when below==above
    # calculate the distance to the nearest bin below and above, assign 1 to the out-of-range elements
    dist_to_below = jnp.where(equal, 1, jnp.abs(self.bins[below] - x))
    dist_to_above = jnp.where(equal, 1, jnp.abs(self.bins[above] - x))
    total = dist_to_below + dist_to_above
    # get the distance weight (0%-100%, abstract for how close to the left/right bin boundary) to the nearest bin below and above
    weight_below = dist_to_above / total
    weight_above = dist_to_below / total
    # get the two-hot encoding for x, weight_below/above same shape as x
    # target shape: (x.shape, len(self.bins))
    target = (
        jax.nn.one_hot(below, len(self.bins)) * weight_below[..., None] +
        jax.nn.one_hot(above, len(self.bins)) * weight_above[..., None])
    # this line is basically calculation the log of the softmax of the logits
    log_pred = self.logits - jax.scipy.special.logsumexp(
        self.logits, -1, keepdims=True)
    return (target * log_pred).sum(-1).sum(self.dims) # -1* cross entropy loss, sum over all classes and the last dim of x (event/feat dim)


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
    """  
    return offset and invscale (calculated from EMA percentile)
    """
    self.update(x)
    return self.stats()

  def update(self, x):
    """  
    Equation (12): EMA for batch percentile, jnp.percentile will flatten all dimensions of x and calculate the percentile
    """
    if parallel():
      # **all-reduce**: to take data distributed across multiple devices, 
      # perform a specified reduction operation (such as summation, averaging, finding minimums or maximums), 
      # and then share the result of this operation back to all devices

      # the following should be used in context of jax.pmap
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
    """  
    return offset and invscale

    offset is the EMA low percentile (5%) 

    invscale is the max(1,S) in Equation (11)
    """
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
