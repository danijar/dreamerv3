import re

import jax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp

f32 = jnp.float32
tfd = tfp.distributions
tree_map = jax.tree_util.tree_map
sg = lambda x: tree_map(jax.lax.stop_gradient, x)

from . import jaxutils
from . import ninjax as nj

cast = jaxutils.cast_to_compute


class RSSM(nj.Module):

  def __init__(
      self, deter=1024, stoch=32, classes=32, unroll=False, initial='learned',
      unimix=0.01, action_clip=1.0, **kw):
    self._deter = deter # deterministic latent state size, h_t
    self._stoch = stoch # stochastic latent state size, z_t
    self._classes = classes # number of classes for discrete latent state, if 0, it is continuous
    self._unroll = unroll
    self._initial = initial
    self._unimix = unimix
    self._action_clip = action_clip
    self._kw = kw

  def initial(self, bs):
    """ 
    Initialize the state of the model

    stoch is a sample of prob (logit /mean+std) ,can be the mode of the prob distribution 
    """
    if self._classes:
      # if it is dicrete/classification, the state includes a one-hot vector logits for each dim in stochastic latent state
      state = dict(
          deter=jnp.zeros([bs, self._deter], f32),
          logit=jnp.zeros([bs, self._stoch, self._classes], f32),
          stoch=jnp.zeros([bs, self._stoch, self._classes], f32))
    else:
      # if it is regression, the state includes a mean and std for value in each dim in stochastic latent state
      state = dict(
          deter=jnp.zeros([bs, self._deter], f32),
          mean=jnp.zeros([bs, self._stoch], f32),
          std=jnp.ones([bs, self._stoch], f32),
          stoch=jnp.zeros([bs, self._stoch], f32))
    if self._initial == 'zeros':
      return cast(state)
    elif self._initial == 'learned':
      deter = self.get('initial', jnp.zeros, state['deter'][0].shape, f32)
      state['deter'] = jnp.repeat(jnp.tanh(deter)[None], bs, 0)  #now shape: (BS,self._deter)  ,add a new batch dim to the front and then replicates the array across this new dimension to match the batch size (bs).
      state['stoch'] = self.get_stoch(cast(state['deter']))  # init stoch sample using the dynamics predictor 'img_out'
      return cast(state)
    else:
      raise NotImplementedError(self._initial)

  def observe(self, embed, action, is_first, state=None):
    """  
    Fig.3(a) --- use real-world embedded observation and action trajectory, for World Model learning
   
    action: (B,T,A), A is the action dim
    Return:
    (posterior dict, prior dict) for all timesteps (containing stacked z_1:T, h_1:T,...) each SHAPE: (B,T,.)
    different from the tuple of dicts having individual z_t, h_t,... as in the output of obs_step
    """
    swap = lambda x: x.transpose([1, 0] + list(range(2, len(x.shape))))
    if state is None:
      state = self.initial(action.shape[0])
    step = lambda prev, inputs: self.obs_step(prev[0], *inputs) # here use the posterior dict and unpack the inputs tuple
    inputs = swap(action), swap(embed), swap(is_first)  # move the second dim to the first, now (T,B,.)
    start = state, state         # init for the first step, so prior=post=state
    post, prior = jaxutils.scan(step, inputs, start, self._unroll)
    post = {k: swap(v) for k, v in post.items()}   # first dim again to B---> (B,T,.)
    prior = {k: swap(v) for k, v in prior.items()}
    return post, prior

  def imagine(self, action, state=None):
    swap = lambda x: x.transpose([1, 0] + list(range(2, len(x.shape))))
    state = self.initial(action.shape[0]) if state is None else state
    assert isinstance(state, dict), state
    action = swap(action)
    prior = jaxutils.scan(self.img_step, action, state, self._unroll)
    prior = {k: swap(v) for k, v in prior.items()}
    return prior

  def get_dist(self, state, argmax=False):
    """ 
    convert logits / mean+std in state dict to a tensor flow distribution 
    """
    if self._classes:
      logit = state['logit'].astype(f32)
      return tfd.Independent(jaxutils.OneHotDist(logit), 1) # specifying that the last dimension of your distribution's output (here, the categorical outcomes) should be treated as part of the independent event space rather than separate instances
    else:
      mean = state['mean'].astype(f32)
      std = state['std'].astype(f32)
      return tfd.MultivariateNormalDiag(mean, std)

  def obs_step(self, prev_state, prev_action, embed, is_first):
    """  
    prev_state: {z_t-1, h_t-1, z_prob_t-1} , each SHAPE: (B,.)
    prev_action: a_t-1
    embed: intermediate representation of the observation
    is_first: boolean indicating whether the current state is the first state in an episode
    
    Output: posterior dict:{z_t, h_t, z_prob_t}, prior dict:{ est_z_t,h_t,est_z_prob_t}
    """
    is_first = cast(is_first)
    prev_action = cast(prev_action)
    if self._action_clip > 0.0: # clip the action above a certain value 
      prev_action *= sg(self._action_clip / jnp.maximum(
          self._action_clip, jnp.abs(prev_action)))
    prev_state, prev_action = jax.tree_util.tree_map(             # ensure that the prev_state and prev_action reset to 0 at the start of each new episode or sequence.
        lambda x: self._mask(x, 1.0 - is_first), (prev_state, prev_action))   
    prev_state = jax.tree_util.tree_map(
        lambda x, y: x + self._mask(y, is_first),
        prev_state, self.initial(len(is_first)))   # replace/add the first state (already 0) with the initial state, len(is_first) is the batch size
    prior = self.img_step(prev_state, prev_action) # include est_z_t
    x = jnp.concatenate([prior['deter'], embed], -1)  # prepare for encoder output
    x = self.get('obs_out', Linear, **self._kw)(x)
    stats = self._stats('obs_stats', x)
    dist = self.get_dist(stats)
    stoch = dist.sample(seed=nj.rng()) # sample from the z_t distribution (posterior)
    post = {'stoch': stoch, 'deter': prior['deter'], **stats}
    return cast(post), cast(prior)

  def img_step(self, prev_state, prev_action):
    """  
    Here performs the dynamics model (GRU) and the dynamics predictor (Linear)

    Input: { z_t-1,h_t-1, z_prob_t-1}, a_t-1  
    Output: { est_z_t,h_t,est_z_prob_t}
    """
    prev_stoch = prev_state['stoch']
    prev_action = cast(prev_action)
    if self._action_clip > 0.0: # ensure clipping of the action
      prev_action *= sg(self._action_clip / jnp.maximum(
          self._action_clip, jnp.abs(prev_action)))
    if self._classes:
      shape = prev_stoch.shape[:-2] + (self._stoch * self._classes,) # (B,T,self._stoch * self._classes)
      prev_stoch = prev_stoch.reshape(shape)
    if len(prev_action.shape) > len(prev_stoch.shape):  # 2D actions.
      shape = prev_action.shape[:-2] + (np.prod(prev_action.shape[-2:]),) # flatten multi-dim actions
      prev_action = prev_action.reshape(shape)
    x = jnp.concatenate([prev_stoch, prev_action], -1) # concatenate the previous stochastic state and the action
    x = self.get('img_in', Linear, **self._kw)(x)  # x---(z_t-1,a_t-1)
    x, deter = self._gru(x, prev_state['deter'])   # Sequence model (dynamics): GRU ,deter:h_t
    x = self.get('img_out', Linear, **self._kw)(x)  # 'img_out':Dynamics predictor 
    stats = self._stats('img_stats', x)           #---> estimated z_t
    dist = self.get_dist(stats)
    stoch = dist.sample(seed=nj.rng()) # sample from the est_z_t distribution (prior)
    prior = {'stoch': stoch, 'deter': deter, **stats}
    return cast(prior)

  def get_stoch(self, deter):
    """ 
    use dynamics predictor 'img_out' to produce a mode of the stochasctic latent var distribution

    using NNs (several linear layers) to convert deterministic input to the mode of a distribution
    """
    x = self.get('img_out', Linear, **self._kw)(deter) # if img_out not exists, create a Linear layer with the params specified in self._kw
    stats = self._stats('img_stats', x)
    dist = self.get_dist(stats)
    return cast(dist.mode()) 

  def _gru(self, x, deter):
    x = jnp.concatenate([deter, x], -1)
    kw = {**self._kw, 'act': 'none', 'units': 3 * self._deter}
    x = self.get('gru', Linear, **kw)(x)
    reset, cand, update = jnp.split(x, 3, -1)
    reset = jax.nn.sigmoid(reset)
    cand = jnp.tanh(reset * cand)
    update = jax.nn.sigmoid(update - 1)
    deter = update * cand + (1 - update) * deter
    return deter, deter

  def _stats(self, name, x):
    """ 
    given input x, apply a Linear layer to x and return a prob. distribution of the stochastic latent state.
    Here have the 1% uniform + 99% NN output trick for discrete problem (in dynamics and encoder)
    """
    if self._classes:
      x = self.get(name, Linear, self._stoch * self._classes)(x)
      logit = x.reshape(x.shape[:-1] + (self._stoch, self._classes))
      if self._unimix:  # 1% uniform + 99% NN output ---> correspond to the last paragraph of world model learning 
        probs = jax.nn.softmax(logit, -1)
        uniform = jnp.ones_like(probs) / probs.shape[-1]
        probs = (1 - self._unimix) * probs + self._unimix * uniform
        logit = jnp.log(probs)
      stats = {'logit': logit}
      return stats
    else:
      x = self.get(name, Linear, 2 * self._stoch)(x)
      mean, std = jnp.split(x, 2, -1) # split the array x into two parts along the last axis 
      std = 2 * jax.nn.sigmoid(std / 2) + 0.1
      return {'mean': mean, 'std': std}

  def _mask(self, value, mask):
    """ 
    applies a batch-wise mask to a multi-dimensional array, (e.g. some batches are masked out)
    """
    return jnp.einsum('b...,b->b...', value, mask.astype(value.dtype))

  def dyn_loss(self, post, prior, impl='kl', free=1.0):
    if impl == 'kl':
      loss = self.get_dist(sg(post)).kl_divergence(self.get_dist(prior))
    elif impl == 'logprob':
      loss = -self.get_dist(prior).log_prob(sg(post['stoch']))
    else:
      raise NotImplementedError(impl)
    if free:
      loss = jnp.maximum(loss, free)
    return loss

  def rep_loss(self, post, prior, impl='kl', free=1.0):
    if impl == 'kl':
      loss = self.get_dist(post).kl_divergence(self.get_dist(sg(prior)))
    elif impl == 'uniform':
      uniform = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), prior)
      loss = self.get_dist(post).kl_divergence(self.get_dist(uniform))
    elif impl == 'entropy':
      loss = -self.get_dist(post).entropy()
    elif impl == 'none':
      loss = jnp.zeros(post['deter'].shape[:-1])
    else:
      raise NotImplementedError(impl)
    if free:
      loss = jnp.maximum(loss, free)
    return loss


class MultiEncoder(nj.Module):

  def __init__(
      self, shapes, cnn_keys=r'.*', mlp_keys=r'.*', mlp_layers=4,
      mlp_units=512, cnn='resize', cnn_depth=48,
      cnn_blocks=2, resize='stride',
      symlog_inputs=False, minres=4, **kw):
    excluded = ('is_first', 'is_last')
    shapes = {k: v for k, v in shapes.items() if (
        k not in excluded and not k.startswith('log_'))}
    self.cnn_shapes = {k: v for k, v in shapes.items() if (
        len(v) == 3 and re.match(cnn_keys, k))}
    self.mlp_shapes = {k: v for k, v in shapes.items() if (
        len(v) in (1, 2) and re.match(mlp_keys, k))}
    self.shapes = {**self.cnn_shapes, **self.mlp_shapes}
    print('Encoder CNN shapes:', self.cnn_shapes)
    print('Encoder MLP shapes:', self.mlp_shapes)
    cnn_kw = {**kw, 'minres': minres, 'name': 'cnn'}
    mlp_kw = {**kw, 'symlog_inputs': symlog_inputs, 'name': 'mlp'}
    if cnn == 'resnet':
      self._cnn = ImageEncoderResnet(cnn_depth, cnn_blocks, resize, **cnn_kw)
    else:
      raise NotImplementedError(cnn)
    if self.mlp_shapes:
      self._mlp = MLP(None, mlp_layers, mlp_units, dist='none', **mlp_kw)

  def __call__(self, data):
    some_key, some_shape = list(self.shapes.items())[0]
    batch_dims = data[some_key].shape[:-len(some_shape)]
    data = {
        k: v.reshape((-1,) + v.shape[len(batch_dims):])
        for k, v in data.items()}
    outputs = []
    if self.cnn_shapes:
      inputs = jnp.concatenate([data[k] for k in self.cnn_shapes], -1)
      output = self._cnn(inputs)
      output = output.reshape((output.shape[0], -1))
      outputs.append(output)
    if self.mlp_shapes:
      inputs = [
          data[k][..., None] if len(self.shapes[k]) == 0 else data[k]
          for k in self.mlp_shapes]
      inputs = jnp.concatenate([x.astype(f32) for x in inputs], -1)
      inputs = jaxutils.cast_to_compute(inputs)
      outputs.append(self._mlp(inputs))
    outputs = jnp.concatenate(outputs, -1)
    outputs = outputs.reshape(batch_dims + outputs.shape[1:])
    return outputs


class MultiDecoder(nj.Module):

  def __init__(
      self, shapes, inputs=['tensor'], cnn_keys=r'.*', mlp_keys=r'.*',
      mlp_layers=4, mlp_units=512, cnn='resize', cnn_depth=48, cnn_blocks=2,
      image_dist='mse', vector_dist='mse', resize='stride', bins=255,
      outscale=1.0, minres=4, cnn_sigmoid=False, **kw):
    excluded = ('is_first', 'is_last', 'is_terminal', 'reward')
    shapes = {k: v for k, v in shapes.items() if k not in excluded}
    self.cnn_shapes = {
        k: v for k, v in shapes.items()
        if re.match(cnn_keys, k) and len(v) == 3}
    self.mlp_shapes = {
        k: v for k, v in shapes.items()
        if re.match(mlp_keys, k) and len(v) == 1}
    self.shapes = {**self.cnn_shapes, **self.mlp_shapes}
    print('Decoder CNN shapes:', self.cnn_shapes)
    print('Decoder MLP shapes:', self.mlp_shapes)
    cnn_kw = {**kw, 'minres': minres, 'sigmoid': cnn_sigmoid}
    mlp_kw = {**kw, 'dist': vector_dist, 'outscale': outscale, 'bins': bins}
    if self.cnn_shapes:
      shapes = list(self.cnn_shapes.values())
      assert all(x[:-1] == shapes[0][:-1] for x in shapes)
      shape = shapes[0][:-1] + (sum(x[-1] for x in shapes),)
      if cnn == 'resnet':
        self._cnn = ImageDecoderResnet(
            shape, cnn_depth, cnn_blocks, resize, **cnn_kw, name='cnn')
      else:
        raise NotImplementedError(cnn)
    if self.mlp_shapes:
      self._mlp = MLP(
          self.mlp_shapes, mlp_layers, mlp_units, **mlp_kw, name='mlp')
    self._inputs = Input(inputs, dims='deter')
    self._image_dist = image_dist

  def __call__(self, inputs, drop_loss_indices=None):
    features = self._inputs(inputs)
    dists = {}
    if self.cnn_shapes:
      feat = features
      if drop_loss_indices is not None:
        feat = feat[:, drop_loss_indices]
      flat = feat.reshape([-1, feat.shape[-1]])
      output = self._cnn(flat)
      output = output.reshape(feat.shape[:-1] + output.shape[1:])
      split_indices = np.cumsum([v[-1] for v in self.cnn_shapes.values()][:-1])
      means = jnp.split(output, split_indices, -1)
      dists.update({
          key: self._make_image_dist(key, mean)
          for (key, shape), mean in zip(self.cnn_shapes.items(), means)})
    if self.mlp_shapes:
      dists.update(self._mlp(features))
    return dists

  def _make_image_dist(self, name, mean):
    mean = mean.astype(f32)
    if self._image_dist == 'normal':
      return tfd.Independent(tfd.Normal(mean, 1), 3)
    if self._image_dist == 'mse':
      return jaxutils.MSEDist(mean, 3, 'sum')
    raise NotImplementedError(self._image_dist)


class ImageEncoderResnet(nj.Module):

  def __init__(self, depth, blocks, resize, minres, **kw):
    self._depth = depth
    self._blocks = blocks
    self._resize = resize
    self._minres = minres
    self._kw = kw

  def __call__(self, x):
    stages = int(np.log2(x.shape[-2]) - np.log2(self._minres))
    depth = self._depth
    x = jaxutils.cast_to_compute(x) - 0.5
    # print(x.shape)
    for i in range(stages):
      kw = {**self._kw, 'preact': False}
      if self._resize == 'stride':
        x = self.get(f's{i}res', Conv2D, depth, 4, 2, **kw)(x)
      elif self._resize == 'stride3':
        s = 2 if i else 3
        k = 5 if i else 4
        x = self.get(f's{i}res', Conv2D, depth, k, s, **kw)(x)
      elif self._resize == 'mean':
        N, H, W, D = x.shape
        x = self.get(f's{i}res', Conv2D, depth, 3, 1, **kw)(x)
        x = x.reshape((N, H // 2, W // 2, 4, D)).mean(-2)
      elif self._resize == 'max':
        x = self.get(f's{i}res', Conv2D, depth, 3, 1, **kw)(x)
        x = jax.lax.reduce_window(
            x, -jnp.inf, jax.lax.max, (1, 3, 3, 1), (1, 2, 2, 1), 'same')
      else:
        raise NotImplementedError(self._resize)
      for j in range(self._blocks):
        skip = x
        kw = {**self._kw, 'preact': True}
        x = self.get(f's{i}b{j}conv1', Conv2D, depth, 3, **kw)(x)
        x = self.get(f's{i}b{j}conv2', Conv2D, depth, 3, **kw)(x)
        x += skip
        # print(x.shape)
      depth *= 2
    if self._blocks:
      x = get_act(self._kw['act'])(x)
    x = x.reshape((x.shape[0], -1))
    # print(x.shape)
    return x


class ImageDecoderResnet(nj.Module):

  def __init__(self, shape, depth, blocks, resize, minres, sigmoid, **kw):
    self._shape = shape
    self._depth = depth
    self._blocks = blocks
    self._resize = resize
    self._minres = minres
    self._sigmoid = sigmoid
    self._kw = kw

  def __call__(self, x):
    stages = int(np.log2(self._shape[-2]) - np.log2(self._minres))
    depth = self._depth * 2 ** (stages - 1)
    x = jaxutils.cast_to_compute(x)
    x = self.get('in', Linear, (self._minres, self._minres, depth))(x)
    for i in range(stages):
      for j in range(self._blocks):
        skip = x
        kw = {**self._kw, 'preact': True}
        x = self.get(f's{i}b{j}conv1', Conv2D, depth, 3, **kw)(x)
        x = self.get(f's{i}b{j}conv2', Conv2D, depth, 3, **kw)(x)
        x += skip
        # print(x.shape)
      depth //= 2
      kw = {**self._kw, 'preact': False}
      if i == stages - 1:
        kw = {}
        depth = self._shape[-1]
      if self._resize == 'stride':
        x = self.get(f's{i}res', Conv2D, depth, 4, 2, transp=True, **kw)(x)
      elif self._resize == 'stride3':
        s = 3 if i == stages - 1 else 2
        k = 5 if i == stages - 1 else 4
        x = self.get(f's{i}res', Conv2D, depth, k, s, transp=True, **kw)(x)
      elif self._resize == 'resize':
        x = jnp.repeat(jnp.repeat(x, 2, 1), 2, 2)
        x = self.get(f's{i}res', Conv2D, depth, 3, 1, **kw)(x)
      else:
        raise NotImplementedError(self._resize)
    if max(x.shape[1:-1]) > max(self._shape[:-1]):
      padh = (x.shape[1] - self._shape[0]) / 2
      padw = (x.shape[2] - self._shape[1]) / 2
      x = x[:, int(np.ceil(padh)): -int(padh), :]
      x = x[:, :, int(np.ceil(padw)): -int(padw)]
    # print(x.shape)
    assert x.shape[-3:] == self._shape, (x.shape, self._shape)
    if self._sigmoid:
      x = jax.nn.sigmoid(x)
    else:
      x = x + 0.5
    return x


class MLP(nj.Module):

  def __init__(
      self, shape, layers, units, inputs=['tensor'], dims=None,
      symlog_inputs=False, **kw):
    assert shape is None or isinstance(shape, (int, tuple, dict)), shape
    if isinstance(shape, int):
      shape = (shape,)
    self._shape = shape
    self._layers = layers
    self._units = units
    self._inputs = Input(inputs, dims=dims)
    self._symlog_inputs = symlog_inputs
    distkeys = (
        'dist', 'outscale', 'minstd', 'maxstd', 'outnorm', 'unimix', 'bins')
    self._dense = {k: v for k, v in kw.items() if k not in distkeys}
    self._dist = {k: v for k, v in kw.items() if k in distkeys}

  def __call__(self, inputs):
    feat = self._inputs(inputs)
    if self._symlog_inputs:
      feat = jaxutils.symlog(feat)
    x = jaxutils.cast_to_compute(feat)
    x = x.reshape([-1, x.shape[-1]])
    for i in range(self._layers):
      x = self.get(f'h{i}', Linear, self._units, **self._dense)(x)
    x = x.reshape(feat.shape[:-1] + (x.shape[-1],))
    if self._shape is None:
      return x
    elif isinstance(self._shape, tuple):
      return self._out('out', self._shape, x)
    elif isinstance(self._shape, dict):
      return {k: self._out(k, v, x) for k, v in self._shape.items()}
    else:
      raise ValueError(self._shape)

  def _out(self, name, shape, x):
    return self.get(f'dist_{name}', Dist, shape, **self._dist)(x)


class Dist(nj.Module):

  def __init__(
      self, shape, dist='mse', outscale=0.1, outnorm=False, minstd=1.0,
      maxstd=1.0, unimix=0.0, bins=255):
    assert all(isinstance(dim, int) for dim in shape), shape
    self._shape = shape
    self._dist = dist
    self._minstd = minstd
    self._maxstd = maxstd
    self._unimix = unimix
    self._outscale = outscale
    self._outnorm = outnorm
    self._bins = bins

  def __call__(self, inputs):
    dist = self.inner(inputs)
    assert tuple(dist.batch_shape) == tuple(inputs.shape[:-1]), (
        dist.batch_shape, dist.event_shape, inputs.shape)
    return dist

  def inner(self, inputs):
    kw = {}
    kw['outscale'] = self._outscale
    kw['outnorm'] = self._outnorm
    shape = self._shape
    if self._dist.endswith('_disc'):
      shape = (*self._shape, self._bins)
    out = self.get('out', Linear, int(np.prod(shape)), **kw)(inputs)
    out = out.reshape(inputs.shape[:-1] + shape).astype(f32)
    if self._dist in ('normal', 'trunc_normal'):
      std = self.get('std', Linear, int(np.prod(self._shape)), **kw)(inputs)
      std = std.reshape(inputs.shape[:-1] + self._shape).astype(f32)
    if self._dist == 'symlog_mse':
      return jaxutils.SymlogDist(out, len(self._shape), 'mse', 'sum')
    if self._dist == 'symlog_disc':
      return jaxutils.DiscDist(
          out, len(self._shape), -20, 20, jaxutils.symlog, jaxutils.symexp)
    if self._dist == 'mse':
      return jaxutils.MSEDist(out, len(self._shape), 'sum')
    if self._dist == 'normal':
      lo, hi = self._minstd, self._maxstd
      std = (hi - lo) * jax.nn.sigmoid(std + 2.0) + lo
      dist = tfd.Normal(jnp.tanh(out), std)
      dist = tfd.Independent(dist, len(self._shape))
      dist.minent = np.prod(self._shape) * tfd.Normal(0.0, lo).entropy()
      dist.maxent = np.prod(self._shape) * tfd.Normal(0.0, hi).entropy()
      return dist
    if self._dist == 'binary':
      dist = tfd.Bernoulli(out)
      return tfd.Independent(dist, len(self._shape))
    if self._dist == 'onehot':
      if self._unimix:
        probs = jax.nn.softmax(out, -1)
        uniform = jnp.ones_like(probs) / probs.shape[-1]
        probs = (1 - self._unimix) * probs + self._unimix * uniform
        out = jnp.log(probs)
      dist = jaxutils.OneHotDist(out)
      if len(self._shape) > 1:
        dist = tfd.Independent(dist, len(self._shape) - 1)
      dist.minent = 0.0
      dist.maxent = np.prod(self._shape[:-1]) * jnp.log(self._shape[-1])
      return dist
    raise NotImplementedError(self._dist)


class Conv2D(nj.Module):

  def __init__(
      self, depth, kernel, stride=1, transp=False, act='none', norm='none',
      pad='same', bias=True, preact=False, winit='uniform', fan='avg'):
    self._depth = depth
    self._kernel = kernel
    self._stride = stride
    self._transp = transp
    self._act = get_act(act)
    self._norm = Norm(norm, name='norm')
    self._pad = pad.upper()
    self._bias = bias and (preact or norm == 'none')
    self._preact = preact
    self._winit = winit
    self._fan = fan

  def __call__(self, hidden):
    if self._preact:
      hidden = self._norm(hidden)
      hidden = self._act(hidden)
      hidden = self._layer(hidden)
    else:
      hidden = self._layer(hidden)
      hidden = self._norm(hidden)
      hidden = self._act(hidden)
    return hidden

  def _layer(self, x):
    if self._transp:
      shape = (self._kernel, self._kernel, self._depth, x.shape[-1])
      kernel = self.get('kernel', Initializer(
          self._winit, fan=self._fan), shape)
      kernel = jaxutils.cast_to_compute(kernel)
      x = jax.lax.conv_transpose(
          x, kernel, (self._stride, self._stride), self._pad,
          dimension_numbers=('NHWC', 'HWOI', 'NHWC'))
    else:
      shape = (self._kernel, self._kernel, x.shape[-1], self._depth)
      kernel = self.get('kernel', Initializer(
          self._winit, fan=self._fan), shape)
      kernel = jaxutils.cast_to_compute(kernel)
      x = jax.lax.conv_general_dilated(
          x, kernel, (self._stride, self._stride), self._pad,
          dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
    if self._bias:
      bias = self.get('bias', jnp.zeros, self._depth, np.float32)
      bias = jaxutils.cast_to_compute(bias)
      x += bias
    return x


class Linear(nj.Module):

  def __init__(
      self, units, act='none', norm='none', bias=True, outscale=1.0,
      outnorm=False, winit='uniform', fan='avg'):
    self._units = tuple(units) if hasattr(units, '__len__') else (units,) #out_features dims
    self._act = get_act(act)
    self._norm = norm
    self._bias = bias and norm == 'none'
    self._outscale = outscale
    self._outnorm = outnorm
    self._winit = winit
    self._fan = fan

  def __call__(self, x):
    """ 
    pass the input x through the linear layer with kernel (weights), bias, normalization and activation function
    and reshape the output to the desired shape (e.g. flattened last dim into multiple dims)
    """
    shape = (x.shape[-1], np.prod(self._units)) # calculate shape of the kernel (input_dim,temp_out_dim)
    kernel = self.get('kernel', Initializer(
        self._winit, self._outscale, fan=self._fan), shape)
    kernel = jaxutils.cast_to_compute(kernel)
    x = x @ kernel
    if self._bias:
      bias = self.get('bias', jnp.zeros, np.prod(self._units), np.float32)
      bias = jaxutils.cast_to_compute(bias)
      x += bias
    if len(self._units) > 1:  # When the length of self._units is more than one, the output should be reshaped into a shape that is not simply flat but has multiple dimensions.
      x = x.reshape(x.shape[:-1] + self._units) # x:(batch_size, temp_out_features)--->(batch_size, unit1, unit2)
    x = self.get('norm', Norm, self._norm)(x)
    x = self._act(x)
    return x


class Norm(nj.Module):

  def __init__(self, impl):
    self._impl = impl

  def __call__(self, x):
    dtype = x.dtype
    if self._impl == 'none':
      return x
    elif self._impl == 'layer':
      x = x.astype(f32)
      x = jax.nn.standardize(x, axis=-1, epsilon=1e-3)
      x *= self.get('scale', jnp.ones, x.shape[-1], f32)
      x += self.get('bias', jnp.zeros, x.shape[-1], f32)
      return x.astype(dtype)
    else:
      raise NotImplementedError(self._impl)


class Input:

  def __init__(self, keys=['tensor'], dims=None):
    assert isinstance(keys, (list, tuple)), keys
    self._keys = tuple(keys)
    self._dims = dims or self._keys[0]

  def __call__(self, inputs):
    if not isinstance(inputs, dict):
      inputs = {'tensor': inputs}
    inputs = inputs.copy()
    for key in self._keys:
      if key.startswith('softmax_'):
        inputs[key] = jax.nn.softmax(inputs[key[len('softmax_'):]])
    if not all(k in inputs for k in self._keys):
      needs = f'{{{", ".join(self._keys)}}}'
      found = f'{{{", ".join(inputs.keys())}}}'
      raise KeyError(f'Cannot find keys {needs} among inputs {found}.')
    values = [inputs[k] for k in self._keys]
    dims = len(inputs[self._dims].shape)
    for i, value in enumerate(values):
      if len(value.shape) > dims:
        values[i] = value.reshape(
            value.shape[:dims - 1] + (np.prod(value.shape[dims - 1:]),))
    values = [x.astype(inputs[self._dims].dtype) for x in values]
    return jnp.concatenate(values, -1)


class Initializer:

  def __init__(self, dist='uniform', scale=1.0, fan='avg'):
    self.scale = scale
    self.dist = dist
    self.fan = fan

  def __call__(self, shape):
    """ 
    calculate the initial value of the kernel (weights) based on the shape of the kernel and the distribution type
    """
    if self.scale == 0.0:
      value = jnp.zeros(shape, f32)
    elif self.dist == 'uniform':
      fanin, fanout = self._fans(shape)
      denoms = {'avg': (fanin + fanout) / 2, 'in': fanin, 'out': fanout}
      scale = self.scale / denoms[self.fan]
      limit = np.sqrt(3 * scale)
      value = jax.random.uniform(
          nj.rng(), shape, f32, -limit, limit)
    elif self.dist == 'normal':
      fanin, fanout = self._fans(shape)
      denoms = {'avg': np.mean((fanin, fanout)), 'in': fanin, 'out': fanout}
      scale = self.scale / denoms[self.fan]
      std = np.sqrt(scale) / 0.87962566103423978
      value = std * jax.random.truncated_normal(
          nj.rng(), -2, 2, shape, f32)
    elif self.dist == 'ortho':
      nrows, ncols = shape[-1], np.prod(shape) // shape[-1]
      matshape = (nrows, ncols) if nrows > ncols else (ncols, nrows)
      mat = jax.random.normal(nj.rng(), matshape, f32)
      qmat, rmat = jnp.linalg.qr(mat)
      qmat *= jnp.sign(jnp.diag(rmat))
      qmat = qmat.T if nrows < ncols else qmat
      qmat = qmat.reshape(nrows, *shape[:-1])
      value = self.scale * jnp.moveaxis(qmat, 0, -1)
    else:
      raise NotImplementedError(self.dist)
    return value

  def _fans(self, shape):
    """ 
    calculate the "fan-in" and "fan-out" values for a given tensor shape, which are commonly used for initializing weights in neural networks. 
    These values help define the scale of weight initialization, ensuring appropriate variance across layers 
    y = X @ W, each row of X is a sample
    Output: fan_in (input dim), fan_out (output dim)
    """
    if len(shape) == 0:
      return 1, 1
    elif len(shape) == 1:
      return shape[0], shape[0]
    elif len(shape) == 2:
      return shape
    else:
      space = int(np.prod(shape[:-2]))
      return shape[-2] * space, shape[-1] * space


def get_act(name):
  """ 
  get the activation function by name 
  """
  if callable(name):
    return name
  elif name == 'none':
    return lambda x: x
  elif name == 'mish':
    return lambda x: x * jnp.tanh(jax.nn.softplus(x))
  elif hasattr(jax.nn, name):
    return getattr(jax.nn, name)
  else:
    raise NotImplementedError(name)
