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
    """wrapper for the Recurrent State Space Model (RSSM) in DreamerV3

    Args:
        deter (int, optional): deterministic latent state size, h_t. Defaults to 1024.
        stoch (int, optional): stochastic latent state size, z_t. Defaults to 32.
        classes (int, optional): number of classes for discrete latent state, if 0, it is continuous. Defaults to 32.
        unroll (bool, optional): whether to manually process the iteration over sequence using for loop. If false, it will use a nj.scan function to handle it. Defaults to False.
        initial (str, optional): initialization method for state dict. Defaults to 'learned'.
        unimix (float, optional): whether to use the 1% uniform + 99% NN output trick for discrete problem. Defaults to 0.01.
        action_clip (float, optional): action clipping threshold. Defaults to 1.0.
    """
    self._deter = deter 
    self._stoch = stoch 
    self._classes = classes 
    self._unroll = unroll  
    self._initial = initial
    self._unimix = unimix
    self._action_clip = action_clip
    self._kw = kw

  def initial(self, bs):
    """Initialize the state of the model.
    stoch is a sample of prob (logit /mean+std) ,can be the mode of the prob distribution 

    Args:
        bs (int): batch size

    Raises:
        NotImplementedError: initial method should be one of 'zeros', 'learned'

    Returns:
        dict: a dict of the initial state
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
    """Fig.3(a) ---Here perform forward pass using real-world embedded observation and action trajectory, for World Model learning

    Args:
        embed (array): embedded observation, shape: (B,T,E), E is the event dim
        action (array): (B,T,A), A is the action dim
        is_first (bool): boolean indicating whether the current step is the first step in an episode, shape: (B,T)
        state (dict, optional): state dict used as init state. Defaults to None.

    Returns:
        tuple: (posterior dict, prior dict) for all timesteps (containing stacked z_1:T, h_1:T,...) each SHAPE: (B,T,.)
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
    """Fig.3(b) ---Forward pass using imagined state and action trajectory, for Actor-Critic learning

    Args:
        action (array): (B,T,A), A is the action dim
        state (dict, optional): initial state dict. Defaults to None.

    Returns:
        dict: prior dict for all timesteps (containing stacked est_z_1:T, h_1:T,...) each SHAPE: (B,T,.)
    
    """
    swap = lambda x: x.transpose([1, 0] + list(range(2, len(x.shape))))
    state = self.initial(action.shape[0]) if state is None else state
    assert isinstance(state, dict), state #if not dict, raising error and print out state.
    action = swap(action)  # now (T,B,.)
    prior = jaxutils.scan(self.img_step, action, state, self._unroll)
    prior = {k: swap(v) for k, v in prior.items()}
    return prior

  def get_dist(self, state, argmax=False):
    """convert logits / mean+std in state dict to a tensor flow distribution 

    Args:
        state (dict): description of the distribution of the stochastic latent state--- containing array for (mean+ std/ logit)
        argmax (bool, optional): no use. Defaults to False.

    Returns:
        obj: a tensor flow distribution object
    """
    if self._classes:
      logit = state['logit'].astype(f32)
      return tfd.Independent(jaxutils.OneHotDist(logit), 1) # the latent state feature dim (-2) be added to event dimension
    else:
      mean = state['mean'].astype(f32)
      std = state['std'].astype(f32)
      return tfd.MultivariateNormalDiag(mean, std)

  def obs_step(self, prev_state, prev_action, embed, is_first):
    """observation step, forward pass of sequence model, encoder (the final step of the encoder after getting embedding), dynamics predictor in Equation (3)

    Args:
        prev_state (dict): posterior {z_t-1, h_t-1, z_prob_t-1}, each SHAPE: (B,...)
        prev_action (array): a_t-1, SHAPE: (B,A)
        embed (array): intermediate representation of the observation, SHAPE: (B,E)
        is_first (bool): boolean indicating whether the current state is the first state in an episode, SHAPE: (B,)

    Returns:
        tuple: posterior dict:{z_t, h_t, z_prob_t}, prior dict:{ est_z_t,h_t,est_z_prob_t}
    
    """
    is_first = cast(is_first)
    prev_action = cast(prev_action)
    if self._action_clip > 0.0: # clip the action above a certain value 
      prev_action *= sg(self._action_clip / jnp.maximum(
          self._action_clip, jnp.abs(prev_action)))
    # the following two lines are to ensure that the prev_state and prev_action reset to 0, if they happen to be the first timestep/state in an episode
    # and if it is the case, then the prev_state will be initialized as the initial state
    prev_state, prev_action = jax.tree_util.tree_map(             # ensure that the prev_state and prev_action reset to 0 at the start of each new episode.
        lambda x: self._mask(x, 1.0 - is_first), (prev_state, prev_action))   
    prev_state = jax.tree_util.tree_map(
        lambda x, y: x + self._mask(y, is_first),
        prev_state, self.initial(len(is_first)))   # replace/add the first state (already 0) with the initial state, len(is_first) is the batch size
    prior = self.img_step(prev_state, prev_action) # include est_z_t
    x = jnp.concatenate([prior['deter'], embed], -1)  # prepare for encoder output acc. to Equation (3)
    x = self.get('obs_out', Linear, **self._kw)(x)
    stats = self._stats('obs_stats', x)
    dist = self.get_dist(stats)
    stoch = dist.sample(seed=nj.rng()) # sample from the z_t distribution (posterior)
    post = {'stoch': stoch, 'deter': prior['deter'], **stats}
    return cast(post), cast(prior)

  def img_step(self, prev_state, prev_action):
    """Here performs the dynamics model (GRU) and the dynamics predictor (Linear)--imagination step

    Args:
        prev_state (dict): { z_t-1,h_t-1, z_prob_t-1}
        prev_action (array): a_t-1  

    Returns:
        dict: prior state dict---{ est_z_t,h_t,est_z_prob_t}
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
    """given input x, apply a Linear layer to x and return a prob. distribution of the stochastic latent state.
    Here have the 1% uniform + 99% NN output trick for discrete problem (in dynamics and encoder and actor, see Unimix categoricals in page 19, Section C: Summary of Differences)

    Args:
        name (str): name of the stats operation
        x (array): input embedding/feature array , shape: (B,...)

    Returns:
        dict: description of the distribution of the stochastic latent state--- containing array for (mean+ std/ logit)
    """
    if self._classes:
      x = self.get(name, Linear, self._stoch * self._classes)(x)
      logit = x.reshape(x.shape[:-1] + (self._stoch, self._classes))
      if self._unimix:  # 1% uniform + 99% NN output ---> correspond to the last paragraph of world model learning 
        probs = jax.nn.softmax(logit, -1)
        uniform = jnp.ones_like(probs) / probs.shape[-1]
        probs = (1 - self._unimix) * probs + self._unimix * uniform
        logit = jnp.log(probs)  # convert probs to logit , convenient for using softmax in the following steps
      stats = {'logit': logit}  
      return stats
    else:
      x = self.get(name, Linear, 2 * self._stoch)(x)
      mean, std = jnp.split(x, 2, -1) # split the array x into two parts (half_1 | half_2) along the last axis 
      std = 2 * jax.nn.sigmoid(std / 2) + 0.1
      return {'mean': mean, 'std': std}

  def _mask(self, value, mask):
    """applies a batch-wise mask to a multi-dimensional array, (e.g. some batches are masked out)

    Args:
        value (array): the array to be masked, shape: (B,...)
        mask (array): the mask array, shape: (B,)

    Returns:
        array: the masked array, shape: (B,...)
    """
    return jnp.einsum('b...,b->b...', value, mask.astype(value.dtype))

  def dyn_loss(self, post, prior, impl='kl', free=1.0):
    """Equation (5).2 in the paper, the loss function for the dynamics predictor, prediction loss

    Args:
        post (dict): posterior state dict of a sequence, not just a single timestep
        prior (dict): prior state dict of a sequence, not just a single timestep
        impl (str, optional): dynamic prediction loss method. Defaults to 'kl'.
        free (float, optional): freebits trick, dyn_loss clipping threshold (lowest bound). Defaults to 1.0.

    Raises:
        NotImplementedError: loss method should be one of 'kl', 'logprob'

    Returns:
        array: dynamics prediction loss over the sequence, shape: (B,T)
    """
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
    """Equation (5).3 in the paper, the loss function for the encoder, representation loss

    Args:
        post (dict): posterior state dict of a sequence, not just a single timestep
        prior (dict): prior state dict of a sequence, not just a single timestep
        impl (str, optional): dynamic prediction loss method. Defaults to 'kl'.
        free (float, optional): freebits trick, dyn_loss clipping threshold (lowest bound). Defaults to 1.0.

    Raises:
        NotImplementedError: loss method should be one of 'kl', 'uniform', 'entropy' or 'none'(zero loss)

    Returns:
        array: representation loss over the sequence, shape: (B,T)
    """
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
    """  
    Args:
    shapes: dict of shapes in tuple (obs_shape) 
    """
    excluded = ('is_first', 'is_last')
    shapes = {k: v for k, v in shapes.items() if (
        k not in excluded and not k.startswith('log_'))}
    # Extract the shapes used for CNN and MLP by matching the cnn/mlp_key from the beginning of the key in dict
    # for CNN: Input event shape (H,W,C)
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
      self._mlp = MLP(None, mlp_layers, mlp_units, dist='none', **mlp_kw) # here the shape is None, so the output is the array, not the distance measure object

  def __call__(self, data):
    """Forward pass of the encoder,

    Args:
        data (dict): input, contain images for CNN or vectors for MLP, also contain action,rewards,continue flag,... (data from a batch of trajectories---> (B,T,...))
        input could contain values with shapes begin with (B,...) or (B,T,...) for batch and timestep dims

    Returns:
        array: encoder output, shape:(batch_dims, event_dims) ----e.g. (B,T,E) or (B,E)
    """
    some_key, some_shape = list(self.shapes.items())[0]  #flatten dict to list of tuples of key and value and pick the first tuple, I guess it contain the event shape
    batch_dims = data[some_key].shape[:-len(some_shape)] # extract batch dims, TODO, need to identify which key is in the data and self.shapes
    data = {
        k: v.reshape((-1,) + v.shape[len(batch_dims):])   # flatten the batch dims into one dim but keep the event dims
        for k, v in data.items()}
    outputs = []
    if self.cnn_shapes:
      inputs = jnp.concatenate([data[k] for k in self.cnn_shapes], -1) # concatenate the input data along the last dim (Channel dim)
      output = self._cnn(inputs)
      output = output.reshape((output.shape[0], -1))  # -->(B*T,E) or (B,E) flatten the output but keep the combined batch dim, repeated again to ensure
      outputs.append(output)
    if self.mlp_shapes:
      inputs = [
          data[k][..., None] if len(self.shapes[k]) == 0 else data[k]
          for k in self.mlp_shapes]
      inputs = jnp.concatenate([x.astype(f32) for x in inputs], -1)
      inputs = jaxutils.cast_to_compute(inputs)
      outputs.append(self._mlp(inputs))
    outputs = jnp.concatenate(outputs, -1)
    outputs = outputs.reshape(batch_dims + outputs.shape[1:]) # recover the combined batch dim (B*T,...) to original batch dim (B,T,...), the event shape E should be one-dim 
    return outputs


class MultiDecoder(nj.Module):

  def __init__(
      self, shapes, inputs=['tensor'], cnn_keys=r'.*', mlp_keys=r'.*',
      mlp_layers=4, mlp_units=512, cnn='resize', cnn_depth=48, cnn_blocks=2,
      image_dist='mse', vector_dist='mse', resize='stride', bins=255,
      outscale=1.0, minres=4, cnn_sigmoid=False, **kw):
    excluded = ('is_first', 'is_last', 'is_terminal', 'reward')
    # TODO: check what shapes contain
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
      assert all(x[:-1] == shapes[0][:-1] for x in shapes) # check the batch dims are same across
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
    """  
    TODO: need to identify the shape of the inputs dict (is the batch dim only B or B,T) and what is the drop_loss_indices
    """
    features = self._inputs(inputs)
    dists = {}
    if self.cnn_shapes:
      feat = features
      if drop_loss_indices is not None:
        feat = feat[:, drop_loss_indices]
      flat = feat.reshape([-1, feat.shape[-1]]) # flat shape: (B,x)
      output = self._cnn(flat)  # output shape: (B,H,W,C)
      # I feel this one is another double check of the shape, should be useless
      output = output.reshape(feat.shape[:-1] + output.shape[1:]) # feat.shape[:-1] should be the batch dims, output.shape[1:] should be the event dims 
      # [:-1] This slice omits the last element of the list. This is necessary because when splitting an array at specified indices with np.split, you do not need to specify the final index since it's implied to be the end of the array.
      split_indices = np.cumsum([v[-1] for v in self.cnn_shapes.values()][:-1]) # calculate the split indices positions for the output in depth channel
      means = jnp.split(output, split_indices, -1)   # output a list of arrays

      # dict.update() is more efficient than dict[key] = value for multiple key-value pairs, it can add all of them simultaneously.
      dists.update({
          key: self._make_image_dist(key, mean)   # the mean of mean is around 0.5
          for (key, shape), mean in zip(self.cnn_shapes.items(), means)})
    if self.mlp_shapes:
      dists.update(self._mlp(features))
    return dists  # return a dict of distance measure objects

  def _make_image_dist(self, name, mean):
    """create image distribution / distance measure object

    Args:
        name (str): arbitrary name for the image distribution
        mean (array): mean of the image distribution, shape (B,H,W,C) or (B,T,H,W,C)

    Raises:
        NotImplementedError: image_distance method should be one of 'normal', 'mse'

    Returns:
        obj: image distance measure object
    """
    mean = mean.astype(f32)
    if self._image_dist == 'normal':
      return tfd.Independent(tfd.Normal(mean, 1), 3) # (H,W,C) now is the event shape
    if self._image_dist == 'mse':
      return jaxutils.MSEDist(mean, 3, 'sum')
    raise NotImplementedError(self._image_dist)


class ImageEncoderResnet(nj.Module):

  def __init__(self, depth, blocks, resize, minres, **kw):
    """wrapper for the ResNet image encoder

    Args:
        depth (int): tensor depth (depth channel) of the first stage
        blocks (int): how many blocks in each stage, each block contains two conv layers with a skip connection
        resize (str): resizing/pooling method between stages, one of 'stride', 'stride3', 'mean', 'max'
        minres (int): min size H W in the ResNet blocks, usually in the last stage
    """
    self._depth = depth
    self._blocks = blocks
    self._resize = resize
    self._minres = minres 
    self._kw = kw

  def __call__(self, x):
    """ImageEncoderResnet forward pass, pooling part different from the original paper version (here with additional conv layers and other pooling methods)

    Args:
        x (array): Input image with shape (B,H,W,C)

    Raises:
        NotImplementedError: resize method should be one of 'stride', 'stride3', 'mean', 'max'

    Returns:
        array: flattened output with shape (B, x)
    """

    stages = int(np.log2(x.shape[-2]) - np.log2(self._minres))
    depth = self._depth  # depth of the first stage
    x = jaxutils.cast_to_compute(x) - 0.5 # normalize the image input from [0,1] to [-0.5,0.5]
    # print(x.shape)
    for i in range(stages):
      kw = {**self._kw, 'preact': False}
      # 1--The beginning of each stage with resizing/pooing to halve the H,W
      if self._resize == 'stride':
        x = self.get(f's{i}res', Conv2D, depth, 4, 2, **kw)(x) #should output the same order of dims as input, but the H,W is halved
      elif self._resize == 'stride3':
        s = 2 if i else 3   # 2x2 stride for the rest of the stages, 3x3 stride for the first stage
        k = 5 if i else 4  # 5x5 kernel for the rest of the stages, 4x4 for the first stage
        x = self.get(f's{i}res', Conv2D, depth, k, s, **kw)(x)
      elif self._resize == 'mean':   # conv+mean pooling, but not traditional 
        N, H, W, D = x.shape
        x = self.get(f's{i}res', Conv2D, depth, 3, 1, **kw)(x)
        x = x.reshape((N, H // 2, W // 2, 4, D)).mean(-2)
      elif self._resize == 'max':    # conv+max pooling, sliding window, more like traditional 
        x = self.get(f's{i}res', Conv2D, depth, 3, 1, **kw)(x)
        x = jax.lax.reduce_window(
            x, -jnp.inf, jax.lax.max, (1, 3, 3, 1), (1, 2, 2, 1), 'same') # "same" to output exactly half the size of the input when stride=2
      else:
        raise NotImplementedError(self._resize)
      # 2--The subsequent blocks of convolutions in ResNet, with skip connections
      for j in range(self._blocks):
        skip = x
        kw = {**self._kw, 'preact': True}
        x = self.get(f's{i}b{j}conv1', Conv2D, depth, 3, **kw)(x)
        x = self.get(f's{i}b{j}conv2', Conv2D, depth, 3, **kw)(x)
        x += skip
        # print(x.shape)
      depth *= 2  # double the depth of the next stage
    if self._blocks:
      x = get_act(self._kw['act'])(x)
    x = x.reshape((x.shape[0], -1)) # flatten the output but keep the batch dim
    # print(x.shape)
    return x


class ImageDecoderResnet(nj.Module):

  def __init__(self, shape, depth, blocks, resize, minres, sigmoid, **kw):
    """wrapper for the ResNet image decoder, output should be an image

    Args:
        shape (tuple): feature/event shape of the output image ---(H,W,C)
        depth (int): depth of the first stage of the original ImageEncoderResnet
        blocks (int): how many blocks in each stage, each block contains two conv layers with a skip connection
        resize (str): resizing/pooling method between stages, one of 'stride', 'stride3', 'mean', 'max'
        minres (int): min size H W in the ResNet blocks, here for decoder it is in the first stage
        sigmoid (bool): whether to apply sigmoid to the output image
    """
    self._shape = shape
    self._depth = depth
    self._blocks = blocks
    self._resize = resize
    self._minres = minres
    self._sigmoid = sigmoid
    self._kw = kw

  def __call__(self, x):
    """forward pass of the ImageDecoderResnet

    Args:
        x (array): input latent feature array with shape (B, x)

    Raises:
        NotImplementedError: resize method should be one of 'stride', 'stride3', 'resize'

    Returns:
        array: output image with shape (B,H,W,C), range [0,1]
    """
    stages = int(np.log2(self._shape[-2]) - np.log2(self._minres))
    depth = self._depth * 2 ** (stages - 1) # calculate the depth of the first stage of the decoder from the one of the encoder
    x = jaxutils.cast_to_compute(x)
    x = self.get('in', Linear, (self._minres, self._minres, depth))(x)  # linear layer to reshape the input to the first stage of the decoder ,shape:(B,H,W,D)
    for i in range(stages):
      for j in range(self._blocks):
        skip = x
        kw = {**self._kw, 'preact': True}
        x = self.get(f's{i}b{j}conv1', Conv2D, depth, 3, **kw)(x)
        x = self.get(f's{i}b{j}conv2', Conv2D, depth, 3, **kw)(x)
        x += skip
        # print(x.shape)
      depth //= 2   # halve the depth of the next stage
      kw = {**self._kw, 'preact': False}
      if i == stages - 1:
        kw = {}     # clear the specified keywords for Conv2D in the last stage
        depth = self._shape[-1]  
      if self._resize == 'stride':
        x = self.get(f's{i}res', Conv2D, depth, 4, 2, transp=True, **kw)(x) # transp=True for transposed convolution, H,W is doubled
      elif self._resize == 'stride3':
        s = 3 if i == stages - 1 else 2
        k = 5 if i == stages - 1 else 4
        x = self.get(f's{i}res', Conv2D, depth, k, s, transp=True, **kw)(x) #H,W is x3 in last stage or x2
      elif self._resize == 'resize':
        x = jnp.repeat(jnp.repeat(x, 2, 1), 2, 2)     # repeat x first along H, then along W-->(B,2*H,2*W,c)  
        x = self.get(f's{i}res', Conv2D, depth, 3, 1, **kw)(x)
      else:
        raise NotImplementedError(self._resize)
    if max(x.shape[1:-1]) > max(self._shape[:-1]): # if the output size (H,W) is larger than the target size, crop the output
      padh = (x.shape[1] - self._shape[0]) / 2
      padw = (x.shape[2] - self._shape[1]) / 2
      x = x[:, int(np.ceil(padh)): -int(padh), :]
      x = x[:, :, int(np.ceil(padw)): -int(padw)]
    # print(x.shape)
    assert x.shape[-3:] == self._shape, (x.shape, self._shape) # check the H,W,C of the output image matches the target shape
    if self._sigmoid:
      x = jax.nn.sigmoid(x)
    else:          # x after layer norm is standardized to mean 0 and std 1, then silu makes it mean concentrated around 0
      x = x + 0.5  # move the x mean from silu act output closer to 0.5 (prepare for creating image distribution), x range is influenced by act and norm in Conv2d
    return x


class MLP(nj.Module):

  def __init__(
      self, shape, layers, units, inputs=['tensor'], dims=None,
      symlog_inputs=False, **kw):
    """MLP wrapper

    Args:
        shape (int,tuple,dict): feature/event shape of the output
        layers (int): number of layers in the MLP
        units (int): number of units per layer
        inputs (list, optional): specified data keys. Defaults to ['tensor'].
        dims (str, optional): target dimension key, all inputs will be reshaped to make sure not exceeding the target dim numbers. Defaults to None.
        symlog_inputs (bool, optional): whether to use symlog to inputs. Defaults to False.
    """
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
    # separate the dense (for building linear layers) and dist parameters (for building distance measure objects) 
    self._dense = {k: v for k, v in kw.items() if k not in distkeys}
    self._dist = {k: v for k, v in kw.items() if k in distkeys}

  def __call__(self, inputs):
    """MLP forward pass

    Args:
        inputs (dict): input data

    Raises:
        ValueError: feature/event shape should be int, tuple or dict

    Returns:
        array/obj: if event shape is None, return the output array (dims number should be same as input, batch dims should be same), otherwise return the distance measure object
    """
    feat = self._inputs(inputs)  # dict to concated array after reshaping and transforming the inputs
    if self._symlog_inputs:          # encoder input is symlog transformed acc. to Page 19, Section C--> Symlog predictions
      feat = jaxutils.symlog(feat)
    x = jaxutils.cast_to_compute(feat)
    x = x.reshape([-1, x.shape[-1]])  # flatten the input but keep the last dim of x
    for i in range(self._layers):
      x = self.get(f'h{i}', Linear, self._units, **self._dense)(x)
    x = x.reshape(feat.shape[:-1] + (x.shape[-1],))  # recover back to the input shape, TODO: is the batch dim of feat same as the output x of linear layers?
    if self._shape is None:
      return x
    # if the event shape is not None, then process further to get the distance measure object from the output x
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
    """ a wrapper for different distance/log_prob types, to get the distance measure using different methods
        including processing for discrete regression result, see Critic Learning section in the paper
    Args:
        shape (tuple): event shape of the distribution, usually length=1; event shape: the shape of the output feature of the last layer; e.g. output 1024 dim embedding, (1024,) is the event shape
        dist (str, optional): distance method. Defaults to 'mse'.
        outscale (float, optional): _description_. Defaults to 0.1.
        outnorm (bool, optional): _description_. Defaults to False.
        minstd (float, optional): for "normal" dist method, will also influence the min entropy of the normal distribution. Defaults to 1.0.
        maxstd (float, optional): for "normal" dist method, will also influence the max entropy of the normal distribution. Defaults to 1.0.
        unimix (float, optional): the mixing extent of uniform distribution with the output distribution. Defaults to 0.0.
        bins (int, optional): only used in discrete settings. Defaults to 255.
    """
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
    """forward pass of the Dist module to get the parametrized distance calculation object

    Args:
        inputs (array): input data

    Returns:
        obj: distance calculation obj, params have been calculated and set by the input data
    """
    dist = self.inner(inputs)
    assert tuple(dist.batch_shape) == tuple(inputs.shape[:-1]), (
        dist.batch_shape, dist.event_shape, inputs.shape)
    return dist

  def inner(self, inputs):
    kw = {}
    kw['outscale'] = self._outscale
    kw['outnorm'] = self._outnorm
    shape = self._shape 
    if self._dist.endswith('_disc'):   # for discrete regression
      shape = (*self._shape, self._bins)  # feature/event_dims + bins_dim
    # 1-- to transform the input to a distribution parameterization by applying a linear layer
    # 2-- or to get an output for distance calculation
    out = self.get('out', Linear, int(np.prod(shape)), **kw)(inputs) # inputs (...,X) ---> out (...,np.prod(shape))
    out = out.reshape(inputs.shape[:-1] + shape).astype(f32)   # out (...,np.prod(shape))---> out (...,shape) shape may be (Z,) or (Z,Y)...
    if self._dist in ('normal', 'trunc_normal'):
      std = self.get('std', Linear, int(np.prod(self._shape)), **kw)(inputs) # this line get the std params output, previous line get the mean params output
      std = std.reshape(inputs.shape[:-1] + self._shape).astype(f32)
    if self._dist == 'symlog_mse':  # mse over symlogged values
      return jaxutils.SymlogDist(out, len(self._shape), 'mse', 'sum') # use the last dim of out to compare the symlog distance
    if self._dist == 'symlog_disc':    # cross entropy over two-hot encoded symlogged values
      return jaxutils.DiscDist(
          out, len(self._shape), -20, 20, jaxutils.symlog, jaxutils.symexp)
    if self._dist == 'mse':
      return jaxutils.MSEDist(out, len(self._shape), 'sum')
    if self._dist == 'normal':
      lo, hi = self._minstd, self._maxstd
      std = (hi - lo) * jax.nn.sigmoid(std + 2.0) + lo  # mapping the std to [lo,hi]
      dist = tfd.Normal(jnp.tanh(out), std)  # build a batched scalar normal distribution with mean and std
      dist = tfd.Independent(dist, len(self._shape)) # specify the last dim as part of the event space rather than separate instances
      # get the min and max entropy of all the normal distribution
      # the total entropy for a joint distribution consisting of multiple independent normal variables
      dist.minent = np.prod(self._shape) * tfd.Normal(0.0, lo).entropy()
      dist.maxent = np.prod(self._shape) * tfd.Normal(0.0, hi).entropy()
      return dist
    if self._dist == 'binary':
      dist = tfd.Bernoulli(out)
      return tfd.Independent(dist, len(self._shape))
    if self._dist == 'onehot':
      # here likely used for actor output
      # trick of 1% uniform + 99% NN output ---> correspond to Unimix categoricals in page 19, Section C: Summary of Differences
      if self._unimix:
        probs = jax.nn.softmax(out, -1)
        uniform = jnp.ones_like(probs) / probs.shape[-1]
        probs = (1 - self._unimix) * probs + self._unimix * uniform
        out = jnp.log(probs)
      dist = jaxutils.OneHotDist(out)
      if len(self._shape) > 1:  # if it is multi-dim action, add the rightmost batch dim to the event space for convenient sampling
        dist = tfd.Independent(dist, len(self._shape) - 1)
      dist.minent = 0.0
      # Here self._shape[:-1] could be 2 if the discrete action is 2D, and self._shape[-1] is the number of classes for each action dim
      dist.maxent = np.prod(self._shape[:-1]) * jnp.log(self._shape[-1]) # max entropy of one-hot distributions (when all probs are equal)
      return dist
    raise NotImplementedError(self._dist)


class Conv2D(nj.Module):

  def __init__(
      self, depth, kernel, stride=1, transp=False, act='none', norm='none',
      pad='same', bias=True, preact=False, winit='uniform', fan='avg'):
    """ convolutional layer wrapper

    Args:
        depth (int): output channel dim
        kernel (int): kernel size
        stride (int, optional): stride size. Defaults to 1.
        transp (bool, optional): whether to use transposed convolution (up-sampling). Defaults to False.
        act (str, optional): name of activation function. Defaults to 'none'.
        norm (str, optional): name of normalization function. Defaults to 'none'.
        pad (str, optional): conv padding method. Defaults to 'same'.
        bias (bool, optional): whether add bias to conv output. Defaults to True.
        preact (bool, optional): whether to apply normalization and activation before convolution. Defaults to False.
        winit (str, optional): distribution for weight initialization, 'uniform' or 'normal' or 'orthogonal'. Defaults to 'uniform'.
        fan (str, optional): for weight initialization, 'avg' or 'in' or 'out'. Defaults to 'avg'.
    """
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
    """convolutional layer

    Args:
        x (array): input tensor/image with shape (N,H,W,C)

    Returns:
        array: tensor with shape (N,H,W,C), if self._pad='same', the output H,W is the same as input shape if stride=1; if stride=2, the output H,W is half of the input shape (special: odd number, output odd//2+1)
    """

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
          dimension_numbers=('NHWC', 'HWIO', 'NHWC')) # input/output dim: (N,H,W,C), kernel dim: (H,W,I,O)
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
      x = x.reshape(x.shape[:-1] + self._units) # x:(batch_size, temp_out_features)--->(batch_size, unit1, unit2,...), e.g. in image decoder, reshape flattened vector to image shape with H,W,C
    x = self.get('norm', Norm, self._norm)(x)
    x = self._act(x)
    return x


class Norm(nj.Module):

  def __init__(self, impl):
    """normalization for the NNs, including None, LayerNorm
    Layer normalization is a technique used to standardize the inputs across features for each data sample individually, which can help stabilize the learning process 

    Args:
        impl (str): which normalization method to use, 'none' or 'layer'
    """
    self._impl = impl

  def __call__(self, x):
    """forward pass of the normalization method, for image input and layer norm, it applies to the last dim (Channel) dim, dtype temporarily converted to f32 during the normalization process
    layer norm makes the output to have zero mean and unit variance
    Args:
        x (array): input to be normalized

    Raises:
        NotImplementedError: only 'none' and 'layer' normalization methods are implemented

    Returns:
        array: normalized output, dtype is the same as the input
    """
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
    """speicify required keys and desired dim-reshaping target (or given a specific key and use its value dim as target, default is the first key in key lists) for input values
    Then it will be make sure all the inputs dim numbers are not exceeding the target dim.

    Args:
        keys (list, optional): specified data keys. Defaults to ['tensor'].
        dims (str, optional): desired dim-reshaping target key. Defaults to None.
    """
    assert isinstance(keys, (list, tuple)), keys
    self._keys = tuple(keys)
    self._dims = dims or self._keys[0]

  def __call__(self, inputs):
    """check whether keys missing in the input dict, apply softmax if specified in key and reshape the value to the target (dims) if exceeded the target dim number,
    at last concat all required values by self._keys together

    Args:
        inputs (dict): input data

    Raises:
        KeyError: all keys in self._keys (specified data keys) must in the input dict

    Returns:
        array: concat all required values by self._keys along last dim together, the batch dims should be same as the input data, otherwise concat will raise error
    """
    if not isinstance(inputs, dict):
      inputs = {'tensor': inputs}
    inputs = inputs.copy()
    for key in self._keys:
      if key.startswith('softmax_'):
        inputs[key] = jax.nn.softmax(inputs[key[len('softmax_'):]])  # do softmax upon the corresponding value to the rest of keywords after 'softmax_' in input dict
    if not all(k in inputs for k in self._keys):  # check all keys in self._keys are in the input dict
      needs = f'{{{", ".join(self._keys)}}}'
      found = f'{{{", ".join(inputs.keys())}}}'
      raise KeyError(f'Cannot find keys {needs} among inputs {found}.')
    values = [inputs[k] for k in self._keys]
    dims = len(inputs[self._dims].shape)
    for i, value in enumerate(values):
      if len(value.shape) > dims:
        values[i] = value.reshape(
            value.shape[:dims - 1] + (np.prod(value.shape[dims - 1:]),)) # reshape the value to match the num of dims, flatten the tails
    values = [x.astype(inputs[self._dims].dtype) for x in values]
    return jnp.concatenate(values, -1)


class Initializer:

  def __init__(self, dist='uniform', scale=1.0, fan='avg'):
    self.scale = scale
    self.dist = dist
    self.fan = fan  #init weight based on 'average of fanin and fanout' or 'fanin' or 'fanout'

  def __call__(self, shape):
    """calculate the initial value of the kernel (weights) based on the shape of the kernel and the distribution type
    
    Args:
        shape (tuple): shape of the kernel (weights)

    Raises:
        NotImplementedError: the distribution type must be one of 'uniform', 'normal', 'ortho'

    Returns:
        array: initial value of the kernel (weights)
    """
    if self.scale == 0.0:
      value = jnp.zeros(shape, f32)
    elif self.dist == 'uniform':   # like a uniform version of He initialization if use avg
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
    """calculate the "fan-in" and "fan-out" values for a given tensor shape, which are commonly used for initializing weights in neural networks. 
    These values help define the scale of weight initialization, ensuring appropriate variance across layers 
    y = X @ W, each row of X is a sample

    Args:
        shape (tuple): shape of the kernel (weights)

    Returns:
        tuple: fan_in (input dim), fan_out (output dim)
    """
    if len(shape) == 0:
      return 1, 1
    elif len(shape) == 1:
      return shape[0], shape[0]
    elif len(shape) == 2:
      return shape
    else:
      space = int(np.prod(shape[:-2])) # think of as the batch size
      return shape[-2] * space, shape[-1] * space # for kernel (H,W,I,O), fan_in=H*W*I, fan_out=H*W*O


def get_act(name):
  """get the activation function by name 

  Args:
      name (str): activation function name

  Raises:
      NotImplementedError: only 'none', 'mish' and functions in jax.nn are implemented

  Returns:
      fn: normalization function itself
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
