import embodied
import jax
import jax.numpy as jnp
import ruamel.yaml as yaml
from pathlib import Path
tree_map = jax.tree_util.tree_map
sg = lambda x: tree_map(jax.lax.stop_gradient, x)

import logging
logger = logging.getLogger()
class CheckTypesFilter(logging.Filter):
  def filter(self, record):
    return 'check_types' not in record.getMessage()
logger.addFilter(CheckTypesFilter())

from . import behaviors
from . import jaxagent
from . import jaxutils
from . import nets
from . import ninjax as nj


@jaxagent.Wrapper
class Agent(nj.Module):

  configs = yaml.YAML(typ='safe').load(
      (Path(__file__).parent / 'configs.yaml'))

  def __init__(self, obs_space, act_space, step, config):
    self.config = config
    self.obs_space = obs_space
    try:
      self.act_space = act_space['action']
    except KeyError:
      self.act_space = act_space
    self.step = step
    self.wm = WorldModel(obs_space, act_space, config, name='wm')
    self.task_behavior = getattr(behaviors, config.task_behavior)(
        self.wm, self.act_space, self.config, name='task_behavior')
    if config.expl_behavior == 'None':
      self.expl_behavior = self.task_behavior
    else:
      self.expl_behavior = getattr(behaviors, config.expl_behavior)(
          self.wm, self.act_space, self.config, name='expl_behavior')

  def policy_initial(self, batch_size):
    return (
        self.wm.initial(batch_size),
        self.task_behavior.initial(batch_size),
        self.expl_behavior.initial(batch_size))

  def train_initial(self, batch_size):
    return self.wm.initial(batch_size)

  def policy(self, obs, state, mode='train'):
    self.config.jax.jit and print('Tracing policy function.')
    obs = self.preprocess(obs)
    (prev_latent, prev_action), task_state, expl_state = state
    embed = self.wm.encoder(obs)
    latent, _ = self.wm.rssm.obs_step(
        prev_latent, prev_action, embed, obs['is_first'])
    self.expl_behavior.policy(latent, expl_state)
    task_outs, task_state = self.task_behavior.policy(latent, task_state)
    expl_outs, expl_state = self.expl_behavior.policy(latent, expl_state)
    if mode == 'eval':
      outs = task_outs
      if type(self.act_space) == dict:
        outs['action'] = {k: w.mode() for k, w in outs['action'].items()}
        outs['log_entropy'] = {k: w.entropy() for k, w in outs['action'].items()}
      else:
        outs['action'] = outs['action'].sample(seed=nj.rng())
        outs['log_entropy'] = jnp.zeros(outs['action'].shape[:1])
    elif mode == 'explore':
      outs = expl_outs
      if type(self.act_space) == dict:
        outs['log_entropy'] = {k: w.entropy() for k, w in outs['action'].items()}
        outs['action'] = {k: w.sample(seed=nj.rng()) for k, w in outs['action'].items()}
      else:
        outs['log_entropy'] = outs['action'].entropy()
        outs['action'] = outs['action'].sample(seed=nj.rng())
    elif mode == 'train':
      outs = task_outs
      if type(self.act_space) == dict:
        outs['log_entropy'] = {k: w.entropy() for k, w in outs['action'].items()}
        outs['action'] = {k: w.sample(seed=nj.rng()) for k, w in outs['action'].items()}
      else:
        outs['log_entropy'] = outs['action'].entropy()
        outs['action'] = outs['action'].sample(seed=nj.rng())
    state = ((latent, outs['action']), task_state, expl_state)
    return outs, state

  def train(self, data, state):
    self.config.jax.jit and print('Tracing train function.')
    metrics = {}
    data = self.preprocess(data)
    state, wm_outs, mets = self.wm.train(data, state)
    metrics.update(mets)
    context = {**data, **wm_outs['post']}
    start = tree_map(lambda x: x.reshape([-1] + list(x.shape[2:])), context)
    _, mets = self.task_behavior.train(self.wm.imagine, start, context)
    metrics.update(mets)
    if self.config.expl_behavior != 'None':
      _, mets = self.expl_behavior.train(self.wm.imagine, start, context)
      metrics.update({'expl_' + key: value for key, value in mets.items()})
    outs = {}
    return outs, state, metrics

  def report(self, data):
    self.config.jax.jit and print('Tracing report function.')
    data = self.preprocess(data)
    report = {}
    report.update(self.wm.report(data))
    mets = self.task_behavior.report(data)
    report.update({f'task_{k}': v for k, v in mets.items()})
    if self.expl_behavior is not self.task_behavior:
      mets = self.expl_behavior.report(data)
      report.update({f'expl_{k}': v for k, v in mets.items()})
    return report

  def preprocess(self, obs):
    obs = obs.copy()
    for key, value in obs.items():
      if key.startswith('log_') or key in ('key',):
        continue
      if len(value.shape) > 3 and value.dtype == jnp.uint8:
        value = jaxutils.cast_to_compute(value) / 255.0
      else:
        value = value.astype(jnp.float32)
      obs[key] = value
    obs['cont'] = 1.0 - obs['is_terminal'].astype(jnp.float32)
    return obs


class WorldModel(nj.Module):

  def __init__(self, obs_space, act_space, config):
    self.obs_space = obs_space
    try:
      self.act_space_shape = act_space['action'].shape
    except KeyError:
      self.act_space_shape = {k: v.shape for k, v in act_space.items() if k != 'reset'}
      #tuple(a+b for a, b in zip(act_space['Continous'].shape, act_space['Discrete'].shape))
    self.config = config
    shapes = {k: tuple(v.shape) for k, v in obs_space.items()}
    shapes = {k: v for k, v in shapes.items() if not k.startswith('log_')}
    self.encoder = nets.MultiEncoder(shapes, **config.encoder, name='enc')
    self.rssm = nets.RSSM(**config.rssm, name='rssm')
    self.heads = {
        'decoder': nets.MultiDecoder(shapes, **config.decoder, name='dec'),
        'reward': nets.MLP((), **config.reward_head, name='rew'),
        'cont': nets.MLP((), **config.cont_head, name='cont')}
    self.opt = jaxutils.Optimizer(name='model_opt', **config.model_opt)
    scales = self.config.loss_scales.copy()
    image, vector = scales.pop('image'), scales.pop('vector')
    scales.update({k: image for k in self.heads['decoder'].cnn_shapes})
    scales.update({k: vector for k in self.heads['decoder'].mlp_shapes})
    self.scales = scales

  def initial(self, batch_size):
    prev_latent = self.rssm.initial(batch_size)
    if type(self.act_space_shape) == dict:
      prev_action = {k: jnp.zeros((batch_size, *v)) for k, v in self.act_space_shape.items()}
    else:
      prev_action = jnp.zeros((batch_size, *self.act_space_shape))
    return prev_latent, prev_action

  def train(self, data, state):
    modules = [self.encoder, self.rssm, *self.heads.values()]
    if 'action' not in data:
      data['action'] = {"Continous": data['Continous'], "Discrete": data['Discrete']}
      data.pop('Continous')
      data.pop('Discrete')
    mets, (state, outs, metrics) = self.opt(
        modules, self.loss, data, state, has_aux=True)
    metrics.update(mets)
    return state, outs, metrics

  def loss(self, data, state):
    embed = self.encoder(data)
    prev_latent, prev_action = state
    # Shape state: (prev_latent, prev_action), action can be a Dict if hybrid (Discrete + Continous)
    if 'action' not in data:
      if len(data['Continous'].shape) == 2:
        data['Continous'] = data['Continous'][..., None]
      if len(data['Discrete'].shape) == 2:
        data['Discrete'] = data['Discrete'][..., None]
      
      data['action'] = {"Continous": data['Continous'], "Discrete": data['Discrete']}
      data.pop('Continous')
      data.pop('Discrete')
    
    if isinstance(data['action'], dict):
      prev_actions = {
        k: jnp.concatenate([prev_action[k][:, None], 
                            data['action'][k][:, :-1]], 1) for k in data['action']}
    else:
      prev_actions = jnp.concatenate([
        prev_action[:, None], data['action'][:, :-1]], 1)
    post, prior = self.rssm.observe(
        embed, prev_actions, data['is_first'], prev_latent)
    dists = {}
    feats = {**post, 'embed': embed}
    for name, head in self.heads.items():
      out = head(feats if name in self.config.grad_heads else sg(feats))
      out = out if isinstance(out, dict) else {name: out}
      dists.update(out)
    losses = {}
    losses['dyn'] = self.rssm.dyn_loss(post, prior, **self.config.dyn_loss)
    losses['rep'] = self.rssm.rep_loss(post, prior, **self.config.rep_loss)
    for key, dist in dists.items():
      loss = -dist.log_prob(data[key].astype(jnp.float32))
      assert loss.shape == embed.shape[:2], (key, loss.shape)
      losses[key] = loss
    scaled = {k: v * self.scales[k] for k, v in losses.items()}
    model_loss = sum(scaled.values())
    out = {'embed':  embed, 'post': post, 'prior': prior}
    out.update({f'{k}_loss': v for k, v in losses.items()})
    last_latent = {k: v[:, -1] for k, v in post.items()}
    if isinstance(data['action'], dict):
      last_action = {k: v[:, -1] for k, v in data['action'].items()}
    else:
      last_action = data['action'][:, -1]
    state = last_latent, last_action
    metrics = self._metrics(data, dists, post, prior, losses, model_loss)
    return model_loss.mean(), (state, out, metrics)

  def imagine(self, policy, start, horizon):
    first_cont = (1.0 - start['is_terminal']).astype(jnp.float32)
    keys = list(self.rssm.initial(1).keys())
    start = {k: v for k, v in start.items() if k in keys}
    start['action'] = policy(start)
    def step(prev, _):
      prev = prev.copy()
      state = self.rssm.img_step(prev, prev.pop('action'))
      return {**state, 'action': policy(state)}
    traj = jaxutils.scan(
        step, jnp.arange(horizon), start, self.config.imag_unroll)
    # traj = 
    traj_ = {
        k: jnp.concatenate([start[k][None], v], 0) for k, v in traj.items() if k != "action"}
    Continous = jnp.concatenate([start["action"]["Continous"][None], traj["action"]["Continous"]], 0)
    Discrete = jnp.concatenate([start["action"]["Discrete"][None], traj["action"]["Discrete"]], 0)
    traj_["action"] = {"Continous": Continous, "Discrete": Discrete}
    traj = traj_
    cont = self.heads['cont'](traj).mode()
    traj['cont'] = jnp.concatenate([first_cont[None], cont[1:]], 0)
    discount = 1 - 1 / self.config.horizon
    traj['weight'] = jnp.cumprod(discount * traj['cont'], 0) / discount
    return traj

  def report(self, data):
    state = self.initial(len(data['is_first']))
    report = {}
    report.update(self.loss(data, state)[-1][-1])
    if isinstance(data['action'], dict):
      action_context, action_post = {}, {}
      for k, v in data['action'].items():
        action_context[k] = v[:6, :5]
        action_post[k] = v[:6, 5:]
    else:
      action_context = data['action'][:6, :5]
      action_post = data['action'][:6, 5:]
    context, _ = self.rssm.observe(
        self.encoder(data)[:6, :5], action_context,
        data['is_first'][:6, :5])
    start = {k: v[:, -1] for k, v in context.items()}
    recon = self.heads['decoder'](context)
    openl = self.heads['decoder'](
        self.rssm.imagine(action_post, start))
    for key in self.heads['decoder'].cnn_shapes.keys():
      truth = data[key][:6].astype(jnp.float32)
      model = jnp.concatenate([recon[key].mode()[:, :5], openl[key].mode()], 1)
      error = (model - truth + 1) / 2
      video = jnp.concatenate([truth, model, error], 2)
      report[f'openl_{key}'] = jaxutils.video_grid(video)
    return report

  def _metrics(self, data, dists, post, prior, losses, model_loss):
    entropy = lambda feat: self.rssm.get_dist(feat).entropy()
    metrics = {}
    metrics.update(jaxutils.tensorstats(entropy(prior), 'prior_ent'))
    metrics.update(jaxutils.tensorstats(entropy(post), 'post_ent'))
    metrics.update({f'{k}_loss_mean': v.mean() for k, v in losses.items()})
    metrics.update({f'{k}_loss_std': v.std() for k, v in losses.items()})
    metrics['model_loss_mean'] = model_loss.mean()
    metrics['model_loss_std'] = model_loss.std()
    metrics['reward_max_data'] = jnp.abs(data['reward']).max()
    metrics['reward_max_pred'] = jnp.abs(dists['reward'].mean()).max()
    if 'reward' in dists and not self.config.jax.debug_nans:
      stats = jaxutils.balance_stats(dists['reward'], data['reward'], 0.1)
      metrics.update({f'reward_{k}': v for k, v in stats.items()})
    if 'cont' in dists and not self.config.jax.debug_nans:
      stats = jaxutils.balance_stats(dists['cont'], data['cont'], 0.5)
      metrics.update({f'cont_{k}': v for k, v in stats.items()})
    return metrics


class ImagActorCritic(nj.Module):

  def __init__(self, critics, scales, act_space, config):
    critics = {k: v for k, v in critics.items() if scales[k]}
    for key, scale in scales.items():
      assert not scale or key in critics, key
    self.critics = {k: v for k, v in critics.items() if scales[k]}
    self.scales = scales
    self.act_space = act_space
    self.config = config
    if type(self.act_space) == dict:
      shape = {k: v.shape for k, v in act_space.items() if k != 'reset'} 
      Discrete = False
      self.grad_disc = config.actor_grad_disc
      self.grad_cont = config.actor_grad_cont
      self.grad = False
    else:
      shape = act_space.shape
      Discrete = act_space.discrete
      self.grad = config.actor_grad_disc if Discrete else config.actor_grad_cont
      
    self.actor = nets.MLP(
        name='actor', dims='deter', shape=shape, **config.actor,
        dist=config.actor_dist_disc if Discrete else config.actor_dist_cont,
        dist_cont=config.actor_dist_cont if not Discrete else None,
        dist_disc=config.actor_dist_disc if not Discrete else None)
    self.retnorms = {
        k: jaxutils.Moments(**config.retnorm, name=f'retnorm_{k}')
        for k in critics}
    self.opt = jaxutils.Optimizer(name='actor_opt', **config.actor_opt)

  def initial(self, batch_size):
    return {}

  def policy(self, state, carry):
    return {'action': self.actor(state)}, carry

  def train(self, imagine, start, context):
    def loss(start):
      if type(self.act_space) == dict:
        policy = lambda s: {k: w.sample(seed=nj.rng()) for k, w in self.actor(s).items()}
      else:         
        policy = lambda s: self.actor(sg(s)).sample(seed=nj.rng())
      traj = imagine(policy, start, self.config.imag_horizon)
      loss, metrics = self.loss(traj)
      return loss, (traj, metrics)
    mets, (traj, metrics) = self.opt(self.actor, loss, start, has_aux=True)
    metrics.update(mets)
    for key, critic in self.critics.items():
      mets = critic.train(traj, self.actor)
      metrics.update({f'{key}_critic_{k}': v for k, v in mets.items()})
    return traj, metrics

  def loss(self, traj):
    metrics = {}
    advs = []
    total = sum(self.scales[k] for k in self.critics)
    for key, critic in self.critics.items():
      rew, ret, base = critic.score(traj, self.actor)
      offset, invscale = self.retnorms[key](ret)
      normed_ret = (ret - offset) / invscale
      normed_base = (base - offset) / invscale
      advs.append((normed_ret - normed_base) * self.scales[key] / total)
      metrics.update(jaxutils.tensorstats(rew, f'{key}_reward'))
      metrics.update(jaxutils.tensorstats(ret, f'{key}_return_raw'))
      metrics.update(jaxutils.tensorstats(normed_ret, f'{key}_return_normed'))
      metrics[f'{key}_return_rate'] = (jnp.abs(ret) >= 0.5).mean()
    adv = jnp.stack(advs).sum(0)
    policy = self.actor(sg(traj))
    if type(self.act_space) == dict:
      logpi = {k: w.log_prob(sg(traj['action'][k]))[:-1] for k, w in policy.items()}
      logpi = sum(logpi.values())
    else:
      logpi = policy.log_prob(sg(traj['action']))[:-1]
    loss = {'backprop': -adv, 'reinforce': -logpi * sg(adv)}
    if self.grad:
      loss = loss[self.grad]
    else:
      loss = sum(loss.values())
    if type(self.act_space) == dict:
      ent = {k: w.entropy()[:-1] for k, w in policy.items()}
      ent = sum(ent.values())
    else:
      ent = policy.entropy()[:-1]
    loss -= self.config.actent * ent
    loss *= sg(traj['weight'])[:-1]
    loss *= self.config.loss_scales.actor
    metrics.update(self._metrics(traj, policy, logpi, ent, adv))
    return loss.mean(), metrics

  def _metrics(self, traj, policy, logpi, ent, adv):
    metrics = {}
    if type(self.act_space) == dict:
      ent = {k: w.entropy()[:-1] for k, w in policy.items()}
      rand = {k: (ent[k] - w.minent) / (w.maxent - w.minent) for k, w in policy.items()}
      act = {k: jnp.argmax(traj['action'][k], -1) for k in traj['action']}
      # act = jnp.concatenate([act[k] for k in traj['action']], -1)

      for k in traj['action']:
        metrics.update(jaxutils.tensorstats(act[k], f'action_{k}'))
      for k in rand:
        metrics.update(jaxutils.tensorstats(rand[k], f'policy_randomness_{k}'))
      for k in ent:
        metrics.update(jaxutils.tensorstats(ent[k], f'policy_entropy_{k}'))
      
    else:
      ent = policy.entropy()[:-1]
      rand = (ent - policy.minent) / (policy.maxent - policy.minent)
      rand = rand.mean(range(2, len(rand.shape)))
      act = traj['action']
      act = jnp.argmax(act, -1) if self.act_space.discrete else act

      metrics.update(jaxutils.tensorstats(act, 'action'))
      metrics.update(jaxutils.tensorstats(rand, 'policy_randomness'))
      metrics.update(jaxutils.tensorstats(ent, 'policy_entropy'))
    metrics.update(jaxutils.tensorstats(logpi, 'policy_logprob'))
    metrics.update(jaxutils.tensorstats(adv, 'adv'))
    metrics['imag_weight_dist'] = jaxutils.subsample(traj['weight'])
    return metrics


class VFunction(nj.Module):

  def __init__(self, rewfn, config):
    self.rewfn = rewfn
    self.config = config
    self.net = nets.MLP((), name='net', dims='deter', **self.config.critic)
    self.slow = nets.MLP((), name='slow', dims='deter', **self.config.critic)
    self.updater = jaxutils.SlowUpdater(
        self.net, self.slow,
        self.config.slow_critic_fraction,
        self.config.slow_critic_update)
    self.opt = jaxutils.Optimizer(name='critic_opt', **self.config.critic_opt)

  def train(self, traj, actor):
    target = sg(self.score(traj)[1])
    mets, metrics = self.opt(self.net, self.loss, traj, target, has_aux=True)
    metrics.update(mets)
    self.updater()
    return metrics

  def loss(self, traj, target):
    metrics = {}
    if type(traj['action']) == dict:
      action_continuous = traj['action']["Continous"]
      action_discrete = traj['action']["Discrete"]
      traj = {k: v[:-1] for k, v in traj.items() if k != 'action'}
      traj["action"] = {"Continous": action_continuous, "Discrete": action_discrete}
    else:
      traj = {k: v[:-1] for k, v in traj.items()}
    dist = self.net(traj)
    loss = -dist.log_prob(sg(target))
    if self.config.critic_slowreg == 'logprob':
      reg = -dist.log_prob(sg(self.slow(traj).mean()))
    elif self.config.critic_slowreg == 'xent':
      reg = -jnp.einsum(
          '...i,...i->...',
          sg(self.slow(traj).probs),
          jnp.log(dist.probs))
    else:
      raise NotImplementedError(self.config.critic_slowreg)
    loss += self.config.loss_scales.slowreg * reg
    loss = (loss * sg(traj['weight'])).mean()
    loss *= self.config.loss_scales.critic
    metrics = jaxutils.tensorstats(dist.mean())
    return loss, metrics

  def score(self, traj, actor=None):
    rew = self.rewfn(traj)
    if type(traj['action']) == dict:
      action_continuous = traj['action']["Continous"]
      action_discrete = traj['action']["Discrete"]
    else:
      assert len(rew) == len(traj['action']) - 1, (
          'should provide rewards for all but last action')
    discount = 1 - 1 / self.config.horizon
    disc = traj['cont'][1:] * discount
    value = self.net(traj).mean()
    vals = [value[-1]]
    interm = rew + disc * value[1:] * (1 - self.config.return_lambda)
    for t in reversed(range(len(disc))):
      vals.append(interm[t] + disc[t] * self.config.return_lambda * vals[-1])
    ret = jnp.stack(list(reversed(vals))[:-1])
    return rew, ret, value[:-1]
