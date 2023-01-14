import jax
import jax.numpy as jnp
tree_map = jax.tree_util.tree_map
sg = lambda x: tree_map(jax.lax.stop_gradient, x)

from . import nets
from . import jaxutils
from . import ninjax as nj


class Disag(nj.Module):

  def __init__(self, wm, act_space, config):
    self.config = config.update({'disag_head.inputs': ['tensor']})
    self.opt = jaxutils.Optimizer(name='disag_opt', **config.expl_opt)
    self.inputs = nets.Input(config.disag_head.inputs, dims='deter')
    self.target = nets.Input(self.config.disag_target, dims='deter')
    self.nets = [
        nets.MLP(shape=None, **self.config.disag_head, name=f'disag{i}')
        for i in range(self.config.disag_models)]

  def __call__(self, traj):
    inp = self.inputs(traj)
    preds = jnp.array([net(inp).mode() for net in self.nets])
    return preds.std(0).mean(-1)[1:]

  def train(self, data):
    return self.opt(self.nets, self.loss, data)

  def loss(self, data):
    inp = sg(self.inputs(data)[:, :-1])
    tar = sg(self.target(data)[:, 1:])
    losses = []
    for net in self.nets:
      net._shape = tar.shape[2:]
      losses.append(-net(inp).log_prob(tar).mean())
    return jnp.array(losses).sum()
