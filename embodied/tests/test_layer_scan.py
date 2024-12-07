import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np

from embodied.jax import utils

f32 = jnp.float32
i32 = jnp.int32



class Layer(nj.Module):

  units: int = 8

  def __call__(self, x, c, k):
    assert x.shape[1:] == (self.units,)
    assert c.shape == (7,)
    assert k.shape == (13, 7)
    shape = (x.shape[-1], self.units)
    winit = lambda: jax.random.normal(nj.seed(), shape, f32)
    x = x @ self.value('kernel', winit)
    if 'outer3' not in nj.context():
      nj.context()['outer3'] = jnp.zeros((), i32)
    nj.context()['outer3'] += 1
    nj.context()['outer1'] += 1
    inner = self.value('inner', jnp.array(0))
    self.write('inner', inner + nj.context()['outer2'])
    return x


class Net(nj.Module):

  layers: int = 4
  units: int = 8

  def __call__(self, x):
    if 'outer1' not in nj.context():
      nj.context()['outer1'] = jnp.ones((), i32)
    if 'outer2' not in nj.context():
      nj.context()['outer2'] = jnp.ones((), i32)
    nj.context()['outer1'] += 1

    module = self.sub('linear', Layer, units=self.units)
    c = jnp.zeros((self.layers, 7))
    k = jnp.zeros((13, 7))
    x = utils.LayerScan(module, self.layers)(x, c, k=k)

    return x

  def loss(self, x):
    return self(x).mean()


class TestLayerScan:

  def test_init(self, L=4, B=2, D=8):
    x = np.random.normal(0, 1, (B, D))
    net = Net(layers=L, units=D, name='net')
    params = nj.init(net)({}, x, seed=0)
    assert set(params.keys()) == {
        'outer1', 'outer2', 'outer3',
        'net/linear/kernel', 'net/linear/inner'}
    assert params['net/linear/kernel'].shape == (L, D, D)
    assert params['outer1'] == 1
    assert params['outer2'] == 1
    assert params['outer3'] == 0
    assert params['net/linear/inner'].shape == (L,)
    assert (params['net/linear/inner'] == 0).all()
    for i in range(1, L):
      assert not jnp.allclose(
          params['net/linear/kernel'][0],
          params['net/linear/kernel'][i])

  def test_apply(self, L=4, B=2, D=8):
    x = np.random.normal(0, 1, (B, D))
    net = Net(layers=L, units=D, name='net')
    params = nj.init(net)({}, x, seed=0)
    params, out = nj.pure(net)(params, x)
    assert out.shape == (B, D)
    assert params['outer1'] == L + 2
    assert params['outer2'] == 1
    assert params['outer3'] == L
    assert params['net/linear/inner'].shape == (L,)
    assert (params['net/linear/inner'] == 1).all()

  def test_grad(self, L=4, B=2, D=8):
    x = np.random.normal(0, 1, (B, D))
    net = Net(layers=L, units=D, name='net')
    def fn(x):
      if nj.creating():
        net(x)
      params = {k: v for k, v in net.values.items() if v.dtype == f32}
      params = {net.path + '/' + k: v for k, v in params.items()}
      loss, _, grads = nj.grad(lambda x: net(x).mean(), params.keys())(x)
      params = {k: v - 0.1 * grads[k] for k, v in params.items()}
      nj.context().update(params)
      return loss
    params = nj.init(net)({}, x, seed=0)
    params, loss = nj.pure(fn)(params, x)
    assert loss.shape == ()
