import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import optax
import pytest

from dreamerv3.agent import Agent
from embodied.jax import opt


def updates_for(transform, gradients, compiled):
  params = {'model/kernel': jnp.array([1.0, -2.0], jnp.float32)}
  state = transform.init(params)
  update = jax.jit(transform.update) if compiled else transform.update
  result = []
  for gradient in gradients:
    updates, state = update(
        {'model/kernel': jnp.asarray(gradient, jnp.float32)}, state, params)
    result.append(np.asarray(updates['model/kernel']))
    params = optax.apply_updates(params, updates)
  return np.stack(result)


@pytest.mark.parametrize('compiled', [False, True])
@pytest.mark.parametrize('beta1', [0.0, 0.6])
@pytest.mark.parametrize('nesterov', [False, True])
def test_disabled_momentum_matches_rms_updates(compiled, beta1, nesterov):
  gradients = np.array([[1.0, -2.0], [-3.0, 4.0], [2.0, 1.0]])
  lr, beta2, eps = 0.1, 0.75, 1e-8
  transform = Agent._make_opt(
      None, lr=lr, agc=0, eps=eps, beta1=beta1, beta2=beta2,
      momentum=False, nesterov=nesterov, warmup=0)
  actual = updates_for(transform, gradients, compiled)

  # Independent recurrence without first-moment smoothing.
  square = np.zeros(2)
  expected = []
  for step, gradient in enumerate(gradients, 1):
    square = beta2 * square + (1 - beta2) * gradient ** 2
    corrected = square / (1 - beta2 ** step)
    expected.append(-lr * gradient / (np.sqrt(corrected) + eps))
  np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=1e-7)


@pytest.mark.parametrize('compiled', [False, True])
@pytest.mark.parametrize('nesterov', [False, True])
def test_enabled_momentum_preserves_existing_transform(compiled, nesterov):
  gradients = np.array([[1.0, -2.0], [-3.0, 4.0], [2.0, 1.0]])
  settings = dict(
      lr=0.1, agc=0.3, eps=1e-8, beta1=0.6, beta2=0.75,
      nesterov=nesterov, warmup=0)
  actual = Agent._make_opt(None, **settings)
  expected = optax.chain(
      opt.clip_by_agc(settings['agc']),
      opt.scale_by_rms(settings['beta2'], settings['eps']),
      opt.scale_by_momentum(settings['beta1'], nesterov),
      optax.scale_by_learning_rate(optax.constant_schedule(settings['lr'])))
  np.testing.assert_array_equal(
      updates_for(actual, gradients, compiled),
      updates_for(expected, gradients, compiled))


@pytest.mark.parametrize('momentum', [False, True])
def test_native_optimizer_steps_match_recurrence(momentum):
  class Parameter(nj.Module):

    def __call__(self):
      return self.value('kernel', lambda: jnp.array([1.0, -2.0]))

  model = Parameter(name='model')
  optimizer = opt.Optimizer(model, Agent._make_opt(
      None, lr=0.1, agc=0, eps=1e-8, beta1=0.6, beta2=0.75,
      momentum=momentum, warmup=0), name='opt')

  def step():
    metrics = optimizer(lambda: jnp.square(model()).sum())
    return model(), metrics

  state = nj.init(step)({}, seed=0)
  parameter = np.array([1.0, -2.0])
  square, first = np.zeros(2), np.zeros(2)
  for count in range(1, 5):
    gradient = 2 * parameter
    square = 0.75 * square + 0.25 * gradient ** 2
    scaled = gradient / (np.sqrt(square / (1 - 0.75 ** count)) + 1e-8)
    if momentum:
      first = 0.6 * first + 0.4 * scaled
      scaled = first / (1 - 0.6 ** count)
    parameter = parameter - 0.1 * scaled
    state, (actual, metrics) = nj.pure(step)(state, seed=0)
    np.testing.assert_allclose(actual, parameter, rtol=3e-6, atol=1e-7)
    assert metrics['opt/updates'] == count
