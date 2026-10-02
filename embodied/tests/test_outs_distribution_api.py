"""Distribution APIs preserve joint event likelihoods and sample shapes."""
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from embodied.jax import outs


@pytest.mark.parametrize('event_shape', [(2,), (2, 3)])
def test_aggregated_binary_probability_is_joint_likelihood(event_shape):
  output = outs.Agg(outs.Binary(jnp.zeros((4,) + event_shape)), len(event_shape))
  event = jnp.ones((4,) + event_shape)
  expected = np.full(4, 0.5 ** math.prod(event_shape))
  np.testing.assert_allclose(output.prob(event), expected, rtol=1e-6)
  np.testing.assert_allclose(output.prob(event), jnp.exp(output.logp(event)))


def test_aggregated_binary_probability_with_unequal_probabilities():
  probabilities = jnp.array([0.2, 0.7])
  logits = jnp.log(probabilities / (1 - probabilities))
  output = outs.Agg(outs.Binary(logits), 1)
  event = jnp.array([[1., 0.], [0., 1.]])
  np.testing.assert_allclose(output.prob(event), [0.06, 0.56], rtol=1e-6)


def test_aggregated_categorical_probability_reduces_only_event_axes():
  output = outs.Agg(outs.Categorical(jnp.zeros((2, 3, 4))), 1)
  event = jnp.array([[0, 1, 2], [3, 2, 1]])
  np.testing.assert_allclose(output.prob(event), np.full(2, 1 / 64), rtol=1e-6)


@pytest.mark.parametrize('stddev', [(1., 2.), (0.1, 0.1)])
def test_aggregated_normal_probability_is_joint_density(stddev):
  output = outs.Agg(outs.Normal(jnp.zeros((3, 2)), jnp.array(stddev)), 1)
  expected = 1 / (2 * math.pi * math.prod(stddev))
  np.testing.assert_allclose(output.prob(jnp.zeros((3, 2))), expected, rtol=1e-6)


def test_loss_aggregation_does_not_change_joint_probability():
  output = outs.Agg(outs.Binary(jnp.zeros((3, 2))), 1, agg=jnp.mean)
  event = jnp.ones((3, 2))
  np.testing.assert_allclose(output.loss(event), math.log(2), rtol=1e-6)
  np.testing.assert_allclose(output.prob(event), 0.25, rtol=1e-6)


def test_zero_event_dimensions_preserve_elementwise_probability():
  binary = outs.Binary(jnp.array([[0., 1.], [-1., 2.]]))
  event = jnp.array([[1., 0.], [0., 1.]])
  np.testing.assert_allclose(outs.Agg(binary, 0).prob(event), binary.prob(event))


def test_joint_probability_is_jittable_and_has_analytic_gradient():
  logits = jnp.array([math.log(0.2 / 0.8), math.log(0.7 / 0.3)])
  event = jnp.array([1., 0.])
  probability = lambda logit: outs.Agg(outs.Binary(logit), 1).prob(event)
  np.testing.assert_allclose(jax.jit(probability)(logits), 0.06, rtol=1e-6)
  np.testing.assert_allclose(
      jax.jit(jax.grad(probability))(logits),
      0.06 * (np.array([1., 0.]) - np.array([0.2, 0.7])), rtol=1e-6)


@pytest.mark.parametrize('batch_shape,sample_shape', [
    ((), ()), ((), (7,)), ((2, 3), ()), ((2, 3), (5, 7))])
def test_binary_sample_preserves_requested_and_batch_shapes(batch_shape, sample_shape):
  logits = jnp.full(batch_shape, 0.8)
  key = jax.random.PRNGKey(7)
  actual = outs.Binary(logits).sample(key, sample_shape)
  expected = jax.random.bernoulli(
      key, jax.nn.sigmoid(logits), shape=sample_shape + batch_shape)
  assert actual.shape == sample_shape + batch_shape
  assert actual.dtype == jnp.bool_
  np.testing.assert_array_equal(actual, expected)


def test_binary_sample_honors_deterministic_extreme_logits():
  logits = jnp.array([-jnp.inf, jnp.inf])
  samples = outs.Binary(logits).sample(jax.random.PRNGKey(4), (20,))
  np.testing.assert_array_equal(samples, np.tile([False, True], (20, 1)))


def test_binary_sample_is_jittable_and_aggregation_preserves_event_shape():
  key = jax.random.PRNGKey(3)
  logits = jnp.array([[0., 1.], [-1., 2.]])
  sample = jax.jit(lambda seed: outs.Agg(outs.Binary(logits), 1).sample(seed, (8,)))
  actual = sample(key)
  assert actual.shape == (8, 2, 2)
  np.testing.assert_array_equal(
      actual, jax.random.bernoulli(key, jax.nn.sigmoid(logits), shape=(8, 2, 2)))


def test_binary_sample_matches_bernoulli_frequencies():
  probabilities = jnp.array([0.2, 0.7])
  logits = jnp.log(probabilities / (1 - probabilities))
  samples = outs.Binary(logits).sample(jax.random.PRNGKey(12), (20000,))
  np.testing.assert_allclose(np.asarray(samples).mean(0), probabilities, atol=0.015)
