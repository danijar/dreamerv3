"""Gaussian entropy remains finite without squaring valid extreme scales."""
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from embodied.jax import outs


@pytest.mark.parametrize('scale', [1e-25, 1e-20, 0.5, 1., 2., 1e10, 1e20, 1e30])
def test_entropy_value_and_scale_gradient_match_analytic_normal(scale):
  scale = jnp.array(scale, jnp.float32)
  entropy = lambda stddev: outs.Normal(jnp.array(0.), stddev).entropy()
  actual, gradient = jax.jit(jax.value_and_grad(entropy))(scale)
  expected = math.log(float(scale)) + 0.5 * math.log(2 * math.pi * math.e)
  assert np.isfinite(actual)
  assert np.isfinite(gradient)
  np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=2e-6)
  np.testing.assert_allclose(gradient, 1 / float(scale), rtol=1e-6)


def test_batched_entropy_matches_native_torch_float64_oracle():
  scales = jnp.array([[1e-25, 0.2, 2.], [0.7, 1e20, 1e30]], jnp.float32)
  means = jnp.arange(6, dtype=jnp.float32).reshape(2, 3)
  actual = outs.Normal(means, scales).entropy()
  expected = torch.distributions.Normal(
      torch.tensor(np.asarray(means), dtype=torch.float64),
      torch.tensor(np.asarray(scales), dtype=torch.float64)).entropy()
  np.testing.assert_allclose(actual, expected.numpy(), rtol=1e-6, atol=2e-6)


def test_broadcast_scale_gradient_counts_all_independent_outputs():
  means = jnp.zeros((2, 3))
  entropy = lambda scale: outs.Normal(means, scale).entropy().sum()
  scale = jnp.array(1e20, jnp.float32)
  actual, gradient = jax.jit(jax.value_and_grad(entropy))(scale)
  expected = 6 * (math.log(float(scale)) + 0.5 * math.log(2 * math.pi * math.e))
  np.testing.assert_allclose(actual, expected, rtol=1e-6)
  np.testing.assert_allclose(gradient, 6 / float(scale), rtol=1e-6)


@pytest.mark.parametrize('aggregation', [jnp.sum, jnp.mean])
def test_aggregated_entropy_preserves_configured_event_reduction(aggregation):
  scales = jnp.array([1e-25, 1e20], jnp.float32)
  output = outs.Agg(outs.Normal(jnp.zeros((3, 2)), scales), 1, aggregation)
  expected_factors = np.log(np.asarray(scales, np.float64)) + 0.5 * math.log(2 * math.pi * math.e)
  expected = expected_factors.sum() if aggregation is jnp.sum else expected_factors.mean()
  actual = output.entropy()
  assert actual.shape == (3,)
  np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=2e-6)
