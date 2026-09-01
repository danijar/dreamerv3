import jax
import jax.numpy as jnp
import numpy as np
import pytest

from embodied.jax import outs


class TestOutputs:

  @pytest.mark.parametrize('sample_shape', [(), (4,), (2, 3)])
  def test_binary_sample_shape(self, sample_shape):
    output = outs.Binary(jnp.zeros((2, 3)))
    sample = output.sample(jax.random.PRNGKey(0), sample_shape)

    assert sample.shape == sample_shape + (2, 3)
    assert sample.dtype == jnp.bool_

  def test_aggregate_probability_is_joint_probability(self):
    output = outs.Agg(outs.Binary(jnp.zeros(2)), 1)
    event = jnp.array([True, False])

    probability = output.prob(event)

    np.testing.assert_allclose(probability, 0.25)
    np.testing.assert_allclose(probability, jnp.exp(output.logp(event)))
