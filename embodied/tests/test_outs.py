import jax
import jax.numpy as jnp
import numpy as np
import pytest

from embodied.jax import nets
from embodied.jax import outs

f32 = jnp.float32


def symexp_bins(nbins):
  # Same construction as the symexp_twohot head in embodied/jax/heads.py.
  if nbins % 2 == 1:
    half = nets.symexp(jnp.linspace(-20, 0, (nbins - 1) // 2 + 1, dtype=f32))
    return jnp.concatenate([half, -half[:-1][::-1]], 0)
  half = nets.symexp(jnp.linspace(-20, 0, nbins // 2, dtype=f32))
  return jnp.concatenate([half, -half[::-1]], 0)


class TestTwoHot:

  @pytest.mark.parametrize('nbins', [255, 256])
  @pytest.mark.parametrize('jit', [False, True])
  def test_pred_is_zero_for_uniform_probabilities(self, nbins, jit):
    bins = symexp_bins(nbins)
    pred = lambda logits: outs.TwoHot(logits, bins).pred()
    if jit:
      pred = jax.jit(pred)
    np.testing.assert_array_equal(pred(jnp.zeros((4, nbins), f32)), 0.0)

  @pytest.mark.parametrize('nbins', [255, 256])
  def test_pred_is_the_probability_weighted_bin_value(self, nbins):
    bins = symexp_bins(nbins)
    logits = jax.random.normal(jax.random.PRNGKey(0), (4, nbins), f32)
    pred = jax.jit(lambda logits: outs.TwoHot(logits, bins).pred())(logits)
    probs = np.asarray(jax.nn.softmax(logits), np.float64)
    terms = probs * np.asarray(bins, np.float64)
    # Each of the n float32 products and additions adds at most 2^-24 relative
    # error, so the result stays within n * 2^-24 * sum(|p * b|).
    atol = nbins * 2.0 ** -24 * np.abs(terms).sum(-1)
    np.testing.assert_array_less(np.abs(pred - terms.sum(-1)), atol)
