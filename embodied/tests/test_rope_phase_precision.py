"""RoPE evaluates positional phases before rounding to the activation dtype."""
import math

import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import pytest

from embodied.jax import nets


def rotation_oracle(x, positions, maxlen=4096, inverse=False):
  values = np.asarray(x, dtype=np.float64)
  dimension = values.shape[-1]
  frequencies = maxlen ** (2 * np.arange(dimension // 2) / dimension)
  angles = np.asarray(positions, dtype=np.float64)[..., None, None] / frequencies
  if inverse:
    angles = -angles
  first, second = np.split(values, 2, axis=-1)
  return np.concatenate([
      first * np.cos(angles) - second * np.sin(angles),
      second * np.cos(angles) + first * np.sin(angles)], axis=-1)


@pytest.mark.parametrize('dtype,tolerance', [
    (jnp.bfloat16, 0.015), (jnp.float16, 0.002), (jnp.float32, 2e-6)])
@pytest.mark.parametrize('inverse', [False, True])
def test_long_position_rotation_matches_independent_trigonometric_oracle(dtype, tolerance, inverse):
  positions = jnp.array([[255, 256, 257, 511, 512, 513, 2047, 2048, 2049, 10000]])
  x = jnp.ones((1, positions.shape[1], 2, 8), dtype)
  actual = nets.rope(x, positions, inverse=inverse)
  assert actual.dtype == dtype
  np.testing.assert_allclose(
      np.asarray(actual, dtype=np.float32), rotation_oracle(x, positions, inverse=inverse),
      atol=tolerance, rtol=0)


@pytest.mark.parametrize('dtype,position,tolerance', [
    (jnp.bfloat16, 256, 0.01), (jnp.float16, 2048, 0.002)])
def test_adjacent_positions_retain_unit_relative_phase(dtype, position, tolerance):
  x = jnp.array([[[[1., 0., 0., 0.]], [[1., 0., 0., 0.]]]], dtype)
  rotated = nets.rope(x, jnp.array([[position, position + 1]]))
  dot = jnp.vdot(rotated[0, 0], rotated[0, 1])
  np.testing.assert_allclose(float(dot), math.cos(1), atol=tolerance, rtol=0)


def test_default_positions_do_not_collapse_at_bfloat16_boundary():
  x = jnp.zeros((1, 259, 1, 4), jnp.bfloat16).at[..., 0].set(1)
  rotated = nets.rope(x)
  assert not np.array_equal(rotated[0, 256], rotated[0, 257])
  np.testing.assert_allclose(float(jnp.vdot(rotated[0, 256], rotated[0, 257])),
                             math.cos(1), atol=0.01, rtol=0)


def test_jitted_low_precision_rotation_matches_oracle():
  x = jnp.array([[[[1., 0., 0., 0.]], [[0., 1., 0., 0.]]]], jnp.bfloat16)
  positions = jnp.array([[10000, 10001]])
  actual = jax.jit(nets.rope)(x, positions)
  np.testing.assert_allclose(np.asarray(actual, dtype=np.float32),
                             rotation_oracle(x, positions), atol=0.008, rtol=0)


def test_rotation_input_gradient_uses_the_actual_position_phase():
  positions = jnp.array([[257]])
  x = jnp.zeros((1, 1, 1, 4), jnp.bfloat16)
  gradient = jax.grad(lambda value: nets.rope(value, positions)[..., 0].astype(jnp.float32).sum())(x)
  expected = np.array([math.cos(257), 0., -math.sin(257), 0.])
  np.testing.assert_allclose(np.asarray(gradient, dtype=np.float32).reshape(4),
                             expected, atol=0.004, rtol=0)


def test_native_attention_is_stable_under_a_common_position_offset():
  x = jax.random.normal(jax.random.PRNGKey(7), (1, 4, 8)).astype(jnp.bfloat16)
  attention = nets.Attention(heads=2, name='attention')
  low = jnp.arange(4)[None]
  high = low + 10000
  params = nj.init(attention)({}, x, ts=low, training=False, seed=3)
  _, first = nj.pure(attention)(params, x, ts=low, training=False)
  _, shifted = nj.pure(attention)(params, x, ts=high, training=False)
  np.testing.assert_allclose(np.asarray(first, dtype=np.float32),
                             np.asarray(shifted, dtype=np.float32), atol=0.035, rtol=0)


def test_float64_activations_retain_float64_phase_precision():
  with jax.experimental.enable_x64():
    x = jnp.ones((1, 2, 1, 4), jnp.float64)
    positions = jnp.array([[10000, 10001]])
    actual = nets.rope(x, positions)
    assert actual.dtype == jnp.float64
    np.testing.assert_allclose(actual, rotation_oracle(x, positions), atol=1e-12, rtol=0)
