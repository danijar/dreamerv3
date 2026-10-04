"""Explicit empty replay selectors must survive construction and receive data."""
import numpy as np
import pytest

from embodied.core import replay as replaylib
from embodied.core import selectors


@pytest.mark.parametrize('factory', [
    selectors.Fifo,
    selectors.Uniform,
    selectors.Prioritized,
    lambda: selectors.Recency(np.ones(8)),
])
def test_constructor_preserves_explicit_empty_selector(factory):
    selector = factory()
    assert len(selector) == 0
    replay = replaylib.Replay(length=1, capacity=8, selector=selector)
    assert replay.sampler is selector
    replay.add({'value': 7})
    assert len(selector) == 1


def test_prioritized_feedback_changes_actual_sampled_sequences():
    selector = selectors.Prioritized(initial=0., seed=4)
    replay = replaylib.Replay(length=1, capacity=8, selector=selector, chunksize=3)
    for value in range(8):
        replay.add({'value': value})

    chunkid, index = replay.items[3]
    stepids = replay._getseq(chunkid, index)['stepid']
    replay.update({'stepid': stepids[None], 'priority': np.ones((1, 1))})
    batch = replay.sample(32)
    np.testing.assert_array_equal(batch['value'], np.full((32, 1), 3))


def test_fifo_selector_controls_actual_sampling():
    selector = selectors.Fifo()
    replay = replaylib.Replay(length=2, capacity=8, selector=selector)
    for value in range(8):
        replay.add({'value': value})
    np.testing.assert_array_equal(replay.sample(16)['value'], np.tile([0, 1], (16, 1)))


def test_explicit_uniform_preserves_requested_rng_seed():
    first = replaylib.Replay(length=1, selector=selectors.Uniform(seed=72), seed=1)
    second = replaylib.Replay(length=1, selector=selectors.Uniform(seed=72), seed=99)
    for value in range(8):
        first.add({'value': value})
        second.add({'value': value})
    np.testing.assert_array_equal(first.sample(100)['value'], second.sample(100)['value'])


def test_none_keeps_default_uniform_selector():
    first = replaylib.Replay(length=1, seed=7)
    second = replaylib.Replay(length=1, seed=7, selector=None)
    assert isinstance(first.sampler, selectors.Uniform)
    assert isinstance(second.sampler, selectors.Uniform)
    for value in range(8):
        first.add({'value': value})
        second.add({'value': value})
    np.testing.assert_array_equal(first.sample(100)['value'], second.sample(100)['value'])
