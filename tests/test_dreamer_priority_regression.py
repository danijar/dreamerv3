"""Regression coverage: dreamer priority."""

import numpy as np
import pytest

def test_delayed_evicted_update_does_not_restore_metadata(load_case):
    s = load_case('dreamer_priority').Prioritized()
    s['live'] = [b'live']
    s['old'] = [b'old']
    del s['old']
    s.prioritize([b'old'], [99.0])
    assert dict(s.prios) == {b'live': 1.0}
    assert dict(s.stepitems) == {b'live': ['live']}
    assert s.tree.entries['live'].uprob == 1.0

def test_long_running_turnover_remains_bounded(load_case):
    s = load_case('dreamer_priority').Prioritized()
    s['live'] = [b'live']
    for i in range(1000):
        step = i.to_bytes(8, 'little')
        s[i] = [step]
        del s[i]
        s.prioritize([step], [float(i + 1)])
    assert len(s.prios) == len(s.stepitems) == 1
    assert len(s.tree) == 1

@pytest.mark.parametrize('exponent,maxfrac', [(1.0, 0.0), (0.5, 0.25), (2.0, 1.0)])
def test_live_and_shared_priorities_still_update(load_case, exponent, maxfrac):
    s = load_case('dreamer_priority').Prioritized(exponent=exponent, maxfrac=maxfrac)
    s['a'] = [b'shared', b'a']
    s['b'] = [b'shared', b'b']
    s.prioritize([b'missing', b'shared'], [100.0, 4.0])
    v = [4.0 ** exponent, 1.0]
    expected = maxfrac * max(v) + (1 - maxfrac) * sum(v) / 2
    assert s.tree.entries['a'].uprob == pytest.approx(expected)
    assert s.tree.entries['b'].uprob == pytest.approx(expected)
    assert b'missing' not in s.stepitems
    assert b'missing' not in s.prios
    del s['a']
    s.prioritize([b'shared'], [9.0])
    assert s.prios[b'shared'] == 9.0

def test_numpy_stepids_and_duplicate_updates(load_case):
    s = load_case('dreamer_priority').Prioritized()
    ids = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    s['a'] = ids[:1]
    s['b'] = ids[1:]
    del s['b']
    s.prioritize(np.array([[3, 4], [1, 2], [1, 2]], dtype=np.uint8), [99.0, 2.0, 3.0])
    assert dict(s.prios) == {ids[0].tobytes(): 3.0}
    assert set(s.stepitems) == {ids[0].tobytes()}

def test_zero_on_sample_still_updates_live_items(load_case):
    s = load_case('dreamer_priority').Prioritized(zero_on_sample=True)
    s['a'] = [b'a']
    s['b'] = [b'b']
    key = s()
    assert s.tree.entries[key].uprob == 0.0
    assert len(s.prios) == 2
import importlib.util
from pathlib import Path

@pytest.fixture
def load_case():
    source = Path(__file__).resolve().parents[1] / 'embodied/core/selectors.py'
    spec = importlib.util.spec_from_file_location('_regression_target', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return lambda _: module
