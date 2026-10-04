"""Regression coverage: dreamer recency."""

import numpy as np
import pytest

@pytest.mark.parametrize('size', [2, 16, 17, 257])
def test_sampling_returns_only_retained_keys(load_case, size):
    m = load_case('dreamer_recency')
    s = m.Recency(1 / np.arange(1, size + 1), seed=7)
    for i in range(size):
        s[i] = []
    for _ in range(100):
        assert 0 <= s() < size

@pytest.mark.parametrize('size', [16, 17, 257])
def test_zero_weight_branches_never_selected(load_case, size):
    m = load_case('dreamer_recency')
    probs = np.zeros(size)
    probs[0] = 1
    s = m.Recency(probs, seed=8)
    for i in range(size):
        s[i] = []
    assert [s() for _ in range(20)] == [size - 1] * 20

def test_seed_reproducibility_and_eviction(load_case):
    m = load_case('dreamer_recency')
    a = m.Recency(np.arange(17, 0, -1, dtype=float), seed=9)
    b = m.Recency(np.arange(17, 0, -1, dtype=float), seed=9)
    for i in range(50):
        a[i] = []
        b[i] = []
        if i >= 17:
            del a[i - 17]
            del b[i - 17]
    aa = [a() for _ in range(100)]
    bb = [b() for _ in range(100)]
    assert aa == bb
    assert all((33 <= x < 50 for x in aa))

def test_recency_distribution(load_case):
    m = load_case('dreamer_recency')
    s = m.Recency(np.array([4.0, 3.0, 2.0, 1.0]), seed=11)
    for i in range(4):
        s[i] = []
    counts = np.bincount([s() for _ in range(10000)], minlength=4) / 10000
    np.testing.assert_allclose(counts, [0.1, 0.2, 0.3, 0.4], atol=0.018)
import importlib.util
from pathlib import Path

@pytest.fixture
def load_case():
    source = Path(__file__).resolve().parents[1] / 'embodied/core/selectors.py'
    spec = importlib.util.spec_from_file_location('_regression_target', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return lambda _: module
