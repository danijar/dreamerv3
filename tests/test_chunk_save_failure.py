"""Exercise save-failure bookkeeping without changing chunk file contents."""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    '_chunk_under_test', Path(__file__).resolve().parents[1] / 'embodied/core/chunk.py')
target = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(target)
import numpy as np
import pytest

def make_chunk(module):
    chunk = object.__new__(module.Chunk)
    chunk.time, chunk.uuid, chunk.succ = ('time', 'uuid', 'succ')
    chunk.length, chunk.size = (2, 4)
    chunk.data = {'value': np.array([1, 2, 999, 999], dtype=np.int32)}
    chunk.saved = False
    return chunk

def test_io_failure_restores_unsaved_state(tmp_path, monkeypatch):
    module = target
    chunk = make_chunk(module)
    with monkeypatch.context() as patch:

        class FailingPath:

            def __init__(self, directory):
                pass

            def __truediv__(self, filename):
                return self

            def write(self, value, mode):
                assert chunk.saved
                raise OSError('simulated storage failure')
        patch.setattr(module.elements, 'Path', FailingPath)
        with pytest.raises(OSError, match='simulated storage failure'):
            chunk.save(tmp_path)
    assert chunk.saved is False
    chunk.save(tmp_path)
    assert chunk.saved is True
    with np.load(tmp_path / chunk.filename) as archive:
        np.testing.assert_array_equal(archive['value'], [1, 2])

def test_serialization_failure_can_be_retried(tmp_path, monkeypatch):
    module = target
    chunk = make_chunk(module)

    def fail(*args, **kwargs):
        raise ValueError('simulated serialization failure')
    with monkeypatch.context() as patch:
        patch.setattr(module.np, 'savez_compressed', fail)
        with pytest.raises(ValueError, match='simulated serialization failure'):
            chunk.save(tmp_path)
    assert chunk.saved is False
    chunk.save(tmp_path)
    assert chunk.saved is True

def test_success_keeps_duplicate_save_guard_and_truncates_capacity(tmp_path):
    chunk = make_chunk(target)
    chunk.save(tmp_path)
    assert chunk.saved is True
    with np.load(tmp_path / chunk.filename) as archive:
        np.testing.assert_array_equal(archive['value'], [1, 2])
        assert archive['value'].dtype == np.int32
    with pytest.raises(AssertionError):
        chunk.save(tmp_path)
