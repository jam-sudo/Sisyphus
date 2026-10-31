"""Timed SBC contexts can be shared across independent case blocks."""

import numpy as np
import pytest

pytest.importorskip("torch")

from scripts.sbi_run_sbc_multi_drug import _load_context_cache, _save_context_cache


def test_timed_context_cache_roundtrip_and_key(tmp_path):
    path = tmp_path / "contexts.npz"
    theta = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    contexts = np.arange(38, dtype=np.float32).reshape(2, 19)
    _save_context_cache(
        path, drug="morphine", seed=7, n_calibration=2,
        thetas=theta, contexts=contexts,
    )
    loaded_theta, loaded_contexts = _load_context_cache(
        path, drug="morphine", seed=7, n_calibration=2,
    )
    np.testing.assert_array_equal(loaded_theta, theta)
    np.testing.assert_array_equal(loaded_contexts, contexts)
    with pytest.raises(ValueError, match="does not match"):
        _load_context_cache(path, drug="digoxin", seed=7, n_calibration=2)
