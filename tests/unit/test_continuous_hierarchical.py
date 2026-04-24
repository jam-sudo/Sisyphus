"""Unit tests for continuous hierarchical observation packing."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from sisyphus.sbi.multi_drug import (  # noqa: E402
    pack_observation_continuous,
    stack_training_pairs_continuous,
)


def test_pack_observation_continuous_shape():
    log10_cmax = np.array([-1.5, -2.0])
    drug_features = np.random.randn(2, 12)
    log_bw = np.array([0.0, -0.5])
    log_age = np.array([1.5, 0.7])
    packed = pack_observation_continuous(log10_cmax, drug_features, log_bw, log_age)
    assert packed.shape == (2, 15)
    np.testing.assert_allclose(packed[:, 0], log10_cmax)
    np.testing.assert_allclose(packed[:, 1:13], drug_features)
    np.testing.assert_allclose(packed[:, 13], log_bw)
    np.testing.assert_allclose(packed[:, 14], log_age)


def test_pack_observation_continuous_broadcast_scalars():
    log10_cmax = np.array([-1.5, -2.0])
    drug_features = np.random.randn(12)
    packed = pack_observation_continuous(log10_cmax, drug_features, 0.0, 1.5)
    assert packed.shape == (2, 15)
    assert packed[0, 13] == pytest.approx(0.0)
    assert packed[1, 14] == pytest.approx(1.5)


def test_stack_training_pairs_continuous_shape():
    theta_dict = {"drug_a": np.random.randn(10, 3)}
    logcmax_dict = {"drug_a": np.random.randn(10, 1)}
    feat_dict = {"drug_a": np.random.randn(12)}
    bw_dict = {"drug_a": np.full(10, 70.0)}
    age_dict = {"drug_a": np.full(10, 30.0)}
    theta, x, feat, bw, age, names = stack_training_pairs_continuous(
        theta_dict, logcmax_dict, feat_dict, bw_dict, age_dict,
    )
    assert theta.shape == (10, 3)
    assert x.shape == (10, 1)
    assert feat.shape == (10, 12)
    assert bw.shape == (10,)
    assert age.shape == (10,)
    assert len(names) == 10


def test_stack_drops_nan():
    logcmax = np.array([[1.0], [np.nan], [2.0]])
    theta_dict = {"d": np.random.randn(3, 3)}
    logcmax_dict = {"d": logcmax}
    feat_dict = {"d": np.random.randn(12)}
    bw_dict = {"d": np.array([70.0, 70.0, 70.0])}
    age_dict = {"d": np.array([30.0, 30.0, 30.0])}
    theta, x, feat, bw, age, names = stack_training_pairs_continuous(
        theta_dict, logcmax_dict, feat_dict, bw_dict, age_dict,
    )
    assert theta.shape[0] == 2
    assert x.shape[0] == 2
