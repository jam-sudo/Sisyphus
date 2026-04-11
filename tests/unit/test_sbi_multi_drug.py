"""Unit tests for the multi-drug conditional SBI utilities."""

from __future__ import annotations

import numpy as np
import pytest

from sisyphus.sbi.multi_drug import (
    DRUG_FEATURE_NAMES,
    DrugSpec,
    MultiDrugSimulator,
    N_DRUG_FEATURES,
    extract_drug_features,
    pack_observation,
    stack_training_pairs,
)


MORPHINE_SMILES = "CN1CCC23C4C1CC5=C2C(=C(C=C5)O)OC3C(C=C4)O"
MORPHINE_DOSE_MG = 30.0


@pytest.fixture(scope="module")
def morphine_sim():
    """Single-drug MultiDrugSimulator used by multiple tests."""
    spec = DrugSpec(name="morphine", smiles=MORPHINE_SMILES, dose_mg=MORPHINE_DOSE_MG)
    return MultiDrugSimulator.for_single(spec)


def test_drug_feature_names_and_count():
    assert len(DRUG_FEATURE_NAMES) == N_DRUG_FEATURES == 12


def test_extract_features_shape_and_range(morphine_sim):
    feats = morphine_sim.get_features("morphine")
    assert feats.shape == (N_DRUG_FEATURES,)
    assert feats.dtype == np.float64
    assert np.all(np.isfinite(feats))

    # Physical sanity on key components
    name_to_idx = {n: i for i, n in enumerate(DRUG_FEATURE_NAMES)}
    assert 0.0 <= feats[name_to_idx["fup"]] <= 1.0
    assert feats[name_to_idx["is_base"]] == 1.0  # morphine is a base
    assert feats[name_to_idx["is_acid"]] == 0.0
    assert feats[name_to_idx["is_neutral"]] == 0.0
    # MW/500 roughly 0.5-0.7 for morphine (MW 285)
    assert 0.4 < feats[name_to_idx["mw_scaled"]] < 0.8


def test_extract_features_logp_override(morphine_sim):
    """Passing an explicit logp should override the default fallback."""
    sim = morphine_sim.simulators["morphine"]
    default_feats = extract_drug_features(sim, logp=None)
    override_feats = extract_drug_features(sim, logp=5.5)
    idx = DRUG_FEATURE_NAMES.index("logp")
    assert default_feats[idx] == 2.0
    assert override_feats[idx] == 5.5


def test_extract_features_kwarg_form(morphine_sim):
    """The (drug=, graph=) kwarg form must agree with the legacy (sim,) form."""
    sim = morphine_sim.simulators["morphine"]
    legacy = extract_drug_features(sim, logp=1.23)
    kw = extract_drug_features(drug=sim.nominal_drug, graph=sim.graph, logp=1.23)
    np.testing.assert_array_equal(legacy, kw)


def test_extract_features_kwarg_requires_both(morphine_sim):
    """Passing drug= without graph= must raise, not silently ignore."""
    sim = morphine_sim.simulators["morphine"]
    with pytest.raises(ValueError):
        extract_drug_features(drug=sim.nominal_drug)
    with pytest.raises(ValueError):
        extract_drug_features(graph=sim.graph)


def test_simulate_single_determinism(morphine_sim):
    theta = np.array([0.0, 0.65, 0.0])
    a = morphine_sim.simulate_single("morphine", theta, seed=123)
    b = morphine_sim.simulate_single("morphine", theta, seed=123)
    assert a == b  # byte-for-byte reproducibility


def test_simulate_batch_shape(morphine_sim):
    thetas = np.array([[0.0, 0.65, 0.0], [0.5, 0.3, 0.2]])
    out = morphine_sim.simulate_batch_per_drug("morphine", thetas, seed=0)
    assert out.shape == (2, 1)


def test_stack_training_pairs_drops_nan():
    theta_per_drug = {
        "a": np.array([[0.0, 0.5, 0.0], [0.1, 0.5, 0.0]]),
        "b": np.array([[0.2, 0.5, 0.0]]),
    }
    log_per_drug = {
        "a": np.array([[-1.0], [np.nan]]),
        "b": np.array([[-2.0]]),
    }
    feat_per_drug = {
        "a": np.ones(N_DRUG_FEATURES) * 0.1,
        "b": np.ones(N_DRUG_FEATURES) * 0.9,
    }
    theta_all, x_all, feat_all, names = stack_training_pairs(
        theta_per_drug, log_per_drug, feat_per_drug
    )
    assert theta_all.shape == (2, 3)
    assert x_all.shape == (2, 1)
    assert feat_all.shape == (2, N_DRUG_FEATURES)
    assert names == ["a", "b"]


def test_pack_observation_roundtrip():
    logc = np.array([[-1.5], [-2.0]])
    feats = np.arange(N_DRUG_FEATURES, dtype=float)
    packed = pack_observation(logc, feats)
    assert packed.shape == (2, 1 + N_DRUG_FEATURES)
    # log10_cmax in column 0, features in the rest
    np.testing.assert_array_equal(packed[:, 0:1], logc)
    np.testing.assert_array_equal(packed[0, 1:], feats)
    np.testing.assert_array_equal(packed[1, 1:], feats)


def test_pack_observation_broadcasts_single_drug():
    logc = np.array([-1.5, -2.0])  # (N,)
    feats = np.arange(N_DRUG_FEATURES, dtype=float)
    packed = pack_observation(logc, feats)
    assert packed.shape == (2, 1 + N_DRUG_FEATURES)


def test_pack_observation_shape_mismatch():
    logc = np.array([[-1.5], [-2.0]])
    feats = np.ones((3, N_DRUG_FEATURES))  # mismatched rows
    with pytest.raises(ValueError):
        pack_observation(logc, feats)


def test_multi_drug_simulator_feature_cached(morphine_sim):
    f1 = morphine_sim.get_features("morphine")
    f2 = morphine_sim.get_features("morphine")
    np.testing.assert_array_equal(f1, f2)
    # get_features returns a copy
    f1[0] = -999
    assert morphine_sim.get_features("morphine")[0] != -999
