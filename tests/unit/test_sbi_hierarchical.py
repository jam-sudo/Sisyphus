"""Unit tests for hierarchical population SBI (Track C1).

Tests cover the population registry, hierarchical simulator construction,
population-independent drug features, one-hot encoding, and the extended
pack/stack functions for hierarchical training data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sisyphus.sbi.multi_drug import (
    DRUG_FEATURE_NAMES,
    DrugSpec,
    HierarchicalMultiDrugSimulator,
    MultiDrugSimulator,
    N_DRUG_FEATURES,
    PopulationSpec,
    extract_drug_features,
    load_populations,
    pack_observation_hierarchical,
    population_onehot,
    stack_training_pairs_hierarchical,
)

MORPHINE_SMILES = "CN1CCC23C4C1CC5=C2C(=C(C=C5)O)OC3C(C=C4)O"
MORPHINE_DOSE_MG = 30.0


# ---------------------------------------------------------------------------
# Population registry
# ---------------------------------------------------------------------------


def test_load_populations():
    """load_populations returns adult and pediatric_5y entries."""
    pops = load_populations()
    assert "adult" in pops
    assert "pediatric_5y" in pops
    assert isinstance(pops["adult"], PopulationSpec)
    assert pops["adult"].body_weight_kg == 70.0
    assert pops["pediatric_5y"].body_weight_kg == 18.0
    assert pops["adult"].physiology_yaml == Path("data/physiology/reference_man.yaml")
    assert pops["pediatric_5y"].physiology_yaml == Path("data/physiology/pediatric_5y.yaml")


def test_load_populations_custom_path(tmp_path):
    """load_populations reads from a custom path."""
    custom = tmp_path / "pops.json"
    custom.write_text(
        '{"populations": {"test_pop": {"physiology_yaml": "some.yaml", "body_weight_kg": 50.0}}}'
    )
    pops = load_populations(custom)
    assert "test_pop" in pops
    assert pops["test_pop"].body_weight_kg == 50.0


# ---------------------------------------------------------------------------
# Population one-hot encoding
# ---------------------------------------------------------------------------


def test_population_onehot_encoding():
    """One-hot vectors have correct shape and values."""
    names = ["adult", "pediatric_5y"]
    adult_oh = population_onehot("adult", names)
    ped_oh = population_onehot("pediatric_5y", names)
    assert adult_oh.shape == (2,)
    assert ped_oh.shape == (2,)
    np.testing.assert_array_equal(adult_oh, [1.0, 0.0])
    np.testing.assert_array_equal(ped_oh, [0.0, 1.0])


def test_population_onehot_three_pops():
    """One-hot generalizes to >2 populations."""
    names = ["a", "b", "c"]
    b_oh = population_onehot("b", names)
    np.testing.assert_array_equal(b_oh, [0.0, 1.0, 0.0])


def test_population_onehot_unknown_raises():
    """Requesting an unknown population raises ValueError."""
    names = ["adult", "pediatric_5y"]
    with pytest.raises(ValueError):
        population_onehot("neonatal", names)


# ---------------------------------------------------------------------------
# Drug features are population-independent
# ---------------------------------------------------------------------------

# These tests use real engine simulators (expensive) — module-scoped fixtures.

@pytest.fixture(scope="module")
def morphine_hierarchical_sim():
    """Build hierarchical simulator for morphine across adult + pediatric."""
    spec = DrugSpec(name="morphine", smiles=MORPHINE_SMILES, dose_mg=MORPHINE_DOSE_MG)
    return HierarchicalMultiDrugSimulator.from_specs([spec])


@pytest.fixture(scope="module")
def morphine_adult_sim():
    """Build a plain adult MultiDrugSimulator for morphine (comparison baseline)."""
    spec = DrugSpec(name="morphine", smiles=MORPHINE_SMILES, dose_mg=MORPHINE_DOSE_MG)
    return MultiDrugSimulator.for_single(spec)


def test_hierarchical_simulator_creates_both_populations(morphine_hierarchical_sim):
    """Hierarchical simulator has entries for both adult and pediatric."""
    hsim = morphine_hierarchical_sim
    assert "adult" in hsim.simulators
    assert "pediatric_5y" in hsim.simulators
    assert "morphine" in hsim.simulators["adult"]
    assert "morphine" in hsim.simulators["pediatric_5y"]
    assert hsim.n_populations == 2
    assert hsim.population_names == ["adult", "pediatric_5y"]


def test_drug_features_population_independent(morphine_hierarchical_sim, morphine_adult_sim):
    """Drug features from hierarchical sim match the plain adult sim features.

    This verifies that HierarchicalMultiDrugSimulator always uses the adult
    graph for feature extraction, so drug features are intrinsic properties.
    """
    hier_feats = morphine_hierarchical_sim.get_features("morphine")
    adult_feats = morphine_adult_sim.get_features("morphine")
    np.testing.assert_array_almost_equal(hier_feats, adult_feats, decimal=10)


def test_hierarchical_drug_features_shape(morphine_hierarchical_sim):
    """Drug features have correct shape and are finite."""
    feats = morphine_hierarchical_sim.get_features("morphine")
    assert feats.shape == (N_DRUG_FEATURES,)
    assert np.all(np.isfinite(feats))


def test_hierarchical_pop_onehot(morphine_hierarchical_sim):
    """Pop one-hot accessors return correct values."""
    adult_oh = morphine_hierarchical_sim.get_pop_onehot("adult")
    ped_oh = morphine_hierarchical_sim.get_pop_onehot("pediatric_5y")
    np.testing.assert_array_equal(adult_oh, [1.0, 0.0])
    np.testing.assert_array_equal(ped_oh, [0.0, 1.0])
    # Returns copy
    adult_oh[0] = -999
    assert morphine_hierarchical_sim.get_pop_onehot("adult")[0] == 1.0


def test_pediatric_simulator_runs(morphine_hierarchical_sim):
    """The pediatric simulator produces a finite log10(Cmax) for morphine.

    This validates that pediatric_5y.yaml loads correctly, the graph compiles,
    and the solver converges for a real drug.
    """
    theta = np.array([0.0, 0.65, 0.0])
    result = morphine_hierarchical_sim.simulate_single(
        "pediatric_5y", "morphine", theta, seed=42
    )
    assert np.isfinite(result), f"Pediatric simulation failed: {result}"
    # Pediatric Cmax should differ from adult due to smaller volumes/cardiac output
    adult_result = morphine_hierarchical_sim.simulate_single(
        "adult", "morphine", theta, seed=42
    )
    assert np.isfinite(adult_result)
    # They should not be identical (different physiology)
    assert result != adult_result


def test_pediatric_vs_adult_determinism(morphine_hierarchical_sim):
    """Same (pop, drug, theta, seed) gives identical results on repeated calls."""
    theta = np.array([0.0, 0.5, 0.0])
    a = morphine_hierarchical_sim.simulate_single("pediatric_5y", "morphine", theta, seed=99)
    b = morphine_hierarchical_sim.simulate_single("pediatric_5y", "morphine", theta, seed=99)
    assert a == b


# ---------------------------------------------------------------------------
# pack_observation_hierarchical
# ---------------------------------------------------------------------------


def test_pack_observation_hierarchical_shape():
    """Packed observation has shape (N, 1 + 12 + n_pop)."""
    n = 5
    logc = np.random.randn(n, 1)
    feats = np.random.randn(N_DRUG_FEATURES)
    pop_oh = np.array([1.0, 0.0])
    packed = pack_observation_hierarchical(logc, feats, pop_oh)
    assert packed.shape == (n, 1 + N_DRUG_FEATURES + 2)


def test_pack_observation_hierarchical_content():
    """Verify the packed array contains the right values in the right columns."""
    logc = np.array([[-1.5], [-2.0]])
    feats = np.arange(N_DRUG_FEATURES, dtype=float)
    pop_oh = np.array([0.0, 1.0])
    packed = pack_observation_hierarchical(logc, feats, pop_oh)
    # Column 0: log10_cmax
    np.testing.assert_array_equal(packed[:, 0:1], logc)
    # Columns 1..12: drug features (broadcast)
    np.testing.assert_array_equal(packed[0, 1:1 + N_DRUG_FEATURES], feats)
    np.testing.assert_array_equal(packed[1, 1:1 + N_DRUG_FEATURES], feats)
    # Last 2 columns: pop one-hot (broadcast)
    np.testing.assert_array_equal(packed[0, -2:], pop_oh)
    np.testing.assert_array_equal(packed[1, -2:], pop_oh)


def test_pack_observation_hierarchical_2d_inputs():
    """Full 2D inputs for all arguments."""
    n = 3
    logc = np.ones((n, 1))
    feats = np.ones((n, N_DRUG_FEATURES))
    pop_oh = np.tile(np.array([1.0, 0.0])[None, :], (n, 1))
    packed = pack_observation_hierarchical(logc, feats, pop_oh)
    assert packed.shape == (n, 1 + N_DRUG_FEATURES + 2)


def test_pack_observation_hierarchical_shape_mismatch():
    """Mismatched row counts raise ValueError."""
    logc = np.ones((3, 1))
    feats = np.ones((2, N_DRUG_FEATURES))
    pop_oh = np.array([1.0, 0.0])
    with pytest.raises(ValueError):
        pack_observation_hierarchical(logc, feats, pop_oh)


# ---------------------------------------------------------------------------
# stack_training_pairs_hierarchical
# ---------------------------------------------------------------------------


def test_stack_training_pairs_hierarchical_shape():
    """Stacked hierarchical training data has correct shapes."""
    n_theta = 3
    d_theta = 3
    pop_names = ["adult", "pediatric_5y"]
    pop_oh = {
        "adult": np.array([1.0, 0.0]),
        "pediatric_5y": np.array([0.0, 1.0]),
    }
    theta_ppd = {
        "adult": {"drug_a": np.random.randn(n_theta, d_theta)},
        "pediatric_5y": {"drug_a": np.random.randn(n_theta, d_theta)},
    }
    logcmax_ppd = {
        "adult": {"drug_a": np.random.randn(n_theta, 1)},
        "pediatric_5y": {"drug_a": np.random.randn(n_theta, 1)},
    }
    feats = {"drug_a": np.random.randn(N_DRUG_FEATURES)}

    theta_all, x_all, feat_all, pop_all, drug_names, pop_name_list = (
        stack_training_pairs_hierarchical(theta_ppd, logcmax_ppd, feats, pop_oh)
    )
    # 2 populations * 1 drug * 3 thetas = 6 rows
    assert theta_all.shape == (6, d_theta)
    assert x_all.shape == (6, 1)
    assert feat_all.shape == (6, N_DRUG_FEATURES)
    assert pop_all.shape == (6, 2)
    assert len(drug_names) == 6
    assert len(pop_name_list) == 6


def test_stack_training_pairs_hierarchical_drops_nan():
    """Non-finite log10(cmax) rows are dropped."""
    pop_oh = {"adult": np.array([1.0, 0.0]), "pediatric_5y": np.array([0.0, 1.0])}
    theta_ppd = {
        "adult": {"d": np.array([[0.0, 0.5, 0.0], [0.1, 0.5, 0.0]])},
        "pediatric_5y": {"d": np.array([[0.2, 0.5, 0.0]])},
    }
    logcmax_ppd = {
        "adult": {"d": np.array([[-1.0], [np.nan]])},  # 1 valid, 1 NaN
        "pediatric_5y": {"d": np.array([[-2.0]])},       # 1 valid
    }
    feats = {"d": np.ones(N_DRUG_FEATURES) * 0.5}

    theta_all, x_all, feat_all, pop_all, drug_names, pop_name_list = (
        stack_training_pairs_hierarchical(theta_ppd, logcmax_ppd, feats, pop_oh)
    )
    # 1 + 1 = 2 valid rows
    assert theta_all.shape[0] == 2
    assert x_all.shape[0] == 2


def test_stack_training_pairs_hierarchical_pop_labels():
    """Population labels are correctly assigned in the output."""
    pop_oh = {"adult": np.array([1.0, 0.0]), "ped": np.array([0.0, 1.0])}
    theta_ppd = {
        "adult": {"x": np.array([[0.0, 0.5, 0.0]])},
        "ped": {"x": np.array([[0.1, 0.3, 0.0]])},
    }
    logcmax_ppd = {
        "adult": {"x": np.array([[-1.0]])},
        "ped": {"x": np.array([[-2.0]])},
    }
    feats = {"x": np.ones(N_DRUG_FEATURES)}

    _, _, _, pop_all, drug_names, pop_name_list = (
        stack_training_pairs_hierarchical(theta_ppd, logcmax_ppd, feats, pop_oh)
    )
    # Sorted by pop name: "adult" before "ped"
    assert pop_name_list[0] == "adult"
    assert pop_name_list[1] == "ped"
    np.testing.assert_array_equal(pop_all[0], [1.0, 0.0])
    np.testing.assert_array_equal(pop_all[1], [0.0, 1.0])
