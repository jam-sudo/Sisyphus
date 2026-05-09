"""Tests for ActiveMetabolite dataclass."""
from __future__ import annotations

import pytest

from sisyphus.core import ActiveMetabolite, Distribution


@pytest.fixture
def bh4_active():
    return ActiveMetabolite(
        name="BH4", mw=241.25,
        fup=Distribution(mean=0.23, cv=0.3),
        CL_per_h=Distribution(mean=40.0, cv=0.35),
        Vd_L=Distribution(mean=150.0, cv=0.3),
        conversion_rate_per_h=Distribution(mean=12.0, cv=0.4),
        conversion_site="gut_wall",
        conversion_yield_fraction=Distribution(mean=0.85, cv=0.1),
    )


def test_active_metabolite_minimal_construction(bh4_active):
    """Required fields produce a valid frozen ActiveMetabolite."""
    assert bh4_active.name == "BH4"
    assert bh4_active.mw == 241.25
    assert bh4_active.conversion_site == "gut_wall"


def test_active_metabolite_is_frozen(bh4_active):
    """ActiveMetabolite must be frozen (immutable)."""
    with pytest.raises(AttributeError):
        bh4_active.name = "different"  # type: ignore[misc]


import numpy as np  # noqa: E402

from sisyphus.core import DrugOnGraph  # noqa: E402


def _minimal_drug(active=None, obs_species="parent"):
    """Construct a DrugOnGraph with all required fields and optional active.

    Used by other tests across this plan as a shared factory.
    """
    return DrugOnGraph(
        name="testdrug",
        smiles="CCO",
        dose_mg=100.0,
        route="oral",
        administration_node="stomach_lumen",
        mw=46.07,
        pka=None,
        compound_type="neutral",
        fup=Distribution(0.5),
        rbp=Distribution(1.0),
        kp_method="rodgers_rowland",
        kp_overrides={},
        peff=Distribution(1e-4),
        solubility=Distribution(100.0),
        enzyme_affinity={},
        renal_clearance=Distribution(0.0),
        active_metabolite=active,
        observation_species=obs_species,
    )


def test_drugongraph_default_no_active():
    """Default values: active_metabolite=None, observation_species='parent'."""
    drug = _minimal_drug()
    assert drug.active_metabolite is None
    assert drug.observation_species == "parent"


def test_drugongraph_with_active_metabolite(bh4_active):
    """Active metabolite stored verbatim; observation_species can be 'active'."""
    drug = _minimal_drug(active=bh4_active, obs_species="active")
    assert drug.active_metabolite is bh4_active
    assert drug.observation_species == "active"


def test_observation_active_without_active_metabolite_fails():
    """observation_species='active' requires active_metabolite to be set."""
    with pytest.raises(ValueError, match="observation_species='active' requires"):
        _minimal_drug(active=None, obs_species="active")


def test_invalid_observation_species_fails():
    """observation_species must be 'parent' or 'active'."""
    with pytest.raises(ValueError, match="observation_species must be"):
        _minimal_drug(obs_species="middle")


def test_drugongraph_sample_resamples_active_metabolite(bh4_active):
    """DrugOnGraph.sample() resamples all ActiveMetabolite Distribution fields."""
    drug = _minimal_drug(active=bh4_active, obs_species="active")
    rng = np.random.default_rng(42)
    sampled = drug.sample(rng)
    assert sampled.active_metabolite is not None
    assert sampled.active_metabolite.name == "BH4"
    # After sampling, all Distribution fields are point-valued (cv=0)
    assert sampled.active_metabolite.fup.cv == 0.0
    assert sampled.active_metabolite.CL_per_h.cv == 0.0
    assert sampled.active_metabolite.Vd_L.cv == 0.0
    assert sampled.active_metabolite.conversion_rate_per_h.cv == 0.0
    assert sampled.active_metabolite.conversion_yield_fraction.cv == 0.0
    # Scalar fields preserved
    assert sampled.active_metabolite.mw == 241.25
    assert sampled.active_metabolite.conversion_site == "gut_wall"
    assert sampled.observation_species == "active"


def test_drugongraph_sample_no_active_preserves_none():
    """sample() on drug without active produces drug with active_metabolite=None."""
    drug = _minimal_drug()
    rng = np.random.default_rng(42)
    sampled = drug.sample(rng)
    assert sampled.active_metabolite is None
    assert sampled.observation_species == "parent"
