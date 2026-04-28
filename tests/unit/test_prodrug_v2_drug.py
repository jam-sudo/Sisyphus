"""Unit tests for DrugOnGraph.enzyme_affinity_for_conversion field."""
from __future__ import annotations

import numpy as np
import pytest

from sisyphus.core import ActiveMetabolite, Distribution, DrugOnGraph


def _minimal_drug(**overrides) -> DrugOnGraph:
    """Construct a DrugOnGraph with minimal valid fields."""
    base = dict(
        name="x", smiles="C", dose_mg=100.0, route="oral",
        administration_node="stomach_lumen",
        mw=200.0, pka=None, compound_type="neutral",
        fup=Distribution(0.5), rbp=Distribution(1.0),
        kp_method="rodgers_rowland", kp_overrides={},
        peff=Distribution(1e-4), solubility=Distribution(1.0),
        enzyme_affinity={},
        renal_clearance=Distribution(0.0),
    )
    base.update(overrides)
    return DrugOnGraph(**base)


def _minimal_active() -> ActiveMetabolite:
    """Construct an ActiveMetabolite with minimal valid fields."""
    return ActiveMetabolite(
        name="A",
        mw=200.0,
        fup=Distribution(0.5),
        CL_per_h=Distribution(10.0),
        Vd_L=Distribution(20.0),
        conversion_rate_per_h=Distribution(1.0),
        conversion_site="liver",
        conversion_yield_fraction=Distribution(1.0),
    )


def test_drug_default_enzyme_affinity_for_conversion_is_empty_dict():
    drug = _minimal_drug()
    assert drug.enzyme_affinity_for_conversion == {}


def test_drug_can_set_enzyme_affinity_for_conversion():
    affinity = {"SPR": Distribution(mean=100.0, cv=0.5)}
    drug = _minimal_drug(
        enzyme_affinity_for_conversion=affinity,
        active_metabolite=_minimal_active(),
        observation_species="parent",
    )
    assert drug.enzyme_affinity_for_conversion == affinity


def test_postinit_rejects_affinity_without_active_metabolite():
    """Non-empty enzyme_affinity_for_conversion requires active_metabolite."""
    with pytest.raises(ValueError, match="enzyme_affinity_for_conversion"):
        _minimal_drug(
            enzyme_affinity_for_conversion={"SPR": Distribution(100.0)},
            active_metabolite=None,
        )


def test_postinit_allows_empty_affinity_with_active_metabolite():
    """Empty dict + active_metabolite is allowed (e.g., during construction)."""
    drug = _minimal_drug(
        enzyme_affinity_for_conversion={},
        active_metabolite=_minimal_active(),
        observation_species="parent",
    )
    assert drug.enzyme_affinity_for_conversion == {}


def test_sample_propagates_enzyme_affinity_for_conversion():
    """drug.sample() must resample enzyme_affinity_for_conversion dict."""
    rng = np.random.default_rng(42)
    drug = _minimal_drug(
        enzyme_affinity_for_conversion={"SPR": Distribution(mean=100.0, cv=0.5)},
        active_metabolite=_minimal_active(),
        observation_species="parent",
    )
    sampled = drug.sample(rng)
    assert "SPR" in sampled.enzyme_affinity_for_conversion
    assert sampled.enzyme_affinity_for_conversion["SPR"].cv == 0.0
    assert np.isfinite(sampled.enzyme_affinity_for_conversion["SPR"].mean)
    assert sampled.enzyme_affinity_for_conversion["SPR"].mean > 0
