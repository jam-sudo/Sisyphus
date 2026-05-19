"""Unit tests for per-enzyme prodrug yield (B-04).

See docs/superpowers/specs/2026-05-17-multi-enzyme-prodrug-yield-design.md
"""
from __future__ import annotations

import numpy as np
import pytest

from sisyphus.core import ActiveMetabolite, Distribution, DrugOnGraph


def _minimal_active(**overrides) -> ActiveMetabolite:
    base = dict(
        name="A",
        mw=200.0,
        fup=Distribution(0.5),
        CL_per_h=Distribution(10.0),
        Vd_L=Distribution(20.0),
        conversion_rate_per_h=Distribution(0.0),
        conversion_site="",
        conversion_yield_fraction=Distribution(1.0),
    )
    base.update(overrides)
    return ActiveMetabolite(**base)


class TestActiveMetaboliteEnzymeYields:
    def test_default_is_empty_dict(self):
        am = _minimal_active()
        assert am.enzyme_yields == {}

    def test_can_set_per_enzyme_yields(self):
        yields = {
            "CES1": Distribution(mean=0.0, cv=0.0),
            "CYP2C19": Distribution(mean=1.0, cv=0.30),
        }
        am = _minimal_active(enzyme_yields=yields)
        assert am.enzyme_yields == yields


def _minimal_drug(**overrides) -> DrugOnGraph:
    """Construct a DrugOnGraph with minimal valid fields.

    Replicates the helper from test_prodrug_v2_drug.py for self-containment.
    """
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


class TestDrugOnGraphPropagatesEnzymeYields:
    def _drug_with_yields(self) -> DrugOnGraph:
        am = _minimal_active(
            enzyme_yields={
                "CES1": Distribution(mean=0.0, cv=0.0),
                "CYP2C19": Distribution(mean=1.0, cv=0.30),
            },
        )
        return _minimal_drug(
            active_metabolite=am,
            observation_species="parent",
            enzyme_affinity_for_conversion={
                "CES1": Distribution(100.0),
                "CYP2C19": Distribution(50.0),
            },
        )

    def test_sample_propagates_enzyme_yields(self):
        drug = self._drug_with_yields()
        rng = np.random.default_rng(42)
        sampled = drug.sample(rng)
        assert set(sampled.active_metabolite.enzyme_yields.keys()) == {"CES1", "CYP2C19"}
        # cv=0 entries must round-trip exactly
        assert sampled.active_metabolite.enzyme_yields["CES1"].mean == 0.0

    def test_realize_means_propagates_enzyme_yields(self):
        drug = self._drug_with_yields()
        realized = drug.realize_means()
        assert realized.active_metabolite.enzyme_yields["CES1"].mean == 0.0
        assert realized.active_metabolite.enzyme_yields["CYP2C19"].mean == 1.0
        # realize_means must produce cv=0 deterministic Distributions
        assert realized.active_metabolite.enzyme_yields["CYP2C19"].cv == 0.0

    def test_sample_propagates_empty_enzyme_yields(self):
        """Backward compat: existing entries (no per-enzyme yields) round-trip empty dict."""
        drug = _minimal_drug(
            active_metabolite=_minimal_active(),  # no enzyme_yields override
            observation_species="parent",
            enzyme_affinity_for_conversion={"SPR": Distribution(100.0)},
        )
        rng = np.random.default_rng(0)
        sampled = drug.sample(rng)
        assert sampled.active_metabolite.enzyme_yields == {}
