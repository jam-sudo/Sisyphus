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


# ---------------------------------------------------------------------------
# TestRegistryParsesPerEnzymeYield (Task 3 — B-04)
# ---------------------------------------------------------------------------

import json
from pathlib import Path

from sisyphus.predict.registry import lookup_active_metabolite
from tests.unit.test_prodrug_v2_registry import _v2_entry


class TestRegistryParsesPerEnzymeYield:
    def _write(self, tmp_path: Path, entries: dict) -> Path:
        p = tmp_path / "registry.json"
        p.write_text(json.dumps(entries))
        return p

    def test_lookup_returns_four_tuple(self, tmp_path):
        """Single-enzyme entry: 4-tuple, empty enzyme_yields dict."""
        reg = self._write(tmp_path, {"C": _v2_entry()})
        result = lookup_active_metabolite("C", registry_path=reg)
        assert result is not None
        assert len(result) == 4
        am, obs, affinities, enzyme_yields = result
        assert affinities["SPR"].mean == 50.0
        assert enzyme_yields == {}

    def test_single_enzyme_with_yield_is_parsed(self, tmp_path):
        """Optional per-enzyme yield on a single-enzyme entry round-trips."""
        entry = _v2_entry(
            enzyme_affinity_for_conversion={
                "SPR": {
                    "mean": 50.0,
                    "cv": 0.5,
                    "yield": {"mean": 0.5, "cv": 0.1},
                }
            }
        )
        reg = self._write(tmp_path, {"C": entry})
        _, _, _, enzyme_yields = lookup_active_metabolite("C", registry_path=reg)
        assert "SPR" in enzyme_yields
        assert enzyme_yields["SPR"].mean == 0.5
        assert enzyme_yields["SPR"].cv == 0.1

    def test_multi_enzyme_with_all_yields_is_parsed(self, tmp_path):
        """Multi-enzyme entry with per-enzyme yield on every enzyme."""
        entry = _v2_entry(
            enzyme_affinity_for_conversion={
                "CES1": {
                    "mean": 100.0, "cv": 0.5,
                    "yield": {"mean": 0.0, "cv": 0.0},
                },
                "CYP2C19": {
                    "mean": 30.0, "cv": 0.4,
                    "yield": {"mean": 1.0, "cv": 0.30},
                },
            }
        )
        reg = self._write(tmp_path, {"C": entry})
        _, _, affinities, enzyme_yields = lookup_active_metabolite("C", registry_path=reg)
        assert set(affinities.keys()) == {"CES1", "CYP2C19"}
        assert set(enzyme_yields.keys()) == {"CES1", "CYP2C19"}
        assert enzyme_yields["CES1"].mean == 0.0
        assert enzyme_yields["CYP2C19"].mean == 1.0

    def test_multi_enzyme_missing_yield_raises(self, tmp_path):
        """Multi-enzyme with partial declaration is rejected at load time."""
        entry = _v2_entry(
            enzyme_affinity_for_conversion={
                "CES1": {
                    "mean": 100.0, "cv": 0.5,
                    "yield": {"mean": 0.0, "cv": 0.0},
                },
                "CYP2C19": {"mean": 30.0, "cv": 0.4},  # no yield
            }
        )
        reg = self._write(tmp_path, {"C": entry})
        with pytest.raises(ValueError, match="yield"):
            lookup_active_metabolite("C", registry_path=reg)

    def test_yield_out_of_range_raises(self, tmp_path):
        """Per-enzyme yield must satisfy 0 <= mean <= 1."""
        entry = _v2_entry(
            enzyme_affinity_for_conversion={
                "SPR": {
                    "mean": 50.0, "cv": 0.5,
                    "yield": {"mean": 1.5, "cv": 0.0},
                }
            }
        )
        reg = self._write(tmp_path, {"C": entry})
        with pytest.raises(ValueError, match="yield"):
            lookup_active_metabolite("C", registry_path=reg)
