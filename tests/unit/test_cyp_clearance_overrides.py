"""Unit tests for the cyp_clearance_overrides registry + scaling path.

These verify three contracts:

1. Registry lookup is SMILES-variant-robust (matches by InChIKey).
2. Default for any unregistered drug is 1.0 (no scaling, current behavior).
3. ``_decompose_clint`` multiplies all per-enzyme affinities by the supplied
   metabolic_fraction.

End-to-end pravastatin / OATP1B1 reconciliation lives in the integration
tests (``test_oatp_pravastatin``, ``test_oatp_ecm_statins``).
"""

from __future__ import annotations

import pytest

from sisyphus.core import Distribution
from sisyphus.predict.cyp_clearance_overrides import lookup_metabolic_fraction
from sisyphus.predict.ivive import _decompose_clint

_PRAVASTATIN_CANONICAL = (
    "CC[C@@H](C)C(=O)O[C@@H]1C[C@@H](O)C=C2[C@@H]"
    "(CC[C@@H](O)C[C@@H](O)CC(=O)O)[C@H](C)CC[C@@H]21"
)
_PRAVASTATIN_VARIANT = (
    "CC[C@@H](C)C(=O)O[C@@H]1C[C@H](C=C2[C@@H]1CC[C@H]"
    "([C@@H]2CC[C@H](C[C@H](CC(=O)O)O)O)C)O"
)
_MORPHINE_SMILES = "CN1CCC23C4C1CC5=C2C(=C(C=C5)O)OC3C(C=C4)O"
_CLOPIDOGREL_SMILES = "COC(=O)[C@H](c1ccccc1Cl)N1CCc2sccc2C1"
_CLOPIDOGREL_NONSTEREO = "COC(=O)C(C1=CC=CC=C1Cl)N2CCC3=C(C2)C=CS3"


class TestLookupMetabolicFraction:
    def test_pravastatin_canonical_returns_zero(self):
        assert lookup_metabolic_fraction(_PRAVASTATIN_CANONICAL) == 0.0

    def test_pravastatin_variant_smiles_returns_zero(self):
        """Different SMILES of the same molecule must resolve via InChIKey."""
        assert lookup_metabolic_fraction(_PRAVASTATIN_VARIANT) == 0.0

    def test_clopidogrel_returns_zero(self):
        """B-03: explicit prodrug edges carry parent hepatic consumption."""
        assert lookup_metabolic_fraction(_CLOPIDOGREL_SMILES) == 0.0

    def test_clopidogrel_nonstereo_falls_back_via_inchikey_connectivity(self):
        """B-03 fix-forward: clinical_pk.json uses non-isomeric SMILES whose
        full InChIKey stereo block differs from the stereospecific override.
        The connectivity-block fallback restores the metabolic_fraction=0
        scaling on the production benchmark path. Without this fallback the
        explicit ProdrugActivationEdges and the default XGBoost CL path run
        in parallel, double-counting clopidogrel hepatic consumption."""
        assert lookup_metabolic_fraction(_CLOPIDOGREL_NONSTEREO) == 0.0

    def test_unregistered_drug_returns_one(self):
        assert lookup_metabolic_fraction(_MORPHINE_SMILES) == 1.0

    def test_invalid_smiles_returns_one(self):
        """Malformed input falls back to no-scaling (fail-safe)."""
        assert lookup_metabolic_fraction("not a smiles") == 1.0

    def test_empty_smiles_returns_one(self):
        assert lookup_metabolic_fraction("") == 1.0


class TestDecomposeClintScaling:
    def _clint(self) -> Distribution:
        return Distribution(mean=10.0, cv=0.3)

    def test_default_fraction_one_unchanged(self):
        """metabolic_fraction=1.0 must reproduce the legacy behavior."""
        baseline = _decompose_clint(self._clint(), compound_type="acid", pka=4.0)
        scaled = _decompose_clint(
            self._clint(), compound_type="acid", pka=4.0,
            metabolic_fraction=1.0,
        )
        assert set(baseline) == set(scaled)
        for k in baseline:
            assert baseline[k].mean == pytest.approx(scaled[k].mean)
            assert baseline[k].cv == pytest.approx(scaled[k].cv)

    def test_fraction_zero_yields_zero_affinities(self):
        result = _decompose_clint(
            self._clint(), compound_type="acid", pka=4.0,
            metabolic_fraction=0.0,
        )
        assert result, "expected non-empty enzyme affinity dict"
        for tag, dist in result.items():
            assert dist.mean == 0.0, f"{tag} affinity must be zero"

    def test_fraction_half_scales_proportionally(self):
        baseline = _decompose_clint(self._clint(), compound_type="acid", pka=4.0)
        half = _decompose_clint(
            self._clint(), compound_type="acid", pka=4.0,
            metabolic_fraction=0.5,
        )
        for tag in baseline:
            assert half[tag].mean == pytest.approx(0.5 * baseline[tag].mean)

    def test_cv_unchanged_by_scaling(self):
        """CV propagates from CLint independent of the scale factor."""
        clint = Distribution(mean=10.0, cv=0.42)
        scaled = _decompose_clint(
            clint, compound_type="acid", pka=4.0,
            metabolic_fraction=0.3,
        )
        for dist in scaled.values():
            assert dist.cv == pytest.approx(0.42)
