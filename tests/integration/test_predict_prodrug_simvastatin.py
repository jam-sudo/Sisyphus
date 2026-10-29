"""Integration test for simvastatin lactone -> acid prodrug routing (v0.3.4 / #11).

Tests that predict() with simvastatin lactone SMILES routes through the
prodrug registry (CES1 hydrolysis), simulates the active acid species,
and returns a non-zero active-acid Cmax. Najib 2003 reported 0.003-0.007 mg/L
at 40 mg PO; the full R&R default predicts 0.00027 mg/L (11-26x under).

The registry entry has disposition_state=ceiling_accepted with explicit
5-50x uncertainty span on CL/V (class-extrapolated from atorvastatin
acid; F_simvastatin_acid not located in primary literature). The
acceptance gate (positive Cmax; <0.10 mg/L upper) is therefore
mechanical-correctness only: confirms routing fired and active species
simulated, without claiming clinical accuracy. Per spec §10 + ceiling
disposition, absolute calibration is downstream.
"""
from __future__ import annotations

import pytest

from sisyphus.pipeline.predict import predict
from sisyphus.predict.registry import lookup_active_metabolite

_SIMVASTATIN_LACTONE = (
    "CCC(C)(C)C(=O)O[C@H]1C[C@H](C)C=C2C=C[C@H](C)[C@H]"
    "(CC[C@@H]3C[C@@H](O)CC(=O)O3)[C@H]21"
)
_CLOPIDOGREL = "COC(=O)[C@H](c1ccccc1Cl)N1CCc2sccc2C1"


@pytest.mark.slow
def test_simvastatin_lactone_returns_active_acid_cmax():
    """predict(simvastatin_lactone, 40mg PO) routes through prodrug registry
    and returns a positive active simvastatin acid Cmax."""
    result = predict(_SIMVASTATIN_LACTONE, dose_mg=40.0, route="oral")
    assert result.engine_pk is not None, (
        "engine_pk None — prodrug routing or simulation failed"
    )
    cmax = result.engine_pk.cmax.mean
    assert any("routed via activation to 'simvastatin'" in w for w in result.warnings)
    assert cmax > 0
    # Upper-bound sanity (10x literature) — catches obvious calibration drift
    assert cmax < 0.10, (
        f"simvastatin acid Cmax {cmax:.5f} mg/L above 0.10 (10x Najib 2003 "
        f"upper of 0.007) — possible double-routing or yield error."
    )


def test_clopidogrel_registry_uses_parent_observation_and_per_enzyme_yields():
    """B-03: clopidogrel uses parent scoring with dual-fate enzyme yields."""
    result = lookup_active_metabolite(_CLOPIDOGREL)
    assert result is not None
    active, observation_species, affinities, enzyme_yields = result

    assert active.name == "R-130964"
    assert observation_species == "parent"
    assert set(affinities) == {"CES1", "CYP3A4", "CYP2C9"}
    assert set(enzyme_yields) == set(affinities)
    assert enzyme_yields["CES1"].mean == 0.0
    assert enzyme_yields["CYP3A4"].mean == 1.0
    assert enzyme_yields["CYP2C9"].mean == 1.0


@pytest.mark.slow
def test_clopidogrel_predict_returns_parent_cmax():
    """predict(clopidogrel, 300mg PO) succeeds and returns parent Cmax."""
    result = predict(_CLOPIDOGREL, dose_mg=300.0, route="oral")
    assert result.engine_pk is not None, (
        "engine_pk None — clopidogrel prodrug routing or parent simulation failed"
    )
    assert result.engine_pk.cmax.mean > 0.0
    assert result.pk.cmax.mean > 0.0
