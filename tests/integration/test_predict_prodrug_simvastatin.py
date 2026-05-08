"""Integration test for simvastatin lactone -> acid prodrug routing (v0.3.4 / #11).

Tests that predict() with simvastatin lactone SMILES routes through the
prodrug registry (CES1 hydrolysis), simulates the active acid species,
and returns a Cmax that is non-zero and within order of magnitude of
clinical literature (Najib 2003: 0.003-0.007 mg/L @ 40 mg PO).

The registry entry has disposition_state=ceiling_accepted with explicit
5-50x uncertainty span on CL/V (class-extrapolated from atorvastatin
acid; F_simvastatin_acid not located in primary literature). The
acceptance gate (>0.0005 mg/L lower; <0.10 mg/L upper) is therefore
mechanical-correctness only: confirms routing fired and active species
simulated, without pinning Cmax magnitude. Per spec §10 + ceiling
disposition, absolute calibration is downstream.
"""
from __future__ import annotations

import pytest

from sisyphus.pipeline.predict import predict


_SIMVASTATIN_LACTONE = (
    "CCC(C)(C)C(=O)O[C@H]1C[C@H](C)C=C2C=C[C@H](C)[C@H]"
    "(CC[C@@H]3C[C@@H](O)CC(=O)O3)[C@H]21"
)


@pytest.mark.slow
def test_simvastatin_lactone_returns_active_acid_cmax():
    """predict(simvastatin_lactone, 40mg PO) routes through prodrug registry
    and returns active simvastatin acid Cmax in plausible range."""
    result = predict(_SIMVASTATIN_LACTONE, dose_mg=40.0, route="oral")
    assert result.engine_pk is not None, (
        "engine_pk None — prodrug routing or simulation failed"
    )
    cmax = result.engine_pk.cmax.mean
    # Lower gate at 0.0005 mg/L: empirical post-Task-2 produces ~0.00088 mg/L
    # (3-8x below Najib 2003 clinical 0.003-0.007); within registry's
    # ceiling_accepted disposition's acknowledged 5-50x uncertainty span.
    # Floor confirms routing fired (non-zero) without pinning calibration.
    assert cmax > 0.0005, (
        f"simvastatin acid Cmax {cmax:.5f} mg/L below floor 0.0005 — registry "
        f"routing may have misfired or active species PK is way off."
    )
    # Upper-bound sanity (10x literature) — catches obvious calibration drift
    assert cmax < 0.10, (
        f"simvastatin acid Cmax {cmax:.5f} mg/L above 0.10 (10x Najib 2003 "
        f"upper of 0.007) — possible double-routing or yield error."
    )
