"""CYP/transporter phenotype propagation regression — back-solve cancellation fix gate.

Prior to v0.3.2 commit (Task 2), `_decompose_clint` back-solved enzyme affinity
from abundance, and `pipeline/predict.py` rebuilt the drug AFTER applying the
phenotype scaling — the rebuild cancelled the scaling exactly. CYP1A2:PM,
CYP2C9:PM, etc. produced ratio 1.000×. SLCO1B1:PM escaped because OATP1B1
uses saturable Michaelis-Menten kinetics (no back-solve).

These tests fail on pre-fix main and pass after the pipeline/predict.py
back-solve cancellation fix.
"""
from __future__ import annotations

import pytest

from sisyphus.pipeline.predict import predict


_CAFFEINE_SMILES = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"
_WARFARIN_SMILES = "CC(=O)CC(c1ccccc1)C1=C(O)c2ccccc2OC1=O"
_PRAVASTATIN_SMILES = (
    "CC[C@@H](C)C(=O)O[C@@H]1C[C@H](C=C2[C@@H]1CC[C@H]"
    "([C@@H]2CC[C@H](C[C@H](CC(=O)O)O)O)C)O"
)


@pytest.mark.slow
def test_caffeine_cyp1a2_pm_propagates():
    """CYP1A2:PM should drop caffeine clearance, raising Cmax > 1.5× EM.

    Caffeine is ~80% CYP1A2-metabolized; PM scaling × 0.10 → CYP1A2
    contribution drops to 0.08, residual ~0.20 from other CYPs → total
    CL ~0.28 of EM → Cmax ~3.5×. Gate at 1.5× is conservative.
    """
    em = predict(_CAFFEINE_SMILES, dose_mg=100.0, phenotypes={"CYP1A2": "EM"})
    pm = predict(_CAFFEINE_SMILES, dose_mg=100.0, phenotypes={"CYP1A2": "PM"})
    assert em.engine_pk is not None and pm.engine_pk is not None
    ratio = pm.engine_pk.cmax.mean / em.engine_pk.cmax.mean
    assert ratio > 1.5, (
        f"CYP1A2:PM/EM Cmax ratio {ratio:.3f} ≤ 1.5 — back-solve cancellation "
        f"may have regressed (PM should drop CL, raising Cmax)."
    )


@pytest.mark.slow
def test_warfarin_cyp2c9_pm_propagates():
    """CYP2C9:PM should drop warfarin clearance, raising Cmax > 1.2× EM.

    Acid compound_type allocates fm CYP2C9 0.40 → PM × 0.10 → 0.04;
    residual 0.60 → total ~0.64 of EM → Cmax ~1.56×. Gate at 1.2× is
    conservative against fm uncertainty.
    """
    em = predict(_WARFARIN_SMILES, dose_mg=10.0, phenotypes={"CYP2C9": "EM"})
    pm = predict(_WARFARIN_SMILES, dose_mg=10.0, phenotypes={"CYP2C9": "PM"})
    assert em.engine_pk is not None and pm.engine_pk is not None
    ratio = pm.engine_pk.cmax.mean / em.engine_pk.cmax.mean
    assert ratio > 1.2, (
        f"CYP2C9:PM/EM Cmax ratio {ratio:.3f} ≤ 1.2 — back-solve cancellation "
        f"may have regressed."
    )


@pytest.mark.slow
def test_pravastatin_slco1b1_pm_still_works():
    """SLCO1B1:PM transporter path is unaffected by back-solve fix.

    OATP1B1 uses saturable Michaelis-Menten kinetics, not affinity back-solve.
    PM:EM ~3× per Niemi 2009 + earlier empirical 3.034 on this codebase.
    Gate at 2.5× backstops both pre-fix and post-fix behavior.
    """
    em = predict(_PRAVASTATIN_SMILES, dose_mg=40.0, phenotypes={"SLCO1B1": "EM"})
    pm = predict(_PRAVASTATIN_SMILES, dose_mg=40.0, phenotypes={"SLCO1B1": "PM"})
    assert em.engine_pk is not None and pm.engine_pk is not None
    ratio = pm.engine_pk.cmax.mean / em.engine_pk.cmax.mean
    assert ratio > 2.5, (
        f"SLCO1B1:PM/EM Cmax ratio {ratio:.3f} ≤ 2.5 — transporter phenotype "
        f"path may have regressed."
    )
