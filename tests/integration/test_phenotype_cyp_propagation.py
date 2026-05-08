"""CYP/transporter phenotype propagation regression — back-solve cancellation fix gate.

Prior to v0.3.2 commit (Task 2), `_decompose_clint` back-solved enzyme affinity
from abundance, and `pipeline/predict.py` rebuilt the drug AFTER applying the
phenotype scaling — the rebuild cancelled the scaling exactly. CYP1A2:PM,
CYP2C9:PM, etc. produced ratio 1.000×. SLCO1B1:PM escaped because OATP1B1
uses saturable Michaelis-Menten kinetics (no back-solve).

These tests fail on pre-fix main and pass after the pipeline/predict.py
back-solve cancellation fix.

Drug selection rationale:
- Tizanidine: DrugBank annotates CYP1A2 ONLY → fm_CYP1A2 = 1.0, moderate CLint
  → PM drops CYP1A2 abundance 10×, total hepatic CLint ~10× lower → Cmax ~1.5×.
- Irbesartan: DrugBank annotates CYP2C9 ONLY → fm_CYP2C9 = 1.0, moderate CLint
  → PM drops CYP2C9 abundance 10×, total hepatic CLint ~10× lower → Cmax ~1.2×.
- Caffeine/warfarin avoided: DrugBank annotates 3–5 CYP enzymes (equal fm each),
  diluting the single-enzyme PM effect to < 1.1×.
"""
from __future__ import annotations

import pytest

from sisyphus.pipeline.predict import predict


# CYP1A2 probe: tizanidine (DrugBank: CYP1A2 only)
_TIZANIDINE_SMILES = "C1CN=C(N1)NC2=C(C=CC3=NSN=C32)Cl"

# CYP2C9 probe: irbesartan (DrugBank: CYP2C9 only)
_IRBESARTAN_SMILES = "CCCCC1=NC2(CCCC2)C(=O)N1CC3=CC=C(C=C3)C4=CC=CC=C4C5=NNN=N5"

# SLCO1B1 probe: pravastatin (ECM / transporter path)
_PRAVASTATIN_SMILES = (
    "CC[C@@H](C)C(=O)O[C@@H]1C[C@H](C=C2[C@@H]1CC[C@H]"
    "([C@@H]2CC[C@H](C[C@H](CC(=O)O)O)O)C)O"
)


@pytest.mark.slow
def test_tizanidine_cyp1a2_pm_propagates():
    """CYP1A2:PM should drop tizanidine clearance, raising Cmax > 1.2× EM.

    Tizanidine is annotated in DrugBank as CYP1A2-only substrate → fm_CYP1A2=1.0.
    PM scaling × 0.10 → total hepatic CLint ~0.10 of EM → Cmax ~1.5× in practice
    (well-stirred model moderates theoretical max). Gate at 1.2× is conservative
    and clearly above the pre-fix 1.000× (exact cancellation).
    """
    em = predict(_TIZANIDINE_SMILES, dose_mg=4.0, phenotypes={"CYP1A2": "EM"})
    pm = predict(_TIZANIDINE_SMILES, dose_mg=4.0, phenotypes={"CYP1A2": "PM"})
    assert em.engine_pk is not None and pm.engine_pk is not None
    ratio = pm.engine_pk.cmax.mean / em.engine_pk.cmax.mean
    assert ratio > 1.2, (
        f"CYP1A2:PM/EM Cmax ratio {ratio:.3f} ≤ 1.2 — back-solve cancellation "
        f"may have regressed (PM should drop CL, raising Cmax). "
        f"Pre-fix canonical: 1.000. Post-fix expected: ~1.5."
    )


@pytest.mark.slow
def test_irbesartan_cyp2c9_pm_propagates():
    """CYP2C9:PM should drop irbesartan clearance, raising Cmax > 1.1× EM.

    Irbesartan is annotated in DrugBank as CYP2C9-only substrate → fm_CYP2C9=1.0.
    PM scaling × 0.10 → total hepatic CLint ~0.10 of EM → Cmax ~1.25× in practice.
    Gate at 1.1× is conservative and clearly above the pre-fix 1.000× (exact
    cancellation). CYP2C9 produces a smaller ratio than CYP1A2 here due to higher
    extraction ratio (lower sensitivity in well-stirred model).
    """
    em = predict(_IRBESARTAN_SMILES, dose_mg=150.0, phenotypes={"CYP2C9": "EM"})
    pm = predict(_IRBESARTAN_SMILES, dose_mg=150.0, phenotypes={"CYP2C9": "PM"})
    assert em.engine_pk is not None and pm.engine_pk is not None
    ratio = pm.engine_pk.cmax.mean / em.engine_pk.cmax.mean
    assert ratio > 1.1, (
        f"CYP2C9:PM/EM Cmax ratio {ratio:.3f} ≤ 1.1 — back-solve cancellation "
        f"may have regressed. Pre-fix canonical: 1.000. Post-fix expected: ~1.25."
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
