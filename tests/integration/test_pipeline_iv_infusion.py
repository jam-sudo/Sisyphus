"""V3.1 integration: IV infusion routes through regimen.solver.

Spec: docs/_internal/specs/2026-04-22-v3.1-iv-infusion-design.md §5 Phase 1.

Invariants to preserve:
- Oral and IV bolus paths unchanged (V3 behavior).
- IV infusion with infusion_duration_min > 0.5 (min) uses regimen.solver,
  producing a physically lower Cmax than the equivalent bolus.
"""

from __future__ import annotations

import pytest

from sisyphus.pipeline.predict import predict

# Reuse substrate from test_pipeline_iv_cmax.py: valsartan is a small-molecule
# drug whose ADME pipeline is fast and stable; it also appears in the ECM
# generalization set. We do NOT need rifampin-specific behavior here — any IV
# drug exercises the route-branching code path.
_VALSARTAN_SMILES = "CCCCC(=O)N([C@@H](C(C)C)C(=O)O)Cc1ccc(-c2ccccc2-c2nnn[nH]2)cc1"
_ATORVASTATIN_SMILES = (
    "CC(C)c1c(/C=C/[C@H](O)C[C@H](O)CC(=O)O)n(CCc2ccc(F)cc2)c(-c2ccccc2)c1-c1ccc(F)cc1"
)


def test_infusion_cmax_lower_than_bolus():
    """IV infusion (30 min) Cmax must be strictly lower than IV bolus Cmax."""
    bolus = predict(_VALSARTAN_SMILES, 20.0, route="iv")
    infusion = predict(_VALSARTAN_SMILES, 20.0, route="iv", infusion_duration_min=30.0)
    assert bolus.engine_pk is not None
    assert infusion.engine_pk is not None
    assert infusion.engine_pk.cmax.mean < bolus.engine_pk.cmax.mean


def test_infusion_tmax_near_end_of_infusion():
    """Infusion Cmax occurs at or shortly after end-of-infusion (0.5h ± buffer)."""
    infusion = predict(_VALSARTAN_SMILES, 20.0, route="iv", infusion_duration_min=30.0)
    assert infusion.engine_pk is not None
    # End-of-infusion is 0.5h; Cmax should be within 0.5h of that.
    tmax = infusion.engine_pk.tmax.mean
    assert 0.4 <= tmax <= 1.5, f"Tmax={tmax} outside expected infusion window"


def test_short_infusion_falls_back_to_bolus():
    """infusion_duration_min <= 0.5 (min) routes to IV bolus path (V3 unchanged)."""
    bolus = predict(_VALSARTAN_SMILES, 20.0, route="iv")
    short = predict(_VALSARTAN_SMILES, 20.0, route="iv", infusion_duration_min=0.3)
    assert bolus.engine_pk is not None
    assert short.engine_pk is not None
    # Both go through bolus path — Cmax must match.
    assert abs(bolus.engine_pk.cmax.mean - short.engine_pk.cmax.mean) < 1e-6


def test_oral_rejects_infusion_duration():
    """Oral route with infusion_duration_min is invalid input."""
    with pytest.raises(ValueError, match="infusion_duration_min"):
        predict(_ATORVASTATIN_SMILES, 20.0, route="oral", infusion_duration_min=30.0)


def test_negative_infusion_duration_rejected():
    """Negative infusion_duration_min raises ValueError."""
    with pytest.raises(ValueError, match="infusion_duration_min"):
        predict(_VALSARTAN_SMILES, 20.0, route="iv", infusion_duration_min=-5.0)


def test_infusion_skips_mc_with_warning():
    """MC on infusion is Phase 2 scope — Phase 1 skips with warning."""
    result = predict(
        _VALSARTAN_SMILES, 20.0, route="iv",
        infusion_duration_min=30.0, n_mc_samples=50,
    )
    assert result.engine_pk is not None
    assert result.cmax_90ci is None
    assert any("infusion" in w.lower() for w in result.warnings)
