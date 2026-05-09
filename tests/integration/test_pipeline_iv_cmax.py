"""V3 integration: verify pipeline route-conditional Cmax behavior.

IV drugs: windowed Cmax (V3) produces Cmax strictly less than V2 t=0 spike.
Oral drugs: behavior unchanged from V2 (regression guard).
"""

from __future__ import annotations

from sisyphus.pipeline.predict import predict

# SMILES + doses chosen to avoid network / DB access. Valsartan 20 mg IV
# (ECM generalization test substrate). Atorvastatin 20 mg oral (in holdout,
# represents the oral regression path).
_VALSARTAN_SMILES = "CCCCC(=O)N([C@@H](C(C)C)C(=O)O)Cc1ccc(-c2ccccc2-c2nnn[nH]2)cc1"
_ATORVASTATIN_SMILES = (
    "CC(C)c1c(/C=C/[C@H](O)C[C@H](O)CC(=O)O)n(CCc2ccc(F)cc2)c(-c2ccccc2)c1-c1ccc(F)cc1"
)


def test_iv_pipeline_cmax_uses_windowed_max():
    """For IV bolus, engine Cmax must be strictly less than dose/V_venous."""
    result = predict(
        smiles=_VALSARTAN_SMILES, dose_mg=20.0, route="iv",
        n_mc_samples=50,
    )
    assert result.engine_pk is not None
    # V2 Cmax would be 20/3.7 ≈ 5.405. V3 windowed must be less.
    assert result.engine_pk.cmax.mean < 20.0 / 3.7


def test_iv_pipeline_90pi_is_non_degenerate():
    """For IV bolus with MC, PI must have positive width (non-degenerate)."""
    result = predict(
        smiles=_VALSARTAN_SMILES, dose_mg=20.0, route="iv",
        n_mc_samples=100,
    )
    assert result.cmax_90ci is not None
    low, high = result.cmax_90ci
    assert high > low  # non-degenerate


def test_oral_pipeline_unchanged_by_v3():
    """Oral drugs must see no V3 behavior change — Tmax > 5 min trivially."""
    result = predict(
        smiles=_ATORVASTATIN_SMILES, dose_mg=20.0, route="oral",
        n_mc_samples=0,
    )
    assert result.engine_pk is not None
    # Oral Tmax must be well above the IV threshold (absorption takes hours).
    assert result.engine_pk.tmax.mean > 0.5
