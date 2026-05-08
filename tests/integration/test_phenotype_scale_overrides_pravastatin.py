"""Integration test for phenotype_scale_overrides via predict() — pravastatin SLCO1B1 (#31).

End-to-end gate: override 0.30 (vs CPIC default 0.10) for SLCO1B1:PM
compresses pravastatin Cmax toward EM. The GenoADME use case is
calibrating Sisyphus's PM/EM AUC ratio toward Niemi 2006 men-stratum
central 3.32 (vs current default ~4.5); the equivalent Cmax-side
ordering test is what we verify here.
"""
from __future__ import annotations

import pytest

from sisyphus.pipeline.predict import predict


_PRAVASTATIN = (
    "CC[C@@H](C)C(=O)O[C@@H]1C[C@H](C=C2[C@@H]1CC[C@H]"
    "([C@@H]2CC[C@H](C[C@H](CC(=O)O)O)O)C)O"
)


@pytest.mark.slow
def test_pravastatin_slco1b1_pm_override_compresses_toward_em():
    """SLCO1B1:PM override 0.30 must produce a Cmax between EM and default-PM.

    Compression ordering: EM < Override-PM < Default-PM.
    Default PM scales OATP1B1 abundance × 0.10 → maximum uptake reduction
    → highest Cmax. Override 0.30 scales × 0.30 → less uptake reduction
    → Cmax closer to EM. EM is unchanged baseline.
    """
    em = predict(_PRAVASTATIN, dose_mg=40.0, phenotypes={"SLCO1B1": "EM"})
    default_pm = predict(_PRAVASTATIN, dose_mg=40.0, phenotypes={"SLCO1B1": "PM"})
    override_pm = predict(
        _PRAVASTATIN, dose_mg=40.0,
        phenotypes={"SLCO1B1": "PM"},
        phenotype_scale_overrides={"SLCO1B1": 0.30},
    )
    assert em.engine_pk is not None and default_pm.engine_pk is not None and override_pm.engine_pk is not None

    em_cmax = em.engine_pk.cmax.mean
    default_pm_cmax = default_pm.engine_pk.cmax.mean
    override_pm_cmax = override_pm.engine_pk.cmax.mean

    assert em_cmax < override_pm_cmax < default_pm_cmax, (
        f"compression ordering violated: EM={em_cmax:.4f}, "
        f"Override-PM={override_pm_cmax:.4f}, Default-PM={default_pm_cmax:.4f}"
    )

    # Override compresses the PM/EM ratio toward unity
    default_ratio = default_pm_cmax / em_cmax
    override_ratio = override_pm_cmax / em_cmax
    assert 1.0 < override_ratio < default_ratio, (
        f"override ratio {override_ratio:.3f} not between 1.0 and "
        f"default {default_ratio:.3f}"
    )


@pytest.mark.slow
def test_pravastatin_no_override_unchanged():
    """Calling predict() without phenotype_scale_overrides must produce
    identical Cmax to omitting the kwarg entirely (backward compat)."""
    a = predict(_PRAVASTATIN, dose_mg=40.0, phenotypes={"SLCO1B1": "PM"})
    b = predict(
        _PRAVASTATIN, dose_mg=40.0, phenotypes={"SLCO1B1": "PM"},
        phenotype_scale_overrides=None,
    )
    c = predict(
        _PRAVASTATIN, dose_mg=40.0, phenotypes={"SLCO1B1": "PM"},
        phenotype_scale_overrides={},
    )
    assert a.engine_pk.cmax.mean == pytest.approx(b.engine_pk.cmax.mean)
    assert a.engine_pk.cmax.mean == pytest.approx(c.engine_pk.cmax.mean)
