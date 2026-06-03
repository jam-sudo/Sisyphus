"""Regression: measured_adme=None must leave the SMILES-only path bit-identical."""
import pytest

from sisyphus.pipeline.predict import predict
from sisyphus.predict.adme import MeasuredADMEInput

# Diverse, valid SMILES — identity-agnostic.
_SMILES = [
    "CC(=O)Oc1ccccc1C(=O)O",                   # aspirin
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",            # caffeine
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",              # ibuprofen
    "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O",   # warfarin
]


@pytest.mark.parametrize("smiles", _SMILES)
def test_none_is_bitidentical(smiles):
    a = predict(smiles, 100.0)
    b = predict(smiles, 100.0, measured_adme=None)
    assert a.pk.cmax.mean == b.pk.cmax.mean
    assert a.engine_pk is not None and b.engine_pk is not None
    assert a.engine_pk.cmax.mean == b.engine_pk.cmax.mean


def test_measured_changes_cmax():
    smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
    base = predict(smiles, 100.0)
    meas = predict(smiles, 100.0,
                   measured_adme=MeasuredADMEInput(fup=0.20, clint=200.0))
    assert base.engine_pk.cmax.mean != meas.engine_pk.cmax.mean


def test_warning_tag_present():
    smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
    r = predict(smiles, 100.0,
                measured_adme=MeasuredADMEInput(fup=0.20, clint=200.0))
    assert any("measured_adme" in w for w in r.warnings)


# ── measured-F routing (f_bioavail) ──────────────────────────────────────

_CAFFEINE = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"


@pytest.mark.parametrize("smiles", _SMILES)
def test_f_bioavail_none_is_bitidentical(smiles):
    # An f_bioavail=None input must not perturb the engine track at all.
    a = predict(smiles, 100.0)
    b = predict(smiles, 100.0, measured_adme=MeasuredADMEInput(f_bioavail=None))
    assert a.engine_pk is not None and b.engine_pk is not None
    assert a.engine_pk.cmax.mean == b.engine_pk.cmax.mean
    assert a.pk.cmax.mean == b.pk.cmax.mean


def test_f_bioavail_scales_cmax_and_auc_together():
    # Exposure-scaling applies one factor k to both Cmax and AUC.
    base = predict(_CAFFEINE, 100.0).engine_pk
    corr = predict(_CAFFEINE, 100.0,
                   measured_adme=MeasuredADMEInput(f_bioavail=0.5)).engine_pk
    cmax_ratio = corr.cmax.mean / base.cmax.mean
    auc_ratio = corr.auc_0t.mean / base.auc_0t.mean
    assert cmax_ratio == pytest.approx(auc_ratio, rel=1e-6)


def test_f_bioavail_cmax_is_linear_in_F():
    # Cmax ∝ F (the engine's own F_engine cancels), so the ratio equals the F
    # ratio exactly — a clean, IV-free check of the core behavior.
    lo = predict(_CAFFEINE, 100.0,
                 measured_adme=MeasuredADMEInput(f_bioavail=0.3)).engine_pk
    hi = predict(_CAFFEINE, 100.0,
                 measured_adme=MeasuredADMEInput(f_bioavail=0.9)).engine_pk
    assert hi.cmax.mean / lo.cmax.mean == pytest.approx(3.0, rel=1e-4)


def test_f_bioavail_target_hitting():
    # After correction, the engine's oral F (corrected AUC / IV AUC) ≈ F_measured.
    iv_auc = predict(_CAFFEINE, 100.0, route="iv").engine_pk.auc_0t.mean
    target = 0.5
    corr = predict(_CAFFEINE, 100.0,
                   measured_adme=MeasuredADMEInput(f_bioavail=target)).engine_pk
    corrected_f = corr.auc_0t.mean / iv_auc
    assert corrected_f == pytest.approx(target, rel=0.05)


def test_f_bioavail_warning_reports_engine_f_and_k():
    r = predict(_CAFFEINE, 100.0,
                measured_adme=MeasuredADMEInput(f_bioavail=0.5))
    assert any("f_bioavail=0.5" in w and "f_engine=" in w and "k=" in w
               for w in r.warnings)


def test_f_bioavail_ignored_for_iv_route():
    # F=1 by definition for IV; f_bioavail must be ignored (no scaling) + warned.
    base = predict(_CAFFEINE, 100.0, route="iv").engine_pk
    r = predict(_CAFFEINE, 100.0, route="iv",
                measured_adme=MeasuredADMEInput(f_bioavail=0.5))
    assert r.engine_pk.cmax.mean == base.cmax.mean
    assert any("ignored for non-oral" in w for w in r.warnings)
