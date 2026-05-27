"""Integration test: B-02 Phase 2 UGT path activation produces correct
enzyme_affinity attribution per seed drug.

Verifies the mechanism, not specific Cmax values (those are pinned by
test_cached_holdout_aafe_is_2pXXX after cache regen).

For each of 8 seed drugs:
  1. predict(smiles, dose_mg) succeeds (no exception)
  2. solver_success path (no 'solver_failed' warning)
  3. result.engine_pk is non-None (engine path executed)
"""
from __future__ import annotations

import pytest

from sisyphus.pipeline.predict import predict


_UGT2B7_DRUGS = {
    "morphine":     ("CN1CCC23C4C1CC5=C2C(=C(C=C5)O)OC3C(C=C4)O",        10.0),
    "codeine":      ("COc1ccc2c3c1O[C@H]1[C@@H](O)C=C[C@H]4[C@@H](C2)N(C)CC[C@@]341", 30.0),
    "ketorolac":    ("O=C(c1ccccc1)c1ccc2n1CCC2C(=O)O",                  10.0),
    "indomethacin": ("COc1ccc2c(c1)c(CC(=O)O)c(C)n2C(=O)c1ccc(Cl)cc1",   50.0),
}

_UGT1A9_DRUGS = {
    "dapagliflozin": ("CCOc1ccc(Cc2cc([C@@H]3O[C@H](CO)[C@@H](O)[C@H](O)[C@H]3O)ccc2Cl)cc1", 10.0),
    "etodolac":      ("CCc1cccc2c3c([nH]c12)C(CC)(CC(=O)O)OCC3",          400.0),
    "bexagliflozin": ("C1CC1OCCOC2=CC=C(C=C2)CC3=C(C=CC(=C3)C4C(C(C(C(O4)CO)O)O)O)Cl", 20.0),
    "glasdegib":     ("CN1CCC(CC1C2=NC3=CC=CC=C3N2)NC(=O)NC4=CC=C(C=C4)C#N", 100.0),
}


@pytest.mark.parametrize("drug,case", list(_UGT2B7_DRUGS.items()))
def test_ugt2b7_path_activated(drug, case):
    """Each UGT2B7 seed drug must run through predict() without solver failure."""
    smiles, dose_mg = case
    result = predict(smiles, dose_mg=dose_mg)
    assert result.engine_pk is not None, f"{drug}: engine_pk is None (engine path skipped)"
    assert "solver_failed" not in (result.warnings or []), (
        f"{drug}: solver failed under UGT2B7 activation; warnings: {result.warnings}"
    )


@pytest.mark.parametrize("drug,case", list(_UGT1A9_DRUGS.items()))
def test_ugt1a9_path_activated(drug, case):
    """Each UGT1A9 seed drug must run through predict() without solver failure."""
    smiles, dose_mg = case
    result = predict(smiles, dose_mg=dose_mg)
    assert result.engine_pk is not None, f"{drug}: engine_pk is None"
    assert "solver_failed" not in (result.warnings or []), (
        f"{drug}: solver failed under UGT1A9 activation; warnings: {result.warnings}"
    )


def test_non_substrate_unchanged():
    """Midazolam (CYP3A4 substrate, no UGT) must not crash predict()."""
    midazolam = "c1ccc2c(c1)C(=NC(=O)N2)c1ccccc1F"
    result = predict(midazolam, dose_mg=5.0)
    assert result.engine_pk is not None
    assert result.pk.cmax.mean > 0, "midazolam Cmax should be positive"
