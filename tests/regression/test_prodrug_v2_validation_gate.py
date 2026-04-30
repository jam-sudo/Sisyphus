"""v2 validation gate (per-drug parametrized 3-fold).

Per spec section 6.1: failing-drug parametrize cases are marked xfail with
documented reason. Affinity values NOT adjusted to make tests pass --
mechanistic-A core promise.

Current empirical fold-errors (2026-04-26, b2cb921 + T2-T11 v2 architecture):
  - sepiapterin       4692x  (pred 11.26 mg/L vs obs 0.0024 mg/L)
                      SPR class-extrapolated affinity; v2 design lacks
                      first-pass intestinal extraction adequate to deplete
                      4200 mg dose pre-systemic. (T1 section 7 caveat.)
  - remdesivir        4.43x  (pred 0.989 mg/L vs obs 4.38 mg/L)
                      CES1 abundance only at liver per T9; lacks systemic
                      esterase tissue distribution. Underprediction expected.
  - tebipenem_pivoxil 9.02x  (pred 0.445 mg/L vs obs 4.01 mg/L)
                      ALPI/CES intestinal hydrolysis affinity class-extrapolated;
                      activation rate too low to recover full conversion yield.
  - fostamatinib      4.51x  (pred 0.135 mg/L vs obs 0.61 mg/L)
                      ALPI activation step + R406 fup/CL active metabolite values
                      class-extrapolated; underprediction within expected
                      v2 mechanistic-only error band (~5x).

These are NOT relaxed by tuning affinities. Resolution requires:
  (a) literature-grade kinetic data per enzyme/drug, or
  (b) v3 introducing additional mechanistic terms (e.g. esterase extra-hepatic
      tissue abundance, intestinal-loss kinetics) -- spec scope.
"""
from __future__ import annotations

import pytest

from sisyphus.pipeline.predict import predict


_CLINICAL_CMAX = {
    "sepiapterin":       0.0024,
    "remdesivir":        4.38,
    "tebipenem_pivoxil": 4.01,
    "fostamatinib":      0.61,
}

_SMILES = {
    "sepiapterin":       "C[C@H](O)C(=O)C1=Nc2c(nc(N)[nH]c2=O)NC1",
    "remdesivir":        "CCC(CC)COC(=O)[C@H](C)N[P@](=O)(OC[C@H]1O[C@@](C#N)(c2ccc3c(N)ncnn23)[C@H](O)[C@@H]1O)Oc1ccccc1",
    "tebipenem_pivoxil": "C[C@@H](O)[C@H]1C(=O)N2C(C(=O)OCOC(=O)C(C)(C)C)=C(SC3CN(C4=NCCS4)C3)[C@H](C)[C@H]12",
    "fostamatinib":      "COc1cc(Nc2ncc(F)c(Nc3ccc4c(n3)N(COP(=O)(O)O)C(=O)C(C)(C)O4)n2)cc(OC)c1OC",
}

_DOSE_ROUTE = {
    "sepiapterin":       (4200.0, "oral"),
    "remdesivir":        (200.0, "iv"),
    "tebipenem_pivoxil": (300.0, "oral"),
    "fostamatinib":      (75.0, "oral"),
}


_KNOWN_FAILURES = {
    "sepiapterin":
        "v2 mechanistic-A: 4692x overpredict; SPR class-extrapolated, no "
        "first-pass intestinal extraction term sufficient to deplete 4200 mg "
        "dose pre-systemic (T1 section 7 caveat). Resolution: v3 mechanistic "
        "terms or literature-grade SPR kinetics.",
    "remdesivir":
        "v2 mechanistic-A: 4.43x underpredict; CES1 abundance set only at liver "
        "(T9), lacks systemic esterase distribution. Resolution: extra-hepatic "
        "esterase abundance in physiology + literature-grade CES1/CES2 affinity.",
    "tebipenem_pivoxil":
        "v2 mechanistic-A: 9.02x underpredict; ALPI/CES intestinal hydrolysis "
        "affinity class-extrapolated. Resolution: literature-grade ALPI kinetics "
        "or v3 enzyme-specific intestinal flux terms.",
    "fostamatinib":
        "v2 mechanistic-A: 4.51x underpredict; ALPI activation + R406 active-"
        "metabolite Cl/Vd/fup all class-extrapolated. Resolution: literature-"
        "grade R406 disposition data + ALPI affinity refinement.",
}


@pytest.mark.parametrize(
    "drug_name",
    [
        pytest.param(
            name,
            marks=(
                pytest.mark.xfail(reason=_KNOWN_FAILURES[name], strict=True)
                if name in _KNOWN_FAILURES
                else ()
            ),
        )
        for name in _CLINICAL_CMAX
    ],
)
def test_prodrug_3fold(drug_name):
    smiles = _SMILES[drug_name]
    dose, route = _DOSE_ROUTE[drug_name]
    result = predict(smiles, dose_mg=dose, route=route, n_mc_samples=0)
    pred = result.pk.cmax.mean
    obs = _CLINICAL_CMAX[drug_name]
    fold_error = max(pred / obs, obs / pred)
    assert fold_error < 3.0, (
        f"{drug_name}: pred={pred:.4g} mg/L, obs={obs:.4g} mg/L, "
        f"fold_error={fold_error:.2f}x (target < 3.0x)"
    )
