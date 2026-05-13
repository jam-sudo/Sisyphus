"""v2/v3 validation gate (per-drug parametrized 3-fold).

Per spec section 6.1: failing-drug parametrize cases are marked xfail with
documented reason. Affinity values NOT adjusted to make tests pass --
mechanistic-A core promise.

v3 update (2026-05-01, prodrug-v3 dispositions per
docs/superpowers/specs/2026-04-29-prodrug-v3-literature.md):
4 of 4 drugs remain xfail. v3 brought 2 literature-applied registry
updates (Items 2 + 3); fold errors essentially unchanged because
conversion-step extraction efficiency (well-stirred E ~1 at high CLint)
dominates over active-compartment CL/Vd disposition.

Public-only update (2026-05-09, audit-driven honesty pass):
Pinned values were generated with DrugBank+logp_correction enrichment.
Under public-clone deterministic state (no DrugBank, no logp_correction),
remdesivir parent Cmax shifts 0.987 → 1.573 mg/L → fold error 2.78x
(under 3.0x gate). Remdesivir promoted out of xfail. The other 3 remain
xfail (sepiapterin 23×→smaller-but-still-fail, tebipenem ~9×, fostamatinib
~9× under public-only).

Current empirical fold-errors (2026-05-01, v3 dispositions applied):
  - sepiapterin       4748x  (pred 11.40 mg/L vs obs 0.0024 mg/L)
                      Item 1 ceiling_accepted (F_sapropterin not located in
                      primary literature; FDA/EMA explicit). v1 Vd=150
                      retained as known-wrong placeholder. SPR class-
                      extrapolated affinity unchanged from v2; conversion
                      saturation at 4200 mg dose dominates.
  - remdesivir        4.44x  (pred 0.987 mg/L vs obs 4.38 mg/L)
                      Item 2 literature_applied — GS-441524 active
                      CL=17.4 L/h, V=535 L (Tamura 2023 + Leegwater 2022
                      geomean). observation_species=parent → active
                      CL/V update doesn't shift parent Cmax. CES1 abundance
                      only at liver still lacks systemic esterase distribution.
  - tebipenem_pivoxil 9.05x  (pred 0.443 mg/L vs obs 4.01 mg/L)
                      Item 4 + Item 6 ceiling_accepted (F_tebipenem absolute
                      not located; CES2/tebipenem in vitro Vmax/Km not
                      located). v2 placeholder Vd=50, CL=17 retained;
                      class-extrapolated CES2 affinity retained.
  - fostamatinib      4.50x  (pred 0.136 mg/L vs obs 0.61 mg/L)
                      Item 3 literature_applied — R406 active CL=15.7 L/h,
                      Vss=256 L (Matsukane 2022 IV microdose). Active
                      compartment ke (CL/V) reduced ~44%; Cmax shift
                      marginal because conversion-step (ALPI extraction)
                      remains rate-limiting.

These are NOT relaxed by tuning affinities. Resolution requires:
  (a) literature-grade in vitro kinetic data per enzyme/drug (Items 5-6
      ceiling: not yet located in literature), or
  (b) extra mechanistic terms beyond v3 scope (e.g., extra-hepatic esterase
      abundance, intestinal-loss kinetics, BH4 first-pass depletion).
"""
from __future__ import annotations

import pytest

from sisyphus.pipeline.predict import predict
from tests._artifact_helpers import skip_if_local_artifacts

_CLINICAL_CMAX = {
    "sepiapterin":       0.0024,
    "remdesivir":        4.38,
    "tebipenem_pivoxil": 4.01,
    "fostamatinib":      0.61,
}

_SMILES = {
    "sepiapterin":       "C[C@H](O)C(=O)C1=Nc2c(nc(N)[nH]c2=O)NC1",
    "remdesivir":        "CCC(CC)COC(=O)[C@H](C)N[P@](=O)(OC[C@H]1O[C@@](C#N)(c2ccc3c(N)ncnn23)[C@H](O)[C@@H]1O)Oc1ccccc1",  # noqa: E501
    "tebipenem_pivoxil": "C[C@@H](O)[C@H]1C(=O)N2C(C(=O)OCOC(=O)C(C)(C)C)=C(SC3CN(C4=NCCS4)C3)[C@H](C)[C@H]12",  # noqa: E501
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
        "v3 4748x overpredict (Item 1 ceiling_accepted — F_sapropterin not "
        "located in primary literature, v1 Vd=150 placeholder retained). "
        "SPR class-extrapolated affinity; 4200 mg dose pre-systemic conversion "
        "saturation dominates. Resolution: literature-grade BH4 IV F primary "
        "study or extra-hepatic SPR proteomic abundance.",
    # remdesivir: removed from xfail 2026-05-09 (public-only honesty pass).
    # Under public-clone deterministic state, parent Cmax 1.573 mg/L vs FDA
    # observed ~4.38 → fold error 2.78x (under 3.0 gate). Local-developer
    # state with DrugBank+logp_correction had Cmax 0.987 → fold 4.44 (xfail).
    "tebipenem_pivoxil":
        "v3 9.05x underpredict (Item 4 + Item 6 stacked ceiling — F_tebipenem "
        "absolute not located, CES2/tebipenem in vitro Vmax/Km not located). "
        "Class-extrapolated CES2 affinity retained. Resolution: literature-grade "
        "in vitro CES2/tebipenem-pivoxil kinetic data.",
    "fostamatinib":
        "v3 4.50x underpredict (Item 3 literature_applied — R406 active CL=15.7, "
        "Vss=256 from Matsukane 2022 IV microdose). Active ke reduced ~44%; "
        "Cmax shift marginal because ALPI extraction remains rate-limiting. "
        "Resolution: literature-grade in vitro ALPI/fostamatinib kinetics.",
}


@skip_if_local_artifacts
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
