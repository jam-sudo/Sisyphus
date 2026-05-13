"""Per-prodrug Cmax snapshot test (+/-5%).

Catches silent drift below the 3-fold validation gate threshold.
Update _PINNED explicitly when intentionally re-baselining.

v3 re-baselined 2026-05-01 (post Items 1-6 dispositions per spec §6.1).

Hardening re-baselined 2026-05-01 (predict() deterministic path uses
BodyGraph.realize_means() instead of graph.sample(rng=42)).

Public-only re-baselined 2026-05-09 (audit-driven honesty pass): pinned
values were generated on a local-developer state with proprietary DrugBank
artifacts (`data/drugbank/`) and a gitignored logp_correction XGBoost
residual model (`models/adme/logp_correction.json`). Both are loaded
conditionally by predict() and silently shifted Cmax. Public-clone
deterministic values (no DrugBank, no logp_correction):
  sepiapterin       1.130e+01 → 8.679e+00 (-23.2%)
  remdesivir        9.837e-01 → 1.573e+00 (+59.9%)
  tebipenem_pivoxil 5.207e-01 → 4.553e-01 (-12.6%)
  fostamatinib      1.260e-01 → 6.675e-02 (-47.0%)
The 4 prodrug 3-fold validation gates remain xfail per spec §6.4
(extraction-step rate-limits dominate active CL/V disposition);
these snapshots verify mechanical correctness, not absolute clinical
accuracy. NOTE: remdesivir's 3-fold gate (test_prodrug_v2_validation_gate)
flips xfail→PASS under public-only Cmax (1.573 mg/L vs FDA observed
~3.0 mg/L → FE 1.91 < 3.0 gate); see that test for the strict→strict-False
adjustment.
"""
from __future__ import annotations

import pytest

from sisyphus.pipeline.predict import predict
from tests._artifact_helpers import skip_if_local_artifacts

_PINNED = {
    "sepiapterin":       8.679384e+00,
    "remdesivir":        1.573162e+00,
    "tebipenem_pivoxil": 4.553034e-01,
    "fostamatinib":      6.675183e-02,
}

_RTOL = 0.05

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


@skip_if_local_artifacts
@pytest.mark.parametrize("drug_name", list(_PINNED.keys()))
def test_cmax_snapshot(drug_name):
    pinned = _PINNED[drug_name]
    smiles = _SMILES[drug_name]
    dose, route = _DOSE_ROUTE[drug_name]

    result = predict(smiles, dose_mg=dose, route=route, n_mc_samples=0)
    actual = result.pk.cmax.mean

    rel_err = abs(actual - pinned) / pinned
    assert rel_err < _RTOL, (
        f"{drug_name} Cmax drifted: actual={actual:.6e}, pinned={pinned:.6e}, "
        f"rel_err={rel_err:.4f} (>{_RTOL}). If intentional, update _PINNED."
    )
