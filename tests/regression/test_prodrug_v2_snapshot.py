"""Per-prodrug Cmax snapshot test (+/-5%).

Catches silent drift below the 3-fold validation gate threshold.
Update _PINNED explicitly when intentionally re-baselining.
"""
from __future__ import annotations

import pytest

from sisyphus.pipeline.predict import predict


_PINNED = {
    "sepiapterin":       1.126096e+01,
    "remdesivir":        9.890671e-01,
    "tebipenem_pivoxil": 4.445988e-01,
    "fostamatinib":      1.351613e-01,
}

_RTOL = 0.05

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
