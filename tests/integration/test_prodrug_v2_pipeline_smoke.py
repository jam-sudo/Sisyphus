"""End-to-end smoke: each registered v2 prodrug runs without error."""
from __future__ import annotations

import pytest

from sisyphus.pipeline.predict import predict


@pytest.mark.parametrize("drug_name,smiles,dose_mg,route", [
    ("sepiapterin",
     "C[C@H](O)C(=O)C1=Nc2c(nc(N)[nH]c2=O)NC1",
     4200.0, "oral"),
    ("remdesivir",
     "CCC(CC)COC(=O)[C@H](C)N[P@](=O)(OC[C@H]1O[C@@](C#N)(c2ccc3c(N)ncnn23)[C@H](O)[C@@H]1O)Oc1ccccc1",
     200.0, "iv"),
    ("tebipenem_pivoxil",
     "C[C@@H](O)[C@H]1C(=O)N2C(C(=O)OCOC(=O)C(C)(C)C)=C(SC3CN(C4=NCCS4)C3)[C@H](C)[C@H]12",
     300.0, "oral"),
    ("fostamatinib",
     "COc1cc(Nc2ncc(F)c(Nc3ccc4c(n3)N(COP(=O)(O)O)C(=O)C(C)(C)O4)n2)cc(OC)c1OC",
     75.0, "oral"),
])
def test_pipeline_runs_for_each_prodrug(drug_name, smiles, dose_mg, route):
    result = predict(smiles, dose_mg=dose_mg, route=route, n_mc_samples=0)
    assert result is not None
    assert result.pk.cmax.mean > 0, f"{drug_name} Cmax should be positive"
