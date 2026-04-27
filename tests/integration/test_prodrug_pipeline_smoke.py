"""Smoke + validation tests for 4 evidence prodrugs.

Validation gate: each drug's predicted Cmax must be within 3-fold of
clinical reference Cmax (from holdout_n50.json).

These tests act as scientific gates -- if any drug fails, spec validation
§7.5 failure response applies (registry value re-verification → routing
override → spec re-open).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from rdkit import Chem


# Reference Cmax (mg/L) from holdout_n50.json
REF_CMAX = {
    "sepiapterin":         0.0024,   # parent, 60 mg/kg fed (2.4 ng/mL)
    "remdesivir":          4.380,    # parent, 200 mg IV
    "tebipenem_pivoxil":   4.006,    # active tebipenem, 300 mg PO
    "fostamatinib":        0.605,    # active R406, 75 mg PO
}

DOSE_ROUTE = {
    "sepiapterin":         (4200.0, "oral"),
    "remdesivir":          (200.0, "iv"),
    "tebipenem_pivoxil":   (300.0, "oral"),
    "fostamatinib":        (75.0, "oral"),
}


def _load_drug_inputs() -> dict[str, str]:
    """Returns drug name -> canonical SMILES from registry+holdout cross-ref."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    with (repo_root / "data" / "reference" / "holdout_n50.json").open() as f:
        n50 = json.load(f)
    drugs = n50.get("drugs", n50)
    with (repo_root / "data" / "sbi" / "prodrug_activation_registry.json").open() as f:
        registry = json.load(f)

    smiles_map = {}
    for name in REF_CMAX.keys():
        if isinstance(drugs, dict):
            entry = drugs.get(name)
        else:
            entry = next((d for d in drugs if d.get("name") == name), None)
        if entry is None:
            continue
        raw_smiles = entry.get("smiles") or entry.get("SMILES")
        if raw_smiles is None:
            continue
        canonical = Chem.MolToSmiles(Chem.MolFromSmiles(raw_smiles), canonical=True)
        if canonical in registry:
            smiles_map[name] = canonical
    return smiles_map


@pytest.mark.parametrize("drug_name", list(REF_CMAX.keys()))
def test_prodrug_pipeline_smoke(drug_name):
    """Each drug runs end-to-end without error; Cmax positive."""
    from sisyphus.pipeline.predict import predict
    smiles_map = _load_drug_inputs()
    if drug_name not in smiles_map:
        pytest.skip(f"SMILES for {drug_name} not found in registry/holdout cross-ref")
    smiles = smiles_map[drug_name]
    dose, route = DOSE_ROUTE[drug_name]
    result = predict(smiles=smiles, dose_mg=dose, route=route, n_mc_samples=0)
    assert result.pk.cmax.mean > 0, (
        f"{drug_name}: Cmax should be positive, got {result.pk.cmax.mean}")
    # Warnings should mention prodrug routing
    assert any("prodrug" in w.lower() or "routed" in w.lower()
               or "active metabolite" in w.lower()
               for w in result.warnings), (
        f"{drug_name}: expected prodrug warning in {result.warnings}")


@pytest.mark.parametrize("drug_name", list(REF_CMAX.keys()))
def test_prodrug_validation_gate_3fold(drug_name):
    """Predicted Cmax within 3-fold of clinical reference.

    Per spec §7.5: failure response is registry re-verification -> routing
    override -> spec re-open. This test is the SUCCESS METRIC for the entire
    plan.
    """
    from sisyphus.pipeline.predict import predict
    smiles_map = _load_drug_inputs()
    if drug_name not in smiles_map:
        pytest.skip(f"SMILES for {drug_name} not found in registry/holdout cross-ref")
    smiles = smiles_map[drug_name]
    dose, route = DOSE_ROUTE[drug_name]
    result = predict(smiles=smiles, dose_mg=dose, route=route, n_mc_samples=0)

    predicted = result.pk.cmax.mean
    reference = REF_CMAX[drug_name]
    fold_error = max(predicted / reference, reference / predicted)
    assert fold_error <= 3.0, (
        f"{drug_name}: predicted Cmax {predicted:.4f} mg/L, "
        f"reference {reference:.4f} mg/L, fold-error {fold_error:.2f}x "
        f"exceeds 3-fold gate"
    )
