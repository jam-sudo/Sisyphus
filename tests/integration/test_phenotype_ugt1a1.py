"""Integration test for UGT1A1 phenotype propagation (issue #10)."""
from __future__ import annotations

import logging

import pytest

from sisyphus.pipeline.predict import predict


# Raltegravir canonical SMILES — Task 4 derived from PubChem CID 54671008.
# This test reads the SMILES from the registry to stay in sync with Task 4.
def _raltegravir_smiles() -> str:
    import json, pathlib
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    data = json.loads((repo_root / "data" / "enzymes" / "ugt1a1_substrates.json").read_text())
    for s in data["substrates"]:
        if s["drug"] == "raltegravir":
            return s["smiles"]
    raise AssertionError("raltegravir not in ugt1a1_substrates.json")


def test_parse_ugt1a1_pm():
    from sisyphus.predict.phenotype import parse_phenotype_spec
    out = parse_phenotype_spec("UGT1A1:PM")
    assert out == {"UGT1A1": "PM"}


def test_apply_ugt1a1_pm_no_warning(caplog):
    from sisyphus.predict.phenotype import apply_phenotype_to_graph
    from sisyphus.graph.builder import build_from_yaml
    import pathlib

    caplog.set_level(logging.WARNING)
    g = build_from_yaml(pathlib.Path("data/physiology/reference_man.yaml"))
    _ = apply_phenotype_to_graph(g, {"UGT1A1": "PM"})
    warnings_about_ugt = [r for r in caplog.records if "UGT1A1" in r.getMessage() and "not found" in r.getMessage()]
    assert not warnings_about_ugt


@pytest.mark.slow
def test_raltegravir_ugt1a1_pm_propagates():
    """UGT1A1:PM should drop raltegravir clearance, raising Cmax > 1.2× EM.

    Iwamoto 2008: UGT1A1*28 carriers ~40% AUC increase. Cmax effect
    smaller; gate at 1.2× is conservative.
    """
    smiles = _raltegravir_smiles()
    em = predict(smiles, dose_mg=400.0, phenotypes={"UGT1A1": "EM"})
    pm = predict(smiles, dose_mg=400.0, phenotypes={"UGT1A1": "PM"})
    assert em.engine_pk is not None and pm.engine_pk is not None
    ratio = pm.engine_pk.cmax.mean / em.engine_pk.cmax.mean
    assert ratio > 1.2, (
        f"UGT1A1:PM/EM Cmax ratio {ratio:.3f} ≤ 1.2 — phenotype propagation gate."
    )
