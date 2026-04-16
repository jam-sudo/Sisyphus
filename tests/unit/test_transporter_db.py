"""Unit tests for the OATP1B1 transporter database loader."""

from __future__ import annotations

import pytest

from sisyphus.core import TransporterKinetics
from sisyphus.predict.transporter_db import load_oatp1b1_kinetics


def test_load_pravastatin():
    kinetics = load_oatp1b1_kinetics("pravastatin")
    assert kinetics is not None
    assert "OATP1B1" in kinetics
    tk = kinetics["OATP1B1"]
    assert isinstance(tk, TransporterKinetics)
    assert tk.jmax.mean == pytest.approx(228.0)
    assert tk.jmax.cv == pytest.approx(0.30)
    assert tk.km.mean == pytest.approx(13.6)
    assert tk.km.cv == pytest.approx(0.25)


def test_load_unknown_drug_returns_none():
    assert load_oatp1b1_kinetics("aspirin") is None


def test_load_is_case_insensitive():
    assert load_oatp1b1_kinetics("Pravastatin") is not None
    assert load_oatp1b1_kinetics("PRAVASTATIN") is not None


def test_build_drug_on_graph_carries_transporter_kinetics():
    """Passing transporter_kinetics to build_drug_on_graph populates the field."""
    from sisyphus.predict.adme import predict_adme
    from sisyphus.predict.chemistry import compute_profile
    from sisyphus.predict.ivive import build_drug_on_graph

    pravastatin_smiles = (
        "CC[C@@H](C)C(=O)O[C@@H]1C[C@H](C=C2[C@@H]1CC[C@H]"
        "([C@@H]2CC[C@H](C[C@H](CC(=O)O)O)O)C)O"
    )
    profile = compute_profile(pravastatin_smiles)
    adme = predict_adme(profile)
    kinetics = load_oatp1b1_kinetics("pravastatin")

    drug = build_drug_on_graph(
        profile, adme, dose_mg=40.0, route="oral",
        transporter_kinetics=kinetics,
    )
    assert "OATP1B1" in drug.transporter_kinetics
    assert drug.transporter_kinetics["OATP1B1"].jmax.mean == pytest.approx(228.0)


def test_build_drug_on_graph_without_transporter_is_empty():
    from sisyphus.predict.adme import predict_adme
    from sisyphus.predict.chemistry import compute_profile
    from sisyphus.predict.ivive import build_drug_on_graph

    profile = compute_profile("CCO")  # ethanol
    adme = predict_adme(profile)
    drug = build_drug_on_graph(profile, adme, dose_mg=10.0, route="oral")
    assert drug.transporter_kinetics == {}
