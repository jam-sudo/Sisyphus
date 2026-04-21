"""Integration smoke test: 2 ECM generalization drugs run end-to-end.

Uses N_SAMPLES=10 for speed; the full pre-registered run uses N=1000.
Assertions are "did it run, did it produce finite numbers" — NOT pass/fail
(that is the actual experiment, only run via scripts/validate_oatp_generalization.py).
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

import sisyphus.engine.flux  # noqa: F401 -- register flux specs
from sisyphus.engine.compiler import ODECompiler
from sisyphus.engine.uncertainty import UncertaintyEngine
from sisyphus.graph.builder import build_from_yaml
from sisyphus.predict.adme import predict_adme
from sisyphus.predict.chemistry import compute_profile
from sisyphus.predict.ivive import build_drug_on_graph
from sisyphus.predict.transporter_db import (
    load_hepatic_ecm_params,
    load_oatp1b1_kinetics,
)

ROOT = pathlib.Path(__file__).resolve().parents[2]
_PHYS = ROOT / "data" / "physiology" / "reference_man.yaml"
_OBS_FILE = ROOT / "data" / "validation" / "oatp_generalization_drugs.json"

_DRUGS = ["valsartan", "glimepiride"]


@pytest.fixture(scope="module")
def obs_data() -> dict:
    with _OBS_FILE.open() as f:
        return json.load(f)


@pytest.fixture(scope="module")
def graph_and_enzymes():
    graph = build_from_yaml(_PHYS)
    liver_enzymes = {tag: dist.mean for tag, dist in graph.nodes["liver"].enzymes.items()}
    return graph, liver_enzymes


@pytest.mark.parametrize("drug_name", _DRUGS)
def test_pipeline_end_to_end_smoke(drug_name: str, obs_data: dict, graph_and_enzymes):
    """MC(N=10) completes, returns finite Cmax, valid 90% PI."""
    graph, liver_enzymes = graph_and_enzymes
    entry = obs_data["drugs"][drug_name]

    profile = compute_profile(entry["smiles"])
    adme = predict_adme(profile)
    drug = build_drug_on_graph(
        profile,
        adme,
        dose_mg=float(entry["dose_mg"]),
        route="iv",
        liver_enzymes=liver_enzymes,
        transporter_kinetics=load_oatp1b1_kinetics(drug_name),
        hepatic_ecm_params=load_hepatic_ecm_params(drug_name),
    )

    compiled = ODECompiler().compile(graph)
    ue = UncertaintyEngine()
    mc = ue.propagate_fast(
        compiled=compiled, graph=graph, drug=drug,
        n_samples=10, seed=42, t_span=(0.0, 24.0),
        observation_node="venous_blood",
    )

    # Structural checks, not scientific assertions
    assert mc.n_samples >= 8, f"{drug_name}: too many MC failures ({mc.n_failures})"
    assert np.all(np.isfinite(mc.cmax_samples))
    assert np.all(mc.cmax_samples > 0)
    pi_low, pi_high = mc.cmax_90ci
    assert pi_low <= pi_high
    assert pi_low > 0
