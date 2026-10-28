"""Torch-free coverage for CLI routing and the IBIS TDM contract."""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from sisyphus.cli import _resolve_auto_method, _route_drug_name
from sisyphus.core import Distribution
from sisyphus.engine.compiler import ODECompiler
from sisyphus.graph.builder import build_from_yaml
from sisyphus.regimen.tdm import Observation, TDMResult, bayesian_update
from sisyphus.regimen.types import DosingRegimen

MORPHINE_SMILES = "CN1CCC23C4C1CC5=C2C(=C(C=C5)O)OC3C(C=C4)O"


@pytest.fixture(scope="module")
def case():
    from sisyphus.predict.adme import predict_adme
    from sisyphus.predict.chemistry import compute_profile
    from sisyphus.predict.ivive import build_drug_on_graph

    root = Path(__file__).resolve().parents[2]
    graph = build_from_yaml(root / "data/physiology/reference_man.yaml")
    compiled = ODECompiler().compile(graph)
    profile = compute_profile(MORPHINE_SMILES)
    drug = build_drug_on_graph(profile, predict_adme(profile), 30.0, "oral")
    return compiled, graph, drug, DosingRegimen.single_oral(30.0)


def test_auto_routing_by_name_and_compound_smiles(capsys):
    assert _resolve_auto_method("morphine") == ("sbi", True)
    assert _resolve_auto_method(_route_drug_name(MORPHINE_SMILES, "morphine")) == (
        "sbi", True
    )
    assert _route_drug_name("Cn1c(=O)c2c(ncn2C)n(C)c1=O", None) == "Caffeine"
    assert _resolve_auto_method("unlisted") == ("ibis", False)
    assert "not in routing table" in capsys.readouterr().err


def test_ibis_end_to_end_moves_toward_observation(case):
    compiled, graph, drug, regimen = case
    observed = 0.01865
    result = bayesian_update(
        compiled, graph, drug, regimen, (Observation(1.0, observed),),
        method="ibis", n_prior=50,
    )
    assert result.method_used == "ibis"
    assert abs(result.posterior_cmax.mean - observed) < abs(result.prior_cmax.mean - observed)
    assert result.cmax_ci_90 is not None
    assert all(math.isfinite(value) for value in result.cmax_ci_90)


def test_sbi_import_error_falls_back_and_discloses_covariates(case, monkeypatch):
    from sisyphus.regimen import tdm_sbi

    def unavailable(*args, **kwargs):
        raise ImportError("torch unavailable")

    monkeypatch.setattr(tdm_sbi, "sbi_update", unavailable)
    compiled, graph, drug, regimen = case
    result = bayesian_update(
        compiled, graph, drug, regimen, (Observation(1.0, 0.01865),),
        method="sbi", n_prior=50, population_class="adult",
        body_weight_kg=70.0, age_years=40.0,
    )
    assert result.method_used == "ibis"
    assert "torch unavailable" in result.warnings[0]
    assert "cannot use population" in result.warnings[0]


def test_sbi_unavailable_raises_without_fallback(case, tmp_path):
    compiled, graph, drug, regimen = case
    with pytest.raises((FileNotFoundError, ImportError)):
        bayesian_update(
            compiled, graph, drug, regimen, (Observation(1.0, 0.01865),),
            method="sbi", sbi_posterior_path=str(tmp_path / "missing.pt"),
            sbi_fallback=False, n_prior=50,
        )


def test_cli_auto_uses_name_and_reports_actual_method(case, monkeypatch, capsys):
    from sisyphus import cli
    from sisyphus.regimen import tdm

    compiled, graph, drug, _ = case
    monkeypatch.setattr(cli, "_build_drug_and_graph", lambda *a, **k: (graph, compiled, drug))
    called = {}

    def update(*args, **kwargs):
        called.update(kwargs)
        return TDMResult(Distribution(0.1), Distribution(0.02),
                         method_used="ibis", warnings=("SBI fallback",))

    monkeypatch.setattr(tdm, "bayesian_update", update)
    args = SimpleNamespace(
        smiles=MORPHINE_SMILES, name="morphine", dose=30.0, route="oral",
        phenotype=None, doses=1, interval=None, obs=["1:0.02"], method="auto",
        n_samples=50, population=None, body_weight=None, age=None, verbose=False,
    )
    cli._run_tdm(args)
    out = capsys.readouterr()
    assert called["method"] == "sbi"
    assert called["sbi_reweight"] is True
    assert "Method: ibis" in out.out
    assert "SBI fallback" in out.err

    args.method = "ibis"
    args.population = "adult"
    with pytest.raises(ValueError, match="require SBI"):
        cli._run_tdm(args)


def test_stale_sbi_model_logs_engine_revision_warning(caplog):
    from sisyphus.regimen.tdm_sbi import _warn_engine_revision

    with caplog.at_level("WARNING"):
        _warn_engine_revision(Path("multi_drug_nsf.pt"))
    assert "Posterior calibration may be stale" in caplog.text


def test_sbi_warns_for_timed_multidose_observation(case, caplog):
    from sisyphus.regimen.tdm_sbi import _warn_observation_scope

    compiled, graph, drug, _ = case
    regimen = DosingRegimen.oral_repeated(30.0, 12.0, 2)
    with caplog.at_level("WARNING"):
        _warn_observation_scope(
            compiled, graph, drug, regimen, Observation(40.0, 0.01865), "venous_blood"
        )
    assert "single-dose Cmax" in caplog.text
    assert "far from engine Tmax" in caplog.text
