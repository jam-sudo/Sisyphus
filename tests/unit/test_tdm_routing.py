"""Torch-free coverage for CLI routing and the IBIS TDM contract."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from sisyphus.cli import _resolve_auto_method, _route_drug_name
from sisyphus.core import Distribution
from sisyphus.engine.compiler import ODECompiler
from sisyphus.graph.builder import build_from_yaml
from sisyphus.regimen.tdm import Observation, TDMResult, bayesian_update
from sisyphus.regimen.types import DosingRegimen
from sisyphus.sbi.importance import LowImportanceESS

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
    drug = build_drug_on_graph(profile, predict_adme(profile), 22.5, "oral")
    return compiled, graph, drug, DosingRegimen.single_oral(22.5)


def test_auto_routing_by_name_and_compound_smiles(capsys):
    root = Path(__file__).resolve().parents[2]
    table = json.loads((root / "data/sbi/method_routing.json").read_text())
    assert _resolve_auto_method("morphine")[0] == table["routes"]["morphine"]
    assert (_resolve_auto_method(_route_drug_name(MORPHINE_SMILES, "morphine"))
            == _resolve_auto_method("morphine"))
    assert _resolve_auto_method("clozapine")[0] == table["routes"]["clozapine"]
    assert _route_drug_name("Cn1c(=O)c2c(ncn2C)n(C)c1=O", None) == "Caffeine"
    assert _resolve_auto_method("unlisted") == ("ibis", False)
    assert "not in routing table" in capsys.readouterr().err


def test_multi_observation_auto_routing_uses_only_calibrated_valid_context():
    root = Path(__file__).resolve().parents[2]
    table = json.loads((root / "data/sbi/method_routing.json").read_text())
    regimen = DosingRegimen.single_oral(10.0)
    observations = (Observation(1.0, 0.02), Observation(4.0, 0.01))
    for name in ("morphine", "ketorolac", "rivaroxaban"):
        decision = table["multi_observation"]["drugs"][name]
        expected = "sbi" if decision["use_native"] else table["routes"][name]
        assert _resolve_auto_method(name, regimen, observations)[0] == expected
    assert _resolve_auto_method(
        "morphine", regimen, observations, SimpleNamespace(smiles="CCO")
    )[0] == table["routes"]["morphine"]
    invalid = (Observation(1.0, 0.02), Observation(3.0, 0.01))
    assert _resolve_auto_method("morphine", regimen, invalid)[0] == table["routes"]["morphine"]
    assert _resolve_auto_method("morphine", regimen, observations[:1])[0] == (
        table["routes"]["morphine"])


def test_multi_observation_model_matches_canonical_smiles():
    from sisyphus.regimen.tdm_sbi import _native_multi_observation_model

    root = Path(__file__).resolve().parents[2]
    validation = json.loads((root / "data/sbi/validation_drug_set.json").read_text())
    routing = json.loads((root / "data/sbi/method_routing.json").read_text())
    for name in ("morphine", "ketorolac", "clozapine", "rivaroxaban"):
        smiles = next(row["smiles"] for row in validation["drugs"] if row["name"] == name)
        expected = routing["multi_observation"]["drugs"][name]["use_native"]
        actual = _native_multi_observation_model(SimpleNamespace(smiles=smiles))
        assert (actual is not None) == expected
    expected_morphine = routing["multi_observation"]["drugs"]["morphine"]["use_native"]
    actual_morphine = _native_multi_observation_model(SimpleNamespace(smiles=MORPHINE_SMILES))
    assert (actual_morphine is not None) == expected_morphine


def test_ibis_end_to_end_moves_toward_observation(case):
    compiled, graph, drug, regimen = case
    observed = 0.0285
    result = bayesian_update(
        compiled, graph, drug, regimen, (Observation(1.0, observed),),
        method="ibis", n_prior=50,
    )
    assert result.method_used == "ibis"
    assert abs(result.posterior_cmax.mean - observed) < abs(result.prior_cmax.mean - observed)
    assert result.cmax_ci_90 is not None
    assert all(math.isfinite(value) for value in result.cmax_ci_90)


@pytest.mark.parametrize("failure", [ImportError("torch unavailable"),
                                     LowImportanceESS("SBC excluded pair")])
def test_sbi_unavailable_falls_back_and_discloses_covariates(case, monkeypatch, failure):
    from sisyphus.regimen import tdm_sbi

    def unavailable(*args, **kwargs):
        raise failure

    monkeypatch.setattr(tdm_sbi, "sbi_update", unavailable)
    compiled, graph, drug, regimen = case
    result = bayesian_update(
        compiled, graph, drug, regimen, (Observation(1.0, 0.0285),),
        method="sbi", n_prior=50, population_class="adult",
        body_weight_kg=70.0, age_years=40.0,
    )
    assert result.method_used == "ibis"
    assert str(failure) in result.warnings[0]
    assert "cannot use population" in result.warnings[0]


def test_sbi_unavailable_raises_without_fallback(case, tmp_path):
    compiled, graph, drug, regimen = case
    with pytest.raises((FileNotFoundError, ImportError)):
        bayesian_update(
            compiled, graph, drug, regimen, (Observation(1.0, 0.0285),),
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
                         method_used="ibis")

    monkeypatch.setattr(tdm, "bayesian_update", update)
    args = SimpleNamespace(
        smiles=MORPHINE_SMILES, name="morphine", dose=22.5, route="oral",
        phenotype=None, doses=1, interval=None, obs=["1:0.0285"], method="auto",
        n_samples=50, population=None, body_weight=None, age=None, verbose=False,
    )
    cli._run_tdm(args)
    out = capsys.readouterr()
    assert called["method"] == "ibis"
    assert "sbi_reweight" not in called
    assert "Method: ibis" in out.out
    assert "routing morphine" in out.err

    args.obs = ["1:0.0285", "4:0.010"]
    cli._run_tdm(args)
    assert called["method"] == "sbi"

    args.method = "ibis"
    args.population = "adult"
    with pytest.raises(ValueError, match="require SBI"):
        cli._run_tdm(args)


def test_stale_sbi_model_logs_engine_revision_warning(tmp_path, monkeypatch, caplog):
    from sisyphus.regimen import tdm_sbi

    stamp_path = tmp_path / "engine_revision.json"
    stamp_path.write_text(json.dumps({
        "models": {
            "multi_drug_nsf.pt": {
                "commit": "previous",
                "sha256": "outdated",
                "paths": ["engine.py"],
            },
        },
    }))
    (tmp_path / "engine.py").write_text("current")
    monkeypatch.setattr(
        tdm_sbi, "get_resource_config",
        lambda: SimpleNamespace(root=tmp_path, model=lambda *args, **kwargs: stamp_path),
    )

    with caplog.at_level("WARNING"):
        tdm_sbi._warn_engine_revision(Path("multi_drug_nsf.pt"))
    assert "Posterior calibration may be stale" in caplog.text


def test_verified_equivalent_sbi_engine_revision(tmp_path, monkeypatch, caplog):
    from sisyphus.regimen import tdm_sbi

    stamp_path = tmp_path / "engine_revision.json"
    (tmp_path / "engine.py").write_text("current")
    digest = hashlib.sha256(b"engine.py\0current").hexdigest()
    stamp_path.write_text(json.dumps({"models": {"multi_drug_nsf.pt": {
        "commit": "training", "sha256": "training-hash", "paths": ["engine.py"],
        "verified_equivalent": {"sha256": digest, "check": "fixed-seed comparison"},
    }}}))
    monkeypatch.setattr(
        tdm_sbi, "get_resource_config",
        lambda: SimpleNamespace(root=tmp_path, model=lambda *args, **kwargs: stamp_path),
    )

    with caplog.at_level("WARNING"):
        tdm_sbi._warn_engine_revision(Path("multi_drug_nsf.pt"))
    assert not caplog.records


def test_validated_sbi_engine_revision(tmp_path, monkeypatch, caplog):
    from sisyphus.regimen import tdm_sbi

    stamp_path = tmp_path / "engine_revision.json"
    (tmp_path / "engine.py").write_text("current")
    digest = hashlib.sha256(b"engine.py\0current").hexdigest()
    stamp_path.write_text(json.dumps({"models": {"multi_observation_nsf.pt": {
        "commit": "training", "sha256": "training-hash", "paths": ["engine.py"],
        "validated_revision": {"sha256": digest, "check": "current-engine SBC"},
    }}}))
    monkeypatch.setattr(
        tdm_sbi, "get_resource_config",
        lambda: SimpleNamespace(root=tmp_path, model=lambda *args, **kwargs: stamp_path),
    )

    with caplog.at_level("WARNING"):
        tdm_sbi._warn_engine_revision(Path("multi_observation_nsf.pt"))
    assert not caplog.records


@pytest.mark.parametrize("model_name", [
    "multi_drug_nsf.pt", "hierarchical_nsf_2k.pt", "multi_observation_nsf.pt",
    "continuous_hierarchical_nsf.pt",
])
def test_retrained_sbi_model_matches_current_engine(model_name, caplog):
    from sisyphus.regimen.tdm_sbi import _warn_engine_revision

    with caplog.at_level("WARNING"):
        _warn_engine_revision(Path(model_name))
    assert "Posterior calibration may be stale" not in caplog.text


def test_routing_rebuild_preserves_tdm_overrides(tmp_path, monkeypatch):
    from scripts.sbi_build_routing_table import main

    root = Path(__file__).resolve().parents[2]
    out = tmp_path / "method_routing.json"
    original = json.loads((root / "data/sbi/method_routing.json").read_text())
    sbc_path = tmp_path / "data/validation/sbi_sbc_multi_drug_corrected.json"
    sbc_path.parent.mkdir(parents=True)
    sbc_path.write_text(json.dumps({
        "gate_passed": False,
        "drugs": [
            {"name": "morphine", "pass": True, "coverage_max_deviation": 0.05,
             "ks_pvalues": [0.5]},
            {"name": "pravastatin", "pass": True, "coverage_max_deviation": 0.05,
             "ks_pvalues": [0.5], "rejection_fallback_cases": 1},
            {"name": "clozapine", "pass": True, "coverage_max_deviation": 0.12,
             "ks_pvalues": [0.5]},
        ],
    }))
    monkeypatch.setattr("scripts.sbi_build_routing_table.ROOT", tmp_path)
    out.write_text(json.dumps(original))
    monkeypatch.setattr(sys, "argv", ["sbi_build_routing_table", "--out", str(out)])
    main()
    rebuilt = json.loads(out.read_text())
    assert rebuilt["source"] == "data/validation/sbi_sbc_multi_drug_corrected.json"
    assert rebuilt["sbi_reweight"] == original["sbi_reweight"]
    assert rebuilt["summary"]["sbi"] == 1
    assert rebuilt["summary"]["ibis"] == 2
    assert rebuilt["summary"]["global_sbc_gate_passed"] is False
    assert rebuilt["summary"]["sbc_exact_rejection_cases_total"] == 1
    assert rebuilt["summary"]["sbc_exact_rejection_cases_on_sbi_routes"] == 0
    assert rebuilt["reasons"]["pravastatin"]["reason"] == "exact_rejection_sbc_fallback"
    assert rebuilt["multi_observation"] == original["multi_observation"]
    assert (rebuilt["reasons"]["morphine"]["tdm_bias_override"]
            == original["reasons"]["morphine"]["tdm_bias_override"])

    original["routes"]["pravastatin"] = "is"
    out.write_text(json.dumps(original))
    main()
    rebuilt = json.loads(out.read_text())
    assert rebuilt["routes"]["pravastatin"] == "is"
    assert rebuilt["summary"]["is"] == 1

    multi_path = tmp_path / "data/validation/sbi_sbc_multi_observation_corrected.json"
    multi_path.write_text(json.dumps({"gate_passed": False, "drugs": [
        {"name": "morphine", "pass": True, "coverage_max_deviation": 0.05,
         "ks_pvalues": [0.2, 0.2, 0.2]},
        {"name": "clozapine", "pass": True, "coverage_max_deviation": 0.12,
         "ks_pvalues": [0.2, 0.2, 0.2]},
        {"name": "pravastatin", "pass": True, "coverage_max_deviation": 0.02,
         "ks_pvalues": [0.2, 0.2, 0.2], "rejection_fallback_cases": 1},
    ]}))
    validation_path = tmp_path / "data/sbi/validation_drug_set.json"
    validation_path.parent.mkdir(parents=True)
    validation_path.write_text(json.dumps({"drugs": [
        {"name": name, "smiles": name} for name in ("morphine", "clozapine", "pravastatin")
    ]}))
    main()
    decisions = json.loads(out.read_text())["multi_observation"]
    assert decisions["native_count"] == 1
    assert decisions["sbc_exact_rejection_cases_total"] == 1
    assert decisions["sbc_exact_rejection_cases_on_native_routes"] == 0
    assert [decisions["drugs"][name]["use_native"] for name in (
        "morphine", "clozapine", "pravastatin"
    )] == [True, False, False]

    pairs = [
        {"population": "adult", "drug": "morphine", "pass": True,
         "coverage_max_deviation": 0.05, "ks_pvalues": [0.2, 0.2, 0.2],
         "rejection_fallback_cases": 0},
        {"population": "adult", "drug": "pravastatin", "pass": True,
         "coverage_max_deviation": 0.05, "ks_pvalues": [0.2, 0.2, 0.2],
         "rejection_fallback_cases": 2},
    ]
    hier_path = tmp_path / "data/validation/sbi_sbc_hierarchical_corrected.json"
    hier_path.write_text(json.dumps({
        "gate_passed": False, "populations": ["adult"], "pairs": pairs,
    }))
    cont_path = tmp_path / "data/validation/sbi_sbc_continuous_corrected.json"
    cont_path.write_text(json.dumps({
        "gate_passed": False,
        "populations": [{"name": "adult", "bw": 70.0, "age": 35.0}],
        "pairs": pairs,
    }))
    main()
    rebuilt = json.loads(out.read_text())
    for flow in ("hierarchical", "continuous"):
        section = rebuilt[flow]
        assert section["sbi_pair_count"] == 1
        assert section["sbc_exact_rejection_cases_total"] == 2
        assert section["sbc_exact_rejection_cases_on_sbi_routes"] == 0
        assert [row["use_sbi"] for row in section["pairs"]] == [True, False]
    assert rebuilt["continuous"]["pairs"][0]["body_weight_kg"] == 70.0
    assert rebuilt["continuous"]["pairs"][0]["age_years"] == 35.0

    incomplete = json.loads(sbc_path.read_text())
    incomplete["drugs"].append({
        "name": "amantadine", "pass": None, "coverage_max_deviation": None,
        "ks_pvalues": [],
    })
    sbc_path.write_text(json.dumps(incomplete))
    main()
    rebuilt = json.loads(out.read_text())
    assert rebuilt["routes"]["amantadine"] == "ibis"
    assert rebuilt["reasons"]["amantadine"]["reason"] == "sampling_incomplete"


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
