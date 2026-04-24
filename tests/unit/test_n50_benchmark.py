"""Unit tests for scripts/run_n50_benchmark.py.

Validates the pure-function helpers and the argparse/mode logic without
invoking the full predict() pipeline (mocked). Mechanism verification.
"""

from __future__ import annotations

import importlib.util
import json
import math
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
SCRIPT = ROOT / "scripts/run_n50_benchmark.py"


@pytest.fixture
def n50_module():
    """Load the benchmark script as an importable module."""
    spec = importlib.util.spec_from_file_location("n50_benchmark", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# _map_route
# ---------------------------------------------------------------------------


def test_map_route_oral(n50_module):
    entry = {"route": "oral"}
    assert n50_module._map_route(entry) == ("oral", None)


def test_map_route_iv_bolus(n50_module):
    entry = {"route": "iv_bolus"}
    assert n50_module._map_route(entry) == ("iv", None)


def test_map_route_iv_infusion(n50_module):
    entry = {"route": "iv_infusion", "infusion_duration_min": 30.0}
    route, dur = n50_module._map_route(entry)
    assert route == "iv"
    assert dur == 30.0


def test_map_route_iv_infusion_missing_duration_rejected(n50_module):
    entry = {"route": "iv_infusion", "_name": "testdrug"}
    with pytest.raises(ValueError, match="iv_infusion requires"):
        n50_module._map_route(entry)


def test_map_route_unknown_rejected(n50_module):
    with pytest.raises(ValueError, match="Unsupported route"):
        n50_module._map_route({"route": "subcutaneous"})


# ---------------------------------------------------------------------------
# _aafe_with_ci
# ---------------------------------------------------------------------------


def test_aafe_with_ci_point_estimate(n50_module):
    """AAFE of [2, 0.5, 2, 0.5] = 10^mean([log10 2]*4) = 2."""
    out = n50_module._aafe_with_ci([2.0, 0.5, 2.0, 0.5], n_bootstrap=1000)
    assert out["n"] == 4
    assert abs(out["aafe"] - 2.0) < 1e-9
    # CI should straddle the point estimate
    assert out["ci_95_low"] <= out["aafe"] <= out["ci_95_high"]


def test_aafe_with_ci_empty(n50_module):
    out = n50_module._aafe_with_ci([])
    assert out == {"aafe": None, "ci_95_low": None, "ci_95_high": None, "n": 0}


def test_aafe_with_ci_filters_nonpositive(n50_module):
    """fold=0 or None is excluded."""
    out = n50_module._aafe_with_ci([None, 0.0, 2.0, 2.0], n_bootstrap=1000)
    assert out["n"] == 2


def test_aafe_with_ci_deterministic(n50_module):
    """Same input → same CI (fixed RNG seed 20260422)."""
    folds = [2.0, 0.5, 3.0, 0.33, 1.5, 0.7, 2.2, 0.4]
    a = n50_module._aafe_with_ci(folds, n_bootstrap=1000)
    b = n50_module._aafe_with_ci(folds, n_bootstrap=1000)
    assert a == b


# ---------------------------------------------------------------------------
# _pct_within_n_fold
# ---------------------------------------------------------------------------


def test_pct_within_2fold(n50_module):
    # |log10(2)| = 0.301 (in), |log10(3)| = 0.477 (out)
    assert n50_module._pct_within_n_fold([2.0, 0.5, 3.0, 1 / 3.0], 2.0) == 50.0


def test_pct_within_n_fold_empty(n50_module):
    assert n50_module._pct_within_n_fold([], 2.0) is None


# ---------------------------------------------------------------------------
# run() with mocked predict
# ---------------------------------------------------------------------------


def _mock_predict_factory(cmax_map: dict[str, float]):
    """Build a fake predict() that returns Cmax from a SMILES→Cmax lookup."""
    class FakeDist:
        def __init__(self, m): self.mean = m
    class FakePK:
        def __init__(self, m): self.cmax = FakeDist(m)
    class FakeResult:
        def __init__(self, m):
            self.pk = FakePK(m)
            self.engine_pk = FakePK(m)
            self.ml_pk = FakePK(m)
            self.in_applicability_domain = True
            self.ad_flags = ()

    def fake_predict(*, smiles, dose_mg, route, infusion_duration_min=None):
        if smiles not in cmax_map:
            raise ValueError(f"unmocked smiles {smiles}")
        return FakeResult(cmax_map[smiles])

    return fake_predict


def test_run_evaluates_verified_only(n50_module, monkeypatch):
    payload = {
        "drugs": {
            "drugA": {
                "status": "VERIFIED", "smiles": "A", "dose_mg": 100,
                "route": "oral", "observed_cmax_mg_l": 1.0,
                "oatp1b1_substrate": False,
            },
            "drugB": {
                "status": "PENDING", "smiles": "B", "dose_mg": 100,
                "route": "oral", "observed_cmax_mg_l": 1.0,
            },
            "drugC": {
                "status": "BLOCKED", "smiles": "C", "dose_mg": 100,
                "route": "oral", "observed_cmax_mg_l": 1.0,
            },
        }
    }
    fake = _mock_predict_factory({"A": 2.0})
    monkeypatch.setitem(sys.modules, "sisyphus.pipeline.predict", type(sys)("stub"))
    sys.modules["sisyphus.pipeline.predict"].predict = fake
    results = n50_module.run(payload)
    assert set(results["per_drug"].keys()) == {"drugA"}
    assert len(results["skipped"]) == 2


def test_run_oatp_subset_separation(n50_module, monkeypatch):
    payload = {
        "drugs": {
            "oatp1": {
                "status": "VERIFIED", "smiles": "O1", "dose_mg": 100,
                "route": "oral", "observed_cmax_mg_l": 1.0,
                "oatp1b1_substrate": True,
            },
            "nonoatp": {
                "status": "VERIFIED", "smiles": "X", "dose_mg": 100,
                "route": "oral", "observed_cmax_mg_l": 1.0,
                "oatp1b1_substrate": False,
            },
        }
    }
    fake = _mock_predict_factory({"O1": 2.0, "X": 0.5})
    sys.modules["sisyphus.pipeline.predict"] = type(sys)("stub")
    sys.modules["sisyphus.pipeline.predict"].predict = fake
    results = n50_module.run(payload)
    assert results["overall"]["n"] == 2
    assert results["oatp1b1_subset"]["n"] == 1
    # OATP subset AAFE = |log10(2)| → 2.0
    assert abs(results["oatp1b1_subset"]["aafe"] - 2.0) < 1e-9


def test_run_passes_infusion_duration(n50_module, monkeypatch):
    """iv_infusion entry must reach predict() with infusion_duration_min set."""
    captured: dict = {}

    def spy_predict(*, smiles, dose_mg, route, infusion_duration_min=None):
        captured["route"] = route
        captured["infusion_duration_min"] = infusion_duration_min
        class FD: mean = 1.0
        class FP: cmax = FD()
        class FR:
            pk = FP(); engine_pk = FP(); ml_pk = FP()
            in_applicability_domain = True; ad_flags = ()
        return FR()

    payload = {
        "drugs": {
            "rif": {
                "status": "VERIFIED", "smiles": "S", "dose_mg": 600,
                "route": "iv_infusion", "infusion_duration_min": 30.0,
                "observed_cmax_mg_l": 17.5, "oatp1b1_substrate": True,
            }
        }
    }
    stub = type(sys)("stub")
    stub.predict = spy_predict
    monkeypatch.setitem(sys.modules, "sisyphus.pipeline.predict", stub)
    n50_module.run(payload)
    assert captured == {"route": "iv", "infusion_duration_min": 30.0}


# ---------------------------------------------------------------------------
# CLI: freeze-run protection
# ---------------------------------------------------------------------------


def _run_cli(args: list[str], cwd: pathlib.Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT)] + args,
        cwd=str(cwd),
        capture_output=True,
        text=True,
    )


def test_cli_requires_mode_flag():
    proc = _run_cli([], ROOT)
    assert proc.returncode != 0
    assert "one of the arguments" in proc.stderr.lower() or proc.returncode == 2


def test_cli_freeze_requires_confirm(tmp_path):
    payload = {"cycle_id": "test", "n_admitted": 50, "n_target": 50, "drugs": {}}
    f = tmp_path / "n50.json"
    f.write_text(json.dumps(payload))
    proc = _run_cli(["--freeze-run", "--n50-file", str(f)], ROOT)
    assert proc.returncode == 2
    assert "--confirm" in proc.stdout or "--confirm" in proc.stderr


def test_cli_freeze_refuses_partial_n50(tmp_path):
    payload = {"cycle_id": "test", "n_admitted": 3, "n_target": 50, "drugs": {}}
    f = tmp_path / "n50.json"
    f.write_text(json.dumps(payload))
    proc = _run_cli(
        ["--freeze-run", "--confirm", "--n50-file", str(f)], ROOT
    )
    assert proc.returncode == 2
    assert "n_admitted" in proc.stdout


def test_cli_freeze_refuses_overwrite(tmp_path, monkeypatch):
    """Existing freeze file for the same cycle must block the run."""
    # Use a cycle_id that won't collide with real files. Place a pre-existing
    # freeze file at data/validation/n50_benchmark_<cycle>.json.
    cycle = "unittest-cycle-2026Q2"
    freeze_file = ROOT / "data/validation" / f"n50_benchmark_{cycle}.json"
    assert not freeze_file.exists(), "test fixture collision"
    freeze_file.write_text("{}")
    try:
        payload = {
            "cycle_id": cycle, "n_admitted": 50, "n_target": 50, "drugs": {},
        }
        n50_f = tmp_path / "n50.json"
        n50_f.write_text(json.dumps(payload))
        proc = _run_cli(
            ["--freeze-run", "--confirm", "--n50-file", str(n50_f)], ROOT
        )
        assert proc.returncode == 2
        assert "already exists" in proc.stdout
    finally:
        freeze_file.unlink()


def test_cli_smoke_test_writes_nothing(tmp_path):
    """--smoke-test must not create any file in data/validation/."""
    payload = {
        "cycle_id": "smoke", "n_admitted": 0, "n_target": 50,
        "drugs": {},  # empty — run() returns empty results, no predictions
    }
    f = tmp_path / "n50.json"
    f.write_text(json.dumps(payload))
    files_before = set((ROOT / "data/validation").glob("n50_benchmark_*.json"))
    proc = _run_cli(["--smoke-test", "--n50-file", str(f)], ROOT)
    files_after = set((ROOT / "data/validation").glob("n50_benchmark_*.json"))
    assert proc.returncode == 0, proc.stderr
    assert files_before == files_after
    assert "NOT A FREEZE RUN" in proc.stdout
