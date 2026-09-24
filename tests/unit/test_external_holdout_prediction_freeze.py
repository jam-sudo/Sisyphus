"""The holdout runner must load resources from its frozen checkout."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def test_resource_root_cannot_be_redirected(monkeypatch, tmp_path):
    script = Path(__file__).resolve().parents[2] / "scripts/predict_external_holdout_manifest.py"
    spec = importlib.util.spec_from_file_location("holdout_prediction", script)
    assert spec and spec.loader
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    monkeypatch.delenv("SISYPHUS_ROOT", raising=False)
    runner._verify_resource_root()
    monkeypatch.setenv("SISYPHUS_ROOT", str(tmp_path))
    with pytest.raises(ValueError, match="frozen checkout"):
        runner._verify_resource_root()


def test_solver_settings_must_match_runtime(tmp_path):
    script = Path(__file__).resolve().parents[2] / "scripts/predict_external_holdout_manifest.py"
    spec = importlib.util.spec_from_file_location("holdout_prediction", script)
    assert spec and spec.loader
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    settings_path = script.parents[1] / "data/validation/solver_settings_v1.json"
    runner._verify_runtime_settings(settings_path)
    changed = json.loads(settings_path.read_text())
    changed["deterministic_solver"]["rtol"] = 1e-4
    copy_path = tmp_path / "settings.json"
    copy_path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="do not match"):
        runner._verify_runtime_settings(copy_path)
