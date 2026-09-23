"""The holdout runner must load resources from its frozen checkout."""

from __future__ import annotations

import importlib.util
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
