"""The displayed development residual band must match the current model and cache."""

import hashlib
import json
import math
from pathlib import Path

from sisyphus.validation.holdout_contract import (
    _PRODUCTION_FITTED_MODELS,
    verify_development_residual_interval,
)

ROOT = Path(__file__).resolve().parents[2]


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_development_residual_artifact_is_current_and_explicit_about_skips():
    artifact = json.loads((ROOT / "data/validation/development_residual_interval.json").read_text())
    assert artifact["method"] == "development_empirical_residual_quantile"
    assert artifact["source_cache_sha256"] == _sha(
        ROOT / "data/training/4track_holdout_predictions.json"
    )
    expected_paths = {path.replace(".meta.json", ".json") for path in _PRODUCTION_FITTED_MODELS}
    assert set(artifact["model_artifact_sha256"]) == expected_paths
    for path, digest in artifact["model_artifact_sha256"].items():
        assert digest == _sha(ROOT / path)
    assert artifact["n_training_reference"] == 67
    assert artifact["n_calibration_meta"] == 63
    assert {row["name"] for row in artifact["skipped_training_reference"]} == {
        "amoxicillin", "atenolol", "metformin", "verapamil"
    }
    assert math.isfinite(artifact["tracks"]["meta"]["0.1"])
    assert verify_development_residual_interval(ROOT) == artifact["tracks"]["meta"]["0.1"]
