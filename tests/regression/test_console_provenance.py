"""Frozen web presets must match the currently shipped model and residual artifacts."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_console_presets_use_current_resources():
    payload = json.loads((ROOT / "web/public/data/console_data.json").read_text())
    cache = json.loads((ROOT / "data/training/4track_holdout_predictions.json").read_text())
    assert payload["benchmark"]["overall"] == cache["overall"]
    assert len(payload["drugs"]) == 8
    for drug in payload["drugs"]:
        provenance = drug["artifactProvenance"]
        assert drug["residualIntervalSource"] == "development_empirical_residual"
        assert drug["meta"]["cmax"] > 0
        for path, digest in provenance.items():
            if path == "resource_profile":
                assert digest == "public"
            else:
                assert digest == hashlib.sha256((ROOT / path).read_bytes()).hexdigest(), path
