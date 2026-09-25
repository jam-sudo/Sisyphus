"""Frozen web presets must match the currently shipped model and residual artifacts."""

import hashlib
import json
import math
from pathlib import Path

from scripts.bootstrap_4track_ci import paired_meta_ml_ratio

ROOT = Path(__file__).resolve().parents[2]


def test_console_presets_use_current_resources():
    payload = json.loads((ROOT / "web/public/data/console_data.json").read_text())
    cache = json.loads((ROOT / "data/training/4track_holdout_predictions.json").read_text())
    interval = json.loads((ROOT / "data/validation/development_residual_interval.json").read_text())
    half_width = 10 ** interval["tracks"]["meta"]["0.1"]
    assert payload["benchmark"]["overall"] == cache["overall"]
    assert len(payload["drugs"]) == 8
    assert next(drug for drug in payload["drugs"] if drug["id"] == "metformin")["dose"] == 389.93
    for drug in payload["drugs"]:
        provenance = drug["artifactProvenance"]
        assert drug["residualIntervalSource"] == "development_empirical_residual"
        assert drug["meta"]["cmax"] > 0
        assert math.isclose(
            drug["residualInterval90"][0], drug["meta"]["cmax"] / half_width, rel_tol=1e-5
        )
        assert math.isclose(
            drug["residualInterval90"][1], drug["meta"]["cmax"] * half_width, rel_tol=1e-5
        )
        for path, digest in provenance.items():
            if path == "resource_profile":
                assert digest == "public"
            else:
                assert digest == hashlib.sha256((ROOT / path).read_bytes()).hexdigest(), path


def test_model_card_interval_matches_current_cache():
    cache_path = ROOT / "data/training/4track_holdout_predictions.json"
    cache = json.loads(cache_path.read_text())
    ci = json.loads(
        (ROOT / "data/validation/4track_ci_2026-09-24_audited_reference.json").read_text()
    )
    card = json.loads((ROOT / "data/model_card.json").read_text())
    dev = card["current_evidence"]["retrospective_development_benchmark"]
    assert ci["source_cache_sha256"] == hashlib.sha256(cache_path.read_bytes()).hexdigest()
    assert dev["meta_aafe"] == cache["overall"]["meta"]["aafe"]
    assert dev["bootstrap_95_ci"] == [
        ci["overall"]["meta"]["ci_95_low"],
        ci["overall"]["meta"]["ci_95_high"],
    ]
    assert ci["overall"]["paired_meta_ml"] == paired_meta_ml_ratio(cache["drugs"])
    assert dev["paired_meta_ml"] == ci["overall"]["paired_meta_ml"]
    console = json.loads((ROOT / "web/public/data/console_data.json").read_text())
    assert console["benchmark"]["paired_meta_ml"] == ci["overall"]["paired_meta_ml"]
    readme = (ROOT / "README.md").read_text()
    assert f"benchmark is {cache['overall']['meta']['aafe']:.4f} on" in readme
    meta = cache["overall"]["meta"]
    meta_ci = ci["overall"]["meta"]
    assert (
        f"Meta AAFE is {meta['aafe']:.3f} [bootstrap 95% CI "
        f"{meta_ci['ci_95_low']:.2f}&ndash;{meta_ci['ci_95_high']:.2f}"
    ) in readme
    for scope, track, title in (
        ("overall", "meta", "**Meta-learner (production)**"),
        ("overall", "engine", "Engine only"),
        ("overall", "ml", "ML only"),
        ("in_domain", "meta", "Meta, in-domain"),
    ):
        metrics = cache[scope][track]
        bounds = ci[scope][track]
        point = f"{metrics['aafe']:.3f}"
        if scope == "overall" and track == "meta":
            point = f"**{point}**†"
        row = (
            f"| {title} | {point} | "
            f"[{bounds['ci_95_low']:.2f}, {bounds['ci_95_high']:.2f}] | "
            f"{metrics['pct_2fold']:.1f}% | {metrics['pct_3fold']:.1f}% | {bounds['n']} |"
        )
        assert row in readme


def test_paired_ratio_resamples_compounds_together():
    rows = [
        {"obs": 1.0, "meta": 2.0, "ml": 4.0},
        {"obs": 1.0, "meta": 4.0, "ml": 8.0},
    ]
    assert paired_meta_ml_ratio(rows, n_bootstrap=100) == {
        "ratio": 0.5, "ci_95_low": 0.5, "ci_95_high": 0.5, "n": 2,
    }
