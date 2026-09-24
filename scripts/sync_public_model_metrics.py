#!/usr/bin/env python3
"""Sync public model-card/web metrics from the canonical prediction cache.

This script does not run or tune a model. It removes hand-copied metric drift by
making the committed cache the only numeric source for README-adjacent surfaces.
"""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "training" / "4track_holdout_predictions.json"
CI = ROOT / "data" / "validation" / "4track_ci_2026-09-24_audited_reference.json"
MODEL_CARD = ROOT / "data" / "model_card.json"
WEB_BENCHMARK = ROOT / "web" / "public" / "data" / "benchmark.json"
WEB_CONSOLE = ROOT / "web" / "public" / "data" / "console_data.json"
RESIDUAL_INTERVAL = ROOT / "data" / "validation" / "development_residual_interval.json"


def _write(path: Path, value: dict, *, pretty: bool = True) -> None:
    if pretty:
        rendered = json.dumps(value, indent=2, sort_keys=False) + "\n"
    else:
        rendered = json.dumps(value, separators=(",", ":")) + "\n"
    path.write_text(rendered)


def main() -> None:
    cache = json.loads(CACHE.read_text())
    ci = json.loads(CI.read_text())
    if ci["source_cache_sha256"] != sha256(CACHE.read_bytes()).hexdigest():
        raise ValueError("Bootstrap CI was not computed from the current prediction cache")
    model_card = json.loads(MODEL_CARD.read_text())
    console = json.loads(WEB_CONSOLE.read_text())

    dev = model_card["current_evidence"]["retrospective_development_benchmark"]
    dev["n_compounds"] = cache["n_holdout"]
    dev["meta_aafe"] = cache["overall"]["meta"]["aafe"]
    dev["bootstrap_95_ci"] = [
        ci["overall"]["meta"]["ci_95_low"],
        ci["overall"]["meta"]["ci_95_high"],
    ]
    dev["paired_meta_ml"] = ci["overall"]["paired_meta_ml"]

    benchmark = {
        "classification": "retrospective_development_benchmark_not_independent_holdout",
        "n_holdout": cache["n_holdout"],
        "n_skipped": cache["n_skipped"],
        "overall": cache["overall"],
        "in_domain": cache["in_domain"],
        "drugs": cache["drugs"],
    }
    scatter = [
        {
            "name": row["name"],
            "obs": row["obs"],
            "eng": row["eng"],
            "ml": row["ml"],
            "meta": row["meta"],
            "in_ad": bool(row.get("in_ad", False)),
        }
        for row in cache["drugs"]
    ]
    console["benchmark"] = {
        "classification": "retrospective_development_benchmark",
        "n_development": cache["n_holdout"],
        "overall": cache["overall"],
        "in_domain": cache["in_domain"],
        "paired_meta_ml": ci["overall"]["paired_meta_ml"],
        "scatter": scatter,
    }
    console["constants"].update(
        {
            "DEVELOPMENT_AAFE": cache["overall"]["meta"]["aafe"],
            "ENGINE_AAFE": cache["overall"]["engine"]["aafe"],
            "ML_AAFE": cache["overall"]["ml"]["aafe"],
        }
    )
    console["constants"].pop("HOLDOUT_AAFE", None)
    console["constants"].pop("INDOMAIN_AAFE", None)
    notes = console.setdefault("meta_info", {}).setdefault("notes", [])
    notice = "This repeatedly accessed development benchmark is not an independent holdout."
    if notice not in notes:
        notes.append(notice)

    _write(MODEL_CARD, model_card)
    model_card_sha = sha256(MODEL_CARD.read_bytes()).hexdigest()
    residual_interval_sha = sha256(RESIDUAL_INTERVAL.read_bytes()).hexdigest()
    for drug in console["drugs"]:
        drug["artifactProvenance"]["data/model_card.json"] = model_card_sha
        drug["artifactProvenance"]["data/validation/development_residual_interval.json"] = (
            residual_interval_sha
        )
    _write(WEB_BENCHMARK, benchmark, pretty=False)
    _write(WEB_CONSOLE, console, pretty=False)
    print("Synchronized model card and web metrics from", CACHE.relative_to(ROOT))


if __name__ == "__main__":
    main()
