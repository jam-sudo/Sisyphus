#!/usr/bin/env python3
"""Regenerate canonical public-clone artifacts after the full R&R Kp merge.

Run through canonical-regen.yml on Linux/Python 3.10 with requirements-lock.txt.
Sequential arms distinguish merged code, full R&R, and retraining on one stack.
"""

from __future__ import annotations

import json
import shutil
import subprocess

from regen_clf_leakfree_canonical import ROOT, _benchmark_meta_aafe, _require_public_clone, _run

CACHE = ROOT / "data/training/4track_holdout_predictions.json"
ATTRIBUTION = ROOT / "data/validation/canonical_regen_attribution_2026-10-29.json"
KP_SOURCES = (
    "src/sisyphus/core.py",
    "src/sisyphus/pipeline/context.py",
    "src/sisyphus/pipeline/predict.py",
    "src/sisyphus/predict/ivive.py",
)


def _benchmark_revision(revision: str) -> dict:
    # shortcut: this revision anchor belongs to the full R&R merge, update for any later regen.
    original = {path: (ROOT / path).read_bytes() for path in KP_SOURCES}
    try:
        for path in KP_SOURCES:
            (ROOT / path).write_bytes(subprocess.check_output(
                ["git", "show", f"{revision}:{path}"], cwd=ROOT
            ))
        _benchmark_meta_aafe()
        return json.loads(CACHE.read_text())
    finally:
        for path, contents in original.items():
            (ROOT / path).write_bytes(contents)


def main() -> None:
    _require_public_clone()
    committed = json.loads(CACHE.read_text())
    before_rr = _benchmark_revision("48a9927")
    _benchmark_meta_aafe()  # full R&R, shipped CL/F and Vd/F models
    merged = json.loads(CACHE.read_text())

    _run(str(ROOT / "scripts/train_clf_vdf_models.py"))
    _benchmark_meta_aafe()  # canonical recipe retrain on the already filtered corpus
    retrained = json.loads(CACHE.read_text())
    assert retrained["n_holdout"] == committed["n_holdout"]
    assert all("clf" in row and "vdss" in row for row in retrained["drugs"])

    ATTRIBUTION.write_text(json.dumps({
        "context": "same canonical stack; merged #132-134, full R&R, recipe retrain",
        "committed": {"overall": committed["overall"], "in_domain": committed["in_domain"]},
        "merged_before_full_rr_shipped_models": {
            "overall": before_rr["overall"], "in_domain": before_rr["in_domain"]
        },
        "merged_shipped_models": {
            "overall": merged["overall"], "in_domain": merged["in_domain"]
        },
        "merged_retrained": {
            "overall": retrained["overall"], "in_domain": retrained["in_domain"]
        },
    }, indent=2) + "\n")

    _run(str(ROOT / "scripts/capture_prodrug_v3_baseline.py"))
    _run(str(ROOT / "scripts/bootstrap_4track_ci.py"), "--date", "2026-10-29",
         "--tag", "canonical", "--context", "EHC, public physiology, prodrug audit, full R&R Kp")
    _run(str(ROOT / "scripts/calibrate_conformal.py"))
    _run(str(ROOT / "scripts/sync_public_model_metrics.py"))
    _run(str(ROOT / "scripts/gen_console_data.py"))
    _run(str(ROOT / "scripts/sync_public_model_metrics.py"))
    for name in ("benchmark.json", "console_data.json"):
        shutil.copy2(ROOT / "web/public/data" / name, ROOT / "app/data" / name)


if __name__ == "__main__":
    main()
