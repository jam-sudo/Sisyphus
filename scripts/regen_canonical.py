#!/usr/bin/env python3
"""Regenerate canonical public-clone artifacts after the B-15 and physics fixes.

Run through canonical-regen.yml on Linux/Python 3.10 with requirements-lock.txt.
The three arms distinguish merged-code effects, recipe retraining, and the B-15
prospective-corpus filter on one numerics stack.
"""

from __future__ import annotations

import csv
import json
import shutil
import subprocess

from regen_clf_leakfree_canonical import ROOT, _benchmark_meta_aafe, _require_public_clone, _run

CACHE = ROOT / "data/training/4track_holdout_predictions.json"
CSV = ROOT / "data/training/clf_training.csv"
ATTRIBUTION = ROOT / "data/validation/canonical_regen_attribution_2026-10-28.json"
SOURCES = (
    "src/sisyphus/core.py",
    "src/sisyphus/engine/flux.py",
    "src/sisyphus/predict/chemistry.py",
    "src/sisyphus/predict/ivive.py",
)


def _benchmark_revision(revision: str) -> dict:
    # shortcut: these revision anchors belong to this one-off merge, update for any later regen.
    original = {path: (ROOT / path).read_bytes() for path in SOURCES}
    try:
        for path in SOURCES:
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
    before_names = {row["name"].lower() for row in csv.DictReader(CSV.open())}

    integration = _benchmark_revision("5de25d7")
    renal = _benchmark_revision("b362d76")
    _benchmark_meta_aafe()  # merged renal + Kp code, shipped CL/F and Vd/F models
    merged = json.loads(CACHE.read_text())

    _run(str(ROOT / "scripts/train_clf_vdf_models.py"))
    _benchmark_meta_aafe()  # same corpus, canonical recipe retrain
    retrained = json.loads(CACHE.read_text())

    _run(str(ROOT / "scripts/build_clf_training_data.py"))
    after_names = {row["name"].lower() for row in csv.DictReader(CSV.open())}
    assert {"aficamten", "gepotidacin"} <= before_names
    assert not {"aficamten", "gepotidacin"} & after_names
    _run(str(ROOT / "scripts/train_clf_vdf_models.py"))
    _benchmark_meta_aafe()  # final filtered model and four-track cache
    filtered = json.loads(CACHE.read_text())
    assert filtered["n_holdout"] == committed["n_holdout"]
    assert all("clf" in row and "vdss" in row for row in filtered["drugs"])

    ATTRIBUTION.write_text(json.dumps({
        "context": "same canonical stack; sequential integration, renal, Kp, retrain, B-15 arms",
        "committed": {"overall": committed["overall"], "in_domain": committed["in_domain"]},
        "integration_shipped_models": {
            "overall": integration["overall"], "in_domain": integration["in_domain"]
        },
        "renal_shipped_models": {
            "overall": renal["overall"], "in_domain": renal["in_domain"]
        },
        "merged_shipped_models": {
            "overall": merged["overall"], "in_domain": merged["in_domain"]
        },
        "merged_retrained_old_corpus": {
            "overall": retrained["overall"], "in_domain": retrained["in_domain"]
        },
        "merged_retrained_b15_corpus": {
            "overall": filtered["overall"], "in_domain": filtered["in_domain"]
        },
        "removed_names": sorted(before_names - after_names),
    }, indent=2) + "\n")

    _run(str(ROOT / "scripts/capture_prodrug_v3_baseline.py"))
    _run(str(ROOT / "scripts/bootstrap_4track_ci.py"), "--date", "2026-10-28",
         "--tag", "canonical", "--context", "B-15 corpus filter plus renal and Kp fixes")
    _run(str(ROOT / "scripts/calibrate_conformal.py"))
    _run(str(ROOT / "scripts/sync_public_model_metrics.py"))
    _run(str(ROOT / "scripts/gen_console_data.py"))
    _run(str(ROOT / "scripts/sync_public_model_metrics.py"))
    for name in ("benchmark.json", "console_data.json"):
        shutil.copy2(ROOT / "web/public/data" / name, ROOT / "app/data" / name)


if __name__ == "__main__":
    main()
