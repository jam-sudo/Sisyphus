#!/usr/bin/env python3
"""CLF leak-free canonical regen — run ONCE in the canonical public-clone Linux env.

Why this exists: the CLF-track training builder gained a structural InChIKey-14
holdout key (PR #90) that removes 5 name-evading stereo/salt holdout collisions
(valacyclovir, darunavir ethanolate, ...). Shipping leak-free CLF/VDF models
requires retraining, which makes the committed benchmark cache + golden pins
stale. They must be regenerated in the canonical numerics environment (locked
requirements, Linux/OpenBLAS), NOT on a developer macOS machine (BLAS/libomp
drift ~4% breaks the cache-pin tolerance).

The committed CLF/VDF models carry `unknown_legacy` provenance and are not
byte-reproducible; retraining from the canonical recipe (train_clf_vdf_models.py)
reproduces the headline to ~0.05% locally. The CLF leak effect is comparable to
retrain nondeterminism, so this script runs a DUAL arm on one stack/run to
isolate it:

    BASELINE arm : retrain from the committed (leaky) clf_training.csv -> benchmark
    LEAK-FREE arm: rebuild clf_training.csv (IK14 filter) -> retrain -> benchmark

    Delta_leak = leakfree_meta - baseline_meta   (pure leak effect, same stack)
    baseline_meta vs committed 2.731             (retrain reproduction check)

The LEAK-FREE arm's artifacts (models, cache, leak-audit baseline, bootstrap CI)
are the canonical outputs to download and commit. Run:

    PYTHONPATH=src python scripts/regen_clf_leakfree_canonical.py

Then apply the printed pin edits, restore artifacts, run the full suite, update
the README headline to the new Meta, commit, merge.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
_ENV = {**os.environ, "PYTHONPATH": str(ROOT / "src")}

_DRUGBANK = ROOT / "data" / "drugbank" / "drugs.csv"
_LOGP = ROOT / "models" / "adme" / "logp_correction.json"
_CSV = ROOT / "data" / "training" / "clf_training.csv"
_CACHE = ROOT / "data" / "training" / "4track_holdout_predictions.json"
_LEAK_BASELINE = ROOT / "tests" / "regression" / "data" / "prodrug_v3_pre_baseline.json"
_CLF_MODEL = ROOT / "models" / "direct_pk" / "xgboost_clf.json"
_VDF_MODEL = ROOT / "models" / "direct_pk" / "xgboost_vdf.json"


def _run(*args: str) -> None:
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True, env=_ENV)


def _require_public_clone() -> None:
    present = [str(p.relative_to(ROOT)) for p in (_DRUGBANK, _LOGP) if p.exists()]
    if present:
        sys.exit(
            "REFUSING: developer artifacts present "
            f"({', '.join(present)}). This regen MUST run in public-clone state.\n"
            "  mv data/drugbank data/drugbank.local-archive\n"
            "  mv models/adme/logp_correction.json{,.local-archive}"
        )


def _benchmark_meta_aafe() -> float:
    """Run the canonical benchmark, overwrite the cache, return meta AAFE."""
    _run(str(ROOT / "scripts" / "run_engine_benchmark.py"), "--save-json", str(_CACHE))
    return json.loads(_CACHE.read_text())["overall"]["meta"]["aafe"]


def main() -> None:
    _require_public_clone()

    # === BASELINE arm: retrain from the committed (leaky) clf_training.csv ===
    # The committed clf_training.csv is unchanged on this branch (PR #90 touched
    # only the builder), so training from it reproduces the shipped-model regime.
    print("== [baseline 1/2] retraining CLF/VDF from committed clf_training.csv ==", flush=True)
    _run(str(ROOT / "scripts" / "train_clf_vdf_models.py"))
    print("== [baseline 2/2] benchmarking baseline ==", flush=True)
    baseline_meta = _benchmark_meta_aafe()

    # === LEAK-FREE arm: rebuild clf_training.csv with the IK14 filter, retrain ===
    print("== [leak-free 1/5] rebuilding clf_training.csv (IK14 holdout filter) ==", flush=True)
    _run(str(ROOT / "scripts" / "build_clf_training_data.py"))
    print("== [leak-free 2/5] retraining CLF/VDF on leak-free data ==", flush=True)
    _run(str(ROOT / "scripts" / "train_clf_vdf_models.py"))
    print("== [leak-free 3/5] regenerating 4track_holdout_predictions.json ==", flush=True)
    _benchmark_meta_aafe()  # writes _CACHE
    cache = json.loads(_CACHE.read_text())
    meta_aafe = cache["overall"]["meta"]["aafe"]
    indom_n = cache["in_domain"]["n"]
    indom_aafe = cache["in_domain"]["meta"]["aafe"]
    eng_aafe = cache["overall"]["engine"]["aafe"]

    from sisyphus.pipeline.predict import predict
    from sisyphus.validation.reference import load_reference

    print("== [leak-free 4/5] regenerating leak-audit baseline + tebipenem pin ==", flush=True)
    refs = [r for r in load_reference() if r.in_holdout]
    baseline: dict[str, float] = {}
    for r in refs:
        try:
            baseline[r.name] = float(predict(r.smiles, r.dose_mg, r.route).pk.cmax.mean)
        except Exception as exc:  # noqa: BLE001
            print(f"  WARN {r.name}: {exc}")
    _LEAK_BASELINE.write_text(json.dumps(baseline, indent=2, sort_keys=True) + "\n")

    teb_smiles = ("C[C@@H](O)[C@H]1C(=O)N2C(C(=O)OCOC(=O)C(C)(C)C)=C(SC3CN("
                  "C4=NCCS4)C3)[C@H](C)[C@H]12")
    teb = predict(teb_smiles, 300.0, "oral").pk.cmax.mean

    print("== [leak-free 5/5] bootstrapping CI ==", flush=True)
    _run(str(ROOT / "scripts" / "bootstrap_4track_ci.py"),
         "--cache", str(_CACHE), "--tag", "clf_leakfree",
         "--context", "CLF InChIKey-14 holdout leak fix (canonical regen)")

    print("\n" + "=" * 72)
    print("CLF LEAK-FREE CANONICAL REGEN COMPLETE")
    print("=" * 72)
    print(f"  BASELINE (retrain, leaky csv)  Meta AAFE = {baseline_meta:.5f}")
    print(f"  LEAK-FREE (retrain, IK14 csv)  Meta AAFE = {meta_aafe:.5f}")
    print(f"  Delta_leak = {meta_aafe - baseline_meta:+.5f}  "
          f"(baseline vs committed 2.731 = {baseline_meta - 2.731:+.5f} retrain drift)")
    print(f"  cache: Meta = {meta_aafe:.5f} | Engine = {eng_aafe:.4f} | "
          f"in_domain.n = {indom_n} | in_domain Meta = {indom_aafe:.5f}")
    print("\n  Apply these edits (LEAK-FREE arm is canonical):")
    print(f"  -> rename/repin tests/integration/test_holdout_regression.py cache pin "
          f"to assert abs(aafe - {meta_aafe:.4f}) < 0.020")
    print(f"  -> _PINNED['tebipenem_pivoxil'] = {teb:.6e}")
    print("  -> README headline table Meta -> new value (reconcile vs cache JSON)")
    print("  -> commit leak-free clf_training.csv + xgboost_clf/vdf.json + cache + "
          "bootstrap + leak baseline")


if __name__ == "__main__":
    main()
