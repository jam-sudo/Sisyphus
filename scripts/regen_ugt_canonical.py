#!/usr/bin/env python3
"""UGT single-path canonical regen — run ONCE in the canonical public-clone Linux env.

The UGT double-allocation fix (route each UGT tag through one fm mechanism) changes
predict() output for the 8 B-02 Phase 2 UGT substrates (morphine, codeine,
ketorolac, indomethacin, dapagliflozin, etodolac, bexagliflozin, glasdegib), all in
the 107-holdout, so the committed benchmark cache + a few golden pins are stale.
They must be regenerated on the canonical numerics stack (locked requirements,
Linux/OpenBLAS), NOT on a developer macOS machine (~4% aggregate BLAS drift).

This is a predict-layer change only (no model retraining, deterministic), so a
single-arm regen suffices: the committed 2.735 was itself generated on this stack,
so the new cache's delta from 2.735 is purely the UGT fix. Run:

    PYTHONPATH=src python scripts/regen_ugt_canonical.py

The regenerated artifacts (cache, leak baseline, bootstrap CI) are the canonical
outputs to download and commit; apply the printed pin edits.
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
_CACHE = ROOT / "data" / "training" / "4track_holdout_predictions.json"
_LEAK_BASELINE = ROOT / "tests" / "regression" / "data" / "prodrug_v3_pre_baseline.json"

_UGT_SEEDS = {
    "morphine", "codeine", "ketorolac", "indomethacin",
    "dapagliflozin", "etodolac", "bexagliflozin", "glasdegib",
}
_PRIOR_HEADLINE = 2.73531  # committed pre-fix canonical Meta AAFE


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


def main() -> None:
    _require_public_clone()
    from sisyphus.pipeline.predict import predict
    from sisyphus.validation.reference import load_reference

    print("== [1/4] regenerating 4track_holdout_predictions.json ==", flush=True)
    _run(str(ROOT / "scripts" / "run_engine_benchmark.py"), "--save-json", str(_CACHE))
    cache = json.loads(_CACHE.read_text())
    meta_aafe = cache["overall"]["meta"]["aafe"]
    indom_n = cache["in_domain"]["n"]
    indom_aafe = cache["in_domain"]["meta"]["aafe"]
    eng_aafe = cache["overall"]["engine"]["aafe"]

    print("== [2/4] regenerating leak-audit baseline ==", flush=True)
    refs = [r for r in load_reference() if r.in_holdout]
    baseline: dict[str, float] = {}
    for r in refs:
        try:
            baseline[r.name] = float(predict(r.smiles, r.dose_mg, r.route).pk.cmax.mean)
        except Exception as exc:  # noqa: BLE001
            print(f"  WARN {r.name}: {exc}")
    _LEAK_BASELINE.write_text(json.dumps(baseline, indent=2, sort_keys=True) + "\n")

    print("== [3/4] recomputing tebipenem pin ==", flush=True)
    teb_smiles = ("C[C@@H](O)[C@H]1C(=O)N2C(C(=O)OCOC(=O)C(C)(C)C)=C(SC3CN("
                  "C4=NCCS4)C3)[C@H](C)[C@H]12")
    teb = predict(teb_smiles, 300.0, "oral").pk.cmax.mean

    print("== [4/4] bootstrapping CI ==", flush=True)
    _run(str(ROOT / "scripts" / "bootstrap_4track_ci.py"),
         "--cache", str(_CACHE), "--tag", "ugt",
         "--context", "UGT single-path fm fix (double-allocation removed; canonical regen)")

    by_name = {r["name"]: r for r in cache["drugs"]}
    print("\n" + "=" * 72)
    print("UGT SINGLE-PATH CANONICAL REGEN COMPLETE")
    print("=" * 72)
    print(f"  Meta AAFE = {meta_aafe:.5f}  (prior canonical {_PRIOR_HEADLINE:.5f}, "
          f"Δ {meta_aafe - _PRIOR_HEADLINE:+.5f})")
    print(f"  Engine = {eng_aafe:.4f} | in_domain.n = {indom_n} | "
          f"in_domain Meta = {indom_aafe:.5f}")
    print("  8 UGT seed drugs (meta fold):")
    for name in sorted(_UGT_SEEDS):
        r = by_name.get(name)
        if r:
            print(f"    {name:16} obs={r['obs']:.4g}  meta={r['meta']:.4g}  "
                  f"fold={r['meta_fold']:+.3f}")
    print("\n  Apply: repin tests/integration/test_holdout_regression.py cache pin to "
          f"abs(aafe - {meta_aafe:.4f}) < 0.020 (rename if the 3-dp value changed);")
    print(f"  _PINNED['tebipenem_pivoxil'] = {teb:.6e};")
    print("  README headline table Meta -> new value (reconcile vs cache JSON); commit "
          "cache + leak baseline + bootstrap.")


if __name__ == "__main__":
    main()
