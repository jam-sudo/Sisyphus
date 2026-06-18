#!/usr/bin/env python3
"""FLUX-1 canonical finalization — run ONCE in the canonical public-clone Linux env.

Why this exists: the FLUX-1 engine fix (PR #65) changed hepatic/gut extraction,
so the committed benchmark cache + a few golden pins are stale. They must be
regenerated in the canonical numerics environment (locked requirements,
Linux/OpenBLAS), NOT on a developer macOS/Accelerate machine (~2.3% aggregate
drift breaks the cache-pin tolerance). This script regenerates the DATA
artifacts and PRINTS the exact test-file edits to apply.

Preconditions (PUBLIC-CLONE STATE — the SSOT):
    mv data/drugbank data/drugbank.local-archive
    mv models/adme/logp_correction.json models/adme/logp_correction.json.local-archive

Run:
    PYTHONPATH=src python scripts/regen_flux1_canonical.py

After it prints the new golden values:
  1. Update tests/integration/test_holdout_regression.py::test_cached_holdout_aafe_is_2p698
     pin to the printed Meta AAFE (and rename / update the docstring).
  2. Update tests/regression/test_prodrug_v2_snapshot.py _PINNED["tebipenem_pivoxil"].
  3. Remove the FLUX-1 xfails from the CACHE/BASELINE-fixable tests (now green):
       - tests/integration/test_ecm_holdout_regression.py::test_ecm_holdout_spot_check_10_drugs
       - tests/regression/test_prodrug_v3_enzyme_leak_audit.py::test_enzyme_leak_audit
       - tests/regression/test_prodrug_v2_snapshot.py  (drop tebipenem from _FLUX1_STALE_PINS)
  4. KEEP the OATP xfails (test_oatp_ecm_statins / test_predict_auto_ecm for
     pravastatin+pitavastatin) until the OATP1B1 abundance is re-anchored to a
     non-holdout substrate — a separate follow-up (do NOT pin the known-wrong
     ECM Cmax values).
  5. Restore artifacts; run the full suite; commit; merge.
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

    # 1. Regenerate the 4track holdout cache (reuses the canonical benchmark script).
    print("== [1/4] regenerating 4track_holdout_predictions.json ==", flush=True)
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_engine_benchmark.py"),
         "--save-json", str(_CACHE)],
        cwd=ROOT, check=True, env=_ENV,
    )
    cache = json.loads(_CACHE.read_text())
    meta_aafe = cache["overall"]["meta"]["aafe"]
    indom_n = cache["in_domain"]["n"]
    indom_aafe = cache["in_domain"]["meta"]["aafe"]
    eng_aafe = cache["overall"]["engine"]["aafe"]

    # 2. Regenerate the leak-audit pre-baseline (public-clone holdout meta Cmax).
    print("== [2/4] regenerating prodrug_v3_pre_baseline.json ==", flush=True)
    refs = [r for r in load_reference() if r.in_holdout]
    baseline: dict[str, float] = {}
    for r in refs:
        try:
            res = predict(r.smiles, r.dose_mg, r.route)
            baseline[r.name] = float(res.pk.cmax.mean)
        except Exception as exc:  # noqa: BLE001
            print(f"  WARN {r.name}: {exc}")
    _LEAK_BASELINE.write_text(json.dumps(baseline, indent=2, sort_keys=True) + "\n")

    # 3. Recompute the regen-fixable golden pins.
    print("== [3/4] recomputing golden pins ==", flush=True)
    teb_smiles = ("C[C@@H](O)[C@H]1C(=O)N2C(C(=O)OCOC(=O)C(C)(C)C)=C(SC3CN("
                  "C4=NCCS4)C3)[C@H](C)[C@H]12")
    teb = predict(teb_smiles, 300.0, "oral").pk.cmax.mean

    # 4. Bootstrap CI on the new cache.
    print("== [4/4] bootstrapping CI ==", flush=True)
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "bootstrap_4track_ci.py"),
         "--cache", str(_CACHE), "--tag", "flux1",
         "--context", "Post-FLUX-1 intrinsic-clearance fix (canonical regen)"],
        cwd=ROOT, check=True, env=_ENV,
    )

    print("\n" + "=" * 72)
    print("CANONICAL FLUX-1 REGEN COMPLETE — apply these test edits:")
    print("=" * 72)
    print(f"  cache: Meta AAFE = {meta_aafe:.5f} | Engine = {eng_aafe:.4f} | "
          f"in_domain.n = {indom_n} | in_domain Meta = {indom_aafe:.5f}")
    print(f"  -> test_cached_holdout_aafe_is_2p698: assert abs(aafe - {meta_aafe:.4f}) < 0.020")
    print(f"  -> _PINNED['tebipenem_pivoxil'] = {teb:.6e}")
    print("  -> remove FLUX-1 xfails: test_ecm_holdout_spot_check_10_drugs, "
          "test_enzyme_leak_audit, and tebipenem from _FLUX1_STALE_PINS")
    print("  -> KEEP OATP xfails (pravastatin/pitavastatin statin + auto_ecm) "
          "until OATP1B1 re-anchor")
    print("  Then: restore artifacts, run full suite, update the README headline "
          "table to the new Meta, commit, merge.")


if __name__ == "__main__":
    main()
