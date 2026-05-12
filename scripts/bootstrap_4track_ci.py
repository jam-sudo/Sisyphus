#!/usr/bin/env python3
"""Bootstrap 95% CIs for the 4-track holdout AAFE numbers.

Reads `data/training/4track_holdout_predictions.json` (the cached per-drug
predictions produced by `scripts/run_engine_benchmark.py`) and writes
`data/validation/4track_ci_<date>_<tag>.json` with point estimates + CIs
for Engine / ML / Meta on the overall N=107 holdout AND the in-domain
subset.

Method (matches `scripts/run_n50_benchmark.py::_aafe_with_ci`):
    AAFE = 10 ^ mean(abs(log10(fold)))
    CI   = 10 ^ percentile(bootstrap_means, [2.5, 97.5])
where the bootstrap resamples `abs(log10(fold))` with replacement,
10,000 iterations, seed=20260422 (deterministic across runs).

Usage:
    python scripts/bootstrap_4track_ci.py
    python scripts/bootstrap_4track_ci.py --tag v0.4 --out data/validation/4track_ci_2026-05-12_v0.4.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE = ROOT / "data" / "training" / "4track_holdout_predictions.json"

N_BOOTSTRAP = 10000
SEED = 20260422


def aafe_with_ci(folds: list[float], n_bootstrap: int = N_BOOTSTRAP) -> dict:
    arr = np.asarray([f for f in folds if f is not None and f > 0], dtype=float)
    if arr.size == 0:
        return {"aafe": None, "ci_95_low": None, "ci_95_high": None, "n": 0}
    log_abs = np.abs(np.log10(arr))
    aafe = float(10.0 ** log_abs.mean())
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, arr.size, size=(n_bootstrap, arr.size))
    boot_means = log_abs[idx].mean(axis=1)
    ci_low = float(10.0 ** np.percentile(boot_means, 2.5))
    ci_high = float(10.0 ** np.percentile(boot_means, 97.5))
    return {"aafe": round(aafe, 4), "ci_95_low": round(ci_low, 4),
            "ci_95_high": round(ci_high, 4), "n": int(arr.size)}


def pct_within_n_fold(folds: list[float], n: float) -> float | None:
    arr = np.asarray([f for f in folds if f is not None and f > 0], dtype=float)
    if arr.size == 0:
        return None
    log_abs = np.abs(np.log10(arr))
    return float(round(100.0 * (log_abs <= np.log10(n)).mean(), 1))


def _track_summary(rows: list[dict], track_key: str) -> dict:
    folds = [r.get(f"{track_key}_fold") for r in rows]
    out = aafe_with_ci(folds)
    out["pct_2fold"] = pct_within_n_fold(folds, 2.0)
    out["pct_3fold"] = pct_within_n_fold(folds, 3.0)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path, default=DEFAULT_CACHE,
                   help="path to 4track_holdout_predictions.json")
    p.add_argument("--tag", default="v0.4",
                   help="artifact tag (date suffix is auto)")
    p.add_argument("--out", type=Path, default=None,
                   help="output JSON path (default: data/validation/4track_ci_<date>_<tag>.json)")
    p.add_argument("--context", default="public-clone deterministic state (no DrugBank, no logp_correction); audit-driven honesty regen.",
                   help="prose context for the artifact")
    p.add_argument("--date", default=None,
                   help="date stamp YYYY-MM-DD (default: today)")
    args = p.parse_args()

    from datetime import datetime, timezone
    date = args.date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_path = args.out or ROOT / "data" / "validation" / f"4track_ci_{date}_{args.tag}.json"

    cached = json.loads(args.cache.read_text())
    drugs: list[dict] = cached["drugs"]
    in_domain = [r for r in drugs if r.get("in_ad", True) and not r.get("ad_flags")]

    report = {
        "computed_at": f"{date}-{args.tag}",
        "source_cache": str(args.cache.relative_to(ROOT)),
        "context": args.context,
        "method": "bootstrap on abs(log10(fold))",
        "n_bootstrap": N_BOOTSTRAP,
        "seed": SEED,
        "overall": {
            "n": len(drugs),
            "engine": _track_summary(drugs, "eng"),
            "ml":     _track_summary(drugs, "ml"),
            "meta":   _track_summary(drugs, "meta"),
        },
        "in_domain": {
            "n": len(in_domain),
            "engine": _track_summary(in_domain, "eng"),
            "ml":     _track_summary(in_domain, "ml"),
            "meta":   _track_summary(in_domain, "meta"),
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {out_path}")
    for slice_name in ("overall", "in_domain"):
        s = report[slice_name]
        print(f"\n{slice_name} (N={s['n']}):")
        for track in ("engine", "ml", "meta"):
            t = s[track]
            print(f"  {track:6s}  AAFE={t['aafe']:.4f}  CI=[{t['ci_95_low']:.4f}, {t['ci_95_high']:.4f}]  "
                  f"%2-fold={t['pct_2fold']}  %3-fold={t['pct_3fold']}")


if __name__ == "__main__":
    main()
