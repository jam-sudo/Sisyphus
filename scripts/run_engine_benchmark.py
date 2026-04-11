#!/usr/bin/env python3
"""Engine-only and meta-learner AAFE benchmark on holdout set.

Reports per-drug fold errors and aggregate metrics for both engine-only
and meta-learner predictions.  Used to measure impact of predict-layer
changes (Kp cap, Peff model, etc.) on engine accuracy independently
of ML contributions.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save-json",
        type=Path,
        default=None,
        help="save per-drug predictions + aggregate metrics to JSON",
    )
    args = parser.parse_args()

    from sisyphus.pipeline.predict import predict
    from sisyphus.validation.metrics import aafe, pct_within_n_fold
    from sisyphus.validation.reference import load_reference

    refs = [r for r in load_reference() if r.in_holdout]
    print(f"Holdout drugs: {len(refs)}")

    rows: list[dict] = []
    skipped = 0

    for i, ref in enumerate(refs):
        try:
            result = predict(ref.smiles, ref.dose_mg, ref.route)

            eng_cmax = result.engine_pk.cmax.mean if result.engine_pk else None
            ml_cmax = result.ml_pk.cmax.mean if result.ml_pk else None
            meta_cmax = result.pk.cmax.mean
            compound_type = "?"
            try:
                from sisyphus.predict.chemistry import compute_profile
                p = compute_profile(ref.smiles)
                compound_type = p.compound_type
            except Exception:
                pass

            eng_fold = eng_cmax / ref.cmax_obs if eng_cmax and eng_cmax > 0 else None
            ml_fold = ml_cmax / ref.cmax_obs if ml_cmax and ml_cmax > 0 else None
            meta_fold = meta_cmax / ref.cmax_obs if meta_cmax > 0 else None

            rows.append({
                "name": ref.name,
                "type": compound_type,
                "obs": ref.cmax_obs,
                "eng": eng_cmax,
                "ml": ml_cmax,
                "meta": meta_cmax,
                "eng_fold": eng_fold,
                "ml_fold": ml_fold,
                "meta_fold": meta_fold,
                "in_ad": bool(getattr(result, "in_applicability_domain", True)),
                "ad_flags": list(getattr(result, "ad_flags", [])),
            })
        except Exception as e:
            skipped += 1
            print(f"  SKIP {ref.name}: {e}")

    # Aggregate metrics
    eng_pred = np.array([r["eng"] for r in rows if r["eng"] is not None and r["eng"] > 0])
    eng_obs = np.array([r["obs"] for r in rows if r["eng"] is not None and r["eng"] > 0])
    ml_pred = np.array([r["ml"] for r in rows if r["ml"] is not None and r["ml"] > 0])
    ml_obs = np.array([r["obs"] for r in rows if r["ml"] is not None and r["ml"] > 0])
    meta_pred = np.array([r["meta"] for r in rows if r["meta"] is not None and r["meta"] > 0])
    meta_obs = np.array([r["obs"] for r in rows if r["meta"] is not None and r["meta"] > 0])

    print(f"\n{'='*80}")
    print("AGGREGATE METRICS")
    print(f"{'='*80}")
    print(f"{'Source':<15s} {'N':>4s} {'AAFE':>8s} {'%2-fold':>8s} {'%3-fold':>8s}")
    print(f"{'-'*45}")
    for label, p, o in [("Engine", eng_pred, eng_obs), ("ML", ml_pred, ml_obs), ("Meta-learner", meta_pred, meta_obs)]:
        if len(p) > 0:
            a = aafe(p, o)
            p2 = pct_within_n_fold(p, o, 2.0)
            p3 = pct_within_n_fold(p, o, 3.0)
            print(f"{label:<15s} {len(p):4d} {a:8.3f} {p2:7.1f}% {p3:7.1f}%")

    # Per-drug table sorted by engine fold error magnitude
    print(f"\n{'='*80}")
    print("PER-DRUG FOLD ERRORS (sorted by |log10(eng_fold)|)")
    print(f"{'='*80}")
    print(f"{'Drug':<25s} {'Type':<8s} {'Obs':>8s} {'Eng':>8s} {'ML':>8s} {'Meta':>8s} {'EngFold':>9s} {'Direction':>10s}")
    print(f"{'-'*90}")

    def sort_key(r):
        f = r["eng_fold"]
        return abs(np.log10(f)) if f and f > 0 else 0

    for r in sorted(rows, key=sort_key, reverse=True):
        ef = r["eng_fold"]
        direction = ""
        if ef is not None:
            direction = "OVER" if ef > 3.0 else ("UNDER" if ef < 1/3 else "ok")
        eng_str = f"{r['eng']:.4f}" if r["eng"] else "N/A"
        ml_str = f"{r['ml']:.4f}" if r["ml"] else "N/A"
        meta_str = f"{r['meta']:.4f}" if r["meta"] else "N/A"
        ef_str = f"{ef:.2f}x" if ef else "N/A"
        print(f"{r['name']:<25s} {r['type']:<8s} {r['obs']:8.4f} {eng_str:>8s} {ml_str:>8s} {meta_str:>8s} {ef_str:>9s} {direction:>10s}")

    print(f"\nSkipped: {skipped}")

    # Over-prediction summary (engine fold > 3)
    over = [r for r in rows if r["eng_fold"] is not None and r["eng_fold"] > 3.0]
    under = [r for r in rows if r["eng_fold"] is not None and r["eng_fold"] < 1/3]
    print(f"\nOver-predictions (>3-fold): {len(over)} drugs")
    for r in sorted(over, key=lambda x: x["eng_fold"], reverse=True):
        print(f"  {r['name']:<25s} {r['eng_fold']:.1f}x")
    print(f"\nUnder-predictions (<1/3-fold): {len(under)} drugs")
    for r in sorted(under, key=lambda x: x["eng_fold"]):
        print(f"  {r['name']:<25s} {r['eng_fold']:.3f}x (={1/r['eng_fold']:.1f}x under)")

    # In-domain subset — drugs where AD flags are empty.
    in_dom = [r for r in rows if r.get("in_ad", True) and not r.get("ad_flags")]
    print(f"\nIn-domain subset: N={len(in_dom)} (no AD flags)")
    for label, getter in [
        ("Engine", lambda r: r["eng"]),
        ("ML", lambda r: r["ml"]),
        ("Meta", lambda r: r["meta"]),
    ]:
        vals = [(getter(r), r["obs"]) for r in in_dom if getter(r) and getter(r) > 0]
        if vals:
            pp = np.array([v[0] for v in vals])
            oo = np.array([v[1] for v in vals])
            a = aafe(pp, oo)
            p2 = pct_within_n_fold(pp, oo, 2.0)
            p3 = pct_within_n_fold(pp, oo, 3.0)
            print(f"  {label:<8s} AAFE={a:.3f}  %2-fold={p2:.1f}%  %3-fold={p3:.1f}%")

    if args.save_json is not None:
        # Build structured output for downstream analyses
        def _aggregate(pred_list: list[dict], pred_key: str) -> dict:
            vv = [(r[pred_key], r["obs"]) for r in pred_list if r[pred_key] and r[pred_key] > 0]
            if not vv:
                return {"n": 0}
            pp = np.array([v[0] for v in vv])
            oo = np.array([v[1] for v in vv])
            return {
                "n": len(vv),
                "aafe": float(aafe(pp, oo)),
                "pct_2fold": float(pct_within_n_fold(pp, oo, 2.0)),
                "pct_3fold": float(pct_within_n_fold(pp, oo, 3.0)),
            }

        payload = {
            "n_holdout": len(rows),
            "n_skipped": skipped,
            "overall": {
                "engine": _aggregate(rows, "eng"),
                "ml": _aggregate(rows, "ml"),
                "meta": _aggregate(rows, "meta"),
            },
            "in_domain": {
                "n": len(in_dom),
                "engine": _aggregate(in_dom, "eng"),
                "ml": _aggregate(in_dom, "ml"),
                "meta": _aggregate(in_dom, "meta"),
            },
            "drugs": rows,
        }
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nWrote {args.save_json}")


if __name__ == "__main__":
    main()
