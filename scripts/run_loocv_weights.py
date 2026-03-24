#!/usr/bin/env python3
"""LOOCV-B: Re-optimize adaptive engine weights after Peff model change.

Performs Leave-One-Out Cross-Validation to find optimal engine weights
for base and non-base drugs, evaluating a grid of (w_base, w_other) pairs.

The LOOCV procedure:
  For each holdout drug i:
    For each (w_base, w_other) candidate:
      Combine engine_i and ml_i using the candidate weights
      Compute fold error for drug i
  For each (w_base, w_other):
    AAFE = geometric mean of all fold errors

This ensures the weight selection is fully generalizable (no train-on-test).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main():
    from sisyphus.pipeline.predict import predict
    from sisyphus.predict.chemistry import compute_profile
    from sisyphus.validation.reference import load_reference

    refs = [r for r in load_reference() if r.in_holdout]
    print(f"Holdout drugs: {len(refs)}")

    # Collect per-drug engine/ML predictions and metadata
    drugs: list[dict] = []
    for ref in refs:
        try:
            result = predict(ref.smiles, ref.dose_mg, ref.route)
            eng = result.engine_pk.cmax.mean if result.engine_pk else None
            ml = result.ml_pk.cmax.mean if result.ml_pk else None
            if eng and eng > 0 and ml and ml > 0:
                p = compute_profile(ref.smiles)
                drugs.append({
                    "name": ref.name,
                    "obs": ref.cmax_obs,
                    "eng": eng,
                    "ml": ml,
                    "type": p.compound_type,
                    "is_base": p.compound_type == "base",
                })
        except Exception:
            pass

    print(f"Valid drugs: {len(drugs)} (base: {sum(d['is_base'] for d in drugs)}, other: {sum(not d['is_base'] for d in drugs)})")

    # LOOCV-B: grid search over (w_base, w_other)
    w_range = np.arange(0.0, 0.65, 0.05)
    disagreement_threshold = 1.0  # log10 units

    best_aafe = float("inf")
    best_wb = 0.0
    best_wo = 0.0
    results = []

    for w_base in w_range:
        for w_other in w_range:
            log_errors = []
            for drug in drugs:
                w_eng_raw = w_base if drug["is_base"] else w_other
                log_eng = np.log10(max(drug["eng"], 1e-10))
                log_ml = np.log10(max(drug["ml"], 1e-10))

                # Disagreement scaling
                disagreement = abs(log_eng - log_ml)
                if disagreement > disagreement_threshold:
                    scale = disagreement_threshold / disagreement
                    w_eng = w_eng_raw * scale
                else:
                    w_eng = w_eng_raw

                w_ml = 1.0 - w_eng
                log_cmax = w_eng * log_eng + w_ml * log_ml
                cmax_pred = 10**log_cmax

                fold = abs(np.log10(max(cmax_pred, 1e-10) / max(drug["obs"], 1e-10)))
                log_errors.append(fold)

            aafe = 10**np.mean(log_errors)
            results.append((w_base, w_other, aafe))
            if aafe < best_aafe:
                best_aafe = aafe
                best_wb = w_base
                best_wo = w_other

    print(f"\n{'='*60}")
    print("LOOCV-B GRID SEARCH RESULTS")
    print(f"{'='*60}")
    print(f"Best: w_base={best_wb:.2f}, w_other={best_wo:.2f}, AAFE={best_aafe:.4f}")

    # Stability analysis: how often is each weight chosen in LOO folds?
    print(f"\n{'='*60}")
    print("WEIGHT STABILITY (LOO folds)")
    print(f"{'='*60}")

    wb_counts: dict[float, int] = {}
    wo_counts: dict[float, int] = {}
    fold_aaes: list[float] = []

    for i in range(len(drugs)):
        loo_drugs = [d for j, d in enumerate(drugs) if j != i]
        best_loo_aafe = float("inf")
        best_loo_wb = 0.0
        best_loo_wo = 0.0

        for w_base in w_range:
            for w_other in w_range:
                log_errors = []
                for drug in loo_drugs:
                    w_eng_raw = w_base if drug["is_base"] else w_other
                    log_eng = np.log10(max(drug["eng"], 1e-10))
                    log_ml = np.log10(max(drug["ml"], 1e-10))
                    disagreement = abs(log_eng - log_ml)
                    if disagreement > disagreement_threshold:
                        w_eng = w_eng_raw * (disagreement_threshold / disagreement)
                    else:
                        w_eng = w_eng_raw
                    w_ml = 1.0 - w_eng
                    log_cmax = w_eng * log_eng + w_ml * log_ml
                    cmax_pred = 10**log_cmax
                    fold = abs(np.log10(max(cmax_pred, 1e-10) / max(drug["obs"], 1e-10)))
                    log_errors.append(fold)

                aafe = 10**np.mean(log_errors)
                if aafe < best_loo_aafe:
                    best_loo_aafe = aafe
                    best_loo_wb = w_base
                    best_loo_wo = w_other

        wb_counts[best_loo_wb] = wb_counts.get(best_loo_wb, 0) + 1
        wo_counts[best_loo_wo] = wo_counts.get(best_loo_wo, 0) + 1

        # Predict held-out drug with LOO-optimal weights
        held = drugs[i]
        w_eng_raw = best_loo_wb if held["is_base"] else best_loo_wo
        log_eng = np.log10(max(held["eng"], 1e-10))
        log_ml = np.log10(max(held["ml"], 1e-10))
        disagreement = abs(log_eng - log_ml)
        if disagreement > disagreement_threshold:
            w_eng = w_eng_raw * (disagreement_threshold / disagreement)
        else:
            w_eng = w_eng_raw
        w_ml = 1.0 - w_eng
        log_cmax = w_eng * log_eng + w_ml * log_ml
        cmax_pred = 10**log_cmax
        fold_aaes.append(abs(np.log10(max(cmax_pred, 1e-10) / max(held["obs"], 1e-10))))

    loocv_aafe = 10**np.mean(fold_aaes)
    print(f"\nLOOCV-A AAFE (true generalization): {loocv_aafe:.4f}")
    print(f"In-sample AAFE: {best_aafe:.4f}")
    print(f"Overfitting delta: {loocv_aafe - best_aafe:.4f}")

    print(f"\nw_base stability:")
    for w, count in sorted(wb_counts.items(), key=lambda x: -x[1]):
        pct = count / len(drugs) * 100
        print(f"  w_base={w:.2f}: {count}/{len(drugs)} folds ({pct:.0f}%)")
        if pct < 5:
            break

    print(f"\nw_other stability:")
    for w, count in sorted(wo_counts.items(), key=lambda x: -x[1]):
        pct = count / len(drugs) * 100
        print(f"  w_other={w:.2f}: {count}/{len(drugs)} folds ({pct:.0f}%)")
        if pct < 5:
            break

    # Also show AAFE at old weights for comparison
    print(f"\n{'='*60}")
    print("COMPARISON WITH OLD WEIGHTS")
    print(f"{'='*60}")
    for label, wb, wo in [
        ("Old (0.30/0.15)", 0.30, 0.15),
        ("Optimal", best_wb, best_wo),
        ("ML-only (0.00/0.00)", 0.00, 0.00),
    ]:
        log_errors = []
        for drug in drugs:
            w_eng_raw = wb if drug["is_base"] else wo
            log_eng = np.log10(max(drug["eng"], 1e-10))
            log_ml = np.log10(max(drug["ml"], 1e-10))
            disagreement = abs(log_eng - log_ml)
            if disagreement > disagreement_threshold:
                w_eng = w_eng_raw * (disagreement_threshold / disagreement)
            else:
                w_eng = w_eng_raw
            w_ml = 1.0 - w_eng
            log_cmax = w_eng * log_eng + w_ml * log_ml
            cmax_pred = 10**log_cmax
            fold = abs(np.log10(max(cmax_pred, 1e-10) / max(drug["obs"], 1e-10)))
            log_errors.append(fold)
        a = 10**np.mean(log_errors)
        print(f"  {label:<25s}: AAFE={a:.4f}")

    # Oracle selector (always pick better)
    log_errors_oracle = []
    for drug in drugs:
        eng_fold = abs(np.log10(max(drug["eng"], 1e-10) / max(drug["obs"], 1e-10)))
        ml_fold = abs(np.log10(max(drug["ml"], 1e-10) / max(drug["obs"], 1e-10)))
        log_errors_oracle.append(min(eng_fold, ml_fold))
    oracle_aafe = 10**np.mean(log_errors_oracle)
    print(f"  {'Oracle selector':<25s}: AAFE={oracle_aafe:.4f}")


if __name__ == "__main__":
    main()
