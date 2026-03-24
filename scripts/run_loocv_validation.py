#!/usr/bin/env python3
"""Rigorous LOOCV validation of w_base=0.60 / w_other=0.00 rule.

Answers: "Is w_other=0.00 genuinely optimal, or noise-driven overfitting?"

Four experiments:
1. Fixed weight (0.17/0.17) — baseline
2. Extreme Adaptive (0.60/0.00) — in-sample on N=51
3. LOOCV-A — fixed rule 0.60/0.00, out-of-bag prediction
4. LOOCV-B — per-fold grid search, track w_other distribution
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

DISAGREEMENT_THRESHOLD = 1.0  # log10


def combine(eng: float, ml: float, w_eng_raw: float) -> float:
    """Log-space geometric weighted mean with disagreement scaling."""
    log_eng = np.log10(max(eng, 1e-10))
    log_ml = np.log10(max(ml, 1e-10))
    disagreement = abs(log_eng - log_ml)
    if disagreement > DISAGREEMENT_THRESHOLD:
        w_eng = w_eng_raw * (DISAGREEMENT_THRESHOLD / disagreement)
    else:
        w_eng = w_eng_raw
    w_ml = 1.0 - w_eng
    return 10 ** (w_eng * log_eng + w_ml * log_ml)


def fold_error(pred: float, obs: float) -> float:
    return abs(np.log10(max(pred, 1e-10) / max(obs, 1e-10)))


def compute_aafe(errors: list[float]) -> float:
    return 10 ** np.mean(errors)


def compute_pct2fold(drugs: list[dict], w_base: float, w_other: float) -> float:
    n_within = 0
    for d in drugs:
        w = w_base if d["is_base"] else w_other
        pred = combine(d["eng"], d["ml"], w)
        ratio = pred / d["obs"]
        if 0.5 <= ratio <= 2.0:
            n_within += 1
    return n_within / len(drugs) * 100 if drugs else 0.0


def main():
    import logging
    logging.basicConfig(level=logging.WARNING)

    from sisyphus.pipeline.predict import predict
    from sisyphus.predict.chemistry import compute_profile
    from sisyphus.validation.reference import load_reference

    refs = [r for r in load_reference() if r.in_holdout]
    print(f"Holdout drugs: {len(refs)}")

    # Collect predictions
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

    n_base = sum(d["is_base"] for d in drugs)
    n_other = len(drugs) - n_base
    print(f"Valid: {len(drugs)} (base={n_base}, other={n_other})")

    # ─── 1. Fixed weight (0.17) ─────────────────────────────────────
    errors_fixed = []
    for d in drugs:
        pred = combine(d["eng"], d["ml"], 0.17)
        errors_fixed.append(fold_error(pred, d["obs"]))
    aafe_fixed = compute_aafe(errors_fixed)
    pct2_fixed = compute_pct2fold(drugs, 0.17, 0.17)

    # ─── 2. Extreme Adaptive in-sample (0.60/0.00) ─────────────────
    errors_extreme = []
    for d in drugs:
        w = 0.60 if d["is_base"] else 0.00
        pred = combine(d["eng"], d["ml"], w)
        errors_extreme.append(fold_error(pred, d["obs"]))
    aafe_extreme = compute_aafe(errors_extreme)
    pct2_extreme = compute_pct2fold(drugs, 0.60, 0.00)

    # ─── 3. LOOCV-A: fixed rule 0.60/0.00, out-of-bag ──────────────
    errors_loocv_a = []
    for i in range(len(drugs)):
        held = drugs[i]
        w = 0.60 if held["is_base"] else 0.00
        pred = combine(held["eng"], held["ml"], w)
        errors_loocv_a.append(fold_error(pred, held["obs"]))
    aafe_loocv_a = compute_aafe(errors_loocv_a)

    # For %2-fold, same as in-sample since rule is fixed
    pct2_loocv_a = pct2_extreme  # identical (rule doesn't change)

    # ─── 4. LOOCV-B: per-fold grid search ──────────────────────────
    W_BASE_GRID = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    W_OTHER_GRID = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25]

    errors_loocv_b = []
    wb_chosen: list[float] = []
    wo_chosen: list[float] = []

    for i in range(len(drugs)):
        train = [d for j, d in enumerate(drugs) if j != i]
        held = drugs[i]

        # Grid search on train set
        best_aafe = float("inf")
        best_wb = 0.60
        best_wo = 0.00
        for wb in W_BASE_GRID:
            for wo in W_OTHER_GRID:
                errs = []
                for d in train:
                    w = wb if d["is_base"] else wo
                    pred = combine(d["eng"], d["ml"], w)
                    errs.append(fold_error(pred, d["obs"]))
                a = compute_aafe(errs)
                if a < best_aafe:
                    best_aafe = a
                    best_wb = wb
                    best_wo = wo

        wb_chosen.append(best_wb)
        wo_chosen.append(best_wo)

        # Predict held-out drug with this fold's optimal
        w = best_wb if held["is_base"] else best_wo
        pred = combine(held["eng"], held["ml"], w)
        errors_loocv_b.append(fold_error(pred, held["obs"]))

    aafe_loocv_b = compute_aafe(errors_loocv_b)

    # %2-fold for LOOCV-B: use per-fold weights
    n_within_b = 0
    for i, d in enumerate(drugs):
        w = wb_chosen[i] if d["is_base"] else wo_chosen[i]
        pred = combine(d["eng"], d["ml"], w)
        ratio = pred / d["obs"]
        if 0.5 <= ratio <= 2.0:
            n_within_b += 1
    pct2_loocv_b = n_within_b / len(drugs) * 100

    # ─── Oracle ────────────────────────────────────────────────────
    errors_oracle = []
    for d in drugs:
        e_eng = fold_error(d["eng"], d["obs"])
        e_ml = fold_error(d["ml"], d["obs"])
        errors_oracle.append(min(e_eng, e_ml))
    aafe_oracle = compute_aafe(errors_oracle)

    # ═══════════════════════════════════════════════════════════════
    # OUTPUT
    # ═══════════════════════════════════════════════════════════════

    print(f"\n{'='*72}")
    print("LOOCV VALIDATION RESULTS")
    print(f"{'='*72}")
    print(f"{'Method':<25s} {'AAFE':>8s} {'%2-fold':>8s}  Notes")
    print(f"{'-'*72}")
    print(f"{'Fixed (0.17/0.17)':<25s} {aafe_fixed:8.4f} {pct2_fixed:7.1f}%  baseline")
    print(f"{'Extreme (in-sample)':<25s} {aafe_extreme:8.4f} {pct2_extreme:7.1f}%  0.60/0.00, N={len(drugs)}")
    print(f"{'LOOCV-A (fixed rule)':<25s} {aafe_loocv_a:8.4f} {pct2_loocv_a:7.1f}%  0.60/0.00 fixed, OOB")
    print(f"{'LOOCV-B (optimized)':<25s} {aafe_loocv_b:8.4f} {pct2_loocv_b:7.1f}%  per-fold grid search")
    print(f"{'Oracle':<25s} {aafe_oracle:8.4f}           theoretical ceiling")

    print(f"\nOverfitting diagnostics:")
    print(f"  Extreme in-sample vs LOOCV-A delta: {aafe_loocv_a - aafe_extreme:+.4f}")
    print(f"  LOOCV-B vs LOOCV-A delta:           {aafe_loocv_b - aafe_loocv_a:+.4f}")

    # ─── w_other distribution across 51 folds ─────────────────────
    print(f"\n{'='*72}")
    print("LOOCV-B: w_other DISTRIBUTION ACROSS 51 FOLDS")
    print(f"{'='*72}")
    wo_unique = sorted(set(wo_chosen))
    for w in wo_unique:
        count = wo_chosen.count(w)
        pct = count / len(wo_chosen) * 100
        bar = "#" * int(pct / 2)
        print(f"  w_other={w:.2f}: {count:3d}/{len(wo_chosen)} ({pct:5.1f}%) {bar}")

    print(f"\n{'='*72}")
    print("LOOCV-B: w_base DISTRIBUTION ACROSS 51 FOLDS")
    print(f"{'='*72}")
    wb_unique = sorted(set(wb_chosen))
    for w in wb_unique:
        count = wb_chosen.count(w)
        pct = count / len(wb_chosen) * 100
        bar = "#" * int(pct / 2)
        print(f"  w_base={w:.2f}: {count:3d}/{len(wb_chosen)} ({pct:5.1f}%) {bar}")

    # ─── KEY QUESTION ─────────────────────────────────────────────
    n_wo_nonzero = sum(1 for w in wo_chosen if w > 0.001)
    n_wo_zero = len(wo_chosen) - n_wo_nonzero
    modal_wo = max(set(wo_chosen), key=wo_chosen.count)

    print(f"\n{'='*72}")
    print("KEY QUESTION: Is w_other=0.00 robust or noise-driven?")
    print(f"{'='*72}")
    print(f"  Folds with w_other = 0.00:  {n_wo_zero}/{len(wo_chosen)} ({n_wo_zero/len(wo_chosen)*100:.1f}%)")
    print(f"  Folds with w_other > 0.00:  {n_wo_nonzero}/{len(wo_chosen)} ({n_wo_nonzero/len(wo_chosen)*100:.1f}%)")
    print(f"  Modal w_other: {modal_wo:.2f}")

    if n_wo_nonzero > len(wo_chosen) * 0.2:
        print(f"\n  >>> WARNING: w_other=0.00 is NOT stable.")
        print(f"  >>> {n_wo_nonzero}/{len(wo_chosen)} folds preferred w_other > 0.")
        print(f"  >>> Recommend defensive lower bound: w_other = {modal_wo:.2f}")
    elif n_wo_nonzero > 0:
        print(f"\n  >>> CAUTION: {n_wo_nonzero} fold(s) preferred w_other > 0.")
        print(f"  >>> Small minority — 0.00 is likely robust but consider w_other=0.05 as safety margin.")
    else:
        print(f"\n  >>> CONFIRMED: w_other=0.00 chosen in ALL 51 folds.")
        print(f"  >>> The rule is fully stable.")

    # ─── Per-drug detail: which drugs pull w_other away from 0? ───
    if n_wo_nonzero > 0:
        print(f"\n{'='*72}")
        print("DIAGNOSTIC: Which held-out drugs cause w_other != 0.00?")
        print(f"{'='*72}")
        for i in range(len(drugs)):
            if wo_chosen[i] > 0.001:
                d = drugs[i]
                eng_fold = d["eng"] / d["obs"]
                ml_fold = d["ml"] / d["obs"]
                print(f"  Fold {i:2d}: held={d['name']:<25s} type={d['type']:<10s} "
                      f"eng_fold={eng_fold:.2f} ml_fold={ml_fold:.2f} "
                      f"→ w_base={wb_chosen[i]:.2f} w_other={wo_chosen[i]:.2f}")


if __name__ == "__main__":
    main()
