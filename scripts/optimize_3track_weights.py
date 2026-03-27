#!/usr/bin/env python3
"""Phase 3-4: LOOCV weight optimization for 3-track meta-learner + benchmark.

Collects engine, ML, and CL/F Cmax predictions for all holdout drugs,
then optimizes 3-track weights via simplex grid (w1+w2+w3=1, step=0.05).

Reports:
- Optimal weights by compound_type (base vs other)
- AAFE comparison: 2-track vs 3-track
- CL/F track orthogonality analysis
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent


def generate_simplex_grid(step: float = 0.05) -> list[tuple[float, float, float]]:
    """Generate all (w1, w2, w3) with w1+w2+w3=1, step=0.05."""
    combos = []
    n_steps = int(round(1.0 / step))
    for i in range(n_steps + 1):
        for j in range(n_steps + 1 - i):
            k = n_steps - i - j
            w1 = round(i * step, 4)
            w2 = round(j * step, 4)
            w3 = round(k * step, 4)
            combos.append((w1, w2, w3))
    return combos


def compute_aafe(predicted: np.ndarray, observed: np.ndarray) -> float:
    """AAFE = 10^(mean(|log10(pred/obs)|))."""
    mask = (predicted > 0) & (observed > 0)
    if mask.sum() == 0:
        return float("inf")
    log_errors = np.abs(np.log10(predicted[mask] / observed[mask]))
    return float(10 ** np.mean(log_errors))


def pct_2fold(predicted: np.ndarray, observed: np.ndarray) -> float:
    """Percentage of predictions within 2-fold."""
    mask = (predicted > 0) & (observed > 0)
    if mask.sum() == 0:
        return 0.0
    ratios = predicted[mask] / observed[mask]
    return float(np.mean((ratios >= 0.5) & (ratios <= 2.0)) * 100)


def combine_3track(
    log_eng: float | None,
    log_ml: float | None,
    log_clf: float | None,
    w_eng: float,
    w_ml: float,
    w_clf: float,
) -> float:
    """Combine 3 tracks with geometric weighting in log space."""
    tracks = []
    weights = []

    if log_eng is not None:
        tracks.append(log_eng)
        weights.append(w_eng)
    if log_ml is not None:
        tracks.append(log_ml)
        weights.append(w_ml)
    if log_clf is not None:
        tracks.append(log_clf)
        weights.append(w_clf)

    if not tracks:
        return 0.0

    total_w = sum(weights)
    if total_w <= 0:
        # Equal weighting
        return float(10 ** np.mean(tracks))

    # Renormalize
    log_cmax = sum(w / total_w * lc for lc, w in zip(tracks, weights))
    return float(10 ** log_cmax)


def main() -> None:
    # ── Step 1: Collect predictions for all holdout drugs ───────────────
    logger.info("Collecting 3-track predictions for holdout drugs...")
    logger.info("(This runs the full pipeline for each drug — may take several minutes)")

    from sisyphus.pipeline.predict import predict
    from sisyphus.validation.reference import load_reference

    refs = load_reference()
    refs = [r for r in refs if r.in_holdout]
    logger.info("Holdout drugs: %d", len(refs))

    # Collect data
    data = []
    skipped = 0

    for i, ref in enumerate(refs):
        try:
            result = predict(ref.smiles, ref.dose_mg, ref.route)

            entry = {
                "name": ref.name,
                "cmax_obs": ref.cmax_obs,
                "compound_type": "unknown",
                "ad_flags": list(result.ad_flags),
            }

            # Determine compound_type from predict layer
            from sisyphus.predict.chemistry import compute_profile
            try:
                profile = compute_profile(ref.smiles)
                entry["compound_type"] = profile.compound_type
            except Exception:
                pass

            if result.engine_pk and result.engine_pk.cmax.mean > 0:
                entry["cmax_engine"] = result.engine_pk.cmax.mean
            if result.ml_pk and result.ml_pk.cmax.mean > 0:
                entry["cmax_ml"] = result.ml_pk.cmax.mean
            if result.clf_pk and result.clf_pk.cmax.mean > 0:
                entry["cmax_clf"] = result.clf_pk.cmax.mean
            entry["cmax_combined"] = result.pk.cmax.mean

            data.append(entry)

            if (i + 1) % 10 == 0:
                logger.info("  [%d/%d] processed", i + 1, len(refs))
        except Exception as e:
            skipped += 1
            logger.warning("  [%d/%d] %s: failed: %s", i + 1, len(refs), ref.name, e)

    logger.info("Collected: %d drugs, %d skipped", len(data), skipped)

    # Save raw predictions for analysis
    out_path = ROOT / "data" / "training" / "3track_holdout_predictions.json"
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info("Saved predictions to %s", out_path)

    # ── Step 2: Individual track AAFE ──────────────────────────────────
    eng_pred = np.array([d.get("cmax_engine", 0) for d in data])
    ml_pred = np.array([d.get("cmax_ml", 0) for d in data])
    clf_pred = np.array([d.get("cmax_clf", 0) for d in data])
    obs = np.array([d["cmax_obs"] for d in data])

    eng_mask = eng_pred > 0
    ml_mask = ml_pred > 0
    clf_mask = clf_pred > 0

    logger.info("\n=== Individual Track Performance ===")
    logger.info("Engine: N=%d, AAFE=%.3f, %%2-fold=%.1f%%",
                eng_mask.sum(),
                compute_aafe(eng_pred[eng_mask], obs[eng_mask]) if eng_mask.any() else float("inf"),
                pct_2fold(eng_pred[eng_mask], obs[eng_mask]) if eng_mask.any() else 0)
    logger.info("ML:     N=%d, AAFE=%.3f, %%2-fold=%.1f%%",
                ml_mask.sum(),
                compute_aafe(ml_pred[ml_mask], obs[ml_mask]) if ml_mask.any() else float("inf"),
                pct_2fold(ml_pred[ml_mask], obs[ml_mask]) if ml_mask.any() else 0)
    logger.info("CL/F:   N=%d, AAFE=%.3f, %%2-fold=%.1f%%",
                clf_mask.sum(),
                compute_aafe(clf_pred[clf_mask], obs[clf_mask]) if clf_mask.any() else float("inf"),
                pct_2fold(clf_pred[clf_mask], obs[clf_mask]) if clf_mask.any() else 0)

    # ── Step 3: LOOCV weight optimization ──────────────────────────────
    # Split by compound_type
    base_idx = [i for i, d in enumerate(data) if d["compound_type"] == "base"]
    other_idx = [i for i, d in enumerate(data) if d["compound_type"] != "base"]

    logger.info("\nBase drugs: %d, Other drugs: %d", len(base_idx), len(other_idx))

    grid = generate_simplex_grid(0.05)
    logger.info("Weight grid: %d combinations", len(grid))

    # Prepare log-space values
    log_eng = np.where(eng_pred > 0, np.log10(np.maximum(eng_pred, 1e-10)), None)
    log_ml = np.where(ml_pred > 0, np.log10(np.maximum(ml_pred, 1e-10)), None)
    log_clf = np.where(clf_pred > 0, np.log10(np.maximum(clf_pred, 1e-10)), None)
    log_obs = np.log10(np.maximum(obs, 1e-10))

    def evaluate_weights(
        indices: list[int], w_eng: float, w_ml: float, w_clf: float
    ) -> float:
        """Compute AAFE for a subset of drugs with given weights."""
        errors = []
        for idx in indices:
            le = float(log_eng[idx]) if eng_pred[idx] > 0 else None
            lm = float(log_ml[idx]) if ml_pred[idx] > 0 else None
            lc = float(log_clf[idx]) if clf_pred[idx] > 0 else None

            pred = combine_3track(le, lm, lc, w_eng, w_ml, w_clf)
            if pred > 0:
                errors.append(abs(np.log10(pred) - log_obs[idx]))

        if not errors:
            return float("inf")
        return float(10 ** np.mean(errors))

    # LOOCV: for each fold (LOO), find best weights on N-1, evaluate on 1
    # But with 231 weight combos and 107 drugs, full LOOCV is cheap
    logger.info("\nRunning LOOCV weight optimization...")

    # First: find globally optimal weights for base and other
    best_aafe_base = float("inf")
    best_w_base = (0.45, 0.55, 0.00)
    for w_eng, w_ml, w_clf in grid:
        if base_idx:
            a = evaluate_weights(base_idx, w_eng, w_ml, w_clf)
            if a < best_aafe_base:
                best_aafe_base = a
                best_w_base = (w_eng, w_ml, w_clf)

    best_aafe_other = float("inf")
    best_w_other = (0.00, 1.00, 0.00)
    for w_eng, w_ml, w_clf in grid:
        if other_idx:
            a = evaluate_weights(other_idx, w_eng, w_ml, w_clf)
            if a < best_aafe_other:
                best_aafe_other = a
                best_w_other = (w_eng, w_ml, w_clf)

    logger.info("\n=== Optimal Weights (Global) ===")
    logger.info("Base:  w_eng=%.2f, w_ml=%.2f, w_clf=%.2f  AAFE=%.3f",
                *best_w_base, best_aafe_base)
    logger.info("Other: w_eng=%.2f, w_ml=%.2f, w_clf=%.2f  AAFE=%.3f",
                *best_w_other, best_aafe_other)

    # LOOCV stability: for each drug, remove it and re-optimize
    loocv_w_base_list = []
    loocv_w_other_list = []
    loocv_errors = []

    all_indices = list(range(len(data)))
    for loo_idx in range(len(data)):
        # Leave one out
        train_base = [i for i in base_idx if i != loo_idx]
        train_other = [i for i in other_idx if i != loo_idx]

        # Re-optimize
        if data[loo_idx]["compound_type"] == "base" and train_base:
            best_a = float("inf")
            best_w = best_w_base
            for w_eng, w_ml, w_clf in grid:
                a = evaluate_weights(train_base, w_eng, w_ml, w_clf)
                if a < best_a:
                    best_a = a
                    best_w = (w_eng, w_ml, w_clf)
            loocv_w_base_list.append(best_w)
            # Predict left-out drug
            le = float(log_eng[loo_idx]) if eng_pred[loo_idx] > 0 else None
            lm = float(log_ml[loo_idx]) if ml_pred[loo_idx] > 0 else None
            lc = float(log_clf[loo_idx]) if clf_pred[loo_idx] > 0 else None
            pred = combine_3track(le, lm, lc, *best_w)
            if pred > 0:
                loocv_errors.append(abs(np.log10(pred) - log_obs[loo_idx]))
        elif train_other:
            best_a = float("inf")
            best_w = best_w_other
            for w_eng, w_ml, w_clf in grid:
                a = evaluate_weights(train_other, w_eng, w_ml, w_clf)
                if a < best_a:
                    best_a = a
                    best_w = (w_eng, w_ml, w_clf)
            loocv_w_other_list.append(best_w)
            le = float(log_eng[loo_idx]) if eng_pred[loo_idx] > 0 else None
            lm = float(log_ml[loo_idx]) if ml_pred[loo_idx] > 0 else None
            lc = float(log_clf[loo_idx]) if clf_pred[loo_idx] > 0 else None
            pred = combine_3track(le, lm, lc, *best_w)
            if pred > 0:
                loocv_errors.append(abs(np.log10(pred) - log_obs[loo_idx]))

    loocv_aafe = float(10 ** np.mean(loocv_errors)) if loocv_errors else float("inf")

    # Weight stability
    if loocv_w_base_list:
        w_base_arr = np.array(loocv_w_base_list)
        base_mode = best_w_base
        base_stability = np.mean(np.all(w_base_arr == base_mode, axis=1)) * 100
    else:
        base_stability = 0

    if loocv_w_other_list:
        w_other_arr = np.array(loocv_w_other_list)
        other_mode = best_w_other
        other_stability = np.mean(np.all(w_other_arr == other_mode, axis=1)) * 100
    else:
        other_stability = 0

    logger.info("\n=== LOOCV Results ===")
    logger.info("LOOCV AAFE:         %.3f", loocv_aafe)
    logger.info("Base stability:     %.0f%% (mode=%s)", base_stability, best_w_base)
    logger.info("Other stability:    %.0f%% (mode=%s)", other_stability, best_w_other)

    # ── Step 4: Comparison with 2-track ────────────────────────────────
    # 2-track baseline: w_eng=0.45, w_ml=0.55, w_clf=0.00 for base
    #                   w_eng=0.00, w_ml=1.00, w_clf=0.00 for other
    baseline_errors = []
    for idx in range(len(data)):
        le = float(log_eng[idx]) if eng_pred[idx] > 0 else None
        lm = float(log_ml[idx]) if ml_pred[idx] > 0 else None
        if data[idx]["compound_type"] == "base":
            pred = combine_3track(le, lm, None, 0.45, 0.55, 0.00)
        else:
            pred = combine_3track(le, lm, None, 0.00, 1.00, 0.00)
        if pred > 0:
            baseline_errors.append(abs(np.log10(pred) - log_obs[idx]))

    baseline_aafe = float(10 ** np.mean(baseline_errors)) if baseline_errors else float("inf")

    # 3-track with optimal weights
    opt3_errors = []
    opt3_pred_arr = np.zeros(len(data))
    for idx in range(len(data)):
        le = float(log_eng[idx]) if eng_pred[idx] > 0 else None
        lm = float(log_ml[idx]) if ml_pred[idx] > 0 else None
        lc = float(log_clf[idx]) if clf_pred[idx] > 0 else None
        if data[idx]["compound_type"] == "base":
            pred = combine_3track(le, lm, lc, *best_w_base)
        else:
            pred = combine_3track(le, lm, lc, *best_w_other)
        opt3_pred_arr[idx] = pred
        if pred > 0:
            opt3_errors.append(abs(np.log10(pred) - log_obs[idx]))

    opt3_aafe = float(10 ** np.mean(opt3_errors)) if opt3_errors else float("inf")
    opt3_pct2 = pct_2fold(opt3_pred_arr, obs)

    # 2-track pct_2fold
    baseline_pred_arr = np.zeros(len(data))
    for idx in range(len(data)):
        le = float(log_eng[idx]) if eng_pred[idx] > 0 else None
        lm = float(log_ml[idx]) if ml_pred[idx] > 0 else None
        if data[idx]["compound_type"] == "base":
            pred = combine_3track(le, lm, None, 0.45, 0.55, 0.00)
        else:
            pred = combine_3track(le, lm, None, 0.00, 1.00, 0.00)
        baseline_pred_arr[idx] = pred
    baseline_pct2 = pct_2fold(baseline_pred_arr, obs)

    logger.info("\n" + "=" * 70)
    logger.info("BENCHMARK COMPARISON")
    logger.info("=" * 70)
    logger.info("%-20s %-12s %-12s %-12s", "Metric", "2-Track", "3-Track", "Delta")
    logger.info("%-20s %-12.3f %-12.3f %-12.3f", "Meta AAFE",
                baseline_aafe, opt3_aafe, opt3_aafe - baseline_aafe)
    logger.info("%-20s %-12.3f %-12.3f %-12.3f", "LOOCV AAFE",
                baseline_aafe, loocv_aafe, loocv_aafe - baseline_aafe)
    logger.info("%-20s %-11.1f%% %-11.1f%% %-11.1f%%", "%2-fold",
                baseline_pct2, opt3_pct2, opt3_pct2 - baseline_pct2)
    logger.info("%-20s %-12s %-12s", "w_eng (base)",
                "0.45", f"{best_w_base[0]:.2f}")
    logger.info("%-20s %-12s %-12s", "w_ml (base)",
                "0.55", f"{best_w_base[1]:.2f}")
    logger.info("%-20s %-12s %-12s", "w_clf (base)",
                "N/A", f"{best_w_base[2]:.2f}")
    logger.info("%-20s %-12s %-12s", "w_eng (other)",
                "0.00", f"{best_w_other[0]:.2f}")
    logger.info("%-20s %-12s %-12s", "w_ml (other)",
                "1.00", f"{best_w_other[1]:.2f}")
    logger.info("%-20s %-12s %-12s", "w_clf (other)",
                "N/A", f"{best_w_other[2]:.2f}")

    # ── Step 5: Orthogonality analysis ─────────────────────────────────
    # Find drugs where CL/F track is better than both engine and ML
    logger.info("\n=== CL/F Track Orthogonality Analysis ===")
    clf_wins = []
    for idx in range(len(data)):
        if "cmax_clf" not in data[idx] or data[idx].get("cmax_clf", 0) <= 0:
            continue
        obs_val = data[idx]["cmax_obs"]
        fe_clf = abs(np.log10(data[idx]["cmax_clf"] / obs_val))
        fe_eng = abs(np.log10(data[idx].get("cmax_engine", 1e-10) / obs_val)) if data[idx].get("cmax_engine", 0) > 0 else float("inf")
        fe_ml = abs(np.log10(data[idx].get("cmax_ml", 1e-10) / obs_val)) if data[idx].get("cmax_ml", 0) > 0 else float("inf")

        if fe_clf < fe_eng and fe_clf < fe_ml:
            clf_wins.append((data[idx]["name"], data[idx]["compound_type"],
                           10**fe_clf, 10**fe_eng, 10**fe_ml))

    logger.info("Drugs where CL/F track is most accurate: %d / %d", len(clf_wins), len(data))
    if clf_wins:
        logger.info("%-25s %-10s %8s %8s %8s", "Drug", "Type", "FE_clf", "FE_eng", "FE_ml")
        for name, ctype, fe_c, fe_e, fe_m in sorted(clf_wins, key=lambda x: x[2]):
            logger.info("%-25s %-10s %8.2f %8.2f %8.2f", name[:25], ctype, fe_c, fe_e, fe_m)

    # Oracle analysis: best of 3 tracks per drug
    oracle_errors = []
    for idx in range(len(data)):
        fes = []
        if eng_pred[idx] > 0:
            fes.append(abs(np.log10(eng_pred[idx] / obs[idx])))
        if ml_pred[idx] > 0:
            fes.append(abs(np.log10(ml_pred[idx] / obs[idx])))
        if clf_pred[idx] > 0:
            fes.append(abs(np.log10(clf_pred[idx] / obs[idx])))
        if fes:
            oracle_errors.append(min(fes))

    oracle_aafe = float(10 ** np.mean(oracle_errors)) if oracle_errors else float("inf")
    logger.info("\nOracle (best of 3 per drug): AAFE=%.3f", oracle_aafe)
    logger.info("  (lower bound — not achievable, shows potential)")


if __name__ == "__main__":
    main()
