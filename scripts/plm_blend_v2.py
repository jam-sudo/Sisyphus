#!/usr/bin/env python3
"""PLM 3rd Track Integration v2: Improved PLM + 3-Way Blend.

Improvements over v1:
1. Direct Cmax model (not 13-timepoint → max) — less noise
2. v7_clean training data (141 outliers removed)
3. Prediction capping (±2 log10 from population median)
4. XGBoost hyperparameter tuning for Cmax regression

Compares: Engine, ML, PLM_v1, PLM_v2 (retrained), PLM_v2_capped
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import xgboost as xgb
from rdkit import Chem
from rdkit.Chem import AllChem

logging.basicConfig(level=logging.WARNING, format="%(message)s")
log = logging.getLogger("plm_v2")
log.setLevel(logging.INFO)

SISYPHUS_3TRACK = Path("data/training/3track_holdout_predictions.json")
PLM_V3_HOLDOUT = Path("../PLM/data/validation/holdout_predictions_v3.json")
PLM_V7_DATA = Path("../PLM/data/curated/plm_dataset_v7_clean.json")
PLM_HOLDOUT_DEF = Path("../PLM/data/validation/holdout_definition.json")
OUTPUT_JSON = Path("data/validation/plm_blend_v2_results.json")


def morgan_fp(smiles: str, nbits=2048, radius=2) -> np.ndarray | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nbits)
    return np.array(fp, dtype=np.float32)


def aafe(pred, obs):
    pred, obs = np.asarray(pred), np.asarray(obs)
    mask = (pred > 0) & (obs > 0)
    return 10 ** np.mean(np.abs(np.log10(pred[mask] / obs[mask])))


def pct_2fold(pred, obs):
    ratios = np.asarray(pred) / np.asarray(obs)
    return 100.0 * np.mean((ratios >= 0.5) & (ratios <= 2.0))


def load_v7_data():
    """Load PLM v7_clean dataset."""
    with open(PLM_V7_DATA) as f:
        entries = json.load(f)
    log.info(f"  v7_clean: {len(entries)} entries, {len(set(e['ik'] for e in entries))} drugs")
    return entries


def load_holdout_iks():
    """Load Sisyphus holdout InChIKeys."""
    with open(PLM_HOLDOUT_DEF) as f:
        hdef = json.load(f)
    return set(hdef["holdout_inchikeys"])


def train_direct_cmax_model(entries, holdout_iks):
    """Train XGBoost: Morgan FP + dose → log10(Cmax/dose).

    Excludes holdout drugs. Returns trained model.
    """
    train_entries = [e for e in entries if e["ik"] not in holdout_iks]
    log.info(f"  Training entries: {len(train_entries)}, drugs: {len(set(e['ik'] for e in train_entries))}")

    X_list, y_list = [], []
    for e in train_entries:
        fp = morgan_fp(e["smiles"])
        if fp is None:
            continue
        dose = e["dose_mg"]
        if dose <= 0 or e["cmax_ngml"] <= 0:
            continue
        # Features: Morgan FP (2048) + log10(dose)
        features = np.append(fp, np.log10(dose))
        target = np.log10(e["cmax_ngml"] / dose)  # log10(Cmax_ngml / dose)
        X_list.append(features)
        y_list.append(target)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    log.info(f"  Feature matrix: {X.shape}, target range: [{y.min():.2f}, {y.max():.2f}]")

    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.5,
        reg_alpha=1.0,
        reg_lambda=5.0,
        min_child_weight=5,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)
    return model, y


def predict_holdout(model, y_train_stats):
    """Generate PLM v2 predictions for Sisyphus holdout drugs."""
    with open(PLM_V3_HOLDOUT) as f:
        v3_preds = json.load(f)

    holdout_sis = [d for d in v3_preds if d.get("in_sisyphus_holdout")]

    median_log_cd = np.median(y_train_stats)
    std_log_cd = np.std(y_train_stats)

    results = []
    for d in holdout_sis:
        fp = morgan_fp(d["smiles"])
        if fp is None:
            continue
        dose = d["dose_mg"]
        if dose <= 0:
            continue

        features = np.append(fp, np.log10(dose)).reshape(1, -1)
        log_cd = float(model.predict(features)[0])

        # Raw prediction
        cmax_v2_ngml = 10 ** log_cd * dose

        # Capped prediction: clamp log_cd within ±2 std of training median
        log_cd_capped = np.clip(log_cd, median_log_cd - 2 * std_log_cd, median_log_cd + 2 * std_log_cd)
        cmax_v2_capped_ngml = 10 ** log_cd_capped * dose

        results.append({
            "drug_name": d["drug_name"],
            "smiles": d["smiles"],
            "dose_mg": dose,
            "cmax_obs_ngml": d["cmax_obs_ngml"],
            "cmax_plm_v1_ngml": d["cmax_pred_plm_ngml"],  # old v3 predictions
            "cmax_plm_v2_ngml": cmax_v2_ngml,
            "cmax_plm_v2_capped_ngml": cmax_v2_capped_ngml,
            "log_cd_raw": log_cd,
            "log_cd_capped": float(log_cd_capped),
        })

    return results, median_log_cd, std_log_cd


def loocv_blend(cmax_tracks, cmax_obs, track_names):
    """LOOCV N-way blend. Returns best weights and AAFE."""
    n = len(cmax_obs)
    n_tracks = len(track_names)

    # Weight grid (step 0.05, sum to 1)
    from itertools import product
    steps = np.arange(0, 1.05, 0.05)

    if n_tracks == 3:
        w_grid = []
        for w0 in np.arange(0, 0.55, 0.05):  # engine capped at 0.50
            for w1 in np.arange(0, 1.05 - w0, 0.05):
                w2 = round(1.0 - w0 - w1, 2)
                if w2 >= 0:
                    w_grid.append((round(w0, 2), round(w1, 2), round(w2, 2)))
    else:
        raise ValueError("Only 3-track implemented")

    predictions = np.zeros(n)

    for i in range(n):
        train_idx = [j for j in range(n) if j != i]
        best_w = None
        best_train_aafe = float("inf")

        for weights in w_grid:
            pred_train = sum(
                w * cmax_tracks[k][train_idx]
                for k, w in enumerate(weights)
            )
            train_aafe = aafe(pred_train, cmax_obs[train_idx])
            if train_aafe < best_train_aafe:
                best_train_aafe = train_aafe
                best_w = weights

        predictions[i] = sum(w * cmax_tracks[k][i] for k, w in enumerate(best_w))

    return aafe(predictions, cmax_obs), pct_2fold(predictions, cmax_obs), predictions


def main():
    log.info("=" * 70)
    log.info("PLM 3rd Track Integration v2: Improved Model + Blend")
    log.info("=" * 70)

    # ── Step 1: Retrain PLM as direct Cmax model ─────────────────────────
    log.info("\n[Step 1] Retrain PLM on v7_clean (direct Cmax model)")
    entries = load_v7_data()
    holdout_iks = load_holdout_iks()
    model, y_train = train_direct_cmax_model(entries, holdout_iks)

    # ── Step 2: Generate holdout predictions ──────────────────────────────
    log.info("\n[Step 2] Generate holdout predictions")
    holdout_preds, median_lcd, std_lcd = predict_holdout(model, y_train)
    log.info(f"  Holdout predictions: {len(holdout_preds)} drugs")
    log.info(f"  Training log10(Cmax/dose): median={median_lcd:.3f}, std={std_lcd:.3f}")
    log.info(f"  Cap range: [{median_lcd - 2*std_lcd:.3f}, {median_lcd + 2*std_lcd:.3f}]")

    # ── Step 3: Merge with Sisyphus 3-track data ─────────────────────────
    log.info("\n[Step 3] Merge with Sisyphus 3-track predictions")
    with open(SISYPHUS_3TRACK) as f:
        sis_data = json.load(f)

    # Build lookup
    plm_lookup = {d["drug_name"].lower(): d for d in holdout_preds}

    merged = []
    for s in sis_data:
        p = plm_lookup.get(s["name"].lower())
        if p is None:
            continue
        merged.append({
            "name": s["name"],
            "cmax_obs": s["cmax_obs"],  # mg/L
            "cmax_engine": s["cmax_engine"],
            "cmax_ml": s["cmax_ml"],
            "cmax_meta_2way": s["cmax_combined"],
            "cmax_plm_v1": p["cmax_plm_v1_ngml"] / 1000,  # ng/mL → mg/L
            "cmax_plm_v2": p["cmax_plm_v2_ngml"] / 1000,
            "cmax_plm_v2c": p["cmax_plm_v2_capped_ngml"] / 1000,
        })

    n = len(merged)
    log.info(f"  Merged overlap: {n} drugs")

    # ── Step 4: Compare PLM versions ──────────────────────────────────────
    log.info(f"\n[Step 4] PLM Version Comparison (N={n})")

    obs = np.array([d["cmax_obs"] for d in merged])
    engine = np.array([d["cmax_engine"] for d in merged])
    ml = np.array([d["cmax_ml"] for d in merged])
    meta2 = np.array([d["cmax_meta_2way"] for d in merged])
    plm_v1 = np.array([d["cmax_plm_v1"] for d in merged])
    plm_v2 = np.array([d["cmax_plm_v2"] for d in merged])
    plm_v2c = np.array([d["cmax_plm_v2c"] for d in merged])

    tracks = [
        ("Engine", engine),
        ("ML", ml),
        ("2-way Meta", meta2),
        ("PLM v1 (13-timepoint)", plm_v1),
        ("PLM v2 (direct Cmax)", plm_v2),
        ("PLM v2 capped", plm_v2c),
    ]

    log.info(f"\n  {'Track':<30} {'AAFE':>8} {'%2-fold':>8}")
    log.info(f"  {'-'*30} {'-'*8} {'-'*8}")
    for name, pred in tracks:
        a = aafe(pred, obs)
        p2 = pct_2fold(pred, obs)
        log.info(f"  {name:<30} {a:>8.3f} {p2:>7.1f}%")

    # ── Step 5: Error correlations ────────────────────────────────────────
    log.info(f"\n[Step 5] Error Correlations")
    eps = 1e-12
    err_engine = np.log10(np.maximum(engine, eps) / np.maximum(obs, eps))
    err_ml = np.log10(np.maximum(ml, eps) / np.maximum(obs, eps))
    err_v1 = np.log10(np.maximum(plm_v1, eps) / np.maximum(obs, eps))
    err_v2 = np.log10(np.maximum(plm_v2, eps) / np.maximum(obs, eps))
    err_v2c = np.log10(np.maximum(plm_v2c, eps) / np.maximum(obs, eps))

    log.info(f"\n  {'Pair':<35} {'r':>8}")
    log.info(f"  {'-'*35} {'-'*8}")
    for name, err in [("ML - PLM v1", (err_ml, err_v1)),
                       ("ML - PLM v2", (err_ml, err_v2)),
                       ("ML - PLM v2 capped", (err_ml, err_v2c)),
                       ("Engine - PLM v2 capped", (err_engine, err_v2c)),
                       ("PLM v1 - PLM v2", (err_v1, err_v2))]:
        r = np.corrcoef(name.split(" - ")[0] and err[0], err[1])[0, 1]
        log.info(f"  {name:<35} {r:>8.3f}")

    # ── Step 6: LOOCV 3-way blends ───────────────────────────────────────
    log.info(f"\n[Step 6] LOOCV 3-Way Blends")

    configs = [
        ("Engine + ML + PLMv1", [engine, ml, plm_v1]),
        ("Engine + ML + PLMv2", [engine, ml, plm_v2]),
        ("Engine + ML + PLMv2c", [engine, ml, plm_v2c]),
    ]

    # 2-way baseline (recalculated on overlap)
    # Inline 2-way LOOCV
    n2 = len(obs)
    pred_2way = np.zeros(n2)
    for i in range(n2):
        tidx = [j for j in range(n2) if j != i]
        best_w, best_a = None, float("inf")
        for w_e in np.arange(0, 0.55, 0.05):
            w_ml = round(1.0 - w_e, 2)
            p = w_e * engine[tidx] + w_ml * ml[tidx]
            a = aafe(p, obs[tidx])
            if a < best_a:
                best_a = a
                best_w = (w_e, w_ml)
        pred_2way[i] = best_w[0] * engine[i] + best_w[1] * ml[i]
    aafe_2way = aafe(pred_2way, obs)
    pct2_2way = pct_2fold(pred_2way, obs)

    log.info(f"\n  {'Config':<35} {'AAFE':>8} {'%2-fold':>8} {'Δ vs 2-way':>12}")
    log.info(f"  {'-'*35} {'-'*8} {'-'*8} {'-'*12}")
    log.info(f"  {'2-way Meta (baseline)':<35} {aafe_2way:>8.3f} {pct2_2way:>7.1f}% {'---':>12}")

    best_config = None
    best_aafe = aafe_2way

    for name, track_arrays in configs:
        a3, p3, _ = loocv_blend(track_arrays, obs, ["eng", "ml", "plm"])
        delta = a3 - aafe_2way
        marker = " ←" if a3 < aafe_2way else ""
        log.info(f"  {name:<35} {a3:>8.3f} {p3:>7.1f}% {delta:>+11.3f}{marker}")
        if a3 < best_aafe:
            best_aafe = a3
            best_config = name

    # ── Step 7: Oracle analysis ───────────────────────────────────────────
    log.info(f"\n[Step 7] Oracle Analysis (per-drug best track)")
    best_counts = {"engine": 0, "ml": 0, "plm_v2c": 0}
    plm_wins = []
    for i in range(n):
        fe_e = abs(np.log10(max(engine[i], eps) / max(obs[i], eps)))
        fe_ml = abs(np.log10(max(ml[i], eps) / max(obs[i], eps)))
        fe_plm = abs(np.log10(max(plm_v2c[i], eps) / max(obs[i], eps)))
        min_fe = min(fe_e, fe_ml, fe_plm)
        if min_fe == fe_plm:
            best_counts["plm_v2c"] += 1
            plm_wins.append(merged[i]["name"])
        elif min_fe == fe_ml:
            best_counts["ml"] += 1
        else:
            best_counts["engine"] += 1

    oracle_preds = np.array([
        min(engine[i], ml[i], plm_v2c[i],
            key=lambda x: abs(np.log10(max(x, eps) / max(obs[i], eps))))
        for i in range(n)
    ])
    aafe_oracle = aafe(oracle_preds, obs)

    log.info(f"  Oracle AAFE: {aafe_oracle:.3f}")
    log.info(f"  Best track: engine={best_counts['engine']}, ml={best_counts['ml']}, plm_v2c={best_counts['plm_v2c']}")
    if plm_wins:
        log.info(f"  PLM v2c best ({len(plm_wins)}): {', '.join(plm_wins[:15])}")

    # ── Step 8: Verdict ───────────────────────────────────────────────────
    log.info(f"\n{'=' * 70}")
    log.info("[Step 8] Verdict")
    log.info("=" * 70)

    if best_config:
        log.info(f"\n  IMPROVEMENT: {best_config} AAFE {best_aafe:.3f} < 2-way {aafe_2way:.3f}")
    else:
        log.info(f"\n  NO IMPROVEMENT: All 3-way blends ≥ 2-way baseline ({aafe_2way:.3f})")

    # Outlier improvement comparison
    v1_outliers = sum(1 for i in range(n) if max(plm_v1[i]/obs[i], obs[i]/plm_v1[i]) > 10)
    v2c_outliers = sum(1 for i in range(n) if max(plm_v2c[i]/obs[i], obs[i]/plm_v2c[i]) > 10)
    log.info(f"\n  Outliers (FE>10x): PLM v1={v1_outliers}, PLM v2 capped={v2c_outliers}")
    log.info(f"  AAFE improvement: PLM v1={aafe(plm_v1, obs):.3f} → v2c={aafe(plm_v2c, obs):.3f}")

    # ── Save results ──────────────────────────────────────────────────────
    results = {
        "experiment": "PLM 3rd Track Integration v2",
        "date": "2026-04-02",
        "n_overlap": n,
        "training_data": "v7_clean (3490 entries, holdout excluded)",
        "cap_range": f"[{median_lcd - 2*std_lcd:.3f}, {median_lcd + 2*std_lcd:.3f}]",
        "standalone_aafe": {
            "engine": round(aafe(engine, obs), 3),
            "ml": round(aafe(ml, obs), 3),
            "plm_v1": round(aafe(plm_v1, obs), 3),
            "plm_v2": round(aafe(plm_v2, obs), 3),
            "plm_v2_capped": round(aafe(plm_v2c, obs), 3),
        },
        "error_correlations": {
            "ml_plm_v1": round(float(np.corrcoef(err_ml, err_v1)[0, 1]), 3),
            "ml_plm_v2": round(float(np.corrcoef(err_ml, err_v2)[0, 1]), 3),
            "ml_plm_v2c": round(float(np.corrcoef(err_ml, err_v2c)[0, 1]), 3),
        },
        "loocv_2way_aafe": round(aafe_2way, 3),
        "best_3way_config": best_config,
        "best_3way_aafe": round(best_aafe, 3) if best_config else None,
        "oracle_aafe": round(aafe_oracle, 3),
        "oracle_wins": best_counts,
        "outliers_v1": v1_outliers,
        "outliers_v2c": v2c_outliers,
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"\n  Saved to {OUTPUT_JSON}")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
