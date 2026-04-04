#!/usr/bin/env python3
"""PLM 3rd Track v3: FDA-Only Data → Truly Independent PLM.

Key change from v2: Train PLM EXCLUSIVELY on FDA table data (pk_table_extracted.json).
No MMPK data → PLM errors are truly independent from Sisyphus ML.

Previous attempts failed because PLM shared 94% training data with ML.
With 618 FDA-only entries (151 drugs), PLM becomes a genuinely different source.
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import xgboost as xgb
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.inchi import MolToInchi, InchiToInchiKey

logging.basicConfig(level=logging.WARNING, format="%(message)s")
log = logging.getLogger("plm_v3")
log.setLevel(logging.INFO)

SISYPHUS_3TRACK = Path("data/training/3track_holdout_predictions.json")
PLM_V3_HOLDOUT = Path("../PLM/data/validation/holdout_predictions_v3.json")
PLM_HOLDOUT_DEF = Path("../PLM/data/validation/holdout_definition.json")
FDA_DATA = Path("../PLM/data/curated/pk_table_extracted.json")
OUTPUT_JSON = Path("data/validation/plm_blend_v3_fda_only_results.json")


def morgan_fp(smiles, nbits=2048, radius=2):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nbits)
    return np.array(fp, dtype=np.float32)


def get_ik(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        inchi = MolToInchi(mol)
        return InchiToInchiKey(inchi)[:14] if inchi else None
    except Exception:
        return None


def aafe(pred, obs):
    pred, obs = np.asarray(pred, dtype=float), np.asarray(obs, dtype=float)
    mask = (pred > 0) & (obs > 0)
    return 10 ** np.mean(np.abs(np.log10(pred[mask] / obs[mask])))


def pct_2fold(pred, obs):
    r = np.asarray(pred) / np.asarray(obs)
    return 100.0 * np.mean((r >= 0.5) & (r <= 2.0))


def load_fda_data():
    """Load and validate FDA-only PK data."""
    with open(FDA_DATA) as f:
        raw = json.load(f)

    valid = []
    for d in raw:
        smi = d.get("smiles", "")
        dose = d.get("dose_mg")
        cmax = d.get("cmax_ngml")
        if not smi or not dose or not cmax or dose <= 0 or cmax <= 0:
            continue
        fp = morgan_fp(smi)
        if fp is None:
            continue
        ik = get_ik(smi)
        if ik is None:
            continue
        valid.append({
            "smiles": smi,
            "dose_mg": dose,
            "cmax_ngml": cmax,
            "ik": ik,
            "drug_name": d.get("drug_name", ""),
            "log_cd": np.log10(cmax / dose),
        })

    return valid


def loocv_3way(engine, ml, plm, obs):
    """LOOCV 3-way linear blend."""
    n = len(obs)
    preds = np.zeros(n)
    w_counts = {}

    for i in range(n):
        t = [j for j in range(n) if j != i]
        best_w, best_a = None, float("inf")
        for w_e in np.arange(0, 0.55, 0.05):
            for w_ml in np.arange(0, 1.05 - w_e, 0.05):
                w_p = round(1.0 - w_e - w_ml, 2)
                if w_p < 0:
                    continue
                p = round(w_e, 2) * engine[t] + round(w_ml, 2) * ml[t] + w_p * plm[t]
                a = aafe(p, obs[t])
                if a < best_a:
                    best_a = a
                    best_w = (round(w_e, 2), round(w_ml, 2), w_p)
        preds[i] = best_w[0] * engine[i] + best_w[1] * ml[i] + best_w[2] * plm[i]
        key = f"{best_w[0]:.2f},{best_w[1]:.2f},{best_w[2]:.2f}"
        w_counts[key] = w_counts.get(key, 0) + 1

    return aafe(preds, obs), pct_2fold(preds, obs), preds, w_counts


def main():
    log.info("=" * 70)
    log.info("PLM 3rd Track v3: FDA-Only Independent Model")
    log.info("=" * 70)

    # ── Step 1: Load FDA-only data ────────────────────────────────────────
    log.info("\n[Step 1] Load FDA-only data")
    fda_entries = load_fda_data()
    log.info(f"  FDA entries: {len(fda_entries)}, drugs: {len(set(d['ik'] for d in fda_entries))}")

    # Exclude holdout drugs
    with open(PLM_HOLDOUT_DEF) as f:
        holdout_iks = set(json.load(f)["holdout_inchikeys"])

    train_entries = [d for d in fda_entries if d["ik"] not in holdout_iks]
    test_iks = set(d["ik"] for d in fda_entries if d["ik"] in holdout_iks)
    log.info(f"  Training (holdout excluded): {len(train_entries)} entries, {len(set(d['ik'] for d in train_entries))} drugs")
    log.info(f"  Holdout overlap: {len(test_iks)} drugs in both FDA and holdout")

    # ── Step 2: Train XGBoost on FDA-only ─────────────────────────────────
    log.info("\n[Step 2] Train XGBoost on FDA-only data")

    X_list, y_list = [], []
    for d in train_entries:
        fp = morgan_fp(d["smiles"])
        if fp is None:
            continue
        features = np.append(fp, np.log10(d["dose_mg"]))
        X_list.append(features)
        y_list.append(d["log_cd"])

    X_train = np.array(X_list, dtype=np.float32)
    y_train = np.array(y_list, dtype=np.float32)
    log.info(f"  Features: {X_train.shape}, target: [{y_train.min():.2f}, {y_train.max():.2f}]")

    model = xgb.XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.5,
        reg_alpha=1.0, reg_lambda=5.0, min_child_weight=5,
        random_state=42, n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # ── Step 3: Predict for Sisyphus holdout ──────────────────────────────
    log.info("\n[Step 3] Predict for Sisyphus holdout")

    with open(PLM_V3_HOLDOUT) as f:
        v3_preds = json.load(f)
    v3_sis = {d["drug_name"].lower(): d for d in v3_preds if d.get("in_sisyphus_holdout")}

    with open(SISYPHUS_3TRACK) as f:
        sis_data = json.load(f)

    merged = []
    for s in sis_data:
        p = v3_sis.get(s["name"].lower())
        if p is None:
            continue

        fp = morgan_fp(p["smiles"])
        if fp is None:
            continue

        features = np.append(fp, np.log10(p["dose_mg"])).reshape(1, -1)
        log_cd = float(model.predict(features)[0])
        cmax_fda = 10 ** log_cd * p["dose_mg"] / 1000  # ng/mL → mg/L

        merged.append({
            "name": s["name"],
            "cmax_obs": s["cmax_obs"],
            "cmax_engine": s["cmax_engine"],
            "cmax_ml": s["cmax_ml"],
            "cmax_meta_2way": s["cmax_combined"],
            "cmax_plm_v1": p["cmax_pred_plm_ngml"] / 1000,
            "cmax_plm_fda": cmax_fda,
        })

    n = len(merged)
    log.info(f"  Merged: {n} drugs")

    obs = np.array([d["cmax_obs"] for d in merged])
    engine = np.array([d["cmax_engine"] for d in merged])
    ml = np.array([d["cmax_ml"] for d in merged])
    plm_v1 = np.array([d["cmax_plm_v1"] for d in merged])
    plm_fda = np.array([d["cmax_plm_fda"] for d in merged])

    # ── Step 4: Standalone comparison ─────────────────────────────────────
    log.info(f"\n[Step 4] Standalone AAFE (N={n})")
    tracks = [
        ("Engine", engine), ("ML", ml),
        ("PLM v1 (shared data)", plm_v1),
        ("PLM v3 (FDA-only)", plm_fda),
    ]
    log.info(f"\n  {'Track':<30} {'AAFE':>8} {'%2-fold':>8}")
    log.info(f"  {'-'*30} {'-'*8} {'-'*8}")
    for name, pred in tracks:
        log.info(f"  {name:<30} {aafe(pred, obs):>8.3f} {pct_2fold(pred, obs):>7.1f}%")

    # ── Step 5: Error correlations ────────────────────────────────────────
    log.info(f"\n[Step 5] Error Correlations")
    eps = 1e-12
    err_ml = np.log10(np.maximum(ml, eps) / np.maximum(obs, eps))
    err_eng = np.log10(np.maximum(engine, eps) / np.maximum(obs, eps))
    err_v1 = np.log10(np.maximum(plm_v1, eps) / np.maximum(obs, eps))
    err_fda = np.log10(np.maximum(plm_fda, eps) / np.maximum(obs, eps))

    pairs = [
        ("Engine - ML", err_eng, err_ml),
        ("ML - PLM v1 (shared)", err_ml, err_v1),
        ("ML - PLM v3 (FDA-only)", err_ml, err_fda),
        ("Engine - PLM v3 (FDA-only)", err_eng, err_fda),
        ("PLM v1 - PLM v3 (FDA)", err_v1, err_fda),
    ]
    log.info(f"\n  {'Pair':<35} {'r':>8}")
    log.info(f"  {'-'*35} {'-'*8}")
    for name, e1, e2 in pairs:
        r = np.corrcoef(e1, e2)[0, 1]
        log.info(f"  {name:<35} {r:>8.3f}")

    r_ml_fda = np.corrcoef(err_ml, err_fda)[0, 1]

    # ── Step 6: LOOCV 3-way blends ───────────────────────────────────────
    log.info(f"\n[Step 6] LOOCV 3-Way Blends")

    # 2-way baseline
    n2 = len(obs)
    pred_2way = np.zeros(n2)
    for i in range(n2):
        t = [j for j in range(n2) if j != i]
        best_w, best_a = None, float("inf")
        for w_e in np.arange(0, 0.55, 0.05):
            w_m = round(1.0 - w_e, 2)
            a = aafe(w_e * engine[t] + w_m * ml[t], obs[t])
            if a < best_a:
                best_a = a
                best_w = (w_e, w_m)
        pred_2way[i] = best_w[0] * engine[i] + best_w[1] * ml[i]
    aafe_2way = aafe(pred_2way, obs)
    pct2_2way = pct_2fold(pred_2way, obs)

    # 3-way with PLM v1
    a_v1, p_v1, _, wc_v1 = loocv_3way(engine, ml, plm_v1, obs)
    # 3-way with PLM FDA-only
    a_fda, p_fda, _, wc_fda = loocv_3way(engine, ml, plm_fda, obs)

    log.info(f"\n  {'Config':<35} {'AAFE':>8} {'%2-fold':>8} {'Δ':>8}")
    log.info(f"  {'-'*35} {'-'*8} {'-'*8} {'-'*8}")
    log.info(f"  {'2-way Meta (baseline)':<35} {aafe_2way:>8.3f} {pct2_2way:>7.1f}% {'---':>8}")
    log.info(f"  {'3-way + PLM v1 (shared)':<35} {a_v1:>8.3f} {p_v1:>7.1f}% {a_v1-aafe_2way:>+7.3f}")
    log.info(f"  {'3-way + PLM v3 (FDA-only)':<35} {a_fda:>8.3f} {p_fda:>7.1f}% {a_fda-aafe_2way:>+7.3f}")

    # Weight stability
    log.info(f"\n  FDA-only weight distribution (LOOCV):")
    for w, c in sorted(wc_fda.items(), key=lambda x: -x[1]):
        log.info(f"    {w}: {c}/{n} ({100*c/n:.1f}%)")

    plm_nonzero = sum(c for w, c in wc_fda.items() if not w.endswith(",0.00") and not w.endswith(",-0.00"))
    log.info(f"  Folds with w_plm > 0: {plm_nonzero}/{n} ({100*plm_nonzero/n:.1f}%)")

    # ── Step 7: Oracle ────────────────────────────────────────────────────
    log.info(f"\n[Step 7] Oracle (per-drug best)")
    wins = {"engine": 0, "ml": 0, "plm_fda": 0}
    plm_win_drugs = []
    for i in range(n):
        fes = {
            "engine": abs(np.log10(max(engine[i], eps) / max(obs[i], eps))),
            "ml": abs(np.log10(max(ml[i], eps) / max(obs[i], eps))),
            "plm_fda": abs(np.log10(max(plm_fda[i], eps) / max(obs[i], eps))),
        }
        best = min(fes, key=fes.get)
        wins[best] += 1
        if best == "plm_fda":
            plm_win_drugs.append(merged[i]["name"])

    oracle = np.array([
        min(engine[i], ml[i], plm_fda[i],
            key=lambda x: abs(np.log10(max(x, eps) / max(obs[i], eps))))
        for i in range(n)
    ])
    log.info(f"  Oracle AAFE: {aafe(oracle, obs):.3f}, %2-fold: {pct_2fold(oracle, obs):.1f}%")
    log.info(f"  Wins: engine={wins['engine']}, ml={wins['ml']}, plm_fda={wins['plm_fda']}")
    if plm_win_drugs:
        log.info(f"  PLM FDA best ({len(plm_win_drugs)}): {', '.join(plm_win_drugs[:20])}")

    # ── Verdict ───────────────────────────────────────────────────────────
    log.info(f"\n{'=' * 70}")
    delta_fda = a_fda - aafe_2way
    if a_fda < aafe_2way:
        log.info(f"VERDICT: PLM FDA-only IMPROVES blend! Δ={delta_fda:+.3f}")
    else:
        log.info(f"VERDICT: PLM FDA-only does NOT improve blend. Δ={delta_fda:+.3f}")
    log.info(f"  r(ML, PLM_FDA) = {r_ml_fda:.3f} (vs shared: {np.corrcoef(err_ml, err_v1)[0,1]:.3f})")
    log.info("=" * 70)

    # ── Save ──────────────────────────────────────────────────────────────
    results = {
        "experiment": "PLM 3rd Track v3 (FDA-only)",
        "date": "2026-04-03",
        "fda_train_entries": len(train_entries),
        "fda_train_drugs": len(set(d["ik"] for d in train_entries)),
        "n_overlap": n,
        "standalone_aafe": {
            "engine": round(aafe(engine, obs), 3),
            "ml": round(aafe(ml, obs), 3),
            "plm_v1_shared": round(aafe(plm_v1, obs), 3),
            "plm_v3_fda_only": round(aafe(plm_fda, obs), 3),
        },
        "error_correlations": {
            "ml_plm_v1": round(float(np.corrcoef(err_ml, err_v1)[0, 1]), 3),
            "ml_plm_fda": round(r_ml_fda, 3),
        },
        "loocv": {
            "2way_aafe": round(aafe_2way, 3),
            "3way_plm_v1_aafe": round(a_v1, 3),
            "3way_plm_fda_aafe": round(a_fda, 3),
            "delta_fda": round(delta_fda, 3),
        },
        "folds_plm_nonzero": plm_nonzero,
        "oracle_aafe": round(aafe(oracle, obs), 3),
        "oracle_wins": wins,
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"\nSaved to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
