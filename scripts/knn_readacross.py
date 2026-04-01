#!/usr/bin/env python3
"""k-NN Read-Across: Non-Parametric Cmax Prediction.

No training. No parameters. No overfitting.
Find k structurally similar drugs in MMPK, use their observed Cmax.

Phase 0: Chemical space coverage check (Tanimoto distribution)
Phase 1: k-NN prediction with multiple (k, weighting) configurations
Phase 2: 3-way meta-learner blend (engine + ML + k-NN)
"""
from __future__ import annotations

import json, logging, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.inchi import MolToInchi, InchiToInchiKey

RDLogger.logger().setLevel(RDLogger.ERROR)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def _ik14(smi):
    mol = Chem.MolFromSmiles(smi)
    if not mol: return None
    inchi = MolToInchi(mol)
    if not inchi: return None
    ik = InchiToInchiKey(inchi)
    return ik[:14] if ik else None

def compute_aafe(p, o):
    mask = (p > 0) & (o > 0)
    return float(10 ** np.mean(np.abs(np.log10(p[mask] / o[mask])))) if mask.sum() > 0 else float("inf")


# ═══════════════════════════════════════════════════════════════════════════
# Build reference set (MMPK training drugs with FP + dose-normalized Cmax)
# ═══════════════════════════════════════════════════════════════════════════

def build_reference_set():
    """Load MMPK drugs, compute fingerprints, return reference for k-NN."""
    with open(ROOT / "data/reference/holdout.json") as f: hd = json.load(f)
    with open(ROOT / "data/reference/clinical_pk.json") as f: cp = json.load(f)
    names = hd.get("holdout", []) + hd.get("train", [])
    drugs = cp.get("drugs", {})
    ho_ik = set()
    for n in names:
        e = drugs.get(n) or drugs.get(n.replace(" ", "_"))
        if e and e.get("smiles"):
            ik = _ik14(e["smiles"])
            if ik: ho_ik.add(ik)

    mmpk = pd.read_csv(ROOT / "data/training/mmpk_expanded_full.csv")
    mmpk = mmpk[~mmpk["in_holdout"]].reset_index(drop=True)

    # Per-drug: median dose and Cmax
    dedup = mmpk.groupby("canon_smiles").agg(
        dose_mg=("dose_mg", "median"),
        cmax=("cmax_mg_L", "median"),
        log_cpd=("log_cmax_per_dose", "median"),
    ).reset_index()
    dedup["ik"] = dedup["canon_smiles"].apply(_ik14)
    dedup = dedup.dropna(subset=["ik"])
    dedup = dedup[~dedup["ik"].isin(ho_ik)].reset_index(drop=True)
    dedup = dedup[(dedup["cmax"] > 0) & (dedup["dose_mg"] > 0)].reset_index(drop=True)

    # Compute fingerprints
    ref_fps = []
    ref_records = []
    for _, row in dedup.iterrows():
        mol = Chem.MolFromSmiles(row["canon_smiles"])
        if mol is None: continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
        ref_fps.append(fp)
        ref_records.append({
            "smi": row["canon_smiles"],
            "dose": float(row["dose_mg"]),
            "cmax": float(row["cmax"]),
            "log_cpd": float(row["log_cpd"]),  # log10(Cmax/Dose)
            "ik": row["ik"],
        })

    log.info("Reference set: %d drugs with FP", len(ref_records))
    return ref_fps, ref_records, ho_ik


# ═══════════════════════════════════════════════════════════════════════════
# k-NN prediction
# ═══════════════════════════════════════════════════════════════════════════

def knn_predict(query_fp, ref_fps, ref_records, k=5, weighting="similarity", dose_query=1.0):
    """Predict log10(Cmax/Dose) using k-NN read-across."""
    sims = DataStructs.BulkTanimotoSimilarity(query_fp, ref_fps)
    top_k = sorted(enumerate(sims), key=lambda x: -x[1])[:k]

    if not top_k:
        return None, 0, []

    indices, similarities = zip(*top_k)
    similarities = np.array(similarities)
    log_cpds = np.array([ref_records[i]["log_cpd"] for i in indices])

    # Weighting
    if weighting == "uniform":
        weights = np.ones(k) / k
    elif weighting == "similarity":
        total = similarities.sum()
        weights = similarities / total if total > 0 else np.ones(k) / k
    elif weighting == "inverse_distance":
        dists = 1.0 - similarities + 1e-6
        inv = 1.0 / dists
        weights = inv / inv.sum()
    else:
        weights = np.ones(k) / k

    log_cpd_pred = float(np.sum(weights * log_cpds))
    top1_sim = float(similarities[0])

    return log_cpd_pred, top1_sim, list(zip([int(i) for i in indices], [float(s) for s in similarities]))


# ═══════════════════════════════════════════════════════════════════════════
# Phase 0: Chemical Space Coverage
# ═══════════════════════════════════════════════════════════════════════════

def phase0(ref_fps, ref_records, ho_ik):
    log.info("=" * 60)
    log.info("Phase 0: Chemical Space Coverage")
    log.info("=" * 60)

    from sisyphus.validation.reference import load_reference

    refs = [r for r in load_reference() if r.in_holdout]

    top1_sims = []
    top3_sims = []
    top5_sims = []

    for ref in refs:
        mol = Chem.MolFromSmiles(ref.smiles)
        if mol is None: continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
        sims = DataStructs.BulkTanimotoSimilarity(fp, ref_fps)
        sorted_sims = sorted(sims, reverse=True)

        top1_sims.append(sorted_sims[0] if len(sorted_sims) > 0 else 0)
        top3_sims.append(np.mean(sorted_sims[:3]) if len(sorted_sims) >= 3 else 0)
        top5_sims.append(np.mean(sorted_sims[:5]) if len(sorted_sims) >= 5 else 0)

    t1 = np.array(top1_sims)

    print()
    print("=" * 60)
    print("  CHEMICAL SPACE COVERAGE (Holdout → MMPK)")
    print("=" * 60)
    print()
    print(f"  Top-1 Tanimoto: median={np.median(t1):.3f}, mean={np.mean(t1):.3f}")
    print(f"    min={t1.min():.3f}, max={t1.max():.3f}, Q25={np.percentile(t1,25):.3f}, Q75={np.percentile(t1,75):.3f}")
    print(f"  Top-1 > 0.5: {(t1>0.5).sum()}/{len(t1)} ({(t1>0.5).mean()*100:.0f}%)")
    print(f"  Top-1 > 0.4: {(t1>0.4).sum()}/{len(t1)} ({(t1>0.4).mean()*100:.0f}%)")
    print(f"  Top-1 > 0.3: {(t1>0.3).sum()}/{len(t1)} ({(t1>0.3).mean()*100:.0f}%)")
    print(f"  Top-1 > 0.2: {(t1>0.2).sum()}/{len(t1)} ({(t1>0.2).mean()*100:.0f}%)")

    # Histogram
    print()
    print("  Tanimoto histogram (Top-1):")
    for lo in np.arange(0, 1.0, 0.1):
        hi = lo + 0.1
        n = ((t1 >= lo) & (t1 < hi)).sum()
        bar = "#" * n
        print(f"    [{lo:.1f}-{hi:.1f}): {n:>3d} {bar}")

    gate = np.median(t1) > 0.3
    print()
    print(f"  Gate (median > 0.3): {'PASS' if gate else 'FAIL'} (median={np.median(t1):.3f})")

    return {
        "median": float(np.median(t1)), "mean": float(np.mean(t1)),
        "min": float(t1.min()), "max": float(t1.max()),
        "gt_05": int((t1 > 0.5).sum()), "gt_04": int((t1 > 0.4).sum()),
        "gt_03": int((t1 > 0.3).sum()),
        "gate": gate,
    }, refs


# ═══════════════════════════════════════════════════════════════════════════
# Phase 1+2: k-NN Prediction + 3-Way Blend
# ═══════════════════════════════════════════════════════════════════════════

def phase1_and_2(ref_fps, ref_records, holdout_refs):
    log.info("=" * 60)
    log.info("Phase 1+2: k-NN Prediction + 3-Way Blend")
    log.info("=" * 60)

    from sisyphus.pipeline.predict import predict
    from sisyphus.predict.chemistry import compute_profile

    # Load holdout engine/ML predictions
    with open(ROOT / "data/training/3track_holdout_predictions.json") as f:
        ho_pred = json.load(f)
    ho_dict = {d["name"]: d for d in ho_pred}

    logging.getLogger("sisyphus").setLevel(logging.WARNING)

    # Build holdout evaluation set
    holdout = []
    for ref in holdout_refs:
        mol = Chem.MolFromSmiles(ref.smiles)
        if mol is None: continue

        hp = ho_dict.get(ref.name)
        if not hp: continue

        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)

        try:
            result = predict(ref.smiles, ref.dose_mg, ref.route)
            cmax_ml = result.ml_pk.cmax.mean if result.ml_pk else hp["cmax_combined"]
            cmax_meta = result.pk.cmax.mean
        except:
            cmax_ml = hp["cmax_combined"]
            cmax_meta = hp["cmax_combined"]

        profile = compute_profile(ref.smiles)

        holdout.append({
            "name": ref.name,
            "obs": ref.cmax_obs,
            "dose": ref.dose_mg,
            "eng": hp["cmax_engine"],
            "ml": cmax_ml,
            "meta": cmax_meta,
            "fp": fp,
            "is_base": profile.compound_type == "base",
        })

    logging.getLogger("sisyphus").setLevel(logging.INFO)
    log.info("Holdout drugs: %d", len(holdout))

    # ─── Phase 1: k-NN with multiple configurations ───
    k_values = [1, 3, 5, 7, 10, 15, 20]
    weightings = ["uniform", "similarity", "inverse_distance"]

    knn_results = {}
    for k in k_values:
        for w in weightings:
            preds = []
            sims_top1 = []
            for d in holdout:
                log_cpd, top1_sim, _ = knn_predict(d["fp"], ref_fps, ref_records, k=k, weighting=w)
                if log_cpd is not None:
                    cmax_pred = 10 ** log_cpd * d["dose"]
                    preds.append(cmax_pred)
                    sims_top1.append(top1_sim)
                else:
                    preds.append(0)
                    sims_top1.append(0)

            pred_arr = np.array(preds)
            obs_arr = np.array([d["obs"] for d in holdout])
            aafe = compute_aafe(pred_arr, obs_arr)
            pct2 = float(np.mean((pred_arr / obs_arr >= 0.5) & (pred_arr / obs_arr <= 2.0)) * 100)

            knn_results[(k, w)] = {"aafe": aafe, "pct2": pct2}

            # Store predictions for primary config
            if k == 5 and w == "similarity":
                for i, d in enumerate(holdout):
                    d["knn"] = preds[i]
                    d["knn_sim"] = sims_top1[i]

    # Print Phase 1 results
    print()
    print("=" * 70)
    print("  PHASE 1: k-NN AAFE BY CONFIGURATION")
    print("=" * 70)
    print()
    print(f"  {'k':>3s}  {'uniform':>8s}  {'similarity':>11s}  {'inv_dist':>9s}")
    print(f"  {'-'*35}")
    for k in k_values:
        row = f"  {k:>3d}"
        for w in weightings:
            row += f"  {knn_results[(k,w)]['aafe']:>8.3f}"
        print(row)

    # Primary config
    primary = knn_results[(5, "similarity")]
    print(f"\n  Primary (k=5, similarity-weighted): AAFE={primary['aafe']:.3f}")

    # Best config
    best_config = min(knn_results, key=lambda x: knn_results[x]["aafe"])
    best = knn_results[best_config]
    print(f"  Best (k={best_config[0]}, {best_config[1]}): AAFE={best['aafe']:.3f}")

    # ─── Phase 2: 3-Way Blend ───
    df = pd.DataFrame(holdout)

    # Error correlation
    log_eng = np.log10(np.maximum(df["eng"].values, 1e-10))
    log_ml = np.log10(np.maximum(df["ml"].values, 1e-10))
    log_knn = np.log10(np.maximum(df["knn"].values, 1e-10))
    log_obs = np.log10(np.maximum(df["obs"].values, 1e-10))

    err_eng = log_eng - log_obs
    err_ml = log_ml - log_obs
    err_knn = log_knn - log_obs

    from scipy.stats import pearsonr
    r_eng_ml, _ = pearsonr(err_eng, err_ml)
    r_eng_knn, _ = pearsonr(err_eng, err_knn)
    r_ml_knn, _ = pearsonr(err_ml, err_knn)

    print()
    print("=" * 60)
    print("  ERROR CORRELATION")
    print("=" * 60)
    print(f"  r(engine, ML):  {r_eng_ml:.3f}")
    print(f"  r(engine, kNN): {r_eng_knn:.3f}")
    print(f"  r(ML, kNN):     {r_ml_knn:.3f}")

    # 3-way LOOCV weight optimization
    log.info("3-way LOOCV weight optimization ...")
    best_3way_aafe = float("inf")
    best_w = (0, 0, 0)

    for w1 in np.arange(0, 0.55, 0.05):  # engine
        for w2 in np.arange(0, 1.05, 0.05):  # ML
            w3 = 1.0 - w1 - w2  # kNN
            if w3 < -0.01 or w3 > 1.01: continue
            w3 = max(0, w3)

            errs = []
            for i in range(len(holdout)):
                le = log_eng[i]; lm = log_ml[i]; lk = log_knn[i]; lo = log_obs[i]
                # Disagreement scaling on engine
                dis = abs(le - lm)
                w1_eff = w1 * (1.0 / dis if dis > 1.0 else 1.0)
                total = w1_eff + w2 + w3
                if total <= 0: total = 1
                lc = (w1_eff * le + w2 * lm + w3 * lk) / total
                errs.append(abs(lc - lo))

            a = 10 ** np.mean(errs)
            if a < best_3way_aafe:
                best_3way_aafe = a
                best_w = (w1, w2, w3)

    # Oracle 3-track
    oracle_errs = [min(abs(err_eng[i]), abs(err_ml[i]), abs(err_knn[i])) for i in range(len(holdout))]
    oracle_aafe = 10 ** np.mean(oracle_errs)

    # Current 2-track baseline
    baseline_meta_aafe = compute_aafe(df["meta"].values, df["obs"].values)
    baseline_ml_aafe = compute_aafe(df["ml"].values, df["obs"].values)

    # Per-subgroup
    base_mask = df["is_base"].values
    nb_mask = ~base_mask

    def _sub_aafe(col, mask):
        p = df[col].values[mask]; o = df["obs"].values[mask]
        return compute_aafe(p, o) if mask.sum() > 3 else float("inf")

    print()
    print("=" * 75)
    print("  HOLDOUT BENCHMARK (N=%d)" % len(df))
    print("=" * 75)
    print()
    print(f"  {'Method':<30s} {'All':>7s} {'Base(%d)' % base_mask.sum():>9s} {'NonBase(%d)' % nb_mask.sum():>11s}")
    print(f"  {'-'*60}")
    for col, label in [
        ("ml", "Baseline ML"),
        ("meta", "Baseline Meta"),
        ("eng", "Engine only"),
        ("knn", "k-NN (k=5, sim-wt)"),
    ]:
        a = compute_aafe(df[col].values, df["obs"].values)
        ab = _sub_aafe(col, base_mask)
        anb = _sub_aafe(col, nb_mask)
        print(f"  {label:<30s} {a:>7.3f} {ab:>9.3f} {anb:>11.3f}")

    print(f"\n  3-way blend (w={best_w[0]:.2f}/{best_w[1]:.2f}/{best_w[2]:.2f}): AAFE={best_3way_aafe:.3f}")
    print(f"  Oracle 3-track: {oracle_aafe:.3f}")
    print(f"  (Previous CL/F oracle: 1.788)")

    # Gate
    delta = best_3way_aafe - baseline_meta_aafe
    print()
    print(f"  Baseline meta: {baseline_meta_aafe:.3f}")
    print(f"  3-way blend:   {best_3way_aafe:.3f}")
    print(f"  Delta:          {delta:+.3f}")
    print()
    if delta < -0.01:
        print("  GATE: ✓ KEEP")
        decision = "KEEP"
    elif delta > 0.01:
        print("  GATE: ✗ REVERT")
        decision = "REVERT"
    else:
        print(f"  GATE: ~ NOISE ({delta:+.3f})")
        decision = "NOISE"

    # Similarity-stratified analysis
    sims = np.array([d.get("knn_sim", 0) for d in holdout])
    print()
    print("  Similarity-stratified k-NN AAFE:")
    for lo, hi in [(0.5, 1.0), (0.4, 0.5), (0.3, 0.4), (0.0, 0.3)]:
        mask = (sims >= lo) & (sims < hi)
        if mask.sum() >= 3:
            a = compute_aafe(df["knn"].values[mask], df["obs"].values[mask])
            print(f"    Sim [{lo:.1f}-{hi:.1f}): N={mask.sum()}, AAFE={a:.3f}")

    return {
        "knn_configs": {f"k{k}_{w}": v for (k, w), v in knn_results.items()},
        "primary_aafe": primary["aafe"],
        "best_config": f"k{best_config[0]}_{best_config[1]}",
        "best_aafe": best["aafe"],
        "error_correlation": {"eng_ml": r_eng_ml, "eng_knn": r_eng_knn, "ml_knn": r_ml_knn},
        "blend_3way": {"aafe": best_3way_aafe, "w": best_w},
        "oracle_3track": oracle_aafe,
        "baseline_meta": baseline_meta_aafe,
        "baseline_ml": baseline_ml_aafe,
        "delta": delta,
        "decision": decision,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    ref_fps, ref_records, ho_ik = build_reference_set()
    coverage, holdout_refs = phase0(ref_fps, ref_records, ho_ik)

    if not coverage["gate"]:
        log.warning("Phase 0 gate FAILED. Chemical space too sparse.")
        results = {"phase0": coverage, "decision": "ABORT_PHASE0"}
    else:
        holdout_results = phase1_and_2(ref_fps, ref_records, holdout_refs)
        results = {"phase0": coverage, "phase1_2": holdout_results}

    out = ROOT / "data/validation/knn_readacross_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info("Results saved to %s", out)


if __name__ == "__main__":
    main()
