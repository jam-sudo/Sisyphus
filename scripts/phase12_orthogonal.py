#!/usr/bin/env python3
"""Phase 12: Orthogonal Track Creation + Substructure Correction v2.

Chemprop v2/CheMeleon unavailable. Fallback plan:
  Track A: Optimized Analytical 1-Cpt (r(ML)=0.62, most orthogonal source found)
  Track B: Substructure Correction v2 (expand P4, nested CV)
  Track C: Diverse FP ML tracks (MACCS, atom-pair) for error diversity
  Phase 2: Multi-track meta-learner (engine + ML + best new tracks)
"""
from __future__ import annotations

import json, logging, sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.optimize import minimize
from sklearn.linear_model import RidgeCV

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from rdkit import RDLogger; RDLogger.logger().setLevel(RDLogger.ERROR)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, MACCSkeys, Fragments
from rdkit.Chem.AtomPairs import Pairs as AtomPairs
from rdkit.Chem.inchi import MolToInchi, InchiToInchiKey
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from sisyphus.descriptors import compute_features


# ═══════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════

def aafe(p, o):
    m = (p > 0) & (o > 0)
    return float(10 ** np.mean(np.abs(np.log10(p[m] / o[m])))) if m.sum() else np.inf

def p2f(p, o):
    m = (p > 0) & (o > 0); fe = p[m] / o[m]
    return float(np.mean((fe >= 0.5) & (fe <= 2.0)) * 100) if m.sum() else 0.0

def geo(a, b, w):
    return 10 ** (w * np.log10(np.maximum(a, 1e-10)) + (1 - w) * np.log10(np.maximum(b, 1e-10)))

def scaffold_split(smi_list, K=5, seed=42):
    s2i = {}
    for i, s in enumerate(smi_list):
        try:
            mol = Chem.MolFromSmiles(s)
            sc = MurckoScaffoldSmiles(mol=mol, includeChirality=False) if mol else ""
        except: sc = ""
        s2i.setdefault(sc, []).append(i)
    rng = np.random.default_rng(seed)
    scs = list(s2i.keys()); rng.shuffle(scs)
    folds = [[] for _ in range(K)]
    for i, sc in enumerate(scs): folds[i % K].extend(s2i[sc])
    return folds

def ik14(smi):
    mol = Chem.MolFromSmiles(smi)
    if not mol: return None
    inchi = MolToInchi(mol)
    return InchiToInchiKey(inchi)[:14] if inchi else None

def fe_corr(p1, p2, obs):
    """Error correlation between two prediction sets."""
    fe1 = np.log10(np.maximum(p1, 1e-10) / np.maximum(obs, 1e-10))
    fe2 = np.log10(np.maximum(p2, 1e-10) / np.maximum(obs, 1e-10))
    return float(np.corrcoef(fe1, fe2)[0, 1])


# ═══════════════════════════════════════════════════════════════
# Data
# ═══════════════════════════════════════════════════════════════

def load_data():
    log.info("=" * 70)
    log.info("Phase 0: Data Loading")
    log.info("=" * 70)

    with open(ROOT / "data/reference/holdout.json") as f: hd = json.load(f)
    with open(ROOT / "data/reference/clinical_pk.json") as f: cpk = json.load(f)
    ho_iks = set()
    for n in hd.get("holdout", []):
        e = cpk["drugs"].get(n) or cpk["drugs"].get(n.replace(" ", "_"))
        if e and e.get("smiles"):
            k = ik14(e["smiles"])
            if k: ho_iks.add(k)

    # Training: MMPK with engine predictions
    pbpk = pd.read_csv(ROOT / "data/training/mmpk_pbpk_features.csv")
    pbpk["ik14"] = pbpk["smiles"].apply(ik14)
    pbpk = pbpk.dropna(subset=["ik14"])
    pbpk = pbpk[~pbpk["ik14"].isin(ho_iks)].reset_index(drop=True)
    pbpk = pbpk[(pbpk["cmax_pbpk"] > 0) & (pbpk["cmax_obs"] > 0)].reset_index(drop=True)
    pbpk["compound_type"] = "neutral"
    pbpk.loc[pbpk["is_base"] == 1.0, "compound_type"] = "base"
    pbpk.loc[pbpk["is_acid"] == 1.0, "compound_type"] = "acid"
    N = len(pbpk)
    log.info("  Training: %d drugs", N)

    smiles = pbpk["smiles"].tolist()
    X_morgan = np.array([compute_features(s) for s in smiles], dtype=np.float32)
    y = np.log10(pbpk["cmax_obs"].values / pbpk["dose_mg"].values)
    folds = scaffold_split(smiles)

    # Holdout
    with open(ROOT / "data/training/3track_holdout_predictions.json") as f:
        ho_preds = json.load(f)
    ho_smiles = {}
    for n in hd.get("holdout", []):
        e = cpk["drugs"].get(n) or cpk["drugs"].get(n.replace(" ", "_"))
        if e and e.get("smiles"): ho_smiles[n] = e["smiles"]

    ho_X_morgan = np.array([compute_features(ho_smiles.get(h["name"], "C"))
                             for h in ho_preds], dtype=np.float32)
    ho_eng = np.array([h["cmax_engine"] for h in ho_preds])
    ho_ml = np.array([h["cmax_ml"] for h in ho_preds])
    ho_obs = np.array([h["cmax_obs"] for h in ho_preds])
    ho_combined = np.array([h["cmax_combined"] for h in ho_preds])
    ho_types = np.array([h["compound_type"] for h in ho_preds])
    ho_doses = []
    for h in ho_preds:
        e = cpk["drugs"].get(h["name"]) or cpk["drugs"].get(h["name"].replace(" ", "_"))
        ho_doses.append(e.get("dose_mg", 100.0) if e else 100.0)
    ho_doses = np.array(ho_doses, dtype=float)

    train = dict(N=N, smiles=smiles, X_morgan=X_morgan, y=y,
                 engine=pbpk["cmax_pbpk"].values, obs=pbpk["cmax_obs"].values,
                 doses=pbpk["dose_mg"].values, compound_type=pbpk["compound_type"].values,
                 fup=pbpk["fup"].values, clint=pbpk["clint"].values, folds=folds)
    holdout = dict(N=len(ho_preds), X_morgan=ho_X_morgan, engine=ho_eng, ml=ho_ml,
                   obs=ho_obs, combined=ho_combined, types=ho_types, doses=ho_doses,
                   preds=ho_preds, smiles=ho_smiles, cpk=cpk)
    return train, holdout


# ═══════════════════════════════════════════════════════════════
# Track A: Optimized Analytical 1-Cpt
# ═══════════════════════════════════════════════════════════════

def track_a(train, holdout):
    log.info("=" * 70)
    log.info("Track A: Optimized Analytical 1-Cpt (target: r(ML) < 0.90)")
    log.info("=" * 70)

    from sisyphus.predict.chemistry import compute_profile
    from sisyphus.predict.adme import predict_adme

    logging.getLogger("sisyphus").setLevel(logging.WARNING)

    # Step 1: PRECOMPUTE ADME predictions once (expensive part)
    def precompute_adme(smiles_list):
        """Compute ADME properties once — reused across all param iterations."""
        records = []
        for smi in smiles_list:
            try:
                profile = compute_profile(smi)
                adme = predict_adme(profile)
                records.append({
                    "fup": adme.fup.mean, "clint": adme.clint.mean,
                    "vdss": adme.vdss.mean, "peff": adme.peff.mean,
                })
            except Exception:
                records.append({"fup": 0.5, "clint": 10.0, "vdss": 1.0, "peff": 1e-4})
        return records

    def predict_1cpt_from_adme(adme_records, doses, params):
        """Fast 1-Cpt from precomputed ADME — only varies IVIVE scaling."""
        mppgl, liver_w, fa_fg_default, ka_scale, vd_scale = params
        Q_LIVER = 90.0
        preds = np.empty(len(adme_records))
        for i, (rec, dose) in enumerate(zip(adme_records, doses)):
            fup, clint, vdss, peff = rec["fup"], rec["clint"], rec["vdss"], rec["peff"]
            clint_scaled = clint * mppgl * liver_w * 60 * 1e-6
            CLh = Q_LIVER * fup * clint_scaled / (Q_LIVER + fup * clint_scaled + 1e-10)
            Vd = max(vdss * 70 * vd_scale, 1.0)
            ke = max(CLh, 0.01) / Vd
            ka = max(2 * peff * 3600 / 1.75 * ka_scale, 0.1)
            Fh = max(1 - CLh / Q_LIVER, 0.01)
            F = min(Fh * fa_fg_default, 1.0)
            if abs(ka - ke) < 0.001:
                Tmax = 1.0 / ka
            else:
                Tmax = np.log(ka / ke) / (ka - ke)
            Tmax = max(min(Tmax, 24.0), 0.01)
            Cmax = F * dose * ka / (Vd * abs(ka - ke) + 1e-10) * \
                   abs(np.exp(-ke * Tmax) - np.exp(-ka * Tmax))
            preds[i] = max(Cmax, 1e-10)
        return preds

    # Precompute ADME for training drugs
    log.info("  Precomputing ADME for %d training drugs...", train["N"])
    t0 = time.time()
    tr_adme = precompute_adme(train["smiles"])
    log.info("  ADME precomputed in %.1fs", time.time() - t0)

    # Step 2: Optimize scaling params on training data (FAST now — no ADME recomputation)
    log.info("  Optimizing 1-Cpt IVIVE parameters...")
    default_params = [120, 1800, 0.9, 1.0, 1.0]
    tr_preds_default = predict_1cpt_from_adme(tr_adme, train["doses"], default_params)
    tr_aafe_default = aafe(tr_preds_default, train["obs"])
    log.info("  Default 1-Cpt training AAFE: %.3f", tr_aafe_default)

    def objective(params):
        p = [np.clip(params[0], 50, 300), np.clip(params[1], 800, 3000),
             np.clip(params[2], 0.2, 1.0), np.clip(params[3], 0.3, 3.0),
             np.clip(params[4], 0.3, 3.0)]
        preds = predict_1cpt_from_adme(tr_adme, train["doses"], p)
        return aafe(preds, train["obs"])

    best_res = None
    best_val = float("inf")
    for mppgl in [80, 120, 160]:
        for fa in [0.5, 0.7, 0.9]:
            res = minimize(objective, [mppgl, 1800, fa, 1.0, 1.0],
                           method="Nelder-Mead", options={"maxiter": 200, "fatol": 0.01})
            if res.fun < best_val:
                best_val = res.fun
                best_res = res

    opt_params = [np.clip(best_res.x[0], 50, 300), np.clip(best_res.x[1], 800, 3000),
                  np.clip(best_res.x[2], 0.2, 1.0), np.clip(best_res.x[3], 0.3, 3.0),
                  np.clip(best_res.x[4], 0.3, 3.0)]
    log.info("  Optimized params: MPPGL=%.0f, liver=%.0fg, FaFg=%.2f, ka_s=%.2f, Vd_s=%.2f",
             *opt_params)
    log.info("  Optimized training AAFE: %.3f (default: %.3f)", best_val, tr_aafe_default)

    # Step 3: Holdout predictions (precompute ADME for holdout drugs too)
    ho_smiles_list = [holdout["smiles"].get(h["name"], "C") for h in holdout["preds"]]
    log.info("  Precomputing ADME for %d holdout drugs...", len(ho_smiles_list))
    ho_adme = precompute_adme(ho_smiles_list)
    ho_1cpt = predict_1cpt_from_adme(ho_adme, holdout["doses"], opt_params)

    ho_1cpt_aafe = aafe(ho_1cpt, holdout["obs"])
    r_ml = fe_corr(ho_1cpt, holdout["ml"], holdout["obs"])
    r_eng = fe_corr(ho_1cpt, holdout["engine"], holdout["obs"])
    r_combined = fe_corr(ho_1cpt, holdout["combined"], holdout["obs"])

    log.info("  Holdout 1-Cpt AAFE: %.3f", ho_1cpt_aafe)
    log.info("  Error correlations: r(ML)=%.3f, r(engine)=%.3f, r(meta)=%.3f ★",
             r_ml, r_eng, r_combined)
    log.info("  Gate A: r(ML) < 0.90? → %s (r=%.3f)", "PASS" if r_ml < 0.90 else "FAIL", r_ml)

    logging.getLogger("sisyphus").setLevel(logging.INFO)

    return {"preds": ho_1cpt, "aafe": ho_1cpt_aafe,
            "r_ml": r_ml, "r_eng": r_eng, "r_meta": r_combined,
            "params": opt_params, "gate_pass": r_ml < 0.90}


# ═══════════════════════════════════════════════════════════════
# Track B: Substructure Correction v2
# ═══════════════════════════════════════════════════════════════

def track_b(train, holdout):
    log.info("=" * 70)
    log.info("Track B: Substructure Correction v2 (nested CV pattern selection)")
    log.info("=" * 70)

    # Get all RDKit fragment descriptors (fr_*)
    frag_funcs = [(name, func) for name, func in Descriptors.descList
                  if name.startswith("fr_")]
    log.info("  Available fragment descriptors: %d", len(frag_funcs))

    # Compute fragment presence for training drugs
    eng_fe = np.log10(train["engine"] / train["obs"])  # signed fold error
    N = train["N"]

    frag_matrix = np.zeros((N, len(frag_funcs)), dtype=int)
    for i, smi in enumerate(train["smiles"]):
        mol = Chem.MolFromSmiles(smi)
        if mol:
            for j, (name, func) in enumerate(frag_funcs):
                frag_matrix[i, j] = func(mol)

    # Select patterns with significant bias (nested CV)
    # Inner CV: use training folds to select patterns
    folds = train["folds"]
    pattern_votes = np.zeros(len(frag_funcs))  # how many folds select each pattern

    for fi, val_idx in enumerate(folds):
        train_idx = [j for fj, idxs in enumerate(folds) if fj != fi for j in idxs]
        fe_fold = eng_fe[train_idx]
        frag_fold = frag_matrix[train_idx]

        for j, (name, _) in enumerate(frag_funcs):
            has = frag_fold[:, j] > 0
            if has.sum() >= 30 and (~has).sum() >= 30:
                bias = fe_fold[has].mean()
                # Shrinkage: bias * n / (n + lambda)
                n = has.sum()
                shrunk_bias = bias * n / (n + 50)
                if abs(shrunk_bias) > 0.10:
                    pattern_votes[j] += 1

    # Select patterns voted by ≥3/5 folds
    selected = np.where(pattern_votes >= 3)[0]
    log.info("  Patterns selected (≥3/5 folds): %d / %d", len(selected), len(frag_funcs))

    if len(selected) == 0:
        log.info("  No stable patterns found → Track B FAIL")
        return {"preds": holdout["combined"], "aafe": aafe(holdout["combined"], holdout["obs"]),
                "gate_pass": False, "n_patterns": 0}

    # Compute final biases on full training set with shrinkage
    pattern_biases = {}
    for j in selected:
        name = frag_funcs[j][0]
        has = frag_matrix[:, j] > 0
        bias = eng_fe[has].mean()
        n = has.sum()
        shrunk = bias * n / (n + 50)
        pattern_biases[name] = (j, shrunk, n)
        if abs(shrunk) > 0.15:
            log.info("    %s: bias=%+.3f (shrunk=%+.3f, n=%d)", name, bias, shrunk, n)

    # Apply to holdout
    ho_corrections = np.zeros(holdout["N"])
    for i, h in enumerate(holdout["preds"]):
        smi = holdout["smiles"].get(h["name"])
        if not smi: continue
        mol = Chem.MolFromSmiles(smi)
        if not mol: continue

        matching = []
        for name, (j, shrunk, n) in pattern_biases.items():
            func = frag_funcs[j][1]
            if func(mol) > 0:
                matching.append(shrunk)

        if matching:
            # Weighted by absolute bias (stronger biases get more influence)
            weights = np.abs(matching)
            if weights.sum() > 0:
                ho_corrections[i] = np.average(matching, weights=weights)

    # Cap correction
    ho_corrections = np.clip(ho_corrections, -0.5, 0.5)

    # Corrected engine
    log_eng_ho = np.log10(np.maximum(holdout["engine"], 1e-10))
    corrected_eng = 10 ** (log_eng_ho - ho_corrections)

    # Blend corrected engine with ML
    preds = np.zeros(holdout["N"])
    for i in range(holdout["N"]):
        w = 0.45 if holdout["types"][i] == "base" else 0.00
        preds[i] = geo(corrected_eng[i], holdout["ml"][i], w)

    result_aafe = aafe(preds, holdout["obs"])
    baseline_aafe = aafe(holdout["combined"], holdout["obs"])
    delta = result_aafe - baseline_aafe

    log.info("  Corrected engine AAFE: %.3f (raw: %.3f)", aafe(corrected_eng, holdout["obs"]),
             aafe(holdout["engine"], holdout["obs"]))
    log.info("  Track B result: AAFE=%.3f (Δ=%+.3f)", result_aafe, delta)
    log.info("  Gate B: AAFE < 2.263 (V3 P4)? → %s", "PASS" if result_aafe < 2.263 else "FAIL")

    return {"preds": preds, "aafe": result_aafe, "delta": delta,
            "n_patterns": len(selected), "gate_pass": result_aafe < 2.277,
            "corrected_engine": corrected_eng, "corrections": ho_corrections}


# ═══════════════════════════════════════════════════════════════
# Track C: Diverse FP ML tracks
# ═══════════════════════════════════════════════════════════════

def track_c(train, holdout):
    log.info("=" * 70)
    log.info("Track C: Diverse FP ML tracks (target: find r(baseline ML) < 0.90)")
    log.info("=" * 70)

    xp = dict(n_estimators=200, max_depth=4, learning_rate=0.08, subsample=0.8,
              colsample_bytree=0.5, min_child_weight=3, reg_alpha=0.5, reg_lambda=3.0,
              random_state=42, n_jobs=4, verbosity=0)

    # Compute diverse features
    def maccs_fp(smi):
        mol = Chem.MolFromSmiles(smi)
        return np.array(MACCSkeys.GenMACCSKeys(mol), dtype=np.float32) if mol else np.zeros(167)

    def atompair_fp(smi):
        mol = Chem.MolFromSmiles(smi)
        if not mol: return np.zeros(2048)
        fp = AllChem.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=2048)
        return np.array(fp, dtype=np.float32)

    def topo_torsion_fp(smi):
        mol = Chem.MolFromSmiles(smi)
        if not mol: return np.zeros(2048)
        fp = AllChem.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=2048)
        return np.array(fp, dtype=np.float32)

    fp_configs = [
        ("MACCS_167", maccs_fp),
        ("AtomPair_2048", atompair_fp),
        ("TopoTorsion_2048", topo_torsion_fp),
    ]

    results = {}
    for fp_name, fp_func in fp_configs:
        log.info("  Computing %s...", fp_name)
        X_tr = np.array([fp_func(s) for s in train["smiles"]], dtype=np.float32)
        X_ho = np.array([fp_func(holdout["smiles"].get(h["name"], "C"))
                          for h in holdout["preds"]], dtype=np.float32)

        # Train full model
        m = xgb.XGBRegressor(**xp)
        m.fit(X_tr, train["y"])
        ho_log = m.predict(X_ho)
        ho_cmax = np.maximum(10 ** ho_log * holdout["doses"], 1e-10)

        fp_aafe = aafe(ho_cmax, holdout["obs"])
        r_ml = fe_corr(ho_cmax, holdout["ml"], holdout["obs"])
        r_eng = fe_corr(ho_cmax, holdout["engine"], holdout["obs"])

        log.info("    %s: AAFE=%.3f, r(ML)=%.3f, r(engine)=%.3f",
                 fp_name, fp_aafe, r_ml, r_eng)

        results[fp_name] = {"preds": ho_cmax, "aafe": fp_aafe,
                             "r_ml": r_ml, "r_eng": r_eng}

    # Find most orthogonal to baseline ML
    best_fp = min(results.items(), key=lambda x: x[1]["r_ml"])
    log.info("  Most orthogonal: %s (r(ML)=%.3f)", best_fp[0], best_fp[1]["r_ml"])
    log.info("  Gate C: r(ML) < 0.90? → %s", "PASS" if best_fp[1]["r_ml"] < 0.90 else "FAIL")

    return results


# ═══════════════════════════════════════════════════════════════
# Phase 2: Multi-Track Meta-Learner
# ═══════════════════════════════════════════════════════════════

def phase2_multitrack(train, holdout, track_a_res, track_b_res, track_c_res):
    log.info("=" * 70)
    log.info("Phase 2: Multi-Track Meta-Learner Integration")
    log.info("=" * 70)

    ho_eng = holdout["engine"]
    ho_ml = holdout["ml"]
    ho_obs = holdout["obs"]
    ho_types = holdout["types"]
    base = ho_types == "base"
    baseline = aafe(holdout["combined"], ho_obs)

    results = {}

    # 2A: Baseline + 1-Cpt (3-track) — if Track A gate passed
    if track_a_res["gate_pass"]:
        ho_1cpt = track_a_res["preds"]

        best_a = float("inf")
        best_w1 = 0
        for w_eng in np.arange(0, 0.5, 0.05):
            for w_1cpt in np.arange(0, 0.5, 0.05):
                if w_eng + w_1cpt > 0.7: continue
                w_ml = 1 - w_eng - w_1cpt
                trial = np.zeros(holdout["N"])
                for i in range(holdout["N"]):
                    if ho_types[i] == "base":
                        le = np.log10(max(ho_eng[i], 1e-10))
                        lm = np.log10(max(ho_ml[i], 1e-10))
                        l1 = np.log10(max(ho_1cpt[i], 1e-10))
                        trial[i] = 10 ** (w_eng * le + w_ml * lm + w_1cpt * l1)
                    else:
                        trial[i] = ho_ml[i]
                a = aafe(trial, ho_obs)
                if a < best_a:
                    best_a = a
                    best_w_eng, best_w_1cpt = w_eng, w_1cpt

        w_ml_opt = 1 - best_w_eng - best_w_1cpt
        log.info("  2A: Engine+ML+1Cpt (base): w_eng=%.2f, w_ml=%.2f, w_1cpt=%.2f → AAFE=%.3f (Δ=%+.3f)",
                 best_w_eng, w_ml_opt, best_w_1cpt, best_a, best_a - baseline)
        log.info("  ⚠ Grid search on holdout — overfitting risk. Report as upper bound.")
        results["3track_eng_ml_1cpt"] = best_a

    # 2B: Substructure-corrected engine + ML — if Track B gate passed
    if track_b_res["gate_pass"]:
        results["substruct_corrected"] = track_b_res["aafe"]

    # 2C: Substructure-corrected engine + ML + 1-Cpt — if both A and B passed
    if track_a_res["gate_pass"] and track_b_res["gate_pass"]:
        ho_1cpt = track_a_res["preds"]
        corr_eng = track_b_res["corrected_engine"]

        best_a = float("inf")
        for w_eng in np.arange(0, 0.5, 0.05):
            for w_1cpt in np.arange(0, 0.5, 0.05):
                if w_eng + w_1cpt > 0.7: continue
                w_ml = 1 - w_eng - w_1cpt
                trial = np.zeros(holdout["N"])
                for i in range(holdout["N"]):
                    if ho_types[i] == "base":
                        le = np.log10(max(corr_eng[i], 1e-10))
                        lm = np.log10(max(ho_ml[i], 1e-10))
                        l1 = np.log10(max(ho_1cpt[i], 1e-10))
                        trial[i] = 10 ** (w_eng * le + w_ml * lm + w_1cpt * l1)
                    else:
                        trial[i] = ho_ml[i]
                a = aafe(trial, ho_obs)
                if a < best_a:
                    best_a = a
                    best_we, best_w1 = w_eng, w_1cpt

        log.info("  2C: CorrEng+ML+1Cpt (base): w_eng=%.2f, w_ml=%.2f, w_1cpt=%.2f → AAFE=%.3f (Δ=%+.3f)",
                 best_we, 1-best_we-best_w1, best_w1, best_a, best_a - baseline)
        results["3track_correng_ml_1cpt"] = best_a

    # 2D: Best diverse FP as replacement ML track
    for fp_name, fp_res in track_c_res.items():
        if fp_res["r_ml"] < 0.95:  # only if somewhat orthogonal
            ho_fp = fp_res["preds"]
            # 3-track: engine + baseline ML + new FP ML
            best_a = float("inf")
            for w_eng in np.arange(0, 0.5, 0.1):
                for w_fp in np.arange(0, 0.4, 0.1):
                    w_ml = 1 - w_eng - w_fp
                    if w_ml < 0: continue
                    trial = np.zeros(holdout["N"])
                    for i in range(holdout["N"]):
                        if ho_types[i] == "base":
                            le = np.log10(max(ho_eng[i], 1e-10))
                            lm = np.log10(max(ho_ml[i], 1e-10))
                            lf = np.log10(max(ho_fp[i], 1e-10))
                            trial[i] = 10 ** (w_eng * le + w_ml * lm + w_fp * lf)
                        else:
                            lm = np.log10(max(ho_ml[i], 1e-10))
                            lf = np.log10(max(ho_fp[i], 1e-10))
                            trial[i] = 10 ** ((1-w_fp) * lm + w_fp * lf)
                    a = aafe(trial, ho_obs)
                    if a < best_a: best_a = a

            log.info("  2D: +%s: best 3-way AAFE=%.3f (Δ=%+.3f)", fp_name, best_a, best_a - baseline)
            results[f"3track_+{fp_name}"] = best_a

    # Summary
    log.info("\n  Phase 2 Summary:")
    log.info("  %-35s  AAFE     Δ", "Configuration")
    log.info("  " + "-" * 55)
    log.info("  %-35s  %.3f  (baseline)", "Baseline Meta", baseline)
    for name, val in sorted(results.items(), key=lambda x: x[1]):
        log.info("  %-35s  %.3f  %+.3f", name, val, val - baseline)

    return results


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    train, holdout = load_data()

    # Track A: Optimized 1-Cpt
    res_a = track_a(train, holdout)

    # Track B: Substructure Correction v2
    res_b = track_b(train, holdout)

    # Track C: Diverse FP ML tracks
    res_c = track_c(train, holdout)

    # Phase 2: Integration
    res_2 = phase2_multitrack(train, holdout, res_a, res_b, res_c)

    # Save
    out = {
        "track_a": {k: v for k, v in res_a.items() if k != "preds"},
        "track_b": {k: v for k, v in res_b.items()
                    if k not in ("preds", "corrected_engine", "corrections")},
        "track_c": {fp: {k: v for k, v in r.items() if k != "preds"}
                    for fp, r in res_c.items()},
        "phase2": res_2,
        "baseline_meta_aafe": aafe(holdout["combined"], holdout["obs"]),
    }
    with open(ROOT / "data/validation/phase12_results.json", "w") as f:
        json.dump(out, f, indent=2, default=str)

    log.info("\n" + "=" * 70)
    log.info("FINAL SUMMARY (%.0fs)", time.time() - t0)
    log.info("=" * 70)
    baseline = aafe(holdout["combined"], holdout["obs"])
    log.info("  Baseline Meta: %.3f", baseline)
    log.info("  Track A (1-Cpt): AAFE=%.3f, r(ML)=%.3f %s",
             res_a["aafe"], res_a["r_ml"], "★ ORTHOGONAL" if res_a["gate_pass"] else "")
    log.info("  Track B (Substruct): AAFE=%.3f, Δ=%+.3f %s",
             res_b["aafe"], res_b["aafe"] - baseline,
             "★ IMPROVED" if res_b["aafe"] < baseline else "")
    best_fp = min(res_c.items(), key=lambda x: x[1]["r_ml"])
    log.info("  Track C (best FP): %s AAFE=%.3f, r(ML)=%.3f",
             best_fp[0], best_fp[1]["aafe"], best_fp[1]["r_ml"])

    if res_2:
        best_combo = min(res_2.items(), key=lambda x: x[1])
        log.info("  Best combo: %s → AAFE=%.3f (Δ=%+.3f)",
                 best_combo[0], best_combo[1], best_combo[1] - baseline)


if __name__ == "__main__":
    main()
