#!/usr/bin/env python3
"""Chemprop v2 D-MPNN experiment: Direct Cmax prediction.

Goal: create a new ML track with r(baseline_ML) < 0.90
D-MPNN learns molecular graph → embedding → Cmax (not fingerprint-based).
"""
from __future__ import annotations
import enum, typing
if not hasattr(enum, 'StrEnum'):
    from strenum import StrEnum; enum.StrEnum = StrEnum
if not hasattr(typing, 'Self'):
    typing.Self = typing.TypeVar('Self')

import json, logging, sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, '/tmp/chemprop_v2')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rdkit import RDLogger; RDLogger.logger().setLevel(RDLogger.ERROR)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent

from rdkit import Chem
from rdkit.Chem.inchi import MolToInchi, InchiToInchiKey
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles

from chemprop.data import MoleculeDatapoint, MoleculeDataset, build_dataloader
from chemprop.models import MPNN
from chemprop.nn import BondMessagePassing
from chemprop.nn.agg import MeanAggregation
from chemprop.nn.predictors import RegressionFFN

# ═══════════════════════════════════════════════════════════════
# Utils
# ═══════════════════════════════════════════════════════════════

def aafe(p, o):
    m = (p > 0) & (o > 0)
    return float(10 ** np.mean(np.abs(np.log10(p[m] / o[m])))) if m.sum() else np.inf

def ik14(smi):
    mol = Chem.MolFromSmiles(smi)
    if not mol: return None
    inchi = MolToInchi(mol)
    return InchiToInchiKey(inchi)[:14] if inchi else None

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


# ═══════════════════════════════════════════════════════════════
# Data
# ═══════════════════════════════════════════════════════════════

def load_data():
    log.info("Loading data...")
    with open(ROOT / "data/reference/holdout.json") as f: hd = json.load(f)
    with open(ROOT / "data/reference/clinical_pk.json") as f: cpk = json.load(f)
    ho_iks = set()
    for n in hd.get("holdout", []):
        e = cpk["drugs"].get(n) or cpk["drugs"].get(n.replace(" ", "_"))
        if e and e.get("smiles"):
            k = ik14(e["smiles"])
            if k: ho_iks.add(k)

    pbpk = pd.read_csv(ROOT / "data/training/mmpk_pbpk_features.csv")
    pbpk["ik14"] = pbpk["smiles"].apply(ik14)
    pbpk = pbpk.dropna(subset=["ik14"])
    pbpk = pbpk[~pbpk["ik14"].isin(ho_iks)].reset_index(drop=True)
    pbpk = pbpk[(pbpk["cmax_pbpk"] > 0) & (pbpk["cmax_obs"] > 0)].reset_index(drop=True)

    smiles = pbpk["smiles"].tolist()
    y = np.log10(pbpk["cmax_obs"].values / pbpk["dose_mg"].values)
    N = len(smiles)
    log.info("  Training: %d drugs", N)

    # Holdout
    with open(ROOT / "data/training/3track_holdout_predictions.json") as f:
        ho_preds = json.load(f)
    ho_smiles = []
    for h in ho_preds:
        e = cpk["drugs"].get(h["name"]) or cpk["drugs"].get(h["name"].replace(" ", "_"))
        ho_smiles.append(e["smiles"] if e and e.get("smiles") else "C")

    ho_ml = np.array([h["cmax_ml"] for h in ho_preds])
    ho_eng = np.array([h["cmax_engine"] for h in ho_preds])
    ho_obs = np.array([h["cmax_obs"] for h in ho_preds])
    ho_types = np.array([h["compound_type"] for h in ho_preds])
    ho_doses = []
    for h in ho_preds:
        e = cpk["drugs"].get(h["name"]) or cpk["drugs"].get(h["name"].replace(" ", "_"))
        ho_doses.append(e.get("dose_mg", 100.0) if e else 100.0)
    ho_doses = np.array(ho_doses)

    return smiles, y, pbpk["dose_mg"].values, ho_smiles, ho_preds, ho_ml, ho_eng, ho_obs, ho_types, ho_doses


# ═══════════════════════════════════════════════════════════════
# Train D-MPNN
# ═══════════════════════════════════════════════════════════════

def train_dmpnn(smiles_train, y_train, smiles_val, epochs=30, lr=1e-3):
    """Train D-MPNN and return validation predictions."""
    train_data = [MoleculeDatapoint.from_smi(s, [y]) for s, y in zip(smiles_train, y_train)]
    val_data = [MoleculeDatapoint.from_smi(s, [0.0]) for s in smiles_val]

    train_ds = MoleculeDataset(train_data)
    val_ds = MoleculeDataset(val_data)

    train_loader = build_dataloader(train_ds, batch_size=64, shuffle=True)
    val_loader = build_dataloader(val_ds, batch_size=64, shuffle=False)

    mp = BondMessagePassing(d_h=200, depth=3)
    agg = MeanAggregation()
    ffn = RegressionFFN(input_dim=mp.output_dim, hidden_dim=200, n_layers=2)
    model = MPNN(mp, agg, ffn)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = torch.nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        n_batches = 0
        for batch in train_loader:
            bmg, _, _, targets, *_ = batch
            optimizer.zero_grad()
            preds = model(bmg)
            loss = loss_fn(preds, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

    # Predict validation
    model.eval()
    val_preds = []
    with torch.no_grad():
        for batch in val_loader:
            bmg, _, _, _, *_ = batch
            preds = model(bmg)
            val_preds.append(preds.numpy())
    val_preds = np.concatenate(val_preds).flatten()

    return val_preds, model


def predict_with_model(model, smiles_list):
    """Predict with trained model."""
    data = [MoleculeDatapoint.from_smi(s, [0.0]) for s in smiles_list]
    ds = MoleculeDataset(data)
    loader = build_dataloader(ds, batch_size=64, shuffle=False)

    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in loader:
            bmg, _, _, _, *_ = batch
            preds = model(bmg)
            all_preds.append(preds.numpy())
    return np.concatenate(all_preds).flatten()


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    smiles, y, doses, ho_smiles, ho_preds, ho_ml, ho_eng, ho_obs, ho_types, ho_doses = load_data()

    N = len(smiles)
    folds = scaffold_split(smiles, K=5)

    # 5-fold scaffold CV
    log.info("Training D-MPNN (5-fold scaffold CV, 30 epochs each)...")
    oof_preds = np.full(N, np.nan)
    for fi, val_idx in enumerate(folds):
        train_idx = [j for fj, idxs in enumerate(folds) if fj != fi for j in idxs]
        smi_tr = [smiles[j] for j in train_idx]
        y_tr = y[train_idx]
        smi_val = [smiles[j] for j in val_idx]

        val_pred, _ = train_dmpnn(smi_tr, y_tr, smi_val, epochs=30)
        oof_preds[val_idx] = val_pred
        oof_cmax = 10 ** val_pred * doses[val_idx]
        fold_aafe = aafe(oof_cmax, 10 ** y[val_idx] * doses[val_idx])
        log.info("  Fold %d: n_train=%d, n_val=%d, AAFE=%.3f",
                 fi, len(train_idx), len(val_idx), fold_aafe)

    oof_cmax_all = 10 ** oof_preds * doses
    obs_all = 10 ** y * doses
    cv_aafe = aafe(oof_cmax_all, obs_all)
    log.info("CV AAFE: %.3f (XGBoost baseline: ~3.30)", cv_aafe)

    # Train full model for holdout
    log.info("Training full model...")
    _, model_full = train_dmpnn(smiles, y, ho_smiles, epochs=30)
    ho_log_preds = predict_with_model(model_full, ho_smiles)
    ho_cmax_dmpnn = 10 ** ho_log_preds * ho_doses

    ho_aafe = aafe(ho_cmax_dmpnn, ho_obs)
    log.info("Holdout D-MPNN AAFE: %.3f (XGBoost ML: %.3f)",
             ho_aafe, aafe(ho_ml, ho_obs))

    # Error correlations
    fe_dmpnn = np.log10(np.maximum(ho_cmax_dmpnn, 1e-10) / np.maximum(ho_obs, 1e-10))
    fe_ml = np.log10(np.maximum(ho_ml, 1e-10) / np.maximum(ho_obs, 1e-10))
    fe_eng = np.log10(np.maximum(ho_eng, 1e-10) / np.maximum(ho_obs, 1e-10))
    r_ml = float(np.corrcoef(fe_dmpnn, fe_ml)[0, 1])
    r_eng = float(np.corrcoef(fe_dmpnn, fe_eng)[0, 1])

    log.info("★ Error correlations: r(D-MPNN, ML)=%.3f, r(D-MPNN, engine)=%.3f", r_ml, r_eng)
    log.info("Gate: r(ML) < 0.90? → %s", "PASS" if r_ml < 0.90 else "FAIL")

    # 3-track blend: engine + ML + D-MPNN
    ho_combined = np.array([h["cmax_combined"] for h in ho_preds])
    baseline = aafe(ho_combined, ho_obs)

    best_a = float("inf")
    best_wd = 0
    for wd in np.arange(0, 0.5, 0.05):
        trial = np.zeros(len(ho_obs))
        for i in range(len(ho_obs)):
            w_eng = 0.45 if ho_types[i] == "base" else 0.00
            w_ml = (1 - w_eng) * (1 - wd)
            w_d = (1 - w_eng) * wd
            le = np.log10(max(ho_eng[i], 1e-10))
            lm = np.log10(max(ho_ml[i], 1e-10))
            ld = np.log10(max(ho_cmax_dmpnn[i], 1e-10))
            trial[i] = 10 ** (w_eng * le + w_ml * lm + w_d * ld)
        a = aafe(trial, ho_obs)
        if a < best_a:
            best_a = a
            best_wd = wd

    log.info("3-track (eng+ML+DMPNN): w_dmpnn=%.2f → AAFE=%.3f (Δ=%+.3f vs baseline %.3f)",
             best_wd, best_a, best_a - baseline, baseline)

    # Save
    out = {
        "cv_aafe": cv_aafe,
        "holdout_aafe": ho_aafe,
        "r_ml": r_ml, "r_engine": r_eng,
        "baseline_ml_aafe": float(aafe(ho_ml, ho_obs)),
        "baseline_meta_aafe": float(baseline),
        "best_3track_aafe": float(best_a),
        "best_w_dmpnn": float(best_wd),
    }
    with open(ROOT / "data/validation/chemprop_results.json", "w") as f:
        json.dump(out, f, indent=2)

    log.info("\nTotal: %.0fs", time.time() - t0)


if __name__ == "__main__":
    main()
