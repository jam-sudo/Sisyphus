#!/usr/bin/env python3
"""Neural Operator Gate 2: End-to-End Fine-tuning on Real MMPK Data.

Architecture:
  SMILES → [Morgan FP 2048 + 9 desc] → ADME Encoder MLP → 12D latent → Surrogate Operator → log10(Cmax)

Phase A: Pre-train operator on 20K synthetic (ADME → Cmax) pairs from Gate 1.
Phase B: Freeze operator, train ADME encoder on MMPK (2057D → 12D → Cmax).
Phase C: Unfreeze operator, fine-tune E2E on MMPK with regularization.
Phase D: Scaffold-stratified CV evaluation. Compare to XGBoost baseline (AAFE 2.34).

Gate criterion: CV AAFE < 2.34 (ML baseline).
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(level=logging.WARNING, format="%(message)s")
log = logging.getLogger("gate2")
log.setLevel(logging.INFO)

for name in ["sisyphus", "sisyphus.predict", "sisyphus.engine", "sisyphus.graph"]:
    logging.getLogger(name).setLevel(logging.WARNING)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_JSON = Path("data/validation/neural_operator_gate2_results.json")


# ── Model Architecture ─────────────────────────────────────────────────────


class SurrogateOperator(nn.Module):
    """Differentiable engine surrogate: 12D ADME params → log10(Cmax)."""

    def __init__(self, input_dim=12, hidden=(256, 256, 128)):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class ADMEEncoder(nn.Module):
    """Morgan FP + descriptors → 12D latent ADME vector.

    Output activations enforce physical constraints:
    - log10_CLint: unbounded (learned)
    - fup: sigmoid → (0, 1)
    - logP: unbounded
    - pKa: 7 + 5*tanh → (2, 12)
    - compound type: 3D softmax
    - MW/500: softplus (positive)
    - log10_dose: passed through from input (not predicted)
    - log10_Peff: unbounded
    - log10_sol: unbounded
    - log10_renal_cl: unbounded
    """

    def __init__(self, input_dim=2057, hidden=(512, 256)):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden:
            layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.1)])
            prev = h
        # Output 12 raw values
        layers.append(nn.Linear(prev, 12))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        raw = self.net(x)
        # Apply physical activations
        out = torch.zeros_like(raw)
        out[:, 0] = raw[:, 0]                         # log10_CLint (free)
        out[:, 1] = torch.sigmoid(raw[:, 1])           # fup ∈ (0,1)
        out[:, 2] = raw[:, 2]                          # logP (free)
        out[:, 3] = 7.0 + 5.0 * torch.tanh(raw[:, 3]) # pKa ∈ (2,12)
        out[:, 4:7] = torch.softmax(raw[:, 4:7], dim=-1)  # compound type
        out[:, 7] = torch.nn.functional.softplus(raw[:, 7])  # MW/500 > 0
        out[:, 8] = raw[:, 8]                          # log10_dose (will be overwritten)
        out[:, 9] = raw[:, 9]                          # log10_Peff (free)
        out[:, 10] = raw[:, 10]                        # log10_sol (free)
        out[:, 11] = raw[:, 11]                        # log10_renal_cl (free)
        return out


class E2EPipeline(nn.Module):
    """Full end-to-end: features → ADME Encoder → Operator → log10(Cmax)."""

    def __init__(self, encoder, operator):
        super().__init__()
        self.encoder = encoder
        self.operator = operator

    def forward(self, features, log10_dose):
        adme = self.encoder(features)
        # Inject actual dose (not predicted from SMILES)
        adme = adme.clone()
        adme[:, 8] = log10_dose
        return self.operator(adme)


# ── Data Loading ───────────────────────────────────────────────────────────


def load_synthetic_data():
    """Load synthetic data from Gate 1 by re-running the generation (cached via seed)."""
    from scripts.neural_operator_gate1 import (
        generate_synthetic_params,
        params_to_features,
    )

    # We need the actual param vectors and Cmax values. Read from Gate 1's run.
    # Since Gate 1 used seed=2026, we can regenerate params, but need Cmax from engine.
    # Instead, let's regenerate quickly from the same seed and re-run.
    # Actually, let's just regenerate the feature matrix and train the operator fresh in PyTorch.
    log.info("  Regenerating synthetic parameter features (seed=2026, reusing Gate 1 data)...")
    rng = np.random.default_rng(2026)
    params = generate_synthetic_params(20000, rng)
    X = params_to_features(params)
    return X, params


def load_mmpk_data():
    """Load MMPK training data with Morgan FP features."""
    from sisyphus.descriptors import compute_features

    df = pd.read_csv("data/training/mmpk_expanded_full.csv")
    train_df = df[~df["in_holdout"]].copy()

    # Aggregate to drug level (mean log_cmax_per_dose per SMILES)
    drug_level = (
        train_df.groupby("canon_smiles")
        .agg({
            "log_cmax_per_dose": "mean",
            "dose_mg": "median",
            "cmax_mg_L": "mean",
            "name": "first",
        })
        .reset_index()
    )

    log.info(f"  MMPK drugs: {len(drug_level)} (aggregated from {len(train_df)} rows)")

    # Compute features
    features_list = []
    valid_idx = []
    for i, row in drug_level.iterrows():
        try:
            feat = compute_features(row["canon_smiles"])
            features_list.append(feat)
            valid_idx.append(i)
        except Exception:
            pass

    X = np.array(features_list)
    drug_level = drug_level.loc[valid_idx].reset_index(drop=True)

    log.info(f"  Valid features: {X.shape[0]}/{len(drug_level)}")

    return X, drug_level


def load_holdout_data():
    """Load holdout drugs for evaluation."""
    from sisyphus.descriptors import compute_features

    with open("data/training/3track_holdout_predictions.json") as f:
        holdout = json.load(f)

    # We need SMILES for holdout drugs - get from MMPK
    df = pd.read_csv("data/training/mmpk_expanded_full.csv")
    holdout_df = df[df["in_holdout"]].copy()

    # Map drug names to SMILES
    name_to_smiles = dict(zip(holdout_df["name"].str.lower(), holdout_df["canon_smiles"]))

    features_list = []
    cmax_obs_list = []
    dose_list = []
    names = []
    for d in holdout:
        smiles = name_to_smiles.get(d["name"].lower())
        if smiles is None:
            continue
        try:
            feat = compute_features(smiles)
            features_list.append(feat)
            cmax_obs_list.append(d["cmax_obs"])
            # Get median dose for this drug
            drug_rows = holdout_df[holdout_df["name"].str.lower() == d["name"].lower()]
            dose = drug_rows["dose_mg"].median() if len(drug_rows) > 0 else 100.0
            dose_list.append(dose)
            names.append(d["name"])
        except Exception:
            pass

    X = np.array(features_list)
    return X, np.array(cmax_obs_list), np.array(dose_list), names


def scaffold_split(smiles_list, n_folds=5, seed=42):
    """Generate scaffold-stratified CV fold assignments."""
    try:
        from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
        from rdkit import Chem

        scaffolds = {}
        for i, smi in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                try:
                    scaf = MurckoScaffoldSmiles(mol=mol, includeChirality=False)
                except Exception:
                    scaf = f"fail_{i}"
            else:
                scaf = f"fail_{i}"
            scaffolds.setdefault(scaf, []).append(i)

        # Assign scaffolds to folds (largest first for balance)
        sorted_scaffolds = sorted(scaffolds.values(), key=len, reverse=True)
        fold_sizes = [0] * n_folds
        fold_assignments = np.zeros(len(smiles_list), dtype=int)

        rng = np.random.default_rng(seed)
        for indices in sorted_scaffolds:
            # Assign to smallest fold
            min_fold = int(np.argmin(fold_sizes))
            for idx in indices:
                fold_assignments[idx] = min_fold
            fold_sizes[min_fold] += len(indices)

        return fold_assignments
    except ImportError:
        # Fallback: random stratified
        rng = np.random.default_rng(seed)
        return rng.integers(0, n_folds, size=len(smiles_list))


# ── Training Functions ─────────────────────────────────────────────────────


def pretrain_operator_pytorch(X_synth, y_synth, epochs=200, lr=1e-3):
    """Pre-train surrogate operator on synthetic data in PyTorch."""
    n = len(X_synth)
    n_train = int(0.9 * n)
    perm = np.random.default_rng(42).permutation(n)

    X_tr = torch.tensor(X_synth[perm[:n_train]], dtype=torch.float32).to(DEVICE)
    y_tr = torch.tensor(y_synth[perm[:n_train]], dtype=torch.float32).to(DEVICE)
    X_va = torch.tensor(X_synth[perm[n_train:]], dtype=torch.float32).to(DEVICE)
    y_va = torch.tensor(y_synth[perm[n_train:]], dtype=torch.float32).to(DEVICE)

    # Standardize inputs
    x_mean = X_tr.mean(0)
    x_std = X_tr.std(0).clamp(min=1e-6)
    X_tr = (X_tr - x_mean) / x_std
    X_va = (X_va - x_mean) / x_std

    operator = SurrogateOperator(input_dim=12, hidden=(256, 256, 128)).to(DEVICE)
    optimizer = torch.optim.Adam(operator.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    dataset = TensorDataset(X_tr, y_tr)
    loader = DataLoader(dataset, batch_size=512, shuffle=True)

    for epoch in range(epochs):
        operator.train()
        epoch_loss = 0
        for xb, yb in loader:
            pred = operator(xb)
            loss = nn.MSELoss()(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)

        operator.eval()
        with torch.no_grad():
            val_pred = operator(X_va)
            val_loss = nn.MSELoss()(val_pred, y_va).item()
            scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in operator.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 30:
                break

    operator.load_state_dict(best_state)

    # Compute R²
    operator.eval()
    with torch.no_grad():
        pred_all = operator((torch.tensor(X_synth, dtype=torch.float32).to(DEVICE) - x_mean) / x_std)
        pred_np = pred_all.cpu().numpy()
    ss_res = np.sum((y_synth - pred_np) ** 2)
    ss_tot = np.sum((y_synth - y_synth.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    return operator, x_mean, x_std, r2


def train_e2e_fold(
    X_train, y_train, doses_train,
    X_val, y_val, doses_val,
    operator, op_x_mean, op_x_std,
    epochs=300, lr=1e-3, freeze_operator=True,
    adme_reg_weight=0.0,
):
    """Train E2E pipeline on one CV fold."""
    X_tr_t = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    y_tr_t = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)
    d_tr_t = torch.tensor(np.log10(np.maximum(doses_train, 0.1)), dtype=torch.float32).to(DEVICE)

    X_va_t = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    y_va_t = torch.tensor(y_val, dtype=torch.float32).to(DEVICE)
    d_va_t = torch.tensor(np.log10(np.maximum(doses_val, 0.1)), dtype=torch.float32).to(DEVICE)

    encoder = ADMEEncoder(input_dim=X_train.shape[1], hidden=(512, 256)).to(DEVICE)

    # Build E2E pipeline with pre-trained operator
    pipeline = E2EPipeline(encoder, operator).to(DEVICE)

    if freeze_operator:
        for p in pipeline.operator.parameters():
            p.requires_grad = False
        opt_params = pipeline.encoder.parameters()
    else:
        opt_params = pipeline.parameters()

    optimizer = torch.optim.Adam(opt_params, lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)

    # Wrap operator to handle standardization
    class WrappedPipeline(nn.Module):
        def __init__(self, pipeline, op_mean, op_std):
            super().__init__()
            self.pipeline = pipeline
            self.register_buffer("op_mean", op_mean)
            self.register_buffer("op_std", op_std)

        def forward(self, features, log10_dose):
            adme = self.pipeline.encoder(features)
            adme = adme.clone()
            adme[:, 8] = log10_dose
            # Standardize for operator
            adme_std = (adme - self.op_mean) / self.op_std
            return self.pipeline.operator(adme_std)

    wrapped = WrappedPipeline(pipeline, op_x_mean, op_x_std).to(DEVICE)

    dataset = TensorDataset(X_tr_t, y_tr_t, d_tr_t)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        wrapped.train()
        epoch_loss = 0
        for xb, yb, db in loader:
            pred = wrapped(xb, db)
            loss = nn.MSELoss()(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(opt_params, 1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(xb)

        wrapped.eval()
        with torch.no_grad():
            val_pred = wrapped(X_va_t, d_va_t)
            val_loss = nn.MSELoss()(val_pred, y_va_t).item()
            scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in wrapped.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 30:
                break

    wrapped.load_state_dict(best_state)
    wrapped.eval()

    with torch.no_grad():
        val_pred = wrapped(X_va_t, d_va_t).cpu().numpy()

    return val_pred, wrapped


# ── Metrics ────────────────────────────────────────────────────────────────


def aafe(pred_cmax, obs_cmax):
    mask = (pred_cmax > 0) & (obs_cmax > 0)
    return 10 ** np.mean(np.abs(np.log10(pred_cmax[mask] / obs_cmax[mask])))


def pct_2fold(pred_cmax, obs_cmax):
    ratios = pred_cmax / obs_cmax
    return 100.0 * np.mean((ratios >= 0.5) & (ratios <= 2.0))


# ── Main ───────────────────────────────────────────────────────────────────


def main():
    log.info("=" * 70)
    log.info("Neural Operator Gate 2: E2E Fine-tuning on Real MMPK Data")
    log.info(f"Device: {DEVICE}")
    log.info("=" * 70)

    # ── Load synthetic data for operator pre-training ─────────────────────
    log.info("\n[Phase A] Pre-train operator on synthetic data")

    # Regenerate synthetic param features + load Cmax from Gate 1
    # We need to re-run the engine or load cached. Let's use a shortcut:
    # Load the Gate 1 results to get the synthetic data characteristics,
    # then re-generate and re-run engine for a smaller set in PyTorch context.

    # Actually, let's re-generate from seed and re-run engine.
    # This is the bottleneck (~30min). To speed up, use fewer samples.
    # OR: run the surrogate pre-training using sklearn MLP from Gate 1 as teacher.

    # Fastest approach: generate synthetic features + run engine for 5K samples
    # (enough for operator pre-training)

    import sisyphus.engine.flux  # noqa: F401
    from sisyphus.engine.compiler import ODECompiler
    from sisyphus.graph.builder import build_from_yaml

    # Import Gate 1 functions via sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from neural_operator_gate1 import (
        build_drug_from_params,
        generate_synthetic_params,
        params_to_features,
        run_engine_batch,
    )

    physiology_dir = Path("data/physiology")
    graph = build_from_yaml(physiology_dir / "reference_man.yaml")
    compiled = ODECompiler().compile(graph)

    # Generate synthetic data (10K for speed — Gate 1 showed even small MLP gets R²>0.998)
    N_SYNTH = 10000
    log.info(f"  Generating {N_SYNTH} synthetic samples...")
    rng = np.random.default_rng(2026)
    synth_params = generate_synthetic_params(N_SYNTH, rng)
    synth_drugs = [build_drug_from_params(p) for p in synth_params]

    t0 = time.time()
    synth_cmax = run_engine_batch(synth_drugs, graph, compiled)
    gen_time = time.time() - t0
    log.info(f"  Generated in {gen_time:.0f}s ({gen_time/60:.1f}min)")

    # Filter valid
    valid_idx = [i for i, c in enumerate(synth_cmax) if c is not None and c > 1e-15]
    synth_features = params_to_features([synth_params[i] for i in valid_idx])
    synth_log_cmax = np.log10(np.array([synth_cmax[i] for i in valid_idx]))
    log.info(f"  Valid synthetic: {len(valid_idx)}/{N_SYNTH}")

    # Pre-train operator in PyTorch
    log.info("  Pre-training operator (PyTorch)...")
    t0 = time.time()
    operator, op_x_mean, op_x_std, op_r2 = pretrain_operator_pytorch(
        synth_features, synth_log_cmax, epochs=300, lr=1e-3
    )
    log.info(f"  Operator R² = {op_r2:.4f} ({time.time()-t0:.1f}s)")

    if op_r2 < 0.99:
        log.info("  WARNING: Operator R² < 0.99. Pre-training may be insufficient.")

    # ── Load MMPK data ───────────────────────────────────────────────────
    log.info("\n[Phase B] Load MMPK training data")
    X_mmpk, drug_df = load_mmpk_data()

    smiles_list = drug_df["canon_smiles"].tolist()
    y_log_cmax_dose = drug_df["log_cmax_per_dose"].values  # log10(Cmax_mg_L / dose)
    doses = drug_df["dose_mg"].values
    cmax_obs = drug_df["cmax_mg_L"].values

    # Target for the pipeline: log10(Cmax) = log_cmax_per_dose + log10(dose)
    y_log_cmax = np.log10(np.maximum(cmax_obs, 1e-10))

    log.info(f"  Training drugs: {len(X_mmpk)}, features: {X_mmpk.shape[1]}")
    log.info(f"  Target: log10(Cmax), range [{y_log_cmax.min():.2f}, {y_log_cmax.max():.2f}]")

    # ── Scaffold CV ──────────────────────────────────────────────────────
    log.info("\n[Phase C] Scaffold-stratified 5-fold CV")
    n_folds = 5
    folds = scaffold_split(smiles_list, n_folds=n_folds, seed=42)

    # Also run XGBoost baseline for direct comparison on same folds
    import xgboost as xgb

    oof_pred_e2e = np.full(len(X_mmpk), np.nan)
    oof_pred_xgb = np.full(len(X_mmpk), np.nan)

    for fold in range(n_folds):
        val_mask = folds == fold
        train_mask = ~val_mask

        X_tr, y_tr, d_tr = X_mmpk[train_mask], y_log_cmax[train_mask], doses[train_mask]
        X_va, y_va, d_va = X_mmpk[val_mask], y_log_cmax[val_mask], doses[val_mask]

        log.info(f"\n  Fold {fold+1}/{n_folds}: train={sum(train_mask)}, val={sum(val_mask)}")

        # E2E pipeline (freeze operator)
        log.info(f"    Training E2E (frozen operator)...")
        t0 = time.time()

        # Clone operator for this fold
        fold_operator = SurrogateOperator(input_dim=12, hidden=(256, 256, 128)).to(DEVICE)
        fold_operator.load_state_dict(operator.state_dict())

        val_pred_e2e, _ = train_e2e_fold(
            X_tr, y_tr, d_tr, X_va, y_va, d_va,
            fold_operator, op_x_mean, op_x_std,
            epochs=300, lr=5e-4, freeze_operator=True,
        )
        oof_pred_e2e[val_mask] = val_pred_e2e
        log.info(f"    E2E done ({time.time()-t0:.1f}s)")

        # XGBoost baseline (same target: log10(Cmax))
        # Include dose as feature
        X_tr_xgb = np.column_stack([X_tr, np.log10(np.maximum(d_tr, 0.1))])
        X_va_xgb = np.column_stack([X_va, np.log10(np.maximum(d_va, 0.1))])

        xgb_model = xgb.XGBRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
        )
        xgb_model.fit(X_tr_xgb, y_tr, eval_set=[(X_va_xgb, y_va)], verbose=False)
        oof_pred_xgb[val_mask] = xgb_model.predict(X_va_xgb)

    # ── Evaluate ─────────────────────────────────────────────────────────
    log.info(f"\n{'=' * 70}")
    log.info("[Phase D] Cross-Validation Results")
    log.info("=" * 70)

    # Convert log10 predictions back to Cmax mg/L
    valid_e2e = ~np.isnan(oof_pred_e2e)
    valid_xgb = ~np.isnan(oof_pred_xgb)

    cmax_pred_e2e = 10 ** oof_pred_e2e[valid_e2e]
    cmax_pred_xgb = 10 ** oof_pred_xgb[valid_xgb]
    cmax_true = cmax_obs[valid_e2e]  # same mask for both

    aafe_e2e = aafe(cmax_pred_e2e, cmax_true)
    aafe_xgb = aafe(cmax_pred_xgb, cmax_true)
    pct2_e2e = pct_2fold(cmax_pred_e2e, cmax_true)
    pct2_xgb = pct_2fold(cmax_pred_xgb, cmax_true)

    # RMSE in log10 space
    rmse_e2e = np.sqrt(np.mean((oof_pred_e2e[valid_e2e] - y_log_cmax[valid_e2e]) ** 2))
    rmse_xgb = np.sqrt(np.mean((oof_pred_xgb[valid_xgb] - y_log_cmax[valid_xgb]) ** 2))

    # Pearson r
    r_e2e = np.corrcoef(oof_pred_e2e[valid_e2e], y_log_cmax[valid_e2e])[0, 1]
    r_xgb = np.corrcoef(oof_pred_xgb[valid_xgb], y_log_cmax[valid_xgb])[0, 1]

    log.info(f"\n  {'Model':<25} {'AAFE':>8} {'%2-fold':>8} {'RMSE':>8} {'r':>8}")
    log.info(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    log.info(f"  {'E2E Neural Operator':<25} {aafe_e2e:>8.3f} {pct2_e2e:>7.1f}% {rmse_e2e:>8.4f} {r_e2e:>8.3f}")
    log.info(f"  {'XGBoost (baseline)':<25} {aafe_xgb:>8.3f} {pct2_xgb:>7.1f}% {rmse_xgb:>8.4f} {r_xgb:>8.3f}")

    # Error correlation between E2E and XGBoost
    err_e2e = oof_pred_e2e[valid_e2e] - y_log_cmax[valid_e2e]
    err_xgb = oof_pred_xgb[valid_xgb] - y_log_cmax[valid_xgb]
    r_errors = np.corrcoef(err_e2e, err_xgb)[0, 1]
    log.info(f"\n  Error correlation (E2E vs XGBoost): r = {r_errors:.3f}")

    # Gate decision
    delta = aafe_e2e - aafe_xgb
    if aafe_e2e < aafe_xgb:
        gate = "PASS"
        log.info(f"\n  GATE 2 PASS: E2E AAFE {aafe_e2e:.3f} < XGBoost {aafe_xgb:.3f} (Δ={delta:+.3f})")
        log.info(f"  Physics-informed E2E training outperforms direct ML!")
    elif aafe_e2e < aafe_xgb * 1.05:
        gate = "MARGINAL"
        log.info(f"\n  GATE 2 MARGINAL: E2E AAFE {aafe_e2e:.3f} ≈ XGBoost {aafe_xgb:.3f} (Δ={delta:+.3f})")
        log.info(f"  E2E competitive but not superior. Consider as ensemble member.")
    else:
        gate = "FAIL"
        log.info(f"\n  GATE 2 FAIL: E2E AAFE {aafe_e2e:.3f} > XGBoost {aafe_xgb:.3f} (Δ={delta:+.3f})")
        log.info(f"  Physics prior insufficient to beat direct ML at N={len(X_mmpk)}.")

    # If errors are orthogonal, there may be blend value even if E2E is worse
    if r_errors < 0.7 and gate == "FAIL":
        log.info(f"\n  NOTE: Error correlation r={r_errors:.3f} < 0.7 → orthogonal errors.")
        log.info(f"  E2E may add blend value despite worse standalone AAFE.")

    # ── Save results ──────────────────────────────────────────────────────
    output = {
        "experiment": "Neural Operator Gate 2",
        "date": "2026-04-02",
        "device": str(DEVICE),
        "n_synthetic_pretrain": len(valid_idx),
        "operator_r2": round(op_r2, 4),
        "n_mmpk_drugs": len(X_mmpk),
        "n_folds": n_folds,
        "cv_results": {
            "e2e": {
                "aafe": round(aafe_e2e, 3),
                "pct_2fold": round(pct2_e2e, 1),
                "rmse_log10": round(rmse_e2e, 4),
                "pearson_r": round(r_e2e, 3),
            },
            "xgboost": {
                "aafe": round(aafe_xgb, 3),
                "pct_2fold": round(pct2_xgb, 1),
                "rmse_log10": round(rmse_xgb, 4),
                "pearson_r": round(r_xgb, 3),
            },
        },
        "error_correlation_e2e_xgb": round(r_errors, 3),
        "delta_aafe": round(delta, 3),
        "gate_result": gate,
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)
    log.info(f"\n  Results saved to {OUTPUT_JSON}")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
