#!/usr/bin/env python3
"""Pharos Training + Evaluation Pipeline.

Phase A: Data loading + preprocessing
Phase B: Model training (MoE with 1-comp PK backbone)
Phase C: Holdout evaluation + Sisyphus comparison

Usage:
    python3 pharos/train.py
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Batch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pharos"))
sys.path.insert(0, str(ROOT / "src"))

from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

from data_pipeline import PharosDataset, load_all_data, scaffold_split, smiles_to_graph, smiles_to_morgan
from model import PharosModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ═══════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════

def collate_fn(batch_records: list[dict], use_morgan: bool = False):
    """Collate batch of records into tensors."""
    graphs = [r["graph"] for r in batch_records]
    fps = torch.stack([r["fp"] for r in batch_records])
    doses = torch.tensor([r["dose"] for r in batch_records], dtype=torch.float)
    bws = torch.tensor([r["bw"] for r in batch_records], dtype=torch.float)

    targets = {}
    masks = {}
    for key in ["cmax", "auc", "thalf", "fup", "clint", "peff"]:
        targets[key] = torch.tensor([r["targets"][key] for r in batch_records], dtype=torch.float)
        masks[key] = torch.tensor([r["masks"][key] for r in batch_records], dtype=torch.bool)

    if use_morgan:
        return fps, doses, bws, targets, masks
    else:
        batch_graph = Batch.from_data_list(graphs)
        return batch_graph, doses, bws, targets, masks


def compute_loss(pred, targets, masks, lambda_entropy=0.1, lambda_aux=0.1):
    """Multi-task masked loss in log10 space + entropy regularization."""
    loss = torch.tensor(0.0, device=pred["cmax"].device)
    n_tasks = 0

    # Primary tasks: log10 Huber loss
    for task in ["cmax", "auc", "thalf"]:
        if masks[task].any():
            p = pred[task][masks[task]]
            t = targets[task][masks[task]]
            # Filter positive values
            valid = (p > 1e-8) & (t > 1e-8)
            if valid.any():
                log_p = torch.log10(p[valid] + 1e-10)
                log_t = torch.log10(t[valid] + 1e-10)
                task_loss = F.smooth_l1_loss(log_p, log_t, beta=1.0)
                loss = loss + task_loss
                n_tasks += 1

    # Auxiliary tasks
    aux_map = {"fup": "aux_fup", "clint": "aux_clint", "peff": "aux_peff"}
    for task, pred_key in aux_map.items():
        if masks[task].any():
            p = pred[pred_key][masks[task]]
            t = targets[task][masks[task]]
            valid = (p > 1e-8) & (t > 1e-8)
            if valid.any():
                if task == "fup":
                    # fup in [0,1]: MSE on raw scale
                    task_loss = F.mse_loss(p[valid], t[valid])
                else:
                    # log-space for CLint, Peff
                    log_p = torch.log10(p[valid] + 1e-10)
                    log_t = torch.log10(t[valid] + 1e-10)
                    task_loss = F.smooth_l1_loss(log_p, log_t, beta=1.0)
                loss = loss + lambda_aux * task_loss
                n_tasks += 1

    # Entropy regularization on gating
    pi = pred["pi"]
    entropy = -(pi * torch.log(pi + 1e-10)).sum(dim=1).mean()
    loss = loss - lambda_entropy * entropy

    return loss / max(n_tasks, 1)


def compute_aafe(pred_vals, true_vals):
    """Compute AAFE (geometric mean absolute fold error)."""
    mask = (pred_vals > 0) & (true_vals > 0)
    if mask.sum() == 0:
        return float("inf")
    log_fold = np.abs(np.log10(pred_vals[mask] / true_vals[mask]))
    return float(10 ** np.mean(log_fold))


def train_epoch(model, dataset, train_idx, batch_size, optimizer, use_morgan,
                lambda_entropy, lambda_aux):
    model.train()
    rng = np.random.default_rng(None)
    indices = rng.permutation(train_idx)
    total_loss = 0
    n_batches = 0

    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start:start + batch_size]
        records = [dataset[i] for i in batch_idx]

        batch_data, doses, bws, targets, masks = collate_fn(records, use_morgan)
        batch_data = batch_data.to(DEVICE) if not use_morgan else batch_data.to(DEVICE)
        doses = doses.to(DEVICE)
        bws = bws.to(DEVICE)
        targets = {k: v.to(DEVICE) for k, v in targets.items()}
        masks = {k: v.to(DEVICE) for k, v in masks.items()}

        pred = model(batch_data, doses, bws)
        loss = compute_loss(pred, targets, masks, lambda_entropy, lambda_aux)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, dataset, eval_idx, batch_size, use_morgan):
    """Evaluate on a subset, return per-task AAFE."""
    model.eval()
    all_pred = {"cmax": [], "auc": [], "thalf": []}
    all_true = {"cmax": [], "auc": [], "thalf": []}

    for start in range(0, len(eval_idx), batch_size):
        batch_idx = eval_idx[start:start + batch_size]
        records = [dataset[i] for i in batch_idx]

        batch_data, doses, bws, targets, masks = collate_fn(records, use_morgan)
        batch_data = batch_data.to(DEVICE) if not use_morgan else batch_data.to(DEVICE)
        doses = doses.to(DEVICE)
        bws = bws.to(DEVICE)

        pred = model(batch_data, doses, bws)

        for task in ["cmax", "auc", "thalf"]:
            m = torch.tensor([r["masks"][task] for r in records])
            if m.any():
                all_pred[task].extend(pred[task][m].cpu().numpy().tolist())
                all_true[task].extend(targets[task][m].cpu().numpy().tolist())

    results = {}
    for task in ["cmax", "auc", "thalf"]:
        if len(all_pred[task]) > 0:
            p = np.array(all_pred[task])
            t = np.array(all_true[task])
            results[f"{task}_aafe"] = compute_aafe(p, t)
            results[f"{task}_n"] = len(p)
        else:
            results[f"{task}_aafe"] = float("inf")
            results[f"{task}_n"] = 0

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Holdout evaluation (Sisyphus benchmark set)
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_holdout(model, use_morgan: bool) -> dict:
    """Evaluate on Sisyphus holdout 107 drugs."""
    model.eval()

    from sisyphus.validation.reference import load_reference

    refs = [r for r in load_reference() if r.in_holdout]
    pred_cmax, obs_cmax = [], []
    per_drug = []

    for ref in refs:
        try:
            if use_morgan:
                fp = smiles_to_morgan(ref.smiles)
                if fp is None:
                    continue
                batch_data = torch.tensor(fp, dtype=torch.float).unsqueeze(0).to(DEVICE)
            else:
                graph = smiles_to_graph(ref.smiles)
                if graph is None:
                    continue
                batch_data = Batch.from_data_list([graph]).to(DEVICE)

            dose = torch.tensor([ref.dose_mg], dtype=torch.float, device=DEVICE)
            bw = torch.tensor([70.0], dtype=torch.float, device=DEVICE)

            out = model(batch_data, dose, bw)
            cmax_pred = out["cmax"].item()

            if cmax_pred > 0 and ref.cmax_obs > 0:
                pred_cmax.append(cmax_pred)
                obs_cmax.append(ref.cmax_obs)
                fe = cmax_pred / ref.cmax_obs
                per_drug.append({
                    "name": ref.name,
                    "pred": round(cmax_pred, 6),
                    "obs": ref.cmax_obs,
                    "fold_error": round(fe, 3),
                    "pi": out["pi"].cpu().numpy().tolist()[0],
                })
        except Exception:
            continue

    pred_arr = np.array(pred_cmax)
    obs_arr = np.array(obs_cmax)

    aafe = compute_aafe(pred_arr, obs_arr)
    within_2fold = np.mean((pred_arr / obs_arr >= 0.5) & (pred_arr / obs_arr <= 2.0)) * 100

    return {
        "n_drugs": len(pred_cmax),
        "aafe": round(aafe, 3),
        "pct_2fold": round(within_2fold, 1),
        "per_drug": per_drug,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main training pipeline
# ═══════════════════════════════════════════════════════════════════════════

def run_training(
    n_experts: int = 3,
    hidden_dim: int = 256,
    lambda_entropy: float = 0.1,
    lambda_aux: float = 0.1,
    use_morgan: bool = False,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 64,
    max_epochs: int = 200,
    patience: int = 15,
    min_epochs: int = 30,
    label: str = "",
) -> dict:
    """Train a Pharos model with given hyperparameters."""
    log.info("=" * 60)
    log.info("Training: %s", label or f"K={n_experts},dim={hidden_dim},morgan={use_morgan}")
    log.info("=" * 60)

    # Load data
    merged = load_all_data()
    dataset = PharosDataset(merged)

    # Determine node_dim from first graph
    node_dim = dataset[0]["graph"].x.shape[1]

    # Scaffold split
    smiles_list = [r["smiles"] for r in dataset.records]
    train_idx, val_idx = scaffold_split(smiles_list, val_frac=0.2)
    log.info("Split: train=%d, val=%d", len(train_idx), len(val_idx))

    # Build model
    model = PharosModel(
        node_dim=node_dim,
        hidden_dim=hidden_dim,
        n_layers=4,
        n_experts=n_experts,
        dropout=0.2,
        use_morgan=use_morgan,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Model parameters: %d", n_params)

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5,
                                  min_lr=1e-6)

    best_val_aafe = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(
            model, dataset, train_idx, batch_size, optimizer, use_morgan,
            lambda_entropy, lambda_aux,
        )

        val_metrics = evaluate(model, dataset, val_idx, batch_size, use_morgan)
        val_aafe = val_metrics.get("cmax_aafe", float("inf"))

        scheduler.step(val_aafe)
        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        if epoch % 10 == 0 or epoch <= 5:
            log.info(
                "Epoch %3d: loss=%.4f val_cmax_aafe=%.3f val_auc_aafe=%.3f lr=%.1e (%.1fs)",
                epoch, train_loss, val_aafe,
                val_metrics.get("auc_aafe", float("inf")),
                current_lr, elapsed,
            )

        if val_aafe < best_val_aafe and epoch >= min_epochs:
            best_val_aafe = val_aafe
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience and epoch >= min_epochs:
            log.info("Early stopping at epoch %d (best val_aafe=%.3f)", epoch, best_val_aafe)
            break

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final validation
    val_final = evaluate(model, dataset, val_idx, batch_size, use_morgan)
    log.info("Final val: cmax_aafe=%.3f, auc_aafe=%.3f, thalf_aafe=%.3f",
             val_final.get("cmax_aafe", 0), val_final.get("auc_aafe", 0),
             val_final.get("thalf_aafe", 0))

    # Holdout evaluation
    log.info("Running holdout evaluation ...")
    holdout = evaluate_holdout(model, use_morgan)
    log.info("Holdout: AAFE=%.3f, %%2-fold=%.1f%%, N=%d",
             holdout["aafe"], holdout["pct_2fold"], holdout["n_drugs"])

    return {
        "label": label,
        "n_experts": n_experts,
        "hidden_dim": hidden_dim,
        "use_morgan": use_morgan,
        "lambda_entropy": lambda_entropy,
        "lambda_aux": lambda_aux,
        "n_params": n_params,
        "best_epoch": max_epochs - epochs_no_improve if best_state else max_epochs,
        "val_cmax_aafe": val_final.get("cmax_aafe"),
        "val_auc_aafe": val_final.get("auc_aafe"),
        "holdout": holdout,
    }


def main():
    log.info("=" * 70)
    log.info("  PHAROS: End-to-End PK Prediction Platform")
    log.info("=" * 70)

    results = {}

    # ─── Main model: GNN + MoE K=3 ───
    results["gnn_moe3"] = run_training(
        n_experts=3, hidden_dim=256, lambda_entropy=0.1, lambda_aux=0.1,
        use_morgan=False, label="GNN+MoE(K=3)",
    )

    # ─── Ablation: single model (K=1) ───
    results["gnn_single"] = run_training(
        n_experts=1, hidden_dim=256, lambda_entropy=0.0, lambda_aux=0.1,
        use_morgan=False, label="GNN+Single(K=1)",
    )

    # ─── Ablation: Morgan FP encoder ───
    results["morgan_moe3"] = run_training(
        n_experts=3, hidden_dim=256, lambda_entropy=0.1, lambda_aux=0.1,
        use_morgan=True, label="Morgan+MoE(K=3)",
    )

    # ─── Ablation: no auxiliary tasks ───
    results["gnn_moe3_noaux"] = run_training(
        n_experts=3, hidden_dim=256, lambda_entropy=0.1, lambda_aux=0.0,
        use_morgan=False, label="GNN+MoE(K=3)+NoAux",
    )

    # ─── Print comparison ───
    print()
    print("=" * 80)
    print("  PHAROS MODEL COMPARISON")
    print("=" * 80)
    print()
    print(f"{'Model':<25s} {'ValCmax':>8s} {'HO-AAFE':>8s} {'HO-%2f':>7s} {'HO-N':>5s} {'Params':>8s}")
    print("-" * 70)
    for key, r in results.items():
        ho = r["holdout"]
        print(f"{r['label']:<25s} {r['val_cmax_aafe']:>8.3f} {ho['aafe']:>8.3f} "
              f"{ho['pct_2fold']:>6.1f}% {ho['n_drugs']:>5d} {r['n_params']:>8d}")

    # Sisyphus baselines for comparison
    print()
    print("  Sisyphus baselines:")
    print(f"  {'Meta-learner':<25s} {'':>8s} {'2.283':>8s} {'53.3':>6s}% {'107':>5s}")
    print(f"  {'ML-only (XGBoost)':<25s} {'':>8s} {'2.336':>8s} {'':>7s} {'107':>5s}")

    # Save results
    output_path = ROOT / "pharos" / "data" / "training_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Strip per_drug from holdout for JSON size
    save_results = {}
    for k, v in results.items():
        r = dict(v)
        r["holdout"] = {k2: v2 for k2, v2 in v["holdout"].items() if k2 != "per_drug"}
        save_results[k] = r

    with open(output_path, "w") as f:
        json.dump(save_results, f, indent=2, default=str)
    log.info("Results saved to %s", output_path)

    # Gate decision
    best_key = min(results, key=lambda k: results[k]["holdout"]["aafe"])
    best_aafe = results[best_key]["holdout"]["aafe"]

    print()
    print("=" * 70)
    print("  GATE DECISION")
    print("=" * 70)
    print()
    print(f"  Best model: {results[best_key]['label']}")
    print(f"  Holdout AAFE: {best_aafe:.3f}")
    print()

    if best_aafe < 2.283:
        print("  ✓ Pharos AAFE < 2.283 — BEATS Sisyphus meta-learner!")
    elif best_aafe < 2.336:
        print("  ~ Pharos between Sisyphus meta (2.283) and ML-only (2.336)")
        print("  Physics backbone has partial value.")
    elif best_aafe < 3.0:
        print("  ✗ Pharos AAFE > 2.336 — worse than Sisyphus ML-only baseline")
        print("  GNN + 1-comp backbone underperforms XGBoost.")
    else:
        print("  ✗✗ Pharos AAFE > 3.0 — fundamental failure")

    return 0


if __name__ == "__main__":
    sys.exit(main())
