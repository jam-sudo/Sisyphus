#!/usr/bin/env python3
"""Neural Operator Gate 1: Can an MLP approximate the Sisyphus PBPK engine?

Generates synthetic ADME parameter sets, runs the Sisyphus engine for each,
and trains an MLP surrogate to map (ADME_params → log10(Cmax)).

Gate criterion: R² > 0.95 on held-out synthetic data.

If this passes, the MLP can serve as a differentiable surrogate for the engine,
enabling end-to-end training: SMILES → ADME Encoder → Neural Operator → Cmax.
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.WARNING, format="%(message)s")
log = logging.getLogger("gate1")
log.setLevel(logging.INFO)

# Suppress noisy loggers
for name in ["sisyphus", "sisyphus.predict", "sisyphus.engine", "sisyphus.graph"]:
    logging.getLogger(name).setLevel(logging.WARNING)

OUTPUT_JSON = Path("data/validation/neural_operator_gate1_results.json")

# ── Synthetic parameter distributions ──────────────────────────────────────
# Based on TDC ADME dataset statistics and pharmacological plausibility.

PARAM_RANGES = {
    "log10_clint": (-0.5, 3.0),     # CLint: 0.3 - 1000 uL/min/1e6 cells
    "fup": (0.005, 1.0),            # fraction unbound
    "logp": (-1.0, 6.0),            # octanol-water partition
    "pka": (2.0, 12.0),             # dissociation constant
    "mw": (150.0, 800.0),           # molecular weight
    "log10_dose": (-0.5, 3.0),      # dose: 0.3 - 1000 mg
    "log10_peff": (-1.0, 2.0),      # Peff: 0.1 - 100 x10^-4 cm/s
    "log10_sol": (-3.0, 2.0),       # solubility: 0.001 - 100 mg/mL
}


def generate_synthetic_params(n: int, rng: np.random.Generator) -> list[dict]:
    """Generate n synthetic ADME parameter sets from uniform distributions."""
    params = []
    for _ in range(n):
        compound_type = rng.choice(["neutral", "acid", "base"])
        pka = float(rng.uniform(*PARAM_RANGES["pka"])) if compound_type != "neutral" else None

        p = {
            "clint": float(10 ** rng.uniform(*PARAM_RANGES["log10_clint"])),
            "fup": float(rng.uniform(*PARAM_RANGES["fup"])),
            "logp": float(rng.uniform(*PARAM_RANGES["logp"])),
            "pka": pka,
            "compound_type": compound_type,
            "mw": float(rng.uniform(*PARAM_RANGES["mw"])),
            "dose_mg": float(10 ** rng.uniform(*PARAM_RANGES["log10_dose"])),
            "peff": float(10 ** rng.uniform(*PARAM_RANGES["log10_peff"])),
            "solubility": float(10 ** rng.uniform(*PARAM_RANGES["log10_sol"])),
        }
        params.append(p)
    return params


def build_drug_from_params(p: dict) -> "DrugOnGraph":
    """Build a DrugOnGraph directly from raw ADME parameters.

    Bypasses predict layer — calls IVIVE functions directly.
    """
    from sisyphus.core import Distribution, DrugOnGraph
    from sisyphus.predict.ivive import (
        _compute_all_kp,
        _decompose_clint,
        _estimate_particle_radius,
    )

    # CLint decomposition → per-enzyme affinities
    clint_dist = Distribution(mean=p["clint"], cv=0.0)
    enzyme_affinity = _decompose_clint(
        clint_dist, p["compound_type"], p["pka"],
        substrate_enzymes=None, ugt_enzymes=None,
    )

    # Kp computation for all tissues
    kp_overrides = _compute_all_kp(
        p["logp"], p["pka"], p["compound_type"],
        kp_method="rodgers_rowland", fup=p["fup"],
    )
    # Set CV=0 for deterministic run
    kp_overrides = {k: Distribution(mean=v.mean, cv=0.0) for k, v in kp_overrides.items()}

    # Renal clearance = GFR * fup
    renal_cl = 7.5 * p["fup"]

    # Solubility-based particle radius
    sol_dist = Distribution(mean=p["solubility"], cv=0.0)
    particle_radius = _estimate_particle_radius(sol_dist)

    return DrugOnGraph(
        name=f"synthetic_{id(p)}",
        smiles="C",  # dummy SMILES
        dose_mg=p["dose_mg"],
        route="oral",
        administration_node="stomach_lumen",
        mw=p["mw"],
        pka=p["pka"],
        compound_type=p["compound_type"],
        fup=Distribution(mean=p["fup"], cv=0.0),
        rbp=Distribution(mean=1.0, cv=0.0),
        kp_method="rodgers_rowland",
        kp_overrides=kp_overrides,
        peff=Distribution(mean=p["peff"], cv=0.0),
        solubility=sol_dist,
        enzyme_affinity=enzyme_affinity,
        renal_clearance=Distribution(mean=renal_cl, cv=0.0),
        particle_radius_um=particle_radius,
    )


def run_engine_batch(drugs: list, graph, compiled) -> list[float | None]:
    """Run engine for a batch of DrugOnGraph objects. Returns list of Cmax (mg/L)."""
    from sisyphus.engine.compiler import ResolvedParams
    from sisyphus.engine.solver import solve

    rng = np.random.default_rng(42)
    cmax_values = []

    for drug in drugs:
        try:
            realized_graph = graph.sample(rng)
            realized_drug = drug.sample(rng)
            params = ResolvedParams(realized_graph, realized_drug)

            y0 = np.zeros(compiled.n_states)
            admin_idx = compiled.state_index[drug.administration_node]
            y0[admin_idx] = drug.dose_mg

            result = solve(compiled, params, y0, t_span=(0, 24))
            if result.solver_success:
                conc = result.concentrations.get("venous_blood")
                if conc is not None and len(conc) > 0:
                    cmax = float(np.max(conc))
                    cmax_values.append(cmax if cmax > 1e-15 else None)
                else:
                    cmax_values.append(None)
            else:
                cmax_values.append(None)
        except Exception:
            cmax_values.append(None)

    return cmax_values


def params_to_features(params: list[dict]) -> np.ndarray:
    """Convert parameter dicts to feature matrix for MLP.

    Features (12D):
    [log10(CLint), fup, logP, pKa_or_7, is_neutral, is_acid, is_base,
     MW/500, log10(dose), log10(Peff), log10(sol), log10(renal_cl)]
    """
    X = np.zeros((len(params), 12))
    for i, p in enumerate(params):
        X[i, 0] = np.log10(max(p["clint"], 1e-6))
        X[i, 1] = p["fup"]
        X[i, 2] = p["logp"]
        X[i, 3] = p["pka"] if p["pka"] is not None else 7.0
        X[i, 4] = 1.0 if p["compound_type"] == "neutral" else 0.0
        X[i, 5] = 1.0 if p["compound_type"] == "acid" else 0.0
        X[i, 6] = 1.0 if p["compound_type"] == "base" else 0.0
        X[i, 7] = p["mw"] / 500.0
        X[i, 8] = np.log10(max(p["dose_mg"], 1e-6))
        X[i, 9] = np.log10(max(p["peff"], 1e-6))
        X[i, 10] = np.log10(max(p["solubility"], 1e-10))
        X[i, 11] = np.log10(max(7.5 * p["fup"], 1e-10))
    return X


def train_mlp_surrogate(X_train, y_train, X_test, y_test, hidden_sizes=(256, 256, 128)):
    """Train MLP surrogate. Returns (model, train_r2, test_r2, test_predictions)."""
    try:
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        log.error("sklearn required. pip install scikit-learn")
        sys.exit(1)

    scaler_x = StandardScaler().fit(X_train)
    scaler_y = StandardScaler().fit(y_train.reshape(-1, 1))

    X_tr = scaler_x.transform(X_train)
    y_tr = scaler_y.transform(y_train.reshape(-1, 1)).ravel()
    X_te = scaler_x.transform(X_test)

    mlp = MLPRegressor(
        hidden_layer_sizes=hidden_sizes,
        activation="relu",
        solver="adam",
        max_iter=2000,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=30,
        random_state=42,
        batch_size=256,
        learning_rate_init=0.001,
    )

    mlp.fit(X_tr, y_tr)

    pred_train = scaler_y.inverse_transform(mlp.predict(X_tr).reshape(-1, 1)).ravel()
    pred_test = scaler_y.inverse_transform(mlp.predict(X_te).reshape(-1, 1)).ravel()

    ss_res_train = np.sum((y_train - pred_train) ** 2)
    ss_tot_train = np.sum((y_train - np.mean(y_train)) ** 2)
    r2_train = 1 - ss_res_train / ss_tot_train

    ss_res_test = np.sum((y_test - pred_test) ** 2)
    ss_tot_test = np.sum((y_test - np.mean(y_test)) ** 2)
    r2_test = 1 - ss_res_test / ss_tot_test

    mae_test = np.mean(np.abs(y_test - pred_test))
    rmse_test = np.sqrt(np.mean((y_test - pred_test) ** 2))

    return mlp, r2_train, r2_test, pred_test, mae_test, rmse_test, scaler_x, scaler_y


def train_xgb_surrogate(X_train, y_train, X_test, y_test):
    """Train XGBoost surrogate for comparison."""
    try:
        import xgboost as xgb
    except ImportError:
        return None, None, None, None, None, None

    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    ss_res_train = np.sum((y_train - pred_train) ** 2)
    ss_tot_train = np.sum((y_train - np.mean(y_train)) ** 2)
    r2_train = 1 - ss_res_train / ss_tot_train

    ss_res_test = np.sum((y_test - pred_test) ** 2)
    ss_tot_test = np.sum((y_test - np.mean(y_test)) ** 2)
    r2_test = 1 - ss_res_test / ss_tot_test

    mae_test = np.mean(np.abs(y_test - pred_test))
    rmse_test = np.sqrt(np.mean((y_test - pred_test) ** 2))

    return model, r2_train, r2_test, pred_test, mae_test, rmse_test


def main():
    log.info("=" * 70)
    log.info("Neural Operator Gate 1: Engine Surrogate Feasibility")
    log.info("=" * 70)

    # ── Phase 0: Speed test (100 samples) ─────────────────────────────────
    log.info("\n[Phase 0] Speed test — 100 synthetic engine runs")

    import sisyphus.engine.flux  # noqa: F401 -- register flux specs
    from sisyphus.engine.compiler import ODECompiler
    from sisyphus.graph.builder import build_from_yaml

    physiology_dir = Path("data/physiology")
    graph = build_from_yaml(physiology_dir / "reference_man.yaml")
    compiler = ODECompiler()
    compiled = compiler.compile(graph)
    log.info(f"  Graph compiled: {compiled.n_states} states")

    rng = np.random.default_rng(42)
    test_params = generate_synthetic_params(100, rng)
    test_drugs = [build_drug_from_params(p) for p in test_params]

    t0 = time.time()
    test_cmax = run_engine_batch(test_drugs, graph, compiled)
    t_100 = time.time() - t0

    n_success = sum(1 for c in test_cmax if c is not None)
    ms_per_run = 1000 * t_100 / 100
    log.info(f"  100 runs: {t_100:.1f}s ({ms_per_run:.1f}ms/run), {n_success}/100 succeeded")

    # Filter valid Cmax
    valid_cmax = [c for c in test_cmax if c is not None and c > 1e-15]
    if valid_cmax:
        log.info(
            f"  Cmax range: [{min(valid_cmax):.2e}, {max(valid_cmax):.2e}] mg/L"
        )

    # Estimate time for full run
    n_target = 20000
    est_time_min = (ms_per_run * n_target) / 60000
    log.info(f"  Est. time for {n_target}: {est_time_min:.1f} min")

    # ── Phase 1: Generate full synthetic dataset ──────────────────────────
    log.info(f"\n[Phase 1] Generating {n_target} synthetic ADME → Cmax pairs")

    rng_full = np.random.default_rng(2026)
    all_params = generate_synthetic_params(n_target, rng_full)
    all_drugs = [build_drug_from_params(p) for p in all_params]

    t0 = time.time()
    batch_size = 1000
    all_cmax = []
    for batch_start in range(0, n_target, batch_size):
        batch_end = min(batch_start + batch_size, n_target)
        batch_cmax = run_engine_batch(
            all_drugs[batch_start:batch_end], graph, compiled
        )
        all_cmax.extend(batch_cmax)
        n_done = batch_end
        n_ok = sum(1 for c in all_cmax if c is not None)
        elapsed = time.time() - t0
        rate = n_done / elapsed if elapsed > 0 else 0
        log.info(
            f"  {n_done}/{n_target} done ({n_ok} valid), "
            f"{rate:.0f} runs/s, "
            f"ETA {(n_target - n_done) / rate / 60:.1f}min"
        )

    total_time = time.time() - t0
    log.info(f"  Total generation: {total_time:.1f}s ({total_time/60:.1f}min)")

    # Filter to valid (param, cmax) pairs
    valid_indices = [
        i for i, c in enumerate(all_cmax)
        if c is not None and c > 1e-15
    ]
    n_valid = len(valid_indices)
    success_rate = 100 * n_valid / n_target
    log.info(f"  Valid samples: {n_valid}/{n_target} ({success_rate:.1f}%)")

    if n_valid < 5000:
        log.info("  GATE FAIL: Too few valid samples (<5000). Engine failures too frequent.")
        sys.exit(1)

    # Build feature matrix and target
    valid_params = [all_params[i] for i in valid_indices]
    valid_cmax_arr = np.array([all_cmax[i] for i in valid_indices])

    X = params_to_features(valid_params)
    y = np.log10(valid_cmax_arr)

    log.info(f"  Feature matrix: {X.shape}, Target range: [{y.min():.2f}, {y.max():.2f}]")

    # ── Phase 2: Train/test split and model training ──────────────────────
    log.info(f"\n[Phase 2] Training surrogate models")

    # 80/20 random split
    n_train = int(0.8 * n_valid)
    perm = rng_full.permutation(n_valid)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    log.info(f"  Train: {len(X_train)}, Test: {len(X_test)}")

    # MLP surrogate (3 sizes)
    results = {}

    for name, hidden in [
        ("MLP-small", (128, 64)),
        ("MLP-medium", (256, 256, 128)),
        ("MLP-large", (512, 512, 256, 128)),
    ]:
        log.info(f"  Training {name}...")
        t0 = time.time()
        _, r2_tr, r2_te, pred_te, mae, rmse, _, _ = train_mlp_surrogate(
            X_train, y_train, X_test, y_test, hidden_sizes=hidden
        )
        dt = time.time() - t0
        log.info(f"    R² train={r2_tr:.4f}, test={r2_te:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f} ({dt:.1f}s)")
        results[name] = {
            "r2_train": round(r2_tr, 4),
            "r2_test": round(r2_te, 4),
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
        }

    # XGBoost surrogate (for comparison)
    log.info("  Training XGBoost...")
    t0 = time.time()
    xgb_model, xgb_r2_tr, xgb_r2_te, xgb_pred_te, xgb_mae, xgb_rmse = train_xgb_surrogate(
        X_train, y_train, X_test, y_test
    )
    dt = time.time() - t0
    if xgb_r2_te is not None:
        log.info(f"    R² train={xgb_r2_tr:.4f}, test={xgb_r2_te:.4f}, MAE={xgb_mae:.4f}, RMSE={xgb_rmse:.4f} ({dt:.1f}s)")
        results["XGBoost"] = {
            "r2_train": round(xgb_r2_tr, 4),
            "r2_test": round(xgb_r2_te, 4),
            "mae": round(xgb_mae, 4),
            "rmse": round(xgb_rmse, 4),
        }

    # ── Phase 3: Results and gate decision ────────────────────────────────
    log.info(f"\n{'=' * 70}")
    log.info("[Phase 3] Results")
    log.info("=" * 70)

    log.info(f"\n  {'Model':<20} {'R² train':>10} {'R² test':>10} {'MAE':>8} {'RMSE':>8}")
    log.info(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*8} {'-'*8}")
    for name, r in results.items():
        log.info(
            f"  {name:<20} {r['r2_train']:>10.4f} {r['r2_test']:>10.4f} "
            f"{r['mae']:>8.4f} {r['rmse']:>8.4f}"
        )

    # Best test R²
    best_model = max(results, key=lambda k: results[k]["r2_test"])
    best_r2 = results[best_model]["r2_test"]
    best_mae = results[best_model]["mae"]

    log.info(f"\n  Best model: {best_model} (R²={best_r2:.4f})")

    # What does MAE mean in practice?
    # MAE in log10 space → geometric fold error
    fold_error = 10 ** best_mae
    log.info(f"  MAE {best_mae:.4f} log10 → mean fold error {fold_error:.3f}x")
    log.info(f"  (surrogate predicts Cmax within {fold_error:.2f}x of engine on average)")

    # Gate decision
    if best_r2 >= 0.95:
        gate = "PASS"
        log.info(f"\n  GATE 1 PASS: R² = {best_r2:.4f} >= 0.95")
        log.info(f"  The engine can be approximated by a differentiable MLP.")
        log.info(f"  Proceed to Gate 2: E2E fine-tuning on real MMPK data.")
    elif best_r2 >= 0.90:
        gate = "MARGINAL"
        log.info(f"\n  GATE 1 MARGINAL: R² = {best_r2:.4f} (0.90-0.95)")
        log.info(f"  Engine approximation is decent but may limit E2E training.")
        log.info(f"  Consider larger network, more data, or feature engineering.")
    else:
        gate = "FAIL"
        log.info(f"\n  GATE 1 FAIL: R² = {best_r2:.4f} < 0.90")
        log.info(f"  PBPK engine dynamics cannot be captured by MLP from ADME params.")
        log.info(f"  Abandon neural operator approach.")

    # Feature importance (XGBoost)
    if xgb_model is not None:
        feature_names = [
            "log10_CLint", "fup", "logP", "pKa", "is_neutral",
            "is_acid", "is_base", "MW/500", "log10_dose",
            "log10_Peff", "log10_sol", "log10_renal_cl"
        ]
        importances = xgb_model.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]
        log.info(f"\n  XGBoost feature importance:")
        for idx in sorted_idx:
            log.info(f"    {feature_names[idx]:<20} {importances[idx]:.3f}")

    # ── Save results ──────────────────────────────────────────────────────
    output = {
        "experiment": "Neural Operator Gate 1",
        "date": "2026-04-02",
        "n_synthetic": n_target,
        "n_valid": n_valid,
        "success_rate_pct": round(success_rate, 1),
        "generation_time_min": round(total_time / 60, 1),
        "ms_per_engine_run": round(ms_per_run, 1),
        "feature_dim": X.shape[1],
        "target": "log10(Cmax_mg_per_L)",
        "target_range": [round(y.min(), 3), round(y.max(), 3)],
        "train_n": n_train,
        "test_n": n_valid - n_train,
        "models": results,
        "best_model": best_model,
        "best_r2_test": best_r2,
        "best_mae_log10": round(best_mae, 4),
        "fold_error_from_mae": round(fold_error, 3),
        "gate_result": gate,
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)
    log.info(f"\n  Results saved to {OUTPUT_JSON}")
    log.info(f"\n{'=' * 70}")


if __name__ == "__main__":
    main()
