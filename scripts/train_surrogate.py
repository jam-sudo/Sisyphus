#!/usr/bin/env python3
"""Train neural surrogate ensemble for fast Cmax prediction.

Generates synthetic ADME parameter sets, runs the Sisyphus engine for each,
and trains an Equinox MLP ensemble to approximate: ADME → log10(Cmax/dose).

Usage:
    python scripts/train_surrogate.py [--n-train 50000] [--n-test 10000] [--n-ensemble 5]

Output:
    models/surrogate/cmax_mlp_{0..4}.eqx     - trained ensemble members
    models/surrogate/scaler.npz              - feature standardization
    models/surrogate/validation.json         - accuracy metrics
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("train_surrogate")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# Gate 1 parameter ranges (proven to cover pharmacological space)
PARAM_RANGES = {
    "log10_clint": (-0.5, 3.0),
    "fup": (0.005, 1.0),
    "logp": (-1.0, 6.0),
    "pka": (2.0, 12.0),
    "mw": (150.0, 800.0),
    "log10_dose": (-0.5, 3.0),
    "log10_peff": (-1.0, 2.0),
    "log10_sol": (-3.0, 2.0),
}


def generate_params(n: int, rng: np.random.Generator) -> list[dict]:
    """Generate synthetic ADME parameter sets."""
    params = []
    for _ in range(n):
        compound_type = rng.choice(["neutral", "acid", "base"])
        pka = float(rng.uniform(*PARAM_RANGES["pka"])) if compound_type != "neutral" else None
        params.append({
            "clint": float(10 ** rng.uniform(*PARAM_RANGES["log10_clint"])),
            "fup": float(rng.uniform(*PARAM_RANGES["fup"])),
            "logp": float(rng.uniform(*PARAM_RANGES["logp"])),
            "pka": pka,
            "compound_type": compound_type,
            "mw": float(rng.uniform(*PARAM_RANGES["mw"])),
            "dose_mg": float(10 ** rng.uniform(*PARAM_RANGES["log10_dose"])),
            "peff": float(10 ** rng.uniform(*PARAM_RANGES["log10_peff"])),
            "solubility": float(10 ** rng.uniform(*PARAM_RANGES["log10_sol"])),
        })
    return params


def params_to_features(params: list[dict]) -> np.ndarray:
    """Convert parameter dicts to 12D feature matrix."""
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


def build_drug_from_params(p: dict):
    """Build DrugOnGraph from raw ADME parameters (same as Gate 1)."""
    from sisyphus.core import Distribution, DrugOnGraph
    from sisyphus.predict.ivive import (
        _compute_all_kp,
        _decompose_clint,
        _estimate_particle_radius,
    )

    clint_dist = Distribution(mean=p["clint"], cv=0.0)
    enzyme_affinity = _decompose_clint(
        clint_dist, p["compound_type"], p["pka"],
        substrate_enzymes=None, ugt_enzymes=None,
    )
    kp_overrides = _compute_all_kp(
        p["logp"], p["pka"], p["compound_type"],
        kp_method="rodgers_rowland", fup=p["fup"],
    )
    kp_overrides = {k: Distribution(mean=v.mean, cv=0.0) for k, v in kp_overrides.items()}
    sol_dist = Distribution(mean=p["solubility"], cv=0.0)
    particle_radius = _estimate_particle_radius(sol_dist)

    return DrugOnGraph(
        name="synthetic",
        smiles="C",
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
        renal_clearance=Distribution(mean=7.5 * p["fup"], cv=0.0),
        particle_radius_um=particle_radius,
    )


def run_engine_batch(params_list, graph, compiled):
    """Run engine for a batch of synthetic drugs. Returns log10(Cmax/dose) array."""
    from sisyphus.engine.compiler import ResolvedParams
    from sisyphus.engine.solver import solve_mc

    results = []
    for i, p in enumerate(params_list):
        try:
            drug = build_drug_from_params(p)
            rg = graph  # no sampling (cv=0 for graph)
            rd = drug   # no sampling (cv=0 for drug)
            params = ResolvedParams(rg, rd)

            y0 = np.zeros(compiled.n_states)
            y0[compiled.state_index["stomach_lumen"]] = p["dose_mg"]

            cmax, tmax, auc, success = solve_mc(
                compiled, params, y0, (0.0, 24.0), "venous_blood",
            )
            if success and cmax > 0:
                log_cd = np.log10(cmax / p["dose_mg"])
                results.append(log_cd)
            else:
                results.append(None)
        except Exception:
            results.append(None)

        if (i + 1) % 500 == 0:
            n_ok = sum(1 for r in results if r is not None)
            log.info("  engine: %d/%d done (%d successful)", i + 1, len(params_list), n_ok)

    return results


def train_ensemble(X_train, y_train, X_test, y_test, n_ensemble=5, n_epochs=500):
    """Train Equinox MLP ensemble."""
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import equinox as eqx
    import optax
    from sisyphus.engine.surrogate import CmaxSurrogate

    # Standardize features
    mean_x = np.mean(X_train, axis=0)
    std_x = np.std(X_train, axis=0)
    std_x = np.maximum(std_x, 1e-10)

    X_tr = jnp.array((X_train - mean_x) / std_x)
    y_tr = jnp.array(y_train)
    X_te = jnp.array((X_test - mean_x) / std_x)
    y_te = jnp.array(y_test)

    models = []
    metrics = []

    for ens_i in range(n_ensemble):
        log.info("Training ensemble member %d/%d...", ens_i + 1, n_ensemble)
        key = jax.random.PRNGKey(ens_i * 1000 + 42)
        model = CmaxSurrogate(key)

        optimizer = optax.adam(1e-3)
        opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

        @eqx.filter_jit
        def loss_fn(model, x, y):
            pred = jax.vmap(model)(x)
            return jnp.mean((pred - y) ** 2)

        @eqx.filter_jit
        def step(model, opt_state, x, y):
            loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x, y)
            updates, opt_state_new = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
            model_new = eqx.apply_updates(model, updates)
            return model_new, opt_state_new, loss

        # Mini-batch training
        batch_size = 512
        n_batches = max(1, len(y_tr) // batch_size)
        best_test_loss = float("inf")
        patience = 30
        no_improve = 0

        for epoch in range(n_epochs):
            # Shuffle
            perm = jax.random.permutation(jax.random.PRNGKey(epoch + ens_i * 10000), len(y_tr))
            X_shuf = X_tr[perm]
            y_shuf = y_tr[perm]

            epoch_loss = 0.0
            for b in range(n_batches):
                xb = X_shuf[b * batch_size:(b + 1) * batch_size]
                yb = y_shuf[b * batch_size:(b + 1) * batch_size]
                model, opt_state, loss = step(model, opt_state, xb, yb)
                epoch_loss += float(loss)

            epoch_loss /= n_batches
            test_loss = float(loss_fn(model, X_te, y_te))

            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_model = model
                no_improve = 0
            else:
                no_improve += 1

            if (epoch + 1) % 50 == 0:
                log.info("  epoch %d: train_mse=%.5f test_mse=%.5f", epoch + 1, epoch_loss, test_loss)

            if no_improve >= patience:
                log.info("  early stopping at epoch %d (best test MSE=%.5f)", epoch + 1, best_test_loss)
                break

        # Evaluate
        pred_te = np.array(jax.vmap(best_model)(X_te))
        ss_res = np.sum((np.array(y_te) - pred_te) ** 2)
        ss_tot = np.sum((np.array(y_te) - np.mean(np.array(y_te))) ** 2)
        r2 = 1 - ss_res / ss_tot

        fold_err = 10 ** np.abs(pred_te - np.array(y_te))
        within_20 = float(np.mean(fold_err <= 1.2))

        log.info("  member %d: R²=%.4f, %%within 20%%=%.1f%%, best_mse=%.5f",
                 ens_i, r2, within_20 * 100, best_test_loss)

        models.append(best_model)
        metrics.append({"r2": float(r2), "within_20pct": float(within_20), "test_mse": float(best_test_loss)})

    return models, metrics, mean_x, std_x


def main():
    parser = argparse.ArgumentParser(description="Train Cmax neural surrogate")
    parser.add_argument("--n-train", type=int, default=50000)
    parser.add_argument("--n-test", type=int, default=10000)
    parser.add_argument("--n-ensemble", type=int, default=5)
    parser.add_argument("--n-epochs", type=int, default=500)
    args = parser.parse_args()

    import jax
    jax.config.update("jax_enable_x64", True)
    import equinox as eqx

    log.info("=" * 70)
    log.info("Neural Surrogate Training — Phase 3 (Priority 4)")
    log.info("=" * 70)

    # Load graph and compile
    from sisyphus.graph.builder import build_from_yaml
    from sisyphus.engine.compiler import ODECompiler

    graph = build_from_yaml(ROOT / "data" / "physiology" / "reference_man.yaml")
    compiled = ODECompiler().compile(graph)
    log.info("Graph: %d nodes, %d edges", compiled.n_states, len(compiled.flux_specs))

    # Generate synthetic parameters
    rng = np.random.default_rng(2026)
    n_total = args.n_train + args.n_test
    log.info("Generating %d synthetic parameter sets...", n_total)
    all_params = generate_params(n_total, rng)

    # Run engine
    log.info("Running engine for %d drugs...", n_total)
    t0 = time.perf_counter()
    all_log_cd = run_engine_batch(all_params, graph, compiled)
    t_engine = time.perf_counter() - t0
    n_ok = sum(1 for r in all_log_cd if r is not None)
    log.info("Engine: %d/%d successful (%.1f s, %.1f drugs/sec)",
             n_ok, n_total, t_engine, n_ok / t_engine)

    # Filter valid
    valid_params = []
    valid_log_cd = []
    for p, lcd in zip(all_params, all_log_cd):
        if lcd is not None and np.isfinite(lcd):
            valid_params.append(p)
            valid_log_cd.append(lcd)

    X = params_to_features(valid_params)
    y = np.array(valid_log_cd)
    log.info("Valid samples: %d (features shape: %s)", len(y), X.shape)

    # Split
    n_train = min(args.n_train, int(len(y) * 0.8))
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    log.info("Train: %d, Test: %d", len(y_train), len(y_test))

    # Train ensemble
    models, metrics, mean_x, std_x = train_ensemble(
        X_train, y_train, X_test, y_test,
        n_ensemble=args.n_ensemble, n_epochs=args.n_epochs,
    )

    # Save
    model_dir = ROOT / "models" / "surrogate"
    model_dir.mkdir(parents=True, exist_ok=True)

    for i, model in enumerate(models):
        path = model_dir / f"cmax_mlp_{i}.eqx"
        eqx.tree_serialise_leaves(str(path), model)
        log.info("Saved: %s", path)

    np.savez(model_dir / "scaler.npz", mean=mean_x, std=std_x)
    log.info("Saved: %s/scaler.npz", model_dir)

    # Ensemble validation
    from sisyphus.engine.surrogate import validate_surrogate
    result = validate_surrogate(
        models, mean_x, std_x, X_test, 10**y_test * np.array([p["dose_mg"] for p in valid_params[n_train:]]),
    )

    result["per_member"] = metrics
    result["n_train"] = n_train
    result["n_test"] = len(y_test)
    result["engine_time_s"] = t_engine
    result["engine_throughput"] = n_ok / t_engine

    with open(model_dir / "validation.json", "w") as f:
        json.dump(result, f, indent=2)
    log.info("Saved: %s/validation.json", model_dir)

    log.info("\n" + "=" * 70)
    log.info("RESULT: coverage=%.1f%% R²=%.4f gate=%s",
             result["coverage_within_20pct"] * 100,
             result["r2_log_scale"],
             "PASS" if result["gate_pass"] else "FAIL")


if __name__ == "__main__":
    main()
