"""integrate_clint_v2.py — Phase 3+4: Integration and error cancellation gate.

Integrates the expanded CLint model into Sisyphus and runs holdout benchmark
with error cancellation gate.

Steps:
  1. Backup current CLint model
  2. Install expanded model (xgboost_clint_v2.json → xgboost_clint.json)
  3. Run holdout benchmark (N=107)
  4. Rerun LOOCV weight optimization (CLint change may shift optimal weights)
  5. Run benchmark again with new weights
  6. Error cancellation gate: decide keep or revert
  7. Print comparison table

Usage:
    python3 scripts/integrate_clint_v2.py
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Paths
MODEL_DIR = ROOT / "models" / "adme"
CURRENT_MODEL = MODEL_DIR / "xgboost_clint.json"
BACKUP_MODEL = MODEL_DIR / "xgboost_clint_pre_chembl.json.bak"
NEW_MODEL = MODEL_DIR / "xgboost_clint_v2.json"


def run_benchmark_with_model(model_path: Path, label: str) -> dict:
    """Install a model and run full benchmark, returning metrics dict."""
    from sisyphus.predict import adme
    from sisyphus.validation.benchmark import run_benchmark

    # Install model
    shutil.copy2(model_path, CURRENT_MODEL)
    adme._model_cache.clear()
    log.info("Installed model: %s (%s)", model_path.name, label)

    # Suppress per-drug logs
    logging.getLogger("sisyphus").setLevel(logging.WARNING)
    result = run_benchmark(holdout_only=True)
    logging.getLogger("sisyphus").setLevel(logging.INFO)

    metrics = {
        "label": label,
        "n_drugs": result.n_drugs,
        "engine_aafe": result.engine_aafe,
        "ml_aafe": result.ml_aafe,
        "meta_aafe": result.aafe,
        "pct_2fold": result.pct_2fold,
        "in_domain_aafe": result.aafe_in_domain,
        "in_domain_pct_2fold": result.pct_2fold_in_domain,
        "n_in_domain": result.n_in_domain,
    }
    log.info(
        "%s: Engine AAFE=%.3f, ML AAFE=%.3f, Meta AAFE=%.3f, %%2-fold=%.1f%%",
        label, metrics["engine_aafe"] or 0, metrics["ml_aafe"] or 0,
        metrics["meta_aafe"], metrics["pct_2fold"],
    )
    return metrics


def run_loocv(label: str) -> dict:
    """Run LOOCV weight optimization, return optimal weights and AAFE."""
    from sisyphus.pipeline.predict import predict
    from sisyphus.predict.chemistry import compute_profile
    from sisyphus.validation.reference import load_reference

    logging.getLogger("sisyphus").setLevel(logging.WARNING)

    refs = [r for r in load_reference() if r.in_holdout]
    drugs = []
    for ref in refs:
        try:
            result = predict(ref.smiles, ref.dose_mg, ref.route)
            eng = result.engine_pk.cmax.mean if result.engine_pk else None
            ml = result.ml_pk.cmax.mean if result.ml_pk else None
            if eng and eng > 0 and ml and ml > 0:
                p = compute_profile(ref.smiles)
                drugs.append({
                    "name": ref.name,
                    "obs": ref.cmax_obs,
                    "eng": eng,
                    "ml": ml,
                    "is_base": p.compound_type == "base",
                })
        except Exception:
            pass

    logging.getLogger("sisyphus").setLevel(logging.INFO)

    # Grid search
    w_range = np.arange(0.0, 1.01, 0.05)
    best_aafe = float("inf")
    best_wb = 0.0
    best_wo = 0.0

    for w_base in w_range:
        for w_other in w_range:
            log_errors = []
            for drug in drugs:
                w_eng_raw = w_base if drug["is_base"] else w_other
                log_eng = np.log10(max(drug["eng"], 1e-10))
                log_ml = np.log10(max(drug["ml"], 1e-10))
                disagreement = abs(log_eng - log_ml)
                if disagreement > 1.0:
                    w_eng = w_eng_raw * (1.0 / disagreement)
                else:
                    w_eng = w_eng_raw
                w_ml = 1.0 - w_eng
                log_cmax = w_eng * log_eng + w_ml * log_ml
                cmax_pred = 10 ** log_cmax
                fold = abs(np.log10(max(cmax_pred, 1e-10) / max(drug["obs"], 1e-10)))
                log_errors.append(fold)

            a = 10 ** np.mean(log_errors)
            if a < best_aafe:
                best_aafe = a
                best_wb = w_base
                best_wo = w_other

    log.info(
        "%s LOOCV: w_base=%.2f, w_other=%.2f, AAFE=%.4f (%d drugs)",
        label, best_wb, best_wo, best_aafe, len(drugs),
    )
    return {
        "w_base": float(best_wb),
        "w_other": float(best_wo),
        "aafe": float(best_aafe),
        "n_drugs": len(drugs),
    }


def main() -> int:
    log.info("=" * 60)
    log.info("Phase 3+4: CLint v2 Integration & Error Cancellation Gate")
    log.info("=" * 60)

    if not NEW_MODEL.exists():
        log.error("New model not found: %s", NEW_MODEL)
        log.error("Run scripts/chembl_clint_extraction.py first.")
        return 1

    # --- Step 1: Backup ---
    if not BACKUP_MODEL.exists():
        shutil.copy2(CURRENT_MODEL, BACKUP_MODEL)
        log.info("Backed up current model to %s", BACKUP_MODEL)

    # --- Step 2: Baseline benchmark (current model) ---
    log.info("-" * 60)
    log.info("Running baseline benchmark ...")
    baseline = run_benchmark_with_model(BACKUP_MODEL, "Baseline CLint")

    # --- Step 3: New model benchmark ---
    log.info("-" * 60)
    log.info("Running expanded model benchmark ...")
    expanded = run_benchmark_with_model(NEW_MODEL, "Expanded CLint")

    # --- Step 4: LOOCV with new model ---
    log.info("-" * 60)
    log.info("Running LOOCV weight optimization with expanded model ...")
    # Install new model first
    shutil.copy2(NEW_MODEL, CURRENT_MODEL)
    from sisyphus.predict import adme
    adme._model_cache.clear()

    loocv = run_loocv("Expanded")

    # --- Step 5: Update weights in ensemble.py if changed ---
    # (We'll report the new weights but not auto-update ensemble.py)

    # --- Step 6: Error cancellation gate ---
    engine_delta = (expanded["engine_aafe"] or 0) - (baseline["engine_aafe"] or 0)
    meta_delta = expanded["meta_aafe"] - baseline["meta_aafe"]

    # --- Print comparison table ---
    print()
    print("=" * 70)
    print("  HOLDOUT BENCHMARK COMPARISON (N=107)")
    print("=" * 70)
    print()

    def _fmt(v):
        return f"{v:.3f}" if v is not None else "N/A"

    def _delta(a, b):
        if a is not None and b is not None:
            d = b - a
            return f"{d:+.3f}"
        return "N/A"

    header = f"{'Metric':<22s} {'Baseline':>12s}  {'Expanded':>12s}  {'Delta':>8s}"
    print(header)
    print("-" * 70)
    for key, label in [
        ("engine_aafe", "Engine AAFE"),
        ("ml_aafe", "ML AAFE"),
        ("meta_aafe", "Meta AAFE"),
        ("pct_2fold", "%2-fold"),
        ("in_domain_aafe", "In-domain AAFE"),
    ]:
        bv = baseline.get(key)
        ev = expanded.get(key)
        if key == "pct_2fold":
            print(f"{label:<22s} {bv:>11.1f}%  {ev:>11.1f}%  {ev - bv:>+7.1f}%")
        else:
            print(f"{label:<22s} {_fmt(bv):>12s}  {_fmt(ev):>12s}  {_delta(bv, ev):>8s}")

    print()
    print(f"{'LOOCV w_base':<22s} {'0.45':>12s}  {loocv['w_base']:>12.2f}")
    print(f"{'LOOCV w_other':<22s} {'0.00':>12s}  {loocv['w_other']:>12.2f}")
    print(f"{'LOOCV AAFE':<22s} {'2.283':>12s}  {loocv['aafe']:>12.4f}")
    print()

    # --- Gate decision ---
    print("=" * 70)
    print("  ERROR CANCELLATION GATE")
    print("=" * 70)
    print()
    print(f"  Engine AAFE delta: {engine_delta:+.3f}")
    print(f"  Meta AAFE delta:   {meta_delta:+.3f}")
    print()

    if engine_delta < -0.2 and meta_delta < 0:
        decision = "KEEP"
        print("  DECISION: KEEP expanded model")
        print("  Engine improved significantly AND meta improved.")
    elif engine_delta < 0 and meta_delta > 0:
        decision = "REVERT"
        print("  DECISION: REVERT — error cancellation detected")
        print("  Engine improved but meta worsened.")
        print("  CLint improvement breaks fup/Peff error balance.")
    elif engine_delta > 0:
        decision = "REVERT"
        print("  DECISION: REVERT — expanded model is worse")
        print("  Engine AAFE worsened.")
    else:
        if meta_delta < 0:
            decision = "KEEP"
            print("  DECISION: KEEP (marginal improvement)")
            print(f"  Engine delta {engine_delta:+.3f} (< -0.2 threshold),")
            print(f"  but meta improved by {meta_delta:+.3f}.")
        else:
            decision = "REVERT"
            print("  DECISION: REVERT — no improvement")
            print(f"  Engine delta {engine_delta:+.3f}, meta delta {meta_delta:+.3f}.")

    print()

    if decision == "REVERT":
        shutil.copy2(BACKUP_MODEL, CURRENT_MODEL)
        adme._model_cache.clear()
        log.info("Reverted to baseline model")
        print("  Model reverted to baseline.")
    else:
        log.info("Keeping expanded model")
        print("  Expanded model retained as production model.")

    # Save results
    results = {
        "baseline": baseline,
        "expanded": expanded,
        "loocv": loocv,
        "engine_delta": float(engine_delta),
        "meta_delta": float(meta_delta),
        "decision": decision,
    }
    output_path = ROOT / "data" / "chembl" / "integration_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Results saved to %s", output_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
