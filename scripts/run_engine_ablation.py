#!/usr/bin/env python3
"""Engine-only ablation: measure engine AAFE independently of meta-learner.

This is the missing diagnostic that was recommended on 2026-03-24 but never
implemented.  It answers the question:

    "Did DrugBank enrichment improve ENGINE predictions,
     even though META predictions showed ±0.01 (noise)?"

Runs two experiments:
  A) DrugBank OFF → engine AAFE, ML AAFE, meta AAFE
  B) DrugBank ON  → engine AAFE, ML AAFE, meta AAFE

The difference ΔAB at the engine level reveals whether the meta-learner
was masking a real improvement (Ceiling 3: meta-learner trap).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def run_with_drugbank(enabled: bool) -> dict:
    """Run holdout benchmark with DrugBank enrichment toggled."""
    # Reset all caches to ensure clean state
    from sisyphus.predict.drugbank import DrugBankConfig, DrugBankLookup, _reset_singleton
    from sisyphus.predict import adme
    adme._model_cache.clear()

    from sisyphus.predict import chemistry
    if hasattr(chemistry.compute_profile, "_logp_model"):
        del chemistry.compute_profile._logp_model

    _reset_singleton()
    config = DrugBankConfig(
        enable_enzyme_fm=enabled,
        enable_fup=enabled,
        enable_pka=enabled,
        enable_logp=enabled,
    )
    from sisyphus.predict import drugbank
    drugbank._INSTANCE = DrugBankLookup(config=config)

    from sisyphus.validation.benchmark import run_benchmark
    result = run_benchmark(holdout_only=True)
    _reset_singleton()

    return {
        "meta_aafe": result.aafe,
        "meta_pct2": result.pct_2fold,
        "engine_aafe": result.engine_aafe,
        "engine_pct2": result.engine_pct_2fold,
        "ml_aafe": result.ml_aafe,
        "ml_pct2": result.ml_pct_2fold,
        "n": result.n_drugs,
        "n_engine": result.n_engine,
        "n_ml": result.n_ml,
    }


def fmt(val, width=8, decimals=3):
    """Format a value, handling None."""
    if val is None:
        return f"{'N/A':>{width}s}"
    return f"{val:>{width}.{decimals}f}"


def main():
    print("=" * 72)
    print("ENGINE-ONLY ABLATION: Diagnosing the meta-learner trap")
    print("=" * 72)
    print()

    # Experiment A: DrugBank OFF
    print("Running Experiment A: DrugBank enrichment OFF ...")
    a = run_with_drugbank(enabled=False)
    print(f"  Done. {a['n']} drugs evaluated, {a['n_engine']} with engine output.")

    # Experiment B: DrugBank ON
    print("Running Experiment B: DrugBank enrichment ON ...")
    b = run_with_drugbank(enabled=True)
    print(f"  Done. {b['n']} drugs evaluated, {b['n_engine']} with engine output.")

    # Results table
    print()
    print("=" * 72)
    print("RESULTS")
    print("=" * 72)
    print()
    header = f"{'':20s} {'A (DB OFF)':>12s} {'B (DB ON)':>12s} {'Δ (B-A)':>10s}"
    print(header)
    print("-" * 72)

    rows = [
        ("Meta AAFE", "meta_aafe"),
        ("Meta %2-fold", "meta_pct2"),
        ("Engine AAFE", "engine_aafe"),
        ("Engine %2-fold", "engine_pct2"),
        ("ML AAFE", "ml_aafe"),
        ("ML %2-fold", "ml_pct2"),
    ]

    for label, key in rows:
        va = a[key]
        vb = b[key]
        if va is not None and vb is not None:
            delta = vb - va
            delta_str = f"{delta:>+10.3f}"
        else:
            delta_str = f"{'N/A':>10s}"
        print(f"{label:<20s} {fmt(va):>12s} {fmt(vb):>12s} {delta_str}")

    print()
    print(f"{'N (total)':<20s} {a['n']:>12d} {b['n']:>12d}")
    print(f"{'N (engine)':<20s} {a['n_engine']:>12d} {b['n_engine']:>12d}")
    print(f"{'N (ML)':<20s} {a['n_ml']:>12d} {b['n_ml']:>12d}")

    # Interpretation
    print()
    print("=" * 72)
    print("INTERPRETATION")
    print("=" * 72)

    eng_a = a.get("engine_aafe")
    eng_b = b.get("engine_aafe")
    meta_a = a.get("meta_aafe")
    meta_b = b.get("meta_aafe")

    if eng_a is not None and eng_b is not None:
        engine_delta = eng_b - eng_a
        meta_delta = (meta_b or 0) - (meta_a or 0)

        if abs(engine_delta) > 0.05:
            print(f"  Engine Δ = {engine_delta:+.3f} (significant)")
            if engine_delta < 0:
                print("  → DrugBank enrichment IMPROVES engine predictions.")
                if abs(meta_delta) < 0.02:
                    print("  → But meta-learner MASKS this improvement (Ceiling 3 confirmed).")
                    print("  → ACTION: Retrain meta-learner with enriched engine predictions.")
                else:
                    print("  → And meta-learner also shows improvement.")
            else:
                print("  → DrugBank enrichment WORSENS engine predictions.")
                print("  → Error cancellation hypothesis confirmed at engine level too.")
        else:
            print(f"  Engine Δ = {engine_delta:+.3f} (noise, |Δ| ≤ 0.05)")
            print("  → DrugBank enrichment has no effect even at engine level.")
            print("  → CLint R²=0.24 ceiling confirmed independently of meta-learner.")
            print("  → Meta-learner trap is a secondary issue; data quality is primary.")
    else:
        print("  Insufficient engine results for interpretation.")

    print()


if __name__ == "__main__":
    main()
