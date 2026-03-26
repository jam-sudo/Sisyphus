#!/usr/bin/env python3
"""pKa + Berezhkovskiy evaluation: measure independent and combined effects.

Runs 4 experiments on the holdout set:

  Baseline: pKa OFF (SMARTS defaults) + Berezhkovskiy OFF (rodgers_rowland)
  Exp 1:    pKa ON  (XGBoost)         + Berezhkovskiy OFF (rodgers_rowland)
  Exp 2:    pKa OFF (SMARTS defaults)  + Berezhkovskiy ON
  Exp 3:    pKa ON  (XGBoost)         + Berezhkovskiy ON   — current default

Reports Engine AAFE, Engine %2-fold, Meta AAFE, Meta %2-fold, and delta vs baseline.
"""

from __future__ import annotations

import functools
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

PKA_ACID = ROOT / "models" / "adme" / "xgboost_pka_acidic.json"
PKA_BASE = ROOT / "models" / "adme" / "xgboost_pka_basic.json"


def _clear_all_caches() -> None:
    """Clear every prediction cache so subsequent runs start fresh."""
    from sisyphus.predict import chemistry, adme
    from sisyphus.predict.drugbank import _reset_singleton

    # pKa model cache
    if hasattr(chemistry._load_pka_models, "_cache"):
        del chemistry._load_pka_models._cache

    # logP correction cache
    if hasattr(chemistry.compute_profile, "_logp_model"):
        del chemistry.compute_profile._logp_model

    # ADME model cache
    adme._model_cache.clear()

    # DrugBank singleton
    _reset_singleton()


def _set_pka(enabled: bool) -> None:
    """Enable/disable pKa XGBoost models by renaming files."""
    acid_off = PKA_ACID.with_suffix(".json.off")
    base_off = PKA_BASE.with_suffix(".json.off")

    if enabled:
        # Restore .json.off -> .json if needed
        if acid_off.exists() and not PKA_ACID.exists():
            acid_off.rename(PKA_ACID)
        if base_off.exists() and not PKA_BASE.exists():
            base_off.rename(PKA_BASE)
    else:
        # Rename .json -> .json.off if needed
        if PKA_ACID.exists():
            PKA_ACID.rename(acid_off)
        if PKA_BASE.exists():
            PKA_BASE.rename(base_off)


def _run_experiment(pka_on: bool, bz_on: bool) -> dict:
    """Run benchmark with given pKa/Berezhkovskiy settings.

    Returns dict with engine_aafe, engine_pct2, meta_aafe, meta_pct2, n, n_engine.
    """
    _clear_all_caches()
    _set_pka(pka_on)

    # Monkey-patch kp_method if Berezhkovskiy is OFF
    from sisyphus.predict import ivive

    _real_build = ivive.build_drug_on_graph
    patched = False

    if not bz_on:
        @functools.wraps(_real_build)
        def _patched_build(*args, kp_method="rodgers_rowland", **kwargs):
            return _real_build(*args, kp_method=kp_method, **kwargs)

        ivive.build_drug_on_graph = _patched_build
        patched = True

    try:
        from sisyphus.validation.benchmark import run_benchmark
        result = run_benchmark(holdout_only=True)
    finally:
        # Always restore
        if patched:
            ivive.build_drug_on_graph = _real_build
        _set_pka(True)  # Always restore pKa files

    return {
        "engine_aafe": result.engine_aafe,
        "engine_pct2": result.engine_pct_2fold,
        "meta_aafe": result.aafe,
        "meta_pct2": result.pct_2fold,
        "n": result.n_drugs,
        "n_engine": result.n_engine,
    }


def fmt(val, width=9, decimals=3):
    """Format a value, handling None."""
    if val is None:
        return f"{'N/A':>{width}s}"
    return f"{val:>{width}.{decimals}f}"


def fmt_pct(val, width=9, decimals=1):
    """Format a percentage value, handling None."""
    if val is None:
        return f"{'N/A':>{width}s}"
    return f"{val:>{width}.{decimals}f}%"


def main():
    experiments = [
        ("Baseline (OFF/OFF)", False, False),
        ("Exp 1 (pKa ON/BZ OFF)", True, False),
        ("Exp 2 (pKa OFF/BZ ON)", False, True),
        ("Exp 3 (pKa ON/BZ ON)", True, True),
    ]

    results = []
    t0 = time.time()

    for name, pka_on, bz_on in experiments:
        pka_str = "ON" if pka_on else "OFF"
        bz_str = "ON" if bz_on else "OFF"
        print(f"Running {name} (pKa={pka_str}, BZ={bz_str}) ...")
        t_start = time.time()
        r = _run_experiment(pka_on, bz_on)
        elapsed = time.time() - t_start
        print(f"  Done in {elapsed:.1f}s. N={r['n']}, N_engine={r['n_engine']}")
        results.append((name, r))

    total = time.time() - t0

    # Print results table
    baseline = results[0][1]
    print()
    print("=" * 80)
    print("pKa + BEREZHKOVSKIY EVALUATION")
    print("=" * 80)
    print()

    header = (
        f"{'Experiment':<28s}"
        f"{'Engine AAFE':>12s}"
        f"{'Engine %2f':>11s}"
        f"{'Meta AAFE':>11s}"
        f"{'Meta %2f':>10s}"
        f"{'Δ Engine':>10s}"
    )
    print(header)
    print("-" * 80)

    for name, r in results:
        eng_aafe = r["engine_aafe"]
        eng_pct2 = r["engine_pct2"]
        meta_aafe = r["meta_aafe"]
        meta_pct2 = r["meta_pct2"]

        base_eng = baseline["engine_aafe"]
        if eng_aafe is not None and base_eng is not None and name != results[0][0]:
            delta = eng_aafe - base_eng
            delta_str = f"{delta:>+10.3f}"
        else:
            delta_str = f"{'---':>10s}"

        print(
            f"{name:<28s}"
            f"{fmt(eng_aafe, 12)}"
            f"{fmt_pct(eng_pct2, 11)}"
            f"{fmt(meta_aafe, 11)}"
            f"{fmt_pct(meta_pct2, 10)}"
            f"{delta_str}"
        )

    print()
    print(f"Total time: {total:.1f}s")
    print()


if __name__ == "__main__":
    main()
