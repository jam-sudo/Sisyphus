#!/usr/bin/env python3
"""Ablation study: 5 experiments measuring individual and combined effects."""

from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
FUP_V2 = ROOT / "models" / "adme" / "xgboost_fup_v2.json"
LOGP_CORR = ROOT / "models" / "adme" / "logp_correction.json"

sys.path.insert(0, str(ROOT / "src"))


def hide(path: Path) -> Path | None:
    """Rename .json → .json.bak to hide model. Returns backup path or None."""
    if path.exists():
        bak = path.with_suffix(".json.bak")
        shutil.move(str(path), str(bak))
        return bak
    return None


def restore(bak: Path | None, target: Path) -> None:
    """Restore .json.bak → .json."""
    if bak and bak.exists():
        shutil.move(str(bak), str(target))


def run_benchmark(lookup_on: bool) -> dict:
    """Run holdout benchmark with specified lookup config."""
    # CRITICAL: clear all caches between experiments
    from sisyphus.predict.drugbank import DrugBankConfig, DrugBankLookup, _reset_singleton
    from sisyphus.predict import adme
    adme._model_cache.clear()  # force model reload

    # Clear logP correction cache if exists
    from sisyphus.predict import chemistry
    if hasattr(chemistry.compute_profile, "_logp_model"):
        del chemistry.compute_profile._logp_model

    _reset_singleton()
    config = DrugBankConfig(
        enable_enzyme_fm=lookup_on,
        enable_fup=lookup_on,
        enable_pka=lookup_on,
        enable_logp=lookup_on,
    )
    from sisyphus.predict import drugbank
    drugbank._INSTANCE = DrugBankLookup(config=config)

    from sisyphus.validation.benchmark import run_benchmark as rb
    result = rb(holdout_only=True)
    _reset_singleton()
    return {
        "aafe": result.aafe,
        "aafe_id": result.aafe_in_domain,
        "pct2": result.pct_2fold,
        "n": result.n_drugs,
    }


def main():
    has_fup = FUP_V2.exists()
    has_logp = LOGP_CORR.exists()
    logger.info("Models: fup_v2=%s, logp_correction=%s", has_fup, has_logp)

    if not has_fup:
        logger.error("xgboost_fup_v2.json not found. Run train_fup_v2.py first.")
        return
    if not has_logp:
        logger.error("logp_correction.json not found. Run train_logp_correction.py first.")
        return

    results = {}

    # Exp 1: Baseline (hide both v2 models)
    logger.info("=== Exp 1: Baseline ===")
    fup_bak = hide(FUP_V2)
    logp_bak = hide(LOGP_CORR)
    try:
        results["1_baseline"] = run_benchmark(lookup_on=False)
    finally:
        restore(fup_bak, FUP_V2)
        restore(logp_bak, LOGP_CORR)

    # Exp 2: fup v2 only
    logger.info("=== Exp 2: fup v2 only ===")
    logp_bak = hide(LOGP_CORR)
    try:
        results["2_fup_v2"] = run_benchmark(lookup_on=False)
    finally:
        restore(logp_bak, LOGP_CORR)

    # Exp 3: logP only
    logger.info("=== Exp 3: logP correction only ===")
    fup_bak = hide(FUP_V2)
    try:
        results["3_logp"] = run_benchmark(lookup_on=False)
    finally:
        restore(fup_bak, FUP_V2)

    # Exp 4: Silver (both, no lookup)
    logger.info("=== Exp 4: Silver ===")
    results["4_silver"] = run_benchmark(lookup_on=False)

    # Exp 5: Gold (both + lookup)
    logger.info("=== Exp 5: Gold ===")
    results["5_gold"] = run_benchmark(lookup_on=True)

    # Summary
    print("\n" + "=" * 65)
    print("ABLATION STUDY RESULTS")
    print("=" * 65)
    print(f"{'Experiment':<25s} {'AAFE':>8s} {'In-domain':>10s} {'%2-fold':>8s} {'N':>4s}")
    print("-" * 65)
    for name, r in results.items():
        print(f"{name:<25s} {r['aafe']:8.3f} {r['aafe_id']:10.3f} {r['pct2']:7.1f}% {r['n']:4d}")

    bl = results["1_baseline"]["aafe"]
    sv = results["4_silver"]["aafe"]
    gd = results["5_gold"]["aafe"]
    print(f"\nTraining effect (Silver - Baseline): {sv - bl:+.3f}")
    print(f"Lookup effect  (Gold - Silver):      {gd - sv:+.3f}")
    print(f"Total effect   (Gold - Baseline):    {gd - bl:+.3f}")


if __name__ == "__main__":
    main()
