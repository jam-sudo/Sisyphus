#!/usr/bin/env python3
"""Simulation-Based Calibration for the trained amortizer.

Runs SBC using the same simulator that generated training data.  This is
the kill-switch gate for POC: if SBC fails (coverage > tolerance off
nominal, or KS p<0.01 on any dim), the amortizer is miscalibrated and we
need to revisit the architecture before downstream validation.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("sbi_sbc")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sisyphus.sbi.amortizer import load_result  # noqa: E402
from sisyphus.sbi.priors import build_box_prior  # noqa: E402
from sisyphus.sbi.sbc import run_sbc  # noqa: E402
from sisyphus.sbi.simulator import EngineSimulator  # noqa: E402

DEFAULT_SMILES = "CN1CCC23C4C1CC5=C2C(=C(C=C5)O)OC3C(C=C4)O"  # morphine
DEFAULT_DOSE_MG = 30.0
DEFAULT_ROUTE = "oral"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--posterior", type=Path, default=ROOT / "models" / "sbi" / "morphine_posterior.pt"
    )
    parser.add_argument("--smiles", default=DEFAULT_SMILES)
    parser.add_argument("--dose-mg", type=float, default=DEFAULT_DOSE_MG)
    parser.add_argument("--route", default=DEFAULT_ROUTE, choices=("oral", "iv"))
    parser.add_argument("--n-calibration", type=int, default=300)
    parser.add_argument("--n-posterior-samples", type=int, default=500)
    parser.add_argument(
        "--out", type=Path, default=ROOT / "data" / "validation" / "sbi_sbc_morphine.json"
    )
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    log.info("Loading trained posterior: %s", args.posterior)
    result = load_result(args.posterior)

    log.info("Building simulator (%s, %.1f mg %s)", args.smiles[:40], args.dose_mg, args.route)
    sim = EngineSimulator.for_drug(args.smiles, args.dose_mg, route=args.route)
    prior = build_box_prior()

    def simulator_fn(theta: np.ndarray) -> float:
        return sim.simulate_single(theta.astype(np.float64), seed=int(np.random.randint(0, 2**31 - 1)))

    def prior_sample_fn(n: int) -> np.ndarray:
        rng = np.random.default_rng(args.seed)
        return prior.sample_numpy(n, rng).astype(np.float64)

    log.info(
        "Running SBC: %d calibration draws × %d posterior samples",
        args.n_calibration,
        args.n_posterior_samples,
    )
    t0 = time.time()
    sbc = run_sbc(
        posterior=result.posterior,
        simulator_fn=simulator_fn,
        prior_sample_fn=prior_sample_fn,
        n_calibration=args.n_calibration,
        n_posterior_samples=args.n_posterior_samples,
        seed=args.seed,
    )
    elapsed = time.time() - t0
    log.info("SBC done in %.1fs", elapsed)
    print()
    print(sbc.summary_str())
    gate = sbc.passes(ks_threshold=0.001, coverage_tol=0.10)
    print(f"\nGate passed (KS p>0.001, coverage within 0.10 of nominal): {gate}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "n_calibration": int(sbc.n_calibration),
        "n_posterior_samples": int(sbc.n_posterior_samples),
        "ks_pvalues": sbc.ks_pvalues.tolist(),
        "ks_stats": sbc.ks_stats.tolist(),
        "coverage": {str(k): v.tolist() for k, v in sbc.coverage.items()},
        "gate_passed": bool(gate),
        "elapsed_seconds": float(elapsed),
    }
    args.out.write_text(json.dumps(report, indent=2))
    log.info("Wrote %s", args.out)

    # Non-zero exit if gate fails (lets CI or orchestration detect)
    sys.exit(0 if gate else 2)


if __name__ == "__main__":
    main()
