#!/usr/bin/env python3
"""Focused TDM 90% CI coverage evaluator for Track D2.

Runs the 5 multi-drug benchmark scenarios (morphine, amantadine, ketorolac,
clozapine, rivaroxaban) × 3 observation schedules (1h / 1h+4h / 1h+4h+8h)
= 15 runs, measures 90% CI containment of the synthetic truth Cmax, and
reports per-method coverage.

Unlike ``scripts/run_tdm_benchmark.py`` this script skips the timepoint
and seed sensitivity sections so it finishes in 5–10 minutes per method
rather than 1–2 hours.

Usage::

    python scripts/evaluate_tdm_ci_coverage.py --method ibis
    python scripts/evaluate_tdm_ci_coverage.py --methods is,ibis,enkf
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)

logging.basicConfig(level=logging.WARNING)

import sisyphus.engine.flux  # noqa: F401
from sisyphus.engine.compiler import ODECompiler, ResolvedParams
from sisyphus.graph.builder import build_from_yaml
from sisyphus.predict.adme import predict_adme
from sisyphus.predict.chemistry import compute_profile
from sisyphus.predict.ivive import build_drug_on_graph
from sisyphus.regimen.solver import solve_regimen
from sisyphus.regimen.tdm import Observation, bayesian_update
from sisyphus.regimen.types import DosingRegimen

PHYSIOLOGY_YAML = ROOT / "data" / "physiology" / "reference_man.yaml"
DEFAULT_OUT = ROOT / "data" / "validation" / "tdm_ci_coverage.json"

N_PRIOR = 2000
ASSAY_CV = 0.10
BASE_SEED = 42

DRUGS = {
    "morphine": {
        "smiles": "CN1CCC23C4C1CC5=C2C(=C(C=C5)O)OC3C(C=C4)O",
        "dose_mg": 30.0,
        "cmax_obs": 0.01865,
        "cmax_engine": 0.03736,
        "compound_type": "base",
    },
    "amantadine": {
        "smiles": "C1C2CC3CC1CC(C2)(C3)N",
        "dose_mg": 100.0,
        "cmax_obs": 0.22,
        "cmax_engine": 0.47592,
        "compound_type": "base",
    },
    "ketorolac": {
        "smiles": "O=C(c1ccccc1)c1ccc2n1CCC2C(=O)O",
        "dose_mg": 10.0,
        "cmax_obs": 0.80,
        "cmax_engine": 0.24623,
        "compound_type": "acid",
    },
    "clozapine": {
        "smiles": "CN1CCN(CC1)C2=NC3=C(C=CC(=C3)Cl)NC4=CC=CC=C42",
        "dose_mg": 75.0,
        "cmax_obs": 0.413,
        "cmax_engine": 0.87545,
        "compound_type": "neutral",
    },
    "rivaroxaban": {
        "smiles": "O=C(NC[C@H]1CN(c2ccc(N3CCOCC3=O)cc2)C(=O)O1)c1ccc(Cl)s1",
        "dose_mg": 10.0,
        "cmax_obs": 0.1432,
        "cmax_engine": 0.06589,
        "compound_type": "neutral",
    },
}

SCENARIOS = {
    "S1_1h":      [1.0],
    "S2_1h_4h":   [1.0, 4.0],
    "S3_1h_4h_8h": [1.0, 4.0, 8.0],
}


def build_pipeline(smiles: str, dose_mg: float):
    graph = build_from_yaml(PHYSIOLOGY_YAML)
    compiled = ODECompiler().compile(graph)
    profile = compute_profile(smiles)
    adme = predict_adme(profile)
    liver_enzymes = None
    if "liver" in graph.nodes and graph.nodes["liver"].enzymes:
        liver_enzymes = {t: d.mean for t, d in graph.nodes["liver"].enzymes.items()}
    drug = build_drug_on_graph(profile, adme, dose_mg, "oral", liver_enzymes=liver_enzymes)
    return graph, compiled, drug


def generate_scaled_observations(
    compiled, graph, drug, regimen, cmax_obs, cmax_engine, obs_times, seed=42
):
    rng = np.random.default_rng(seed)
    realized_graph = graph.sample(np.random.default_rng(0))
    realized_drug = drug.sample(np.random.default_rng(0))
    params = ResolvedParams(realized_graph, realized_drug)
    t_total = max(max(obs_times) + 12.0, 24.0)
    sim = solve_regimen(compiled, params, regimen, t_total_h=t_total, dt_output=0.1)
    if not sim.solver_success:
        return None
    conc = sim.concentrations.get("venous_blood")
    if conc is None or len(conc) == 0:
        return None
    engine_cmax_sim = float(np.max(conc))
    if engine_cmax_sim <= 0:
        return None
    scale = cmax_obs / engine_cmax_sim
    observations = []
    for t in obs_times:
        c_engine = float(np.interp(t, sim.time_h, conc))
        c_scaled = c_engine * scale
        noise = rng.lognormal(0, ASSAY_CV)
        c_measured = max(c_scaled * noise, 1e-8)
        observations.append(Observation(time_h=t, concentration=c_measured, cv=ASSAY_CV))
    return observations


def run_method_on_drug(
    method: str, compiled, graph, drug, regimen, observations,
    seed: int = 42, n_prior: int = N_PRIOR, min_ci_floor: float = 0.0,
):
    result = bayesian_update(
        compiled, graph, drug, regimen,
        observations=observations,
        n_prior=n_prior,
        seed=seed,
        method=method,
        min_ci_half_width_fraction=min_ci_floor,
    )
    if result.cmax_ci_90 is not None:
        ci_lo, ci_hi = result.cmax_ci_90
    else:
        ci_lo = ci_hi = float(result.posterior_cmax.mean)
    return result, ci_lo, ci_hi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--methods", default="importance_sampling",
        help="comma-separated list from {importance_sampling, ibis, enkf}",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n-prior", type=int, default=N_PRIOR)
    parser.add_argument(
        "--min-ci-floor",
        type=float,
        default=0.0,
        help="conformal CI half-width floor as fraction of posterior mean",
    )
    args = parser.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    print(f"Evaluating CI coverage for methods: {methods}")

    report: dict = {
        "n_prior": args.n_prior,
        "min_ci_floor": args.min_ci_floor,
        "assay_cv": ASSAY_CV,
        "methods": {},
    }

    for method in methods:
        print()
        print(f"=== method = {method} ===")
        t0 = time.time()
        runs = []
        cover_flags = []
        for drug_name, info in DRUGS.items():
            graph, compiled, drug = build_pipeline(info["smiles"], info["dose_mg"])
            regimen = DosingRegimen.single_oral(info["dose_mg"])
            for scen_name, obs_times in SCENARIOS.items():
                obs = generate_scaled_observations(
                    compiled, graph, drug, regimen,
                    info["cmax_obs"], info["cmax_engine"],
                    obs_times, seed=BASE_SEED,
                )
                if obs is None:
                    continue
                result, ci_lo, ci_hi = run_method_on_drug(
                    method, compiled, graph, drug, regimen, obs,
                    seed=BASE_SEED, n_prior=args.n_prior,
                    min_ci_floor=args.min_ci_floor,
                )
                covered = bool(ci_lo <= info["cmax_obs"] <= ci_hi)
                cover_flags.append(covered)
                rec = {
                    "drug": drug_name,
                    "scenario": scen_name,
                    "n_obs": len(obs_times),
                    "cmax_obs": info["cmax_obs"],
                    "prior_cmax_mean": float(result.prior_cmax.mean),
                    "prior_cmax_cv": float(result.prior_cmax.cv),
                    "posterior_cmax_mean": float(result.posterior_cmax.mean),
                    "posterior_cmax_cv": float(result.posterior_cmax.cv),
                    "ci90_lo": float(ci_lo),
                    "ci90_hi": float(ci_hi),
                    "covered": covered,
                    "ess": float(result.ess),
                    "n_successful": int(result.n_successful),
                }
                runs.append(rec)
                mark = "✅" if covered else "❌"
                print(
                    f"  {mark} {drug_name:<12s} {scen_name:<12s}  "
                    f"obs={info['cmax_obs']:.4f}  "
                    f"CI=[{ci_lo:.4f}, {ci_hi:.4f}]  "
                    f"post={result.posterior_cmax.mean:.4f} (CV={result.posterior_cmax.cv*100:.0f}%)  "
                    f"ESS={result.ess:.0f}"
                )
        n = len(cover_flags)
        n_covered = int(sum(cover_flags))
        coverage = n_covered / n if n > 0 else 0.0
        elapsed = time.time() - t0
        print(
            f"  SUMMARY [{method}]: {n_covered}/{n} = {coverage*100:.0f}%  "
            f"({elapsed:.0f}s wall)"
        )
        report["methods"][method] = {
            "n_runs": n,
            "n_covered": n_covered,
            "coverage": coverage,
            "wall_seconds": elapsed,
            "runs": runs,
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print()
    print(f"Wrote {args.out}")

    # Final cross-method comparison
    print()
    print("Cross-method coverage:")
    for method, m in report["methods"].items():
        print(f"  {method:<20s} {m['n_covered']}/{m['n_runs']} = {m['coverage']*100:.0f}%  ({m['wall_seconds']:.0f}s)")


if __name__ == "__main__":
    main()
