#!/usr/bin/env python3
"""Targeted verification: does the conformal CI floor recover rivaroxaban multi-obs coverage?

Runs only the 5 scenarios that the n=2000 baseline failed on:
  ketorolac × {S1, S2, S3}      (engine-level fup error, expected unfixable)
  rivaroxaban × {S2, S3}        (over-tight posterior, expected fixable by floor)

Two evaluations: floor=0.0 (baseline) and floor=0.20 (proposed fix). Compares
coverage and CI widths.

Usage:
  python scripts/verify_tdm_ci_floor.py
"""

from __future__ import annotations

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


CASES = [
    # Easy controls — these were covered in the n=2000 baseline; floor must
    # not destroy them.
    ("morphine",   "CN1CCC23C4C1CC5=C2C(=C(C=C5)O)OC3C(C=C4)O", 30.0, 0.01865, "S1_1h",      [1.0]),
    ("amantadine", "C1C2CC3CC1CC(C2)(C3)N",                     100.0, 0.22,    "S1_1h",      [1.0]),
    ("clozapine",  "CN1CCN(CC1)C2=NC3=C(C=CC(=C3)Cl)NC4=CC=CC=C42", 75.0, 0.413, "S1_1h",     [1.0]),
    # Hard cases — failed in baseline.
    ("ketorolac",  "O=C(c1ccccc1)c1ccc2n1CCC2C(=O)O",  10.0, 0.80, "S1_1h",      [1.0]),
    ("ketorolac",  "O=C(c1ccccc1)c1ccc2n1CCC2C(=O)O",  10.0, 0.80, "S2_1h_4h",   [1.0, 4.0]),
    ("ketorolac",  "O=C(c1ccccc1)c1ccc2n1CCC2C(=O)O",  10.0, 0.80, "S3_1h_4h_8h",[1.0, 4.0, 8.0]),
    ("rivaroxaban", "O=C(NC[C@H]1CN(c2ccc(N3CCOCC3=O)cc2)C(=O)O1)c1ccc(Cl)s1", 10.0, 0.1432, "S1_1h", [1.0]),
    ("rivaroxaban", "O=C(NC[C@H]1CN(c2ccc(N3CCOCC3=O)cc2)C(=O)O1)c1ccc(Cl)s1", 10.0, 0.1432, "S2_1h_4h", [1.0, 4.0]),
    ("rivaroxaban", "O=C(NC[C@H]1CN(c2ccc(N3CCOCC3=O)cc2)C(=O)O1)c1ccc(Cl)s1", 10.0, 0.1432, "S3_1h_4h_8h", [1.0, 4.0, 8.0]),
]

ASSAY_CV = 0.10
SEED = 42
N_PRIOR = 2000


def build_pipeline(smiles: str, dose_mg: float):
    graph = build_from_yaml(ROOT / "data" / "physiology" / "reference_man.yaml")
    compiled = ODECompiler().compile(graph)
    profile = compute_profile(smiles)
    adme = predict_adme(profile)
    le = None
    if "liver" in graph.nodes and graph.nodes["liver"].enzymes:
        le = {t: d.mean for t, d in graph.nodes["liver"].enzymes.items()}
    drug = build_drug_on_graph(profile, adme, dose_mg, "oral", liver_enzymes=le)
    return graph, compiled, drug


def gen_obs(compiled, graph, drug, regimen, cmax_obs, obs_times, seed=SEED):
    rng = np.random.default_rng(seed)
    realized_g = graph.sample(np.random.default_rng(0))
    realized_d = drug.sample(np.random.default_rng(0))
    params = ResolvedParams(realized_g, realized_d)
    sim = solve_regimen(compiled, params, regimen, t_total_h=24.0, dt_output=0.1)
    conc = sim.concentrations["venous_blood"]
    scale = cmax_obs / float(np.max(conc))
    out = []
    for t in obs_times:
        c = float(np.interp(t, sim.time_h, conc)) * scale * rng.lognormal(0, ASSAY_CV)
        out.append(Observation(time_h=t, concentration=c, cv=ASSAY_CV))
    return out


def main():
    from sisyphus.regimen.tdm import apply_ci_floor

    print(f"Verifying CI floor on {len(CASES)} scenarios at n_prior={N_PRIOR}")
    print()

    # Run bayesian_update ONCE per case at floor=0, cache results.
    base_results = {}
    print("Phase 1: running bayesian_update (no floor)...")
    for name, smi, dose, cmax_obs, scen, ts in CASES:
        graph, compiled, drug = build_pipeline(smi, dose)
        regimen = DosingRegimen.single_oral(dose)
        obs = gen_obs(compiled, graph, drug, regimen, cmax_obs, ts)
        t0 = time.time()
        r = bayesian_update(
            compiled, graph, drug, regimen, obs,
            n_prior=N_PRIOR, seed=SEED,
            method="importance_sampling",
            min_ci_half_width_fraction=0.0,
        )
        wall = time.time() - t0
        base_results[(name, scen)] = (r, cmax_obs)
        ci = r.cmax_ci_90 if r.cmax_ci_90 is not None else (0.0, 0.0)
        print(f"  {name:<12s} {scen:<12s} post={r.posterior_cmax.mean:.4f} CI=[{ci[0]:.4f}, {ci[1]:.4f}] ({wall:.0f}s)")

    # Phase 2: apply each floor as post-processing.
    rows = []
    for floor in (0.0, 0.20, 0.50, 1.0):
        print(f"\n=== floor = {floor:.2f} ===")
        n_covered = 0
        for (name, scen), (r, cmax_obs) in base_results.items():
            r_floored = apply_ci_floor(r, floor)
            ci = r_floored.cmax_ci_90 if r_floored.cmax_ci_90 is not None else (0.0, 0.0)
            covered = ci[0] <= cmax_obs <= ci[1]
            if covered:
                n_covered += 1
            mark = "✓" if covered else "✗"
            print(
                f"  {mark} {name:<12s} {scen:<12s}  obs={cmax_obs:.4f}  "
                f"post={r.posterior_cmax.mean:.4f}  CI=[{ci[0]:.4f}, {ci[1]:.4f}]"
            )
            rows.append({
                "drug": name,
                "scenario": scen,
                "floor": floor,
                "cmax_obs": cmax_obs,
                "posterior_cmax_mean": float(r.posterior_cmax.mean),
                "posterior_cmax_cv": float(r.posterior_cmax.cv),
                "ci_lo": float(ci[0]),
                "ci_hi": float(ci[1]),
                "covered": covered,
            })
        print(f"  COVERAGE: {n_covered}/{len(CASES)}  ({n_covered/len(CASES)*100:.0f}%)")

    out = {
        "n_prior": N_PRIOR,
        "cases": rows,
    }
    out_path = ROOT / "data" / "validation" / "tdm_ci_floor_verification.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
