#!/usr/bin/env python3
"""Abundance re-calibration sweep for OATP1B1 hepatic uptake.

Goal: find abundance that (a) preserves pravastatin Phase 1 calibration
(Cmax target ~0.045, ratio 0.5-2.0 vs observed OK), and (b) lets stiff
non-pravastatin statins converge within 10 seconds/solve.

Sweeps abundance on the reference_man graph post-build, and for each value
solves pravastatin and rosuvastatin (our canary stiff drug) in both OFF/ON
transporter modes.

Writes ``data/validation/oatp_abundance_sweep.json``.
"""

from __future__ import annotations

import json
import pathlib
import sys
import time
from dataclasses import replace

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import sisyphus.engine.flux  # noqa: F401,E402
from sisyphus.core import Distribution  # noqa: E402
from sisyphus.engine.compiler import ODECompiler, ResolvedParams  # noqa: E402
from sisyphus.engine.solver import solve as solve_scipy  # noqa: E402
from sisyphus.graph.builder import build_from_yaml  # noqa: E402
from sisyphus.graph.body import BodyGraph  # noqa: E402
from sisyphus.predict.adme import predict_adme  # noqa: E402
from sisyphus.predict.chemistry import compute_profile  # noqa: E402
from sisyphus.predict.ivive import build_drug_on_graph  # noqa: E402
from sisyphus.predict.transporter_db import load_oatp1b1_kinetics  # noqa: E402

_PHYS = ROOT / "data" / "physiology" / "reference_man.yaml"
_OUT = ROOT / "data" / "validation" / "oatp_abundance_sweep.json"

# (name, smiles, dose, observed_cmax or None)
_CANARIES = [
    ("pravastatin",  "CC[C@@H](C)C(=O)O[C@@H]1C[C@H](C=C2[C@@H]1CC[C@H]([C@@H]2CC[C@H](C[C@H](CC(=O)O)O)O)C)O", 40.0, 0.045),
    ("rosuvastatin", "CC(C)C1=NC(=NC(=C1C=CC(CC(CC(=O)O)O)O)C2=CC=C(C=C2)F)N(C)S(=O)(=O)C", 20.0, 0.0066),
]

_ABUNDANCES = [1.0e9, 3.0e9, 1.0e10, 3.0e10, 1.0e11]


def _set_abundance(graph: BodyGraph, value: float) -> BodyGraph:
    """Return new graph with liver.OATP1B1 abundance replaced."""
    liver = graph.nodes["liver"]
    old = liver.transporters["OATP1B1"]
    new_transporters = dict(liver.transporters)
    new_transporters["OATP1B1"] = Distribution(
        mean=value, cv=old.cv, dist_type=old.dist_type,
    )
    new_liver = replace(liver, transporters=new_transporters)
    g = BodyGraph()
    g.nodes = dict(graph.nodes)
    g.nodes["liver"] = new_liver
    g.edges = list(graph.edges)
    g.global_params = dict(graph.global_params)
    return g


def _solve_once(graph, drug, t_end: float, timeout_s: float) -> dict:
    rng = np.random.default_rng(42)
    realized_graph = graph.sample(rng)
    realized_drug = drug.sample(rng)
    compiler = ODECompiler()
    compiled = compiler.compile(realized_graph)
    params = ResolvedParams(realized_graph, realized_drug)
    y0 = np.zeros(compiled.n_states)
    y0[compiled.state_index[drug.administration_node]] = drug.dose_mg

    t0 = time.time()
    try:
        # SciPy solver doesn't accept a hard timeout kwarg; rely on caller
        # orchestration or small t_end. Here we just attempt and record
        # wall time.
        result = solve_scipy(compiled, params, y0, t_span=(0, t_end))
    except Exception as exc:
        return {"success": False, "error": str(exc), "wall_s": time.time() - t0}
    wall = time.time() - t0
    if not result.solver_success:
        return {"success": False, "error": "solver_success=False", "wall_s": wall}
    cmax = float(np.max(result.concentrations["venous_blood"]))
    return {"success": True, "cmax": cmax, "wall_s": wall, "mbe": result.mass_balance_error}


def main() -> None:
    base_graph = build_from_yaml(_PHYS)
    results = []

    for abundance in _ABUNDANCES:
        print(f"\n=== abundance = {abundance:.2e} ===", flush=True)
        graph = _set_abundance(base_graph, abundance)
        liver_enzymes = {
            tag: dist.mean for tag, dist in graph.nodes["liver"].enzymes.items()
        }
        per_drug = []

        for name, smiles, dose, observed in _CANARIES:
            print(f"  [{name}] dose={dose}mg observed={observed}", flush=True)
            profile = compute_profile(smiles)
            adme = predict_adme(profile)

            drug_off = build_drug_on_graph(
                profile, adme, dose_mg=dose, route="oral",
                liver_enzymes=liver_enzymes, transporter_kinetics=None,
            )
            drug_on = build_drug_on_graph(
                profile, adme, dose_mg=dose, route="oral",
                liver_enzymes=liver_enzymes,
                transporter_kinetics=load_oatp1b1_kinetics(name),
            )

            # Use t_end=4h, enough for oral Cmax to happen (Tmax 1-2h typical)
            res_off = _solve_once(graph, drug_off, t_end=4.0, timeout_s=15.0)
            print(f"    OFF: {res_off}", flush=True)
            res_on = _solve_once(graph, drug_on, t_end=4.0, timeout_s=15.0)
            print(f"    ON:  {res_on}", flush=True)

            fold_err_on = None
            if res_on.get("success") and observed:
                fold_err_on = max(res_on["cmax"] / observed, observed / res_on["cmax"])
                print(f"    fold_error_ON: {fold_err_on:.2f}", flush=True)

            per_drug.append({
                "drug": name,
                "dose_mg": dose,
                "observed_cmax": observed,
                "off": res_off,
                "on": res_on,
                "fold_error_on": fold_err_on,
            })

        results.append({"abundance": abundance, "drugs": per_drug})

    _OUT.parent.mkdir(parents=True, exist_ok=True)
    with _OUT.open("w") as f:
        json.dump({
            "phase": "OATP abundance re-calibration sweep",
            "sweep": results,
        }, f, indent=2)
    print(f"\nwrote {_OUT}", flush=True)


if __name__ == "__main__":
    main()
