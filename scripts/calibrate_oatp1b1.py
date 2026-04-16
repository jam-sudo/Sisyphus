"""One-shot calibration for liver.transporters.OATP1B1 abundance.

Runs a log-spaced grid of OATP1B1 abundances and prints Cmax vs the
FDA-label observed value for pravastatin 40 mg oral (0.045 mg/L).
The best abundance (ratio closest to 1.0) is printed at the end.
"""

from __future__ import annotations

import dataclasses
import pathlib
import sys

import numpy as np

import sisyphus.engine.flux  # noqa: F401 -- register flux specs
from sisyphus.core import Distribution
from sisyphus.engine.compiler import ODECompiler, ResolvedParams
from sisyphus.engine.solver import solve
from sisyphus.graph.builder import build_from_yaml
from sisyphus.predict.adme import predict_adme
from sisyphus.predict.chemistry import compute_profile
from sisyphus.predict.ivive import build_drug_on_graph
from sisyphus.predict.transporter_db import load_oatp1b1_kinetics

_PRAVASTATIN = (
    "CC[C@@H](C)C(=O)O[C@@H]1C[C@H](C=C2[C@@H]1CC[C@H]"
    "([C@@H]2CC[C@H](C[C@H](CC(=O)O)O)O)C)O"
)
_OBSERVED_CMAX_MG_L = 0.045
_PHYS = pathlib.Path("data/physiology/reference_man.yaml")


def _simulate_cmax(drug, graph, t_end: float = 24.0) -> float:
    """Run a deterministic ODE solve and return venous_blood Cmax."""
    rng = np.random.default_rng(42)
    realized_graph = graph.sample(rng)
    realized_drug = drug.sample(rng)

    compiler = ODECompiler()
    compiled = compiler.compile(realized_graph)
    params = ResolvedParams(realized_graph, realized_drug)

    y0 = np.zeros(compiled.n_states)
    admin_idx = compiled.state_index[drug.administration_node]
    y0[admin_idx] = drug.dose_mg

    result = solve(compiled, params, y0, t_span=(0, t_end))

    if not result.solver_success:
        print(f"WARNING: solver did not converge", file=sys.stderr)
        return float("nan")

    return float(np.max(result.concentrations["venous_blood"]))


def main() -> None:
    # Load base graph and drug profile once
    base_graph = build_from_yaml(_PHYS)
    profile = compute_profile(_PRAVASTATIN)
    adme = predict_adme(profile)

    # Use liver enzymes from the base graph
    liver_enzymes = {
        tag: dist.mean for tag, dist in base_graph.nodes["liver"].enzymes.items()
    }

    # Build drug with OATP kinetics (fixed across all grid points)
    transporter_kinetics = load_oatp1b1_kinetics("pravastatin")
    drug = build_drug_on_graph(
        profile, adme, dose_mg=40.0, route="oral",
        liver_enzymes=liver_enzymes,
        transporter_kinetics=transporter_kinetics,
    )

    # Grid: fine around the seed 1.8e11, spanning 1e10 to 1e12
    abundances = np.logspace(10, 12, 20)

    print(f"\nCalibrating OATP1B1 abundance vs observed Cmax = {_OBSERVED_CMAX_MG_L:.4f} mg/L")
    print(f"{'abundance':>14s}  {'cmax_mg_l':>10s}  {'ratio_to_obs':>14s}")
    print("-" * 44)

    results: list[tuple[float, float, float]] = []
    for ab in abundances:
        # Create a modified liver node with this abundance
        liver_node = base_graph.nodes["liver"]
        new_transporters = dict(liver_node.transporters)
        new_transporters["OATP1B1"] = Distribution(mean=ab, cv=0.0)
        new_liver = dataclasses.replace(liver_node, transporters=new_transporters)

        # Build a graph copy with the modified liver node
        graph = build_from_yaml(_PHYS)
        graph.nodes["liver"] = new_liver

        cmax = _simulate_cmax(drug, graph)
        ratio = cmax / _OBSERVED_CMAX_MG_L
        results.append((ab, cmax, ratio))
        print(f"{ab:>14.3e}  {cmax:>10.4f}  {ratio:>14.3f}")

    # Find best abundance (ratio closest to 1.0)
    best = min(results, key=lambda r: abs(r[2] - 1.0))
    best_ab, best_cmax, best_ratio = best

    print()
    print(f"Best abundance : {best_ab:.3e}")
    print(f"Best Cmax      : {best_cmax:.4f} mg/L  (observed: {_OBSERVED_CMAX_MG_L:.4f} mg/L)")
    print(f"Ratio          : {best_ratio:.3f}  (ideal 1.000)")

    # Also verify that OATP is actually reducing Cmax (cmax_on < cmax_off)
    drug_off = build_drug_on_graph(
        profile, adme, dose_mg=40.0, route="oral",
        liver_enzymes=liver_enzymes,
        transporter_kinetics=None,
    )
    cmax_off = _simulate_cmax(drug_off, base_graph)
    print()
    print(f"Sanity: cmax_off (no OATP) = {cmax_off:.4f} mg/L")
    print(f"Sanity: cmax_on  (best ab) = {best_cmax:.4f} mg/L")
    print(f"Sanity: ratio on/off       = {best_cmax / cmax_off:.3f}  (must be < 0.95)")

    if best_cmax / cmax_off >= 0.95:
        print("WARNING: on/off ratio >= 0.95 — OATP effect is too weak at best abundance!")


if __name__ == "__main__":
    main()
