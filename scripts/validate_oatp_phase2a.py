#!/usr/bin/env python3
"""Phase 2A OATP1B1 statin engine validation.

For each of 5 statins, compute engine Cmax with OATP kinetics OFF vs ON, and
compare to observed Cmax (where available). Reports fold error vs observation
and the OATP-induced Cmax reduction ratio (cmax_on/cmax_off).

Writes ``data/validation/oatp_phase2a_statins.json``.

Usage::

    python scripts/validate_oatp_phase2a.py
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import sisyphus.engine.flux  # noqa: F401,E402 -- register flux specs
from sisyphus.engine.compiler import ODECompiler, ResolvedParams  # noqa: E402
from sisyphus.engine.solver import solve  # noqa: E402
from sisyphus.graph.builder import build_from_yaml  # noqa: E402
from sisyphus.predict.adme import predict_adme  # noqa: E402
from sisyphus.predict.chemistry import compute_profile  # noqa: E402
from sisyphus.predict.ivive import build_drug_on_graph  # noqa: E402
from sisyphus.predict.transporter_db import load_oatp1b1_kinetics  # noqa: E402
from sisyphus.validation.reference import load_reference  # noqa: E402

_PHYS = ROOT / "data" / "physiology" / "reference_man.yaml"
_OUT = ROOT / "data" / "validation" / "oatp_phase2a_statins.json"

# SMILES hand-curated to match FDA-approved statin acid forms (OATP substrates).
_STATINS = {
    "pravastatin":  "CC[C@@H](C)C(=O)O[C@@H]1C[C@H](C=C2[C@@H]1CC[C@H]([C@@H]2CC[C@H](C[C@H](CC(=O)O)O)O)C)O",
    "rosuvastatin": "CC(C)C1=NC(=NC(=C1C=CC(CC(CC(=O)O)O)O)C2=CC=C(C=C2)F)N(C)S(=O)(=O)C",
    "atorvastatin": "CC(C)C1=C(C(=C(N1CCC(CC(CC(=O)O)O)O)C2=CC=C(C=C2)F)C3=CC=CC=C3)C(=O)NC4=CC=CC=C4",
    "pitavastatin": "O=C(O)C[C@H](O)C[C@H](O)/C=C/c1c(C2CC2)nc2ccccc2c1-c1ccc(F)cc1",
    "fluvastatin":  "CC(C)N1C2=CC=CC=C2C(=C1C=CC(CC(CC(=O)O)O)O)C3=CC=C(C=C3)F",
}


def _simulate_cmax(graph, drug, t_end: float = 24.0) -> float:
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
        return float("nan")
    return float(np.max(result.concentrations["venous_blood"]))


def _fold_error(pred: float, obs: float) -> float:
    if pred <= 0 or obs <= 0 or np.isnan(pred):
        return float("nan")
    return max(pred / obs, obs / pred)


def main() -> None:
    graph = build_from_yaml(_PHYS)
    liver_enzymes = {
        tag: dist.mean for tag, dist in graph.nodes["liver"].enzymes.items()
    }
    refs = {r.name.lower(): r for r in load_reference()}

    results = []
    for name, smiles in _STATINS.items():
        ref = refs.get(name)
        dose = ref.dose_mg if ref else 40.0  # fallback dose for no-ref drugs
        route = ref.route if ref else "oral"
        observed_cmax = ref.cmax_obs if ref else None

        print(f"\n[{name}] dose={dose}mg route={route} observed={observed_cmax}")

        profile = compute_profile(smiles)
        adme = predict_adme(profile)

        drug_off = build_drug_on_graph(
            profile, adme, dose_mg=dose, route=route,
            liver_enzymes=liver_enzymes,
            transporter_kinetics=None,
        )
        drug_on = build_drug_on_graph(
            profile, adme, dose_mg=dose, route=route,
            liver_enzymes=liver_enzymes,
            transporter_kinetics=load_oatp1b1_kinetics(name),
        )

        t0 = time.time()
        cmax_off = _simulate_cmax(graph, drug_off)
        cmax_on = _simulate_cmax(graph, drug_on)
        elapsed = time.time() - t0

        ratio = cmax_on / cmax_off if cmax_off > 0 else float("nan")
        fe_off = _fold_error(cmax_off, observed_cmax) if observed_cmax else None
        fe_on = _fold_error(cmax_on, observed_cmax) if observed_cmax else None

        print(f"  cmax_off = {cmax_off:.4f}  cmax_on = {cmax_on:.4f}  "
              f"ratio = {ratio:.3f}  wall = {elapsed:.1f}s")
        if observed_cmax:
            print(f"  fold_error off → on: {fe_off:.2f} → {fe_on:.2f}")

        results.append({
            "drug": name,
            "dose_mg": dose,
            "route": route,
            "observed_cmax_mg_l": observed_cmax,
            "cmax_off_mg_l": cmax_off,
            "cmax_on_mg_l": cmax_on,
            "oatp_ratio": ratio,
            "fold_error_off": fe_off,
            "fold_error_on": fe_on,
            "wall_seconds": elapsed,
        })

    # summary stats for drugs with observations
    with_obs = [r for r in results if r["observed_cmax_mg_l"] is not None]
    fe_off_arr = np.array([r["fold_error_off"] for r in with_obs])
    fe_on_arr = np.array([r["fold_error_on"] for r in with_obs])

    report = {
        "phase": "OATP Phase 2A statin expansion",
        "drugs": results,
        "summary_with_obs": {
            "n_drugs": len(with_obs),
            "geomean_fold_error_off": float(np.exp(np.mean(np.log(fe_off_arr)))),
            "geomean_fold_error_on": float(np.exp(np.mean(np.log(fe_on_arr)))),
            "drugs_with_obs": [r["drug"] for r in with_obs],
        },
        "sanity_check_oatp_reduces_cmax": {
            r["drug"]: r["oatp_ratio"] < 0.95 for r in results
        },
    }

    _OUT.parent.mkdir(parents=True, exist_ok=True)
    with _OUT.open("w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {_OUT}")
    print(f"  geomean fold error: off={report['summary_with_obs']['geomean_fold_error_off']:.2f}"
          f" → on={report['summary_with_obs']['geomean_fold_error_on']:.2f} (N={len(with_obs)})")


if __name__ == "__main__":
    main()
