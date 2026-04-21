#!/usr/bin/env python3
"""Calibrate liver.OATP1B1 abundance under the ECM clearance model.

Sweeps abundance on a log grid, runs pravastatin 40 mg oral through the
engine (ECM flux active, hepatic_ecm.json params loaded), reports Cmax
fold-error vs observed 0.045 mg/L. Picks the abundance minimizing
|ln(FE)| with soft preference for PS_active ∈ [0.5, 2.0] L/h
(Watanabe 2009 PS_inf literature range).

Writes ``data/validation/oatp_ecm_abundance_calibration.json`` and prints
the recommended abundance to stdout.

Usage:
    python3 scripts/calibrate_oatp_abundance_ecm.py
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
from sisyphus.engine.solver import solve  # noqa: E402
from sisyphus.graph.body import BodyGraph  # noqa: E402
from sisyphus.graph.builder import build_from_yaml  # noqa: E402
from sisyphus.predict.adme import predict_adme  # noqa: E402
from sisyphus.predict.chemistry import compute_profile  # noqa: E402
from sisyphus.predict.ivive import build_drug_on_graph  # noqa: E402
from sisyphus.predict.transporter_db import (  # noqa: E402
    load_hepatic_ecm_params,
    load_oatp1b1_kinetics,
)

_PHYS = ROOT / "data" / "physiology" / "reference_man.yaml"
_OUT = ROOT / "data" / "validation" / "oatp_ecm_abundance_calibration.json"
_PRAVA_SMILES = (
    "CC[C@@H](C)C(=O)O[C@@H]1C[C@H](C=C2[C@@H]1CC[C@H]"
    "([C@@H]2CC[C@H](C[C@H](CC(=O)O)O)O)C)O"
)
_OBS_CMAX = 0.045  # mg/L, 40 mg oral pravastatin (FDA label)

# Abundance grid calibrated for the ECM formula unit system.
# PS_active = abundance × (Jmax/Km) × ivive_scaling
# = abundance × (228/13.6) × 6e-5 ≈ abundance × 1.006e-3  L/h
# Watanabe 2009 PS_inf target: 0.5–2.0 L/h → abundance ≈ 500–2000.
# Extended grid to cover FE minimum empirically (literature-range soft pref).
_ABUNDANCES = [1e4, 1e5, 3e5, 5e5, 7e5, 1e6, 3e6, 1e7, 1e8]


def _set_oatp(graph: BodyGraph, value: float) -> BodyGraph:
    liver = graph.nodes["liver"]
    old = liver.transporters["OATP1B1"]
    new_tr = dict(liver.transporters)
    new_tr["OATP1B1"] = Distribution(
        mean=value, cv=old.cv, dist_type=old.dist_type,
    )
    new_liver = replace(liver, transporters=new_tr)
    g = BodyGraph()
    g.nodes = dict(graph.nodes)
    g.nodes["liver"] = new_liver
    g.edges = list(graph.edges)
    g.global_params = dict(graph.global_params)
    return g


def _ps_active_est(abundance: float, jmax: float, km: float, ivive: float) -> float:
    """PS_active = abundance × Jmax/Km × ivive (linear regime)."""
    return abundance * jmax / km * ivive


def _cmax(graph, drug, t_end: float = 24.0):
    rng = np.random.default_rng(42)
    rg = graph.sample(rng)
    rd = drug.sample(rng)
    compiler = ODECompiler()
    compiled = compiler.compile(rg)
    params = ResolvedParams(rg, rd)
    y0 = np.zeros(compiled.n_states)
    y0[compiled.state_index[drug.administration_node]] = drug.dose_mg
    t0 = time.time()
    result = solve(compiled, params, y0, t_span=(0, t_end))
    wall = time.time() - t0
    if not result.solver_success:
        return float("nan"), wall
    return float(np.max(result.concentrations["venous_blood"])), wall


def main() -> None:
    base = build_from_yaml(_PHYS)
    profile = compute_profile(_PRAVA_SMILES)
    adme = predict_adme(profile)
    oatp_kin = load_oatp1b1_kinetics("pravastatin")
    ecm_params = load_hepatic_ecm_params("pravastatin")

    sweep = []
    for abundance in _ABUNDANCES:
        print(f"\n=== abundance = {abundance:.2e} ===", flush=True)
        graph = _set_oatp(base, abundance)
        liver_enz = {t: d.mean for t, d in graph.nodes["liver"].enzymes.items()}
        drug = build_drug_on_graph(
            profile, adme, dose_mg=40.0, route="oral",
            liver_enzymes=liver_enz,
            transporter_kinetics=oatp_kin,
            hepatic_ecm_params=ecm_params,
        )
        cmax, wall_s = _cmax(graph, drug)
        fe = max(cmax / _OBS_CMAX, _OBS_CMAX / cmax) if cmax > 0 else float("nan")

        j = oatp_kin["OATP1B1"]
        ps_act = _ps_active_est(
            abundance=abundance,
            jmax=j.jmax.mean,
            km=j.km.mean,
            ivive=graph.nodes["liver"].ivive_scaling,
        )
        print(
            f"  Cmax = {cmax:.4f} mg/L, FE = {fe:.3f}, "
            f"PS_active = {ps_act:.3f} L/h, wall = {wall_s:.2f}s",
            flush=True,
        )
        sweep.append({
            "abundance": abundance,
            "cmax_mg_L": cmax,
            "observed_cmax_mg_L": _OBS_CMAX,
            "fold_error": fe,
            "ps_active_L_per_h_linear_est": ps_act,
            "wall_s": wall_s,
        })

    in_lit = [
        s for s in sweep
        if 0.5 <= s["ps_active_L_per_h_linear_est"] <= 2.0
        and np.isfinite(s["fold_error"])
    ]
    cands = in_lit if in_lit else [s for s in sweep if np.isfinite(s["fold_error"])]
    if not cands:
        print("\nAll sweep points failed or produced NaN Cmax.", flush=True)
        sys.exit(2)
    best = min(cands, key=lambda s: abs(np.log(s["fold_error"])))

    _OUT.parent.mkdir(parents=True, exist_ok=True)
    with _OUT.open("w") as f:
        json.dump({
            "phase": "OATP ECM abundance calibration (pravastatin)",
            "sweep": sweep,
            "recommended_abundance": best["abundance"],
            "recommended_fold_error": best["fold_error"],
            "recommended_ps_active": best["ps_active_L_per_h_linear_est"],
            "ps_active_in_literature_range": bool(in_lit),
        }, f, indent=2)
    print(
        f"\nRecommended abundance: {best['abundance']:.2e} "
        f"(FE={best['fold_error']:.3f}, "
        f"PS_active={best['ps_active_L_per_h_linear_est']:.3f} L/h)",
        flush=True,
    )
    print(f"Report written to {_OUT}", flush=True)


if __name__ == "__main__":
    main()
