#!/usr/bin/env python3
"""Per-drug probe for Phase 2A stiff-ODE diagnosis.

Runs ONE statin at a time with unbuffered output so we can pinpoint which
drug stalls LSODA. Invoke with drug name as argv[1].

Usage::

    python -u scripts/probe_oatp_phase2a_per_drug.py pravastatin
"""

from __future__ import annotations

import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import sisyphus.engine.flux  # noqa: F401,E402
from sisyphus.engine.compiler import ODECompiler, ResolvedParams  # noqa: E402
from sisyphus.engine.solver import solve as solve_scipy  # noqa: E402
from sisyphus.engine import solver_jax as _sj  # noqa: E402

# Loosen default tolerances for stiff OATP MM kinetics. Publication-grade but
# converges where rtol=1e-8 LSODA stalls on low-Km statins (rosuvastatin Km=6.5).
_sj._DEFAULT_RTOL = 1e-6
_sj._DEFAULT_ATOL = 1e-8
_sj._DEFAULT_MAX_STEPS = 65536
solve = _sj.solve
from sisyphus.graph.builder import build_from_yaml  # noqa: E402
from sisyphus.predict.adme import predict_adme  # noqa: E402
from sisyphus.predict.chemistry import compute_profile  # noqa: E402
from sisyphus.predict.ivive import build_drug_on_graph  # noqa: E402
from sisyphus.predict.transporter_db import load_oatp1b1_kinetics  # noqa: E402

_PHYS = ROOT / "data" / "physiology" / "reference_man.yaml"

_STATINS = {
    "pravastatin":  ("CC[C@@H](C)C(=O)O[C@@H]1C[C@H](C=C2[C@@H]1CC[C@H]([C@@H]2CC[C@H](C[C@H](CC(=O)O)O)O)C)O", 40.0),
    "rosuvastatin": ("CC(C)C1=NC(=NC(=C1C=CC(CC(CC(=O)O)O)O)C2=CC=C(C=C2)F)N(C)S(=O)(=O)C", 20.0),
    "atorvastatin": ("CC(C)C1=C(C(=C(N1CCC(CC(CC(=O)O)O)O)C2=CC=C(C=C2)F)C3=CC=CC=C3)C(=O)NC4=CC=CC=C4", 40.0),
    "pitavastatin": ("O=C(O)C[C@H](O)C[C@H](O)/C=C/c1c(C2CC2)nc2ccccc2c1-c1ccc(F)cc1", 2.0),
    "fluvastatin":  ("CC(C)N1C2=CC=CC=C2C(=C1C=CC(CC(CC(=O)O)O)O)C3=CC=C(C=C3)F", 40.0),
}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in _STATINS:
        print(f"usage: {sys.argv[0]} <{'|'.join(_STATINS)}> [t_end_hours]", flush=True)
        sys.exit(1)
    name = sys.argv[1]
    t_end = float(sys.argv[2]) if len(sys.argv) >= 3 else 6.0

    smiles, dose = _STATINS[name]
    print(f"[{name}] dose={dose}mg t_end={t_end}h", flush=True)

    t0 = time.time()
    graph = build_from_yaml(_PHYS)
    print(f"  graph built in {time.time()-t0:.1f}s", flush=True)
    liver_enzymes = {
        tag: dist.mean for tag, dist in graph.nodes["liver"].enzymes.items()
    }
    t1 = time.time()
    profile = compute_profile(smiles)
    print(f"  chemistry profile in {time.time()-t1:.1f}s mw={profile.mw:.1f} logp={profile.logp:.2f}", flush=True)
    t2 = time.time()
    adme = predict_adme(profile)
    print(f"  adme predicted in {time.time()-t2:.1f}s fup={adme.fup.mean:.3f} clint={adme.clint.mean:.1f}", flush=True)
    kin = load_oatp1b1_kinetics(name)
    oatp = kin["OATP1B1"]
    print(f"  OATP kinetics: Jmax={oatp.jmax.mean:.1f} Km={oatp.km.mean:.3f} "
          f"total_setup={time.time()-t0:.1f}s", flush=True)

    for mode, transporter in [("OFF", None), ("ON", kin)]:
        drug = build_drug_on_graph(
            profile, adme, dose_mg=dose, route="oral",
            liver_enzymes=liver_enzymes,
            transporter_kinetics=transporter,
        )
        rng = np.random.default_rng(42)
        realized_graph = graph.sample(rng)
        realized_drug = drug.sample(rng)
        compiler = ODECompiler()
        compiled = compiler.compile(realized_graph)
        params = ResolvedParams(realized_graph, realized_drug)
        y0 = np.zeros(compiled.n_states)
        y0[compiled.state_index[drug.administration_node]] = drug.dose_mg

        print(f"  solving mode={mode} ...", flush=True)
        tsolve = time.time()
        result = solve(compiled, params, y0, t_span=(0, t_end))
        elapsed = time.time() - tsolve
        if not result.solver_success:
            print(f"    FAIL mode={mode} wall={elapsed:.1f}s", flush=True)
            continue
        cmax = float(np.max(result.concentrations["venous_blood"]))
        print(f"    mode={mode} Cmax={cmax:.4f} mg/L wall={elapsed:.1f}s "
              f"MBE={result.mass_balance_error:.2e}", flush=True)


if __name__ == "__main__":
    main()
