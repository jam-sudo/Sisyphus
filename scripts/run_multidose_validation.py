#!/usr/bin/env python3
"""Multi-dose validation — Track B.

Runs multi-dose regimen simulations for drugs with known steady-state
PK from FDA labels and compares Css_max and accumulation ratio.

This validates the regimen solver's correctness, not the engine's
ADME predictions — the solver must correctly handle dose accumulation,
state vector injection, and segment concatenation.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

import sisyphus.engine.flux  # noqa: F401
from sisyphus.engine.compiler import ODECompiler, ResolvedParams
from sisyphus.graph.builder import build_from_yaml
from sisyphus.pk.endpoints import compute_endpoints
from sisyphus.predict.adme import predict_adme
from sisyphus.predict.chemistry import compute_profile
from sisyphus.predict.ivive import build_drug_on_graph
from sisyphus.regimen.solver import solve_regimen
from sisyphus.regimen.profile import compute_steady_state_metrics
from sisyphus.regimen.types import DosingRegimen

_PHYSIOLOGY_DIR = _REPO / "data" / "physiology"


@dataclass
class ValidationDrug:
    """Drug with known multi-dose PK from FDA labels."""
    name: str
    smiles: str
    dose_mg: float
    interval_h: float
    n_doses: int
    css_max_fda: float      # mg/L from FDA label
    accum_ratio_lit: float  # literature accumulation ratio
    t_half_lit_h: float     # literature t½ (hours)
    source: str             # FDA label reference


# FDA label steady-state data
# Sources: FDA-approved prescribing information (Drugs@FDA)
VALIDATION_DRUGS = [
    ValidationDrug(
        name="ciprofloxacin",
        smiles="O=C(O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O",
        dose_mg=500, interval_h=12, n_doses=14,
        css_max_fda=3.5,       # FDA label: "Css_max 3.5 µg/mL at 500mg q12h"
        accum_ratio_lit=1.2,   # minimal accumulation (t½ < interval)
        t_half_lit_h=4.0,
        source="Cipro FDA label §12.3",
    ),
    ValidationDrug(
        name="montelukast",
        smiles="CC(C)(O)c1ccccc1CC/C(=C\\c1cccc(/C=C/c2ccc3ccc(CC(=O)O)c(Cl)c3c2)c1)SCC1(CC(=O)O)CC1",
        dose_mg=10, interval_h=24, n_doses=14,
        css_max_fda=0.56,      # FDA label: "mean Css_max 0.56 µg/mL at 10mg QD"
        accum_ratio_lit=1.14,  # minimal (t½ ~5.5h)
        t_half_lit_h=5.5,
        source="Singulair FDA label §12.3",
    ),
    ValidationDrug(
        name="carbamazepine",
        smiles="NC(=O)N1c2ccccc2C=Cc2ccccc21",
        dose_mg=200, interval_h=12, n_doses=14,
        css_max_fda=6.0,       # FDA label: 4-12 µg/mL therapeutic range, ~6 typical
        accum_ratio_lit=2.0,   # moderate (initial t½ ~25h, auto-induces to ~12h)
        t_half_lit_h=16.0,     # average after mild autoinduction
        source="Tegretol FDA label §12.3",
    ),
    ValidationDrug(
        name="famotidine",
        smiles="NC(=N)NC1=NC(CSCCC(=N)NS(N)(=O)=O)=CS1",
        dose_mg=20, interval_h=12, n_doses=14,
        css_max_fda=0.09,      # FDA label: "Css_max 70-100 ng/mL at 20mg BID"
        accum_ratio_lit=1.0,   # negligible (t½ 2.5-3.5h)
        t_half_lit_h=3.0,
        source="Pepcid FDA label §12.3",
    ),
    ValidationDrug(
        name="cetirizine",
        smiles="OC(=O)COCCN1CCN(C(c2ccccc2)c2ccc(Cl)cc2)CC1",
        dose_mg=10, interval_h=24, n_doses=14,
        css_max_fda=0.31,      # FDA label: "Css_max ~310 ng/mL at 10mg QD"
        accum_ratio_lit=1.05,  # minimal (t½ ~8h)
        t_half_lit_h=8.0,
        source="Zyrtec FDA label §12.3",
    ),
]


def run_single_drug(drug: ValidationDrug, graph, compiled, liver_enzymes):
    """Run single-dose and multi-dose for one validation drug."""
    profile = compute_profile(drug.smiles)
    adme = predict_adme(profile)
    dog = build_drug_on_graph(profile, adme, drug.dose_mg, "oral", liver_enzymes=liver_enzymes)

    rng = np.random.default_rng(42)
    realized_graph = graph.sample(rng)
    realized_drug = dog.sample(rng)
    params = ResolvedParams(realized_graph, realized_drug)

    # Single dose
    regimen_single = DosingRegimen.single_oral(drug.dose_mg)
    sim_single = solve_regimen(compiled, params, regimen_single, t_total_h=24.0)
    pk_single = compute_endpoints(sim_single)

    # Multi-dose
    regimen_multi = DosingRegimen.oral_repeated(drug.dose_mg, drug.interval_h, drug.n_doses)
    t_total = drug.interval_h * drug.n_doses + 24.0
    sim_multi = solve_regimen(compiled, params, regimen_multi, t_total_h=t_total)

    # Steady-state metrics
    ss = compute_steady_state_metrics(sim_multi, regimen_multi)

    return pk_single, sim_multi, ss


def main():
    print("Building graph and compiling ODE...")
    graph = build_from_yaml(_PHYSIOLOGY_DIR / "reference_man.yaml")
    compiler = ODECompiler()
    compiled = compiler.compile(graph)

    liver_enzymes = {}
    if "liver" in graph.nodes and graph.nodes["liver"].enzymes:
        liver_enzymes = {tag: d.mean for tag, d in graph.nodes["liver"].enzymes.items()}

    print(f"\n{'='*80}")
    print(f"{'Drug':<18s} {'Dose':<12s} {'C1max':>8s} {'Css_max':>8s} {'FDA':>8s} "
          f"{'Fold':>6s} {'AR_pred':>7s} {'AR_lit':>6s} {'SS?':>4s} {'Verdict':>8s}")
    print(f"{'='*80}")

    results = []
    for drug in VALIDATION_DRUGS:
        try:
            pk1, sim_multi, ss = run_single_drug(drug, graph, compiled, liver_enzymes)

            c1_max = pk1.cmax.mean
            css_max = ss.css_max
            fold = css_max / drug.css_max_fda if drug.css_max_fda > 0 else float("inf")
            within_2fold = 0.5 <= fold <= 2.0
            verdict = "PASS" if within_2fold else "FAIL"

            print(f"{drug.name:<18s} {drug.dose_mg:>4.0f}mg q{drug.interval_h:.0f}h "
                  f"{c1_max:>8.4f} {css_max:>8.4f} {drug.css_max_fda:>8.4f} "
                  f"{fold:>6.2f} {ss.accumulation_ratio:>7.2f} {drug.accum_ratio_lit:>6.2f} "
                  f"{'Y' if ss.is_steady_state else 'N':>4s} {verdict:>8s}")

            results.append({
                "name": drug.name, "fold": fold, "within_2fold": within_2fold,
                "ar_pred": ss.accumulation_ratio, "ar_lit": drug.accum_ratio_lit,
                "ss": ss.is_steady_state,
            })
        except Exception as e:
            print(f"{drug.name:<18s} ERROR: {e}")

    print(f"{'='*80}")

    n_pass = sum(1 for r in results if r["within_2fold"])
    n_total = len(results)
    print(f"\nCss_max within 2-fold: {n_pass}/{n_total}")

    # Accumulation ratio check
    ar_errors = []
    for r in results:
        if r["ar_lit"] > 0:
            ar_errors.append(abs(r["ar_pred"] - r["ar_lit"]) / r["ar_lit"])
    if ar_errors:
        pct_within_50 = 100 * sum(1 for e in ar_errors if e < 0.5) / len(ar_errors)
        print(f"Accumulation ratio within ±50%: {sum(1 for e in ar_errors if e < 0.5)}/{len(ar_errors)} ({pct_within_50:.0f}%)")


if __name__ == "__main__":
    main()
