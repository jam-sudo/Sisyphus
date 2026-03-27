#!/usr/bin/env python3
"""Proof of Concept: measured ADME vs predicted ADME in the engine.

Demonstrates that the engine architecture is sound by showing that
replacing XGBoost-predicted ADME with measured literature values
improves engine Cmax accuracy.

This is a proof of concept, not a statistical validation.
N=8-12 drugs, engine-only mode, no meta-learner.
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import dataclasses
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

_dc_replace = dataclasses.replace
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Measured ADME data (literature + TDC) ─────────────────────────────────
# CLint: µL/min/10⁶ hepatocytes (TDC Hepatocyte_AZ, geometric mean of replicates)
# fup: DrugBank experimental fraction unbound in plasma
# Peff: not available for most drugs → Tier 2 (keep predicted)

@dataclass
class MeasuredADME:
    drug: str
    fup: float
    fup_source: str
    clint_hep: float  # µL/min/10⁶ cells
    clint_n_sources: int
    clint_source: str
    peff: float | None = None  # ×10⁻⁴ cm/s, None = keep predicted
    peff_source: str = ""
    tier: str = "2"  # "1A", "1B", or "2"

MEASURED = [
    MeasuredADME("alprazolam", 0.20, "DrugBank", 13.0, 2,
                 "TDC Hepatocyte_AZ (gmean, N=2, range 4.0-42.7)"),
    MeasuredADME("carbamazepine", 0.25, "DrugBank", 10.2, 2,
                 "TDC Hepatocyte_AZ (gmean, N=2, range 5.2-19.9)"),
    MeasuredADME("clozapine", 0.03, "DrugBank", 31.8, 2,
                 "TDC Hepatocyte_AZ (gmean, N=2, range 6.8-150)"),
    MeasuredADME("diclofenac", 0.003, "DrugBank", 83.5, 2,
                 "TDC Hepatocyte_AZ (gmean, N=2, range 80-87.1)"),
    MeasuredADME("sildenafil", 0.04, "DrugBank", 49.9, 2,
                 "TDC Hepatocyte_AZ (gmean, N=2, range 16.6-150)"),
    MeasuredADME("etodolac", 0.01, "DrugBank", 12.9, 1,
                 "TDC Hepatocyte_AZ (N=1)"),
    MeasuredADME("montelukast", 0.01, "DrugBank", 24.7, 2,
                 "TDC Hepatocyte_AZ (gmean, N=2, range 14-43.6)"),
    MeasuredADME("quinine", 0.30, "DrugBank", 21.1, 2,
                 "TDC Hepatocyte_AZ (gmean, N=2, range 6.2-72.4)"),
    MeasuredADME("febuxostat", 0.008, "DrugBank", 9.4, 2,
                 "TDC Hepatocyte_AZ (gmean, N=2, range 7.0-12.7)"),
    MeasuredADME("dasatinib", 0.04, "DrugBank", 28.2, 1,
                 "TDC Hepatocyte_AZ (N=1)"),
    MeasuredADME("clopidogrel", 0.2175, "DrugBank", 137.0, 1,
                 "TDC Hepatocyte_AZ (N=1)"),
    MeasuredADME("abiraterone", 0.01, "DrugBank", 55.0, 1,
                 "TDC Hepatocyte_AZ (N=1)"),
]


def run_engine_only(smiles: str, dose_mg: float, route: str, adme_override=None):
    """Run engine-only prediction with optional ADME override.

    Returns engine Cmax (mg/L) or None on failure.
    """
    import sisyphus.engine.flux  # noqa: F401
    from sisyphus.core import Distribution
    from sisyphus.engine.compiler import ODECompiler, ResolvedParams
    from sisyphus.engine.solver import solve
    from sisyphus.graph.builder import build_from_yaml
    from sisyphus.pk.endpoints import compute_endpoints
    from sisyphus.predict.adme import predict_adme
    from sisyphus.predict.chemistry import compute_profile
    from sisyphus.predict.ivive import build_drug_on_graph

    physiology_dir = ROOT / "data" / "physiology"

    profile = compute_profile(smiles)
    adme = predict_adme(profile)

    # Apply measured overrides
    if adme_override:
        if "fup" in adme_override:
            fup_val = adme_override["fup"]
            adme = _dc_replace(adme, fup=Distribution(mean=fup_val, cv=0.15))
        if "clint" in adme_override:
            clint_val = adme_override["clint"]
            cv = 0.30 if adme_override.get("clint_n", 1) == 1 else 0.25
            adme = _dc_replace(adme, clint=Distribution(mean=clint_val, cv=cv))
        if "peff" in adme_override:
            peff_val = adme_override["peff"]
            adme = _dc_replace(adme, peff=Distribution(mean=peff_val, cv=0.30))

    graph = build_from_yaml(physiology_dir / "reference_man.yaml")

    # Get liver enzymes from graph
    liver_enzymes = None
    if "liver" in graph.nodes and graph.nodes["liver"].enzymes:
        liver_enzymes = {
            tag: dist.mean for tag, dist in graph.nodes["liver"].enzymes.items()
        }

    drug = build_drug_on_graph(profile, adme, dose_mg, route, liver_enzymes=liver_enzymes)

    compiler = ODECompiler()
    compiled = compiler.compile(graph)

    rng = np.random.default_rng(42)
    realized_graph = graph.sample(rng)
    realized_drug = drug.sample(rng)
    params = ResolvedParams(realized_graph, realized_drug)

    y0 = np.zeros(compiled.n_states)
    admin_idx = compiled.state_index[drug.administration_node]
    y0[admin_idx] = drug.dose_mg

    sim_result = solve(compiled, params, y0, t_span=(0, 24))
    if sim_result.solver_success:
        pk = compute_endpoints(sim_result)
        return pk.cmax.mean, adme
    return None, adme


def main():
    from sisyphus.predict.adme import predict_adme
    from sisyphus.predict.chemistry import compute_profile

    import json
    with open(ROOT / "data" / "reference" / "clinical_pk.json") as f:
        cpk = json.load(f)

    print("=" * 90)
    print("PROOF OF CONCEPT: Measured ADME vs Predicted ADME (Engine-Only)")
    print("=" * 90)

    results = []
    for m in MEASURED:
        d = cpk["drugs"].get(m.drug, {})
        smiles = d.get("smiles", "")
        dose = d.get("dose_mg", 0)
        cmax_obs = d.get("pk_params", {}).get("cmax_mg_L", 0)
        if not smiles or dose <= 0 or cmax_obs <= 0:
            print(f"  SKIP {m.drug}: missing data")
            continue

        # A. Engine + predicted ADME
        cmax_pred, adme_pred = run_engine_only(smiles, dose, "oral")

        # B. Engine + measured ADME
        override = {"fup": m.fup, "clint": m.clint_hep, "clint_n": m.clint_n_sources}
        if m.peff is not None:
            override["peff"] = m.peff
        cmax_meas, adme_meas = run_engine_only(smiles, dose, "oral", adme_override=override)

        if cmax_pred is None or cmax_meas is None:
            print(f"  SKIP {m.drug}: engine failed")
            continue

        fe_pred = max(cmax_pred / cmax_obs, cmax_obs / cmax_pred)
        fe_meas = max(cmax_meas / cmax_obs, cmax_obs / cmax_meas)
        improved = fe_meas < fe_pred

        results.append({
            "drug": m.drug,
            "tier": m.tier,
            "cmax_obs": cmax_obs,
            "cmax_eng_pred": cmax_pred,
            "cmax_eng_meas": cmax_meas,
            "fe_pred": fe_pred,
            "fe_meas": fe_meas,
            "improved": improved,
            "fup_pred": adme_pred.fup.mean,
            "fup_meas": m.fup,
            "clint_pred": adme_pred.clint.mean,
            "clint_meas": m.clint_hep,
            "peff_pred": adme_pred.peff.mean,
            "peff_meas": m.peff,
        })

    # ── Output 1: Cmax comparison ──
    print(f"\n{'─'*90}")
    print("OUTPUT 1: Cmax Comparison (Engine-Only)")
    print(f"{'─'*90}")
    print(f"{'Drug':<16} {'Tier':>4} {'Obs Cmax':>10} {'Eng(pred)':>10} {'FE(pred)':>8} {'Eng(meas)':>10} {'FE(meas)':>8} {'Δ':>3}")
    print(f"{'':16} {'':>4} {'(mg/L)':>10} {'(mg/L)':>10} {'':>8} {'(mg/L)':>10} {'':>8}")
    print("-" * 90)
    for r in results:
        arrow = "✓" if r["improved"] else "✗"
        print(f"{r['drug']:<16} {r['tier']:>4} {r['cmax_obs']:>10.4f} "
              f"{r['cmax_eng_pred']:>10.4f} {r['fe_pred']:>8.2f} "
              f"{r['cmax_eng_meas']:>10.4f} {r['fe_meas']:>8.2f} {arrow:>3}")

    # ── Output 2: ADME parameter comparison ──
    print(f"\n{'─'*90}")
    print("OUTPUT 2: ADME Parameter Comparison")
    print(f"{'─'*90}")
    print(f"{'Drug':<16} {'Param':>8} {'Predicted':>10} {'Measured':>10} {'Fold diff':>10} {'N_sources':>10}")
    print("-" * 90)
    for r in results:
        fup_fold = max(r["fup_pred"] / r["fup_meas"], r["fup_meas"] / r["fup_pred"]) if r["fup_meas"] > 0 else 0
        clint_fold = max(r["clint_pred"] / r["clint_meas"], r["clint_meas"] / r["clint_pred"]) if r["clint_meas"] > 0 else 0
        print(f"{r['drug']:<16} {'fup':>8} {r['fup_pred']:>10.4f} {r['fup_meas']:>10.4f} {fup_fold:>10.2f}")
        clint_n = next((m.clint_n_sources for m in MEASURED if m.drug == r["drug"]), 1)
        print(f"{'':16} {'CLint':>8} {r['clint_pred']:>10.1f} {r['clint_meas']:>10.1f} {clint_fold:>10.2f} {clint_n:>10}")
        if r["peff_meas"] is not None:
            peff_fold = max(r["peff_pred"] / r["peff_meas"], r["peff_meas"] / r["peff_pred"])
            print(f"{'':16} {'Peff':>8} {r['peff_pred']:>10.2f} {r['peff_meas']:>10.2f} {peff_fold:>10.2f}")

    # ── Output 3: Summary ──
    print(f"\n{'─'*90}")
    print("OUTPUT 3: Summary")
    print(f"{'─'*90}")
    fe_pred_all = [r["fe_pred"] for r in results]
    fe_meas_all = [r["fe_meas"] for r in results]

    print(f"{'Metric':<35} {'Engine(predicted)':>18} {'Engine(measured)':>18}")
    print("-" * 72)
    print(f"{'AAFE (geometric mean fold error)':<35} {np.exp(np.mean(np.log(fe_pred_all))):>18.3f} {np.exp(np.mean(np.log(fe_meas_all))):>18.3f}")
    print(f"{'Mean fold error':<35} {np.mean(fe_pred_all):>18.3f} {np.mean(fe_meas_all):>18.3f}")
    print(f"{'Median fold error':<35} {np.median(fe_pred_all):>18.3f} {np.median(fe_meas_all):>18.3f}")
    print(f"{'N within 1.5-fold':<35} {sum(1 for f in fe_pred_all if f <= 1.5):>18d} {sum(1 for f in fe_meas_all if f <= 1.5):>18d}")
    print(f"{'N within 2-fold':<35} {sum(1 for f in fe_pred_all if f <= 2.0):>18d} {sum(1 for f in fe_meas_all if f <= 2.0):>18d}")
    print(f"{'N within 3-fold':<35} {sum(1 for f in fe_pred_all if f <= 3.0):>18d} {sum(1 for f in fe_meas_all if f <= 3.0):>18d}")
    print(f"{'N improved':<35} {'':>18} {sum(1 for r in results if r['improved']):>18d}/{len(results)}")

    # ── Interpretation ──
    median_meas = np.median(fe_meas_all)
    n_improved = sum(1 for r in results if r["improved"])
    n_total = len(results)

    print(f"\n{'─'*90}")
    print("INTERPRETATION")
    print(f"{'─'*90}")
    if median_meas <= 1.5:
        print("Pattern A: Engine(measured) median fold error ≤ 1.5")
        print("→ Architecture is sound. Input quality is the bottleneck.")
    elif median_meas <= 2.0:
        print("Pattern C: Engine(measured) median fold error 1.5-2.0")
        print("→ Engine has minor systematic bias, but input quality dominates.")
    elif median_meas > 2.0:
        print("Pattern B: Engine(measured) median fold error > 2.0")
        print("→ Engine may have structural issues beyond input quality.")

    if n_improved < n_total * 0.5:
        print(f"\nPattern D: Only {n_improved}/{n_total} improved with measured ADME.")
        print("→ Error cancellation is active. Measured inputs break existing balance.")


if __name__ == "__main__":
    main()
