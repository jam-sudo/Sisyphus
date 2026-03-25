#!/usr/bin/env python3
"""MMPK CLint Deconvolution — Track A.

For each MMPK drug (excluding holdout), uses the Sisyphus engine in reverse:
given observed Cmax and fixed fup/Peff/Kp, find the apparent CLint that makes
the engine match the observed Cmax.

The deconvolved CLint is NOT real in vivo CLint — it is the value that, given
the current engine's structure and all other fixed parameters, reproduces the
observed Cmax.  It absorbs model error from every other parameter.

Usage:
    python3 scripts/run_deconvolution.py [--max-drugs N] [--output FILE]
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

# Ensure repo root on sys.path
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

import sisyphus.engine.flux  # noqa: F401 — register flux specs
from sisyphus.core import Distribution
from sisyphus.engine.compiler import ODECompiler, ResolvedParams
from sisyphus.engine.solver import solve
from sisyphus.graph.builder import build_from_yaml
from sisyphus.pk.endpoints import compute_endpoints
from sisyphus.predict.adme import predict_adme
from sisyphus.predict.chemistry import compute_profile
from sisyphus.predict.ivive import build_drug_on_graph

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_PHYSIOLOGY_DIR = _REPO / "data" / "physiology"
_TRAINING_DIR = _REPO / "data" / "training"

# Physical bounds for CLint (uL/min/10^6 cells)
_CLINT_MIN = 0.1
_CLINT_MAX = 10000.0
# Bounds in log10 space for brentq
_LOG_CLINT_LO = np.log10(_CLINT_MIN)  # -1
_LOG_CLINT_HI = np.log10(_CLINT_MAX)  # 4


@dataclass
class DeconvolutionResult:
    """Result for a single drug's CLint deconvolution."""
    name: str
    smiles: str
    dose_mg: float
    cmax_obs: float
    cmax_pred_original: float  # engine Cmax with XGBoost CLint
    clint_original: float     # XGBoost CLint prediction
    clint_deconv: float | None  # deconvolved apparent CLint (None if failed)
    cmax_at_deconv: float | None  # engine Cmax at deconvolved CLint
    status: str  # "success", "no_root", "nonphysical", "error"
    error_msg: str = ""


def _engine_cmax_for_clint(
    log_clint: float,
    profile,
    adme,
    dose_mg: float,
    route: str,
    graph,
    compiled,
    liver_enzymes: dict[str, float],
) -> float:
    """Compute engine Cmax for a given log10(CLint).

    Returns log10(Cmax) or NaN on failure.
    """
    clint_val = 10 ** log_clint
    # Replace CLint in ADME with the test value (cv=0 for deterministic)
    adme_mod = type(adme)(
        fup=adme.fup,
        clint=Distribution(mean=clint_val, cv=0.0),
        peff=adme.peff,
        solubility=adme.solubility,
        vdss=adme.vdss,
        rbp=adme.rbp,
    )
    drug = build_drug_on_graph(profile, adme_mod, dose_mg, route, liver_enzymes=liver_enzymes)

    rng = np.random.default_rng(42)
    realized_graph = graph.sample(rng)
    realized_drug = drug.sample(rng)
    params = ResolvedParams(realized_graph, realized_drug)

    y0 = np.zeros(compiled.n_states)
    admin_idx = compiled.state_index[drug.administration_node]
    y0[admin_idx] = drug.dose_mg

    sim = solve(compiled, params, y0, t_span=(0, 24))
    if not sim.solver_success:
        return float("nan")

    pk = compute_endpoints(sim)
    cmax = pk.cmax.mean
    return np.log10(max(cmax, 1e-15))


def deconvolve_one(
    name: str,
    smiles: str,
    dose_mg: float,
    cmax_obs: float,
    graph,
    compiled,
    liver_enzymes: dict[str, float],
) -> DeconvolutionResult:
    """Deconvolve CLint for a single drug."""
    try:
        profile = compute_profile(smiles)
        adme = predict_adme(profile)
    except Exception as e:
        return DeconvolutionResult(
            name=name, smiles=smiles, dose_mg=dose_mg, cmax_obs=cmax_obs,
            cmax_pred_original=0.0, clint_original=0.0,
            clint_deconv=None, cmax_at_deconv=None,
            status="error", error_msg=f"profile/adme: {e}",
        )

    clint_original = adme.clint.mean

    # Get original engine Cmax
    try:
        drug_orig = build_drug_on_graph(profile, adme, dose_mg, "oral", liver_enzymes=liver_enzymes)
        rng = np.random.default_rng(42)
        rg = graph.sample(rng)
        rd = drug_orig.sample(rng)
        params = ResolvedParams(rg, rd)
        y0 = np.zeros(compiled.n_states)
        y0[compiled.state_index[drug_orig.administration_node]] = drug_orig.dose_mg
        sim = solve(compiled, params, y0, t_span=(0, 24))
        cmax_orig = compute_endpoints(sim).cmax.mean if sim.solver_success else 0.0
    except Exception:
        cmax_orig = 0.0

    log_cmax_obs = np.log10(max(cmax_obs, 1e-15))

    def residual(log_clint: float) -> float:
        log_cmax_pred = _engine_cmax_for_clint(
            log_clint, profile, adme, dose_mg, "oral", graph, compiled, liver_enzymes,
        )
        if np.isnan(log_cmax_pred):
            return 1.0  # push solver toward other direction
        return log_cmax_pred - log_cmax_obs

    # Check if root exists by evaluating at bounds
    try:
        f_lo = residual(_LOG_CLINT_LO)
        f_hi = residual(_LOG_CLINT_HI)
    except Exception as e:
        return DeconvolutionResult(
            name=name, smiles=smiles, dose_mg=dose_mg, cmax_obs=cmax_obs,
            cmax_pred_original=cmax_orig, clint_original=clint_original,
            clint_deconv=None, cmax_at_deconv=None,
            status="error", error_msg=f"bound eval: {e}",
        )

    if np.isnan(f_lo) or np.isnan(f_hi):
        return DeconvolutionResult(
            name=name, smiles=smiles, dose_mg=dose_mg, cmax_obs=cmax_obs,
            cmax_pred_original=cmax_orig, clint_original=clint_original,
            clint_deconv=None, cmax_at_deconv=None,
            status="error", error_msg="NaN at bounds",
        )

    if f_lo * f_hi > 0:
        return DeconvolutionResult(
            name=name, smiles=smiles, dose_mg=dose_mg, cmax_obs=cmax_obs,
            cmax_pred_original=cmax_orig, clint_original=clint_original,
            clint_deconv=None, cmax_at_deconv=None,
            status="no_root",
            error_msg=f"f(lo)={f_lo:.3f}, f(hi)={f_hi:.3f} — same sign",
        )

    # Find root
    try:
        log_clint_sol = brentq(residual, _LOG_CLINT_LO, _LOG_CLINT_HI, xtol=0.01, maxiter=30)
        clint_deconv = 10 ** log_clint_sol
    except Exception as e:
        return DeconvolutionResult(
            name=name, smiles=smiles, dose_mg=dose_mg, cmax_obs=cmax_obs,
            cmax_pred_original=cmax_orig, clint_original=clint_original,
            clint_deconv=None, cmax_at_deconv=None,
            status="error", error_msg=f"brentq: {e}",
        )

    # Verify Cmax at deconvolved CLint
    log_cmax_at_deconv = _engine_cmax_for_clint(
        log_clint_sol, profile, adme, dose_mg, "oral", graph, compiled, liver_enzymes,
    )
    cmax_at_deconv = 10 ** log_cmax_at_deconv if not np.isnan(log_cmax_at_deconv) else None

    return DeconvolutionResult(
        name=name, smiles=smiles, dose_mg=dose_mg, cmax_obs=cmax_obs,
        cmax_pred_original=cmax_orig, clint_original=clint_original,
        clint_deconv=clint_deconv, cmax_at_deconv=cmax_at_deconv,
        status="success",
    )


def load_mmpk_data() -> list[dict]:
    """Load MMPK data, excluding holdout drugs."""
    exclusion_path = _TRAINING_DIR / "mmpk_sisyphus_holdout_exclusions.json"
    with open(exclusion_path) as f:
        exclusions = set(json.load(f).keys())

    data = []
    seen = set()  # deduplicate by name (take first dose entry)
    with open(_TRAINING_DIR / "mmpk_expanded_full.csv") as f:
        for row in csv.DictReader(f):
            name = row["name"]
            if name in exclusions or name in seen:
                continue
            seen.add(name)
            try:
                data.append({
                    "name": name,
                    "smiles": row["canon_smiles"],
                    "dose_mg": float(row["dose_mg"]),
                    "cmax_obs": float(row["cmax_mg_L"]),
                })
            except (ValueError, KeyError):
                continue
    return data


def main():
    parser = argparse.ArgumentParser(description="MMPK CLint Deconvolution")
    parser.add_argument("--max-drugs", type=int, default=None)
    parser.add_argument("--output", type=str, default="data/deconvolution_results.tsv")
    args = parser.parse_args()

    print("Loading MMPK data...")
    drugs = load_mmpk_data()
    if args.max_drugs:
        drugs = drugs[: args.max_drugs]
    print(f"  {len(drugs)} drugs after holdout exclusion")

    print("Building graph and compiling ODE...")
    graph = build_from_yaml(_PHYSIOLOGY_DIR / "reference_man.yaml")
    compiler = ODECompiler()
    compiled = compiler.compile(graph)

    liver_enzymes: dict[str, float] = {}
    if "liver" in graph.nodes and graph.nodes["liver"].enzymes:
        liver_enzymes = {tag: d.mean for tag, d in graph.nodes["liver"].enzymes.items()}

    results: list[DeconvolutionResult] = []
    t0 = time.time()

    for i, drug in enumerate(drugs):
        if (i + 1) % 50 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"  [{i+1}/{len(drugs)}] {drug['name']:<30s} ({rate:.1f} drugs/min)")

        result = deconvolve_one(
            drug["name"], drug["smiles"], drug["dose_mg"], drug["cmax_obs"],
            graph, compiled, liver_enzymes,
        )
        results.append(result)

    elapsed = time.time() - t0

    # Summary stats
    n_success = sum(1 for r in results if r.status == "success")
    n_no_root = sum(1 for r in results if r.status == "no_root")
    n_error = sum(1 for r in results if r.status == "error")

    print(f"\n{'='*60}")
    print(f"Deconvolution complete: {len(results)} drugs in {elapsed:.0f}s")
    print(f"  Success: {n_success} ({100*n_success/len(results):.1f}%)")
    print(f"  No root: {n_no_root} ({100*n_no_root/len(results):.1f}%)")
    print(f"  Error:   {n_error} ({100*n_error/len(results):.1f}%)")

    # Save TSV
    output_path = _REPO / args.output
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow([
            "name", "smiles", "dose_mg", "cmax_obs",
            "cmax_pred_original", "clint_original",
            "clint_deconv", "cmax_at_deconv", "status", "error_msg",
        ])
        for r in results:
            writer.writerow([
                r.name, r.smiles, r.dose_mg, r.cmax_obs,
                f"{r.cmax_pred_original:.6f}" if r.cmax_pred_original else "",
                f"{r.clint_original:.4f}" if r.clint_original else "",
                f"{r.clint_deconv:.4f}" if r.clint_deconv else "",
                f"{r.cmax_at_deconv:.6f}" if r.cmax_at_deconv else "",
                r.status, r.error_msg,
            ])
    print(f"Results saved to {output_path}")

    # Go/No-Go diagnostic: deconvolved vs original CLint correlation
    if n_success >= 20:
        deconv_log = []
        orig_log = []
        for r in results:
            if r.status == "success" and r.clint_deconv and r.clint_original > 0:
                deconv_log.append(np.log10(r.clint_deconv))
                orig_log.append(np.log10(r.clint_original))

        if len(deconv_log) >= 20:
            deconv_arr = np.array(deconv_log)
            orig_arr = np.array(orig_log)
            corr = np.corrcoef(orig_arr, deconv_arr)[0, 1]
            r_squared = corr ** 2

            print(f"\n{'='*60}")
            print("GO/NO-GO DIAGNOSTIC")
            print(f"  log10(CLint_deconv) vs log10(CLint_XGBoost)")
            print(f"  N = {len(deconv_log)}")
            print(f"  R² = {r_squared:.3f}")
            print(f"  Pearson r = {corr:.3f}")
            if r_squared > 0.5:
                print("  → DIAGNOSTIC PASS (R² > 0.5): gap is systematic")
                print("  → Proceed to XGBoost training on deconvolved targets")
            elif r_squared > 0.3:
                print("  → DIAGNOSTIC MARGINAL (0.3 < R² < 0.5): proceed with caution")
            else:
                print("  → DIAGNOSTIC FAIL (R² < 0.3): gap is random")
                print("  → Track A ABORTED. Redirect to Track B.")

            # Ratio analysis
            ratios = 10 ** (deconv_arr - orig_arr)
            print(f"\n  Deconv/Original ratio:")
            print(f"    Median: {np.median(ratios):.2f}x")
            print(f"    IQR: [{np.percentile(ratios, 25):.2f}, {np.percentile(ratios, 75):.2f}]x")
            print(f"    Outliers (>100x): {sum(ratios > 100)}")


if __name__ == "__main__":
    main()
