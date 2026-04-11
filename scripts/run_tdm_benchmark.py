#!/usr/bin/env python3
"""TDM Bayesian Update Benchmark — Multi-drug, multi-observation validation.

Evidence for the claim: "TDM is the only viable path to bypass the CLint R²=0.24 ceiling."

Design:
  - 5 holdout drugs (2 base, 1 acid, 2 neutral), fold error 2-5x
  - Synthetic patient observations generated from engine C(t) scaled to observed Cmax
  - 3 scenarios per drug: 1, 2, 3 observations
  - Timepoint sensitivity analysis (single drug)
  - Seed sensitivity analysis

Assay noise: 10% CV (log-normal multiplicative). Note: this is a simplification;
real clinical assays have higher CV (20-30%) near LLOQ.

Usage:
    python3 scripts/run_tdm_benchmark.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger(__name__)

import sisyphus.engine.flux  # noqa: F401 — register flux specs
from sisyphus.engine.compiler import ODECompiler, ResolvedParams
from sisyphus.graph.builder import build_from_yaml
from sisyphus.predict.adme import predict_adme
from sisyphus.predict.chemistry import compute_profile
from sisyphus.predict.ivive import build_drug_on_graph
from sisyphus.regimen.tdm import Observation, bayesian_update
from sisyphus.regimen.types import DosingRegimen
from sisyphus.regimen.solver import solve_regimen

PHYSIOLOGY_YAML = ROOT / "data" / "physiology" / "reference_man.yaml"
OUTPUT_DIR = ROOT / "data" / "validation"
OUTPUT_JSON = OUTPUT_DIR / "tdm_benchmark.json"

N_PRIOR = 2000
ASSAY_CV = 0.10
BASE_SEED = 42

# ═══════════════════════════════════════════════════════════════════════════
# Drug definitions (from holdout: 2 base + 1 acid + 2 neutral, FE 2-5x)
# ═══════════════════════════════════════════════════════════════════════════
DRUGS = {
    "morphine": {
        "smiles": "CN1CCC23C4C1CC5=C2C(=C(C=C5)O)OC3C(C=C4)O",
        "dose_mg": 30.0,
        "cmax_obs": 0.01865,
        "cmax_engine": 0.03736,
        "compound_type": "base",
        "fold_error": 2.00,
    },
    "amantadine": {
        "smiles": "C1C2CC3CC1CC(C2)(C3)N",
        "dose_mg": 100.0,
        "cmax_obs": 0.22,
        "cmax_engine": 0.47592,
        "compound_type": "base",
        "fold_error": 2.16,
    },
    "ketorolac": {
        "smiles": "O=C(c1ccccc1)c1ccc2n1CCC2C(=O)O",
        "dose_mg": 10.0,
        "cmax_obs": 0.80,
        "cmax_engine": 0.24623,
        "compound_type": "acid",
        "fold_error": 3.25,
    },
    "clozapine": {
        "smiles": "CN1CCN(CC1)C2=NC3=C(C=CC(=C3)Cl)NC4=CC=CC=C42",
        "dose_mg": 75.0,
        "cmax_obs": 0.413,
        "cmax_engine": 0.87545,
        "compound_type": "neutral",
        "fold_error": 2.12,
    },
    "rivaroxaban": {
        "smiles": "O=C(NC[C@H]1CN(c2ccc(N3CCOCC3=O)cc2)C(=O)O1)c1ccc(Cl)s1",
        "dose_mg": 10.0,
        "cmax_obs": 0.1432,
        "cmax_engine": 0.06589,
        "compound_type": "neutral",
        "fold_error": 2.17,
    },
}

# Observation scenarios
SCENARIOS = {
    "S1_tmax": None,       # single obs at tmax (computed per drug)
    "S2_1h_4h": [1.0, 4.0],
    "S3_1h_4h_8h": [1.0, 4.0, 8.0],
}

# Timepoint sensitivity (morphine, matching a base drug for the analysis)
TIMEPOINT_SENSITIVITY_DRUG = "morphine"
TIMEPOINT_TIMES = [0.5, 1.0, 2.0, 4.0, 8.0]

# Seed sensitivity
SEED_SENSITIVITY_DRUG = "morphine"
SEED_SENSITIVITY_SEEDS = [42, 123, 456]


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def build_pipeline(smiles: str, dose_mg: float):
    """Build full pipeline: SMILES → graph + compiled + drug."""
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
    """Generate synthetic patient observations by scaling engine C(t) to observed Cmax.

    Method:
      1. Simulate C(t) with median prior parameters (seed for deterministic realization)
      2. Scale: C_scaled(t) = C_engine(t) × (cmax_obs / cmax_engine)
      3. Add assay noise: C_obs = C_scaled(t) × exp(N(0, assay_cv))

    Note: 10% assay CV is a simplification. Real assays have higher CV (20-30%) near LLOQ.
    """
    rng = np.random.default_rng(seed)

    # Deterministic simulation with median parameters
    realized_graph = graph.sample(np.random.default_rng(0))  # fixed seed=0 for median
    realized_drug = drug.sample(np.random.default_rng(0))
    params = ResolvedParams(realized_graph, realized_drug)

    t_total = max(max(obs_times) + 12.0, 24.0)
    sim = solve_regimen(compiled, params, regimen, t_total_h=t_total, dt_output=0.1)

    if not sim.solver_success:
        return None, None

    conc = sim.concentrations.get("venous_blood")
    if conc is None or len(conc) == 0:
        return None, None

    # Scale factor: observed / engine Cmax
    engine_cmax_sim = float(np.max(conc))
    if engine_cmax_sim <= 0:
        return None, None
    scale = cmax_obs / engine_cmax_sim

    # Find tmax for S1 scenario
    tmax = float(sim.time_h[int(np.argmax(conc))])

    observations = []
    true_concs = []
    for t in obs_times:
        c_engine = float(np.interp(t, sim.time_h, conc))
        c_scaled = c_engine * scale
        # Add log-normal noise (multiplicative)
        noise = rng.lognormal(0, ASSAY_CV)
        c_measured = max(c_scaled * noise, 1e-8)
        observations.append(Observation(time_h=t, concentration=c_measured, cv=ASSAY_CV))
        true_concs.append(c_scaled)

    return observations, tmax


def run_tdm(compiled, graph, drug, regimen, observations, seed=42, method="importance_sampling"):
    """Run TDM Bayesian update and extract metrics.

    Returns the TDMResult together with the 90 % credible interval on the
    posterior Cmax. Prefers the ``TDMResult.cmax_ci_90`` field (empirical
    quantile of the raw weighted samples, populated by every dispatch path
    as of the D2 fix). Falls back to the legacy lognormal approximation
    only if that field is missing — this preserves backwards compatibility
    with older TDMResult instances loaded from JSON.
    """
    result = bayesian_update(
        compiled, graph, drug, regimen,
        observations=observations,
        n_prior=N_PRIOR,
        seed=seed,
        method=method,
    )

    if result.cmax_ci_90 is not None:
        ci90_lower, ci90_upper = result.cmax_ci_90
    elif result.n_successful > 0:
        post_mean = result.posterior_cmax.mean
        post_cv = result.posterior_cmax.cv
        if post_mean > 0 and post_cv > 0:
            sigma_log = np.sqrt(np.log(1 + post_cv**2))
            mu_log = np.log(post_mean) - 0.5 * sigma_log**2
            ci90_lower = float(np.exp(mu_log + sigma_log * (-1.645)))
            ci90_upper = float(np.exp(mu_log + sigma_log * 1.645))
        else:
            ci90_lower = ci90_upper = post_mean
    else:
        ci90_lower = ci90_upper = 0.0

    return result, ci90_lower, ci90_upper


def assess_ess(ess, n_prior=N_PRIOR):
    """Classify ESS health."""
    pct = ess / n_prior * 100
    if ess >= 200:
        return "healthy"
    elif ess >= 100:
        return "caution"
    else:
        return "degenerate"


# ═══════════════════════════════════════════════════════════════════════════
# Main benchmark
# ═══════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        default="importance_sampling",
        choices=["importance_sampling", "ibis", "enkf"],
        help="TDM dispatch method used for the benchmark",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Override default output JSON path",
    )
    args = parser.parse_args()
    global OUTPUT_JSON
    if args.out is not None:
        OUTPUT_JSON = args.out
    else:
        tag = {"importance_sampling": "is", "ibis": "ibis", "enkf": "enkf"}[args.method]
        OUTPUT_JSON = OUTPUT_DIR / f"tdm_benchmark_{tag}.json"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    method = args.method
    all_results = []
    print(f"TDM benchmark method: {method}  →  {OUTPUT_JSON.name}")

    print("=" * 85)
    print("  TDM BAYESIAN UPDATE BENCHMARK")
    print("  5 drugs × 3 scenarios + timepoint sensitivity + seed sensitivity")
    print("=" * 85)

    # ─────────────────────────────────────────────────────────────
    # Part A: Multi-drug, multi-observation (15 runs)
    # ─────────────────────────────────────────────────────────────
    print()
    print("PART A: Multi-drug × Multi-observation")
    print("-" * 85)

    for drug_name, drug_info in DRUGS.items():
        print(f"\n  {drug_name} ({drug_info['compound_type']}, "
              f"{drug_info['dose_mg']:.0f}mg, FE={drug_info['fold_error']:.2f})")

        graph, compiled, drug = build_pipeline(drug_info["smiles"], drug_info["dose_mg"])
        regimen = DosingRegimen.single_oral(drug_info["dose_mg"])

        # First, get tmax for S1
        dummy_obs, tmax = generate_scaled_observations(
            compiled, graph, drug, regimen,
            drug_info["cmax_obs"], drug_info["cmax_engine"],
            [1.0], seed=BASE_SEED,
        )
        if tmax is None:
            print(f"    SKIP: simulation failed")
            continue

        scenarios = {
            "S1_tmax": [round(tmax, 1)],
            "S2_1h_4h": [1.0, 4.0],
            "S3_1h_4h_8h": [1.0, 4.0, 8.0],
        }

        for scen_name, obs_times in scenarios.items():
            obs, _ = generate_scaled_observations(
                compiled, graph, drug, regimen,
                drug_info["cmax_obs"], drug_info["cmax_engine"],
                obs_times, seed=BASE_SEED,
            )
            if obs is None:
                continue

            tdm, ci90_lo, ci90_hi = run_tdm(compiled, graph, drug, regimen, obs, seed=BASE_SEED, method=method)

            prior_cv = tdm.prior_cmax.cv * 100
            post_cv = tdm.posterior_cmax.cv * 100
            cv_red = (1 - post_cv / prior_cv) * 100 if prior_cv > 0 else 0
            prior_err = abs(tdm.prior_cmax.mean - drug_info["cmax_obs"]) / drug_info["cmax_obs"] * 100
            post_err = abs(tdm.posterior_cmax.mean - drug_info["cmax_obs"]) / drug_info["cmax_obs"] * 100
            err_red = (1 - post_err / prior_err) * 100 if prior_err > 0 else 0
            ci_contains = ci90_lo <= drug_info["cmax_obs"] <= ci90_hi
            ess_health = assess_ess(tdm.ess)

            rec = {
                "drug": drug_name,
                "compound_type": drug_info["compound_type"],
                "dose_mg": drug_info["dose_mg"],
                "cmax_observed": drug_info["cmax_obs"],
                "scenario": scen_name,
                "n_obs": len(obs_times),
                "obs_times": obs_times,
                "prior_cmax_mean": round(tdm.prior_cmax.mean, 6),
                "prior_cmax_cv": round(prior_cv, 1),
                "posterior_cmax_mean": round(tdm.posterior_cmax.mean, 6),
                "posterior_cmax_cv": round(post_cv, 1),
                "cv_reduction_pct": round(cv_red, 1),
                "ess": round(tdm.ess, 1),
                "ess_pct": round(tdm.ess / tdm.n_successful * 100, 1) if tdm.n_successful > 0 else 0,
                "ess_health": ess_health,
                "n_successful": tdm.n_successful,
                "prior_error_pct": round(prior_err, 1),
                "posterior_error_pct": round(post_err, 1),
                "error_reduction_pct": round(err_red, 1),
                "ci90_lower": round(ci90_lo, 6),
                "ci90_upper": round(ci90_hi, 6),
                "ci90_contains_true": ci_contains,
                "section": "main",
            }
            all_results.append(rec)

            tag = "✅" if ess_health == "healthy" else ("⚠" if ess_health == "caution" else "❌")
            print(f"    {scen_name:<15s} obs={len(obs_times)} "
                  f"CVred={cv_red:>5.1f}% ErrRed={err_red:>6.1f}% "
                  f"PostCV={post_cv:>5.1f}% ESS={tdm.ess:>6.1f} {tag}")

    # ─────────────────────────────────────────────────────────────
    # Part B: Timepoint sensitivity (morphine)
    # ─────────────────────────────────────────────────────────────
    print()
    print("PART B: Timepoint Sensitivity (morphine)")
    print("-" * 85)

    tp_drug = DRUGS[TIMEPOINT_SENSITIVITY_DRUG]
    graph, compiled, drug = build_pipeline(tp_drug["smiles"], tp_drug["dose_mg"])
    regimen = DosingRegimen.single_oral(tp_drug["dose_mg"])

    for t_obs in TIMEPOINT_TIMES:
        obs, _ = generate_scaled_observations(
            compiled, graph, drug, regimen,
            tp_drug["cmax_obs"], tp_drug["cmax_engine"],
            [t_obs], seed=BASE_SEED,
        )
        if obs is None:
            continue

        tdm, ci90_lo, ci90_hi = run_tdm(compiled, graph, drug, regimen, obs, seed=BASE_SEED, method=method)

        prior_cv = tdm.prior_cmax.cv * 100
        post_cv = tdm.posterior_cmax.cv * 100
        cv_red = (1 - post_cv / prior_cv) * 100 if prior_cv > 0 else 0
        prior_err = abs(tdm.prior_cmax.mean - tp_drug["cmax_obs"]) / tp_drug["cmax_obs"] * 100
        post_err = abs(tdm.posterior_cmax.mean - tp_drug["cmax_obs"]) / tp_drug["cmax_obs"] * 100
        err_red = (1 - post_err / prior_err) * 100 if prior_err > 0 else 0
        ci_contains = ci90_lo <= tp_drug["cmax_obs"] <= ci90_hi

        rec = {
            "drug": TIMEPOINT_SENSITIVITY_DRUG,
            "compound_type": tp_drug["compound_type"],
            "dose_mg": tp_drug["dose_mg"],
            "cmax_observed": tp_drug["cmax_obs"],
            "scenario": f"TP_t={t_obs}h",
            "n_obs": 1,
            "obs_times": [t_obs],
            "prior_cmax_mean": round(tdm.prior_cmax.mean, 6),
            "prior_cmax_cv": round(prior_cv, 1),
            "posterior_cmax_mean": round(tdm.posterior_cmax.mean, 6),
            "posterior_cmax_cv": round(post_cv, 1),
            "cv_reduction_pct": round(cv_red, 1),
            "ess": round(tdm.ess, 1),
            "ess_pct": round(tdm.ess / tdm.n_successful * 100, 1) if tdm.n_successful > 0 else 0,
            "ess_health": assess_ess(tdm.ess),
            "n_successful": tdm.n_successful,
            "prior_error_pct": round(prior_err, 1),
            "posterior_error_pct": round(post_err, 1),
            "error_reduction_pct": round(err_red, 1),
            "ci90_lower": round(ci90_lo, 6),
            "ci90_upper": round(ci90_hi, 6),
            "ci90_contains_true": ci_contains,
            "section": "timepoint",
        }
        all_results.append(rec)

        print(f"    t={t_obs:<4.1f}h  CVred={cv_red:>5.1f}%  ErrRed={err_red:>6.1f}%  "
              f"PostCV={post_cv:>5.1f}%  ESS={tdm.ess:>6.1f}")

    # ─────────────────────────────────────────────────────────────
    # Part C: Seed sensitivity (morphine, t=1h)
    # ─────────────────────────────────────────────────────────────
    print()
    print("PART C: Seed Sensitivity (morphine, t=1h)")
    print("-" * 85)

    seed_cv_reds = []
    for seed in SEED_SENSITIVITY_SEEDS:
        obs, _ = generate_scaled_observations(
            compiled, graph, drug, regimen,
            tp_drug["cmax_obs"], tp_drug["cmax_engine"],
            [1.0], seed=seed,
        )
        if obs is None:
            continue

        tdm, ci90_lo, ci90_hi = run_tdm(compiled, graph, drug, regimen, obs, seed=seed, method=method)

        prior_cv = tdm.prior_cmax.cv * 100
        post_cv = tdm.posterior_cmax.cv * 100
        cv_red = (1 - post_cv / prior_cv) * 100 if prior_cv > 0 else 0
        seed_cv_reds.append(cv_red)

        rec = {
            "drug": SEED_SENSITIVITY_DRUG,
            "compound_type": tp_drug["compound_type"],
            "dose_mg": tp_drug["dose_mg"],
            "cmax_observed": tp_drug["cmax_obs"],
            "scenario": f"SEED_{seed}",
            "n_obs": 1,
            "obs_times": [1.0],
            "prior_cmax_mean": round(tdm.prior_cmax.mean, 6),
            "prior_cmax_cv": round(prior_cv, 1),
            "posterior_cmax_mean": round(tdm.posterior_cmax.mean, 6),
            "posterior_cmax_cv": round(post_cv, 1),
            "cv_reduction_pct": round(cv_red, 1),
            "ess": round(tdm.ess, 1),
            "ess_pct": round(tdm.ess / tdm.n_successful * 100, 1) if tdm.n_successful > 0 else 0,
            "ess_health": assess_ess(tdm.ess),
            "n_successful": tdm.n_successful,
            "prior_error_pct": round(abs(tdm.prior_cmax.mean - tp_drug["cmax_obs"]) / tp_drug["cmax_obs"] * 100, 1),
            "posterior_error_pct": round(abs(tdm.posterior_cmax.mean - tp_drug["cmax_obs"]) / tp_drug["cmax_obs"] * 100, 1),
            "error_reduction_pct": 0,  # computed later
            "ci90_lower": round(ci90_lo, 6),
            "ci90_upper": round(ci90_hi, 6),
            "ci90_contains_true": ci90_lo <= tp_drug["cmax_obs"] <= ci90_hi,
            "section": "seed",
        }
        all_results.append(rec)
        print(f"    seed={seed}: CVred={cv_red:.1f}%  PostCV={post_cv:.1f}%  ESS={tdm.ess:.1f}")

    if seed_cv_reds:
        seed_range = max(seed_cv_reds) - min(seed_cv_reds)
        print(f"    Range: {seed_range:.1f}% {'✅ seed-robust' if seed_range <= 5 else '⚠ seed-sensitive'}")

    # ═══════════════════════════════════════════════════════════════════════
    # Summary tables
    # ═══════════════════════════════════════════════════════════════════════

    main_results = [r for r in all_results if r["section"] == "main"]
    tp_results = [r for r in all_results if r["section"] == "timepoint"]

    print()
    print("=" * 95)
    print("  TABLE 1: MAIN BENCHMARK (5 drugs × 3 scenarios)")
    print("=" * 95)
    print()
    print(f"{'Drug':<14s} {'Type':<8s} {'Scen':<16s} {'Obs':>3s} "
          f"{'PriCV%':>7s} {'PstCV%':>7s} {'CVRed%':>7s} "
          f"{'PriErr%':>8s} {'PstErr%':>8s} {'ErrRed%':>8s} "
          f"{'ESS':>6s} {'CI90':>4s}")
    print("-" * 95)
    for r in main_results:
        ci = "✓" if r["ci90_contains_true"] else "✗"
        print(f"{r['drug']:<14s} {r['compound_type']:<8s} {r['scenario']:<16s} {r['n_obs']:>3d} "
              f"{r['prior_cmax_cv']:>7.1f} {r['posterior_cmax_cv']:>7.1f} {r['cv_reduction_pct']:>7.1f} "
              f"{r['prior_error_pct']:>8.1f} {r['posterior_error_pct']:>8.1f} {r['error_reduction_pct']:>8.1f} "
              f"{r['ess']:>6.1f} {ci:>4s}")

    print()
    print("=" * 75)
    print("  TABLE 2: TIMEPOINT SENSITIVITY (morphine, single obs)")
    print("=" * 75)
    print()
    print(f"{'t(h)':>5s} {'PriCV%':>7s} {'PstCV%':>7s} {'CVRed%':>7s} "
          f"{'PriErr%':>8s} {'PstErr%':>8s} {'ErrRed%':>8s} {'ESS':>6s}")
    print("-" * 55)
    for r in tp_results:
        print(f"{r['obs_times'][0]:>5.1f} {r['prior_cmax_cv']:>7.1f} {r['posterior_cmax_cv']:>7.1f} "
              f"{r['cv_reduction_pct']:>7.1f} {r['prior_error_pct']:>8.1f} "
              f"{r['posterior_error_pct']:>8.1f} {r['error_reduction_pct']:>8.1f} {r['ess']:>6.1f}")

    # ─── Aggregate statistics ──────────────────────────────────
    print()
    print("=" * 75)
    print("  SUMMARY")
    print("=" * 75)
    print()

    import pandas as pd
    df_main = pd.DataFrame(main_results)

    # By n_obs
    print("  By number of observations:")
    for n in sorted(df_main["n_obs"].unique()):
        sub = df_main[df_main["n_obs"] == n]
        print(f"    {n} obs: mean CVred={sub['cv_reduction_pct'].mean():.1f}% "
              f"mean ErrRed={sub['error_reduction_pct'].mean():.1f}% "
              f"mean PostCV={sub['posterior_cmax_cv'].mean():.1f}%")

    # CI coverage
    ci_hits = sum(1 for r in main_results if r["ci90_contains_true"])
    print(f"\n  90% CI coverage: {ci_hits}/{len(main_results)} ({ci_hits/len(main_results)*100:.0f}%)")

    # ESS health
    ess_vals = [r["ess"] for r in main_results]
    ess_healthy = sum(1 for e in ess_vals if e >= 200)
    ess_caution = sum(1 for e in ess_vals if 100 <= e < 200)
    ess_degen = sum(1 for e in ess_vals if e < 100)
    print(f"  ESS range: [{min(ess_vals):.0f}, {max(ess_vals):.0f}]")
    print(f"  ESS health: {ess_healthy} healthy, {ess_caution} caution, {ess_degen} degenerate")

    # Timepoint sensitivity
    if tp_results:
        best_tp = max(tp_results, key=lambda r: r["cv_reduction_pct"])
        print(f"\n  Most informative timepoint (morphine): t={best_tp['obs_times'][0]:.1f}h "
              f"(CVred={best_tp['cv_reduction_pct']:.1f}%)")

    # Seed sensitivity
    if seed_cv_reds:
        print(f"  Seed sensitivity (morphine, t=1h): "
              f"CV reduction range = [{min(seed_cv_reds):.1f}%, {max(seed_cv_reds):.1f}%] "
              f"(Δ={max(seed_cv_reds)-min(seed_cv_reds):.1f}%)")

    # Overall
    print(f"\n  Overall (15 main runs):")
    print(f"    Mean CV reduction:    {df_main['cv_reduction_pct'].mean():.1f}%")
    print(f"    Median CV reduction:  {df_main['cv_reduction_pct'].median():.1f}%")
    print(f"    Mean error reduction: {df_main['error_reduction_pct'].mean():.1f}%")
    print(f"    CV improved in:       {(df_main['posterior_cmax_cv'] < df_main['prior_cmax_cv']).sum()}/{len(df_main)} runs")

    # Save results
    with open(OUTPUT_JSON, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
