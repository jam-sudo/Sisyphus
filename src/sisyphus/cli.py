"""Sisyphus CLI entry point.

Usage::

    sisyphus predict --smiles "CC(=O)Oc1ccccc1C(=O)O" --dose 500
    sisyphus simulate --smiles "CN(C)C(=N)NC(=N)N" --dose 500 --interval 12 --doses 14
    sisyphus tdm --smiles "Clc1ccc2c(c1)..." --dose 5 --obs "1.0:0.015"
    sisyphus benchmark --holdout
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Sisyphus PBPK Prediction")
    subparsers = parser.add_subparsers(dest="command")

    # predict command
    pred_parser = subparsers.add_parser("predict", help="Predict PK for a SMILES")
    pred_parser.add_argument("--smiles", required=True, help="SMILES string")
    pred_parser.add_argument("--dose", type=float, required=True, help="Dose in mg")
    pred_parser.add_argument("--route", default="oral", choices=["oral", "iv"])
    pred_parser.add_argument("--verbose", "-v", action="store_true")

    # simulate command (multi-dose)
    sim_parser = subparsers.add_parser("simulate", help="Multi-dose regimen simulation")
    sim_parser.add_argument("--smiles", required=True, help="SMILES string")
    sim_parser.add_argument("--dose", type=float, required=True, help="Dose per administration (mg)")
    sim_parser.add_argument("--interval", type=float, required=True, help="Dosing interval (hours)")
    sim_parser.add_argument("--doses", type=int, required=True, help="Number of doses")
    sim_parser.add_argument("--route", default="oral", choices=["oral", "iv"])
    sim_parser.add_argument("--verbose", "-v", action="store_true")

    # tdm command (Bayesian update)
    tdm_parser = subparsers.add_parser("tdm", help="TDM Bayesian update")
    tdm_parser.add_argument("--smiles", required=True, help="SMILES string")
    tdm_parser.add_argument("--dose", type=float, required=True, help="Dose (mg)")
    tdm_parser.add_argument(
        "--obs", required=True, nargs="+",
        help="Observations as time_h:conc_mg_L pairs (e.g. '1.0:0.015' '6.0:0.008')",
    )
    tdm_parser.add_argument("--interval", type=float, default=None, help="Dosing interval (h), for multi-dose regimens")
    tdm_parser.add_argument("--doses", type=int, default=1, help="Number of doses administered (default: 1)")
    tdm_parser.add_argument("--route", default="oral", choices=["oral", "iv"])
    tdm_parser.add_argument("--n-samples", type=int, default=2000, help="Prior MC samples (default: 2000)")
    tdm_parser.add_argument("--method", default="is", choices=["is", "enkf"], help="TDM method: is (importance sampling) or enkf (Ensemble Kalman Filter)")
    tdm_parser.add_argument("--verbose", "-v", action="store_true")

    # ddi command
    ddi_parser = subparsers.add_parser("ddi", help="DDI prediction (inhibition/induction)")
    ddi_parser.add_argument("--smiles", required=True, help="Victim drug SMILES")
    ddi_parser.add_argument("--dose", type=float, required=True, help="Victim dose (mg)")
    ddi_parser.add_argument("--route", default="oral", choices=["oral", "iv"])
    ddi_parser.add_argument(
        "--inhibitor", default=None,
        choices=["ketoconazole", "fluconazole", "quinidine"],
        help="Preset inhibitor name",
    )
    ddi_parser.add_argument(
        "--inducer", default=None,
        choices=["rifampin"],
        help="Preset inducer name",
    )
    ddi_parser.add_argument("--verbose", "-v", action="store_true")

    # dose-adjust command (MIPD)
    da_parser = subparsers.add_parser("dose-adjust", help="MIPD dose recommendation from TDM")
    da_parser.add_argument("--smiles", required=True, help="SMILES string")
    da_parser.add_argument("--dose", type=float, required=True, help="Current dose (mg)")
    da_parser.add_argument(
        "--obs", required=True, nargs="+",
        help="Observations as time_h:conc_mg_L pairs (e.g. '1.0:0.015')",
    )
    da_parser.add_argument("--target-css", type=float, required=True, help="Target Css_max (mg/L)")
    da_parser.add_argument("--interval", type=float, default=None, help="Dosing interval (h)")
    da_parser.add_argument("--doses", type=int, default=1, help="Number of doses administered")
    da_parser.add_argument("--route", default="oral", choices=["oral", "iv"])
    da_parser.add_argument("--n-samples", type=int, default=2000, help="Prior MC samples")
    da_parser.add_argument("--method", default="is", choices=["is", "enkf"], help="TDM method: is (importance sampling) or enkf (Ensemble Kalman Filter)")
    da_parser.add_argument("--verbose", "-v", action="store_true")

    # benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Run holdout benchmark")
    bench_parser.add_argument("--holdout", action="store_true", help="Run on holdout set only")
    bench_parser.add_argument("--max-drugs", type=int, default=None, help="Limit number of drugs")
    bench_parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    if args.command == "predict":
        _run_predict(args)
    elif args.command == "simulate":
        _run_simulate(args)
    elif args.command == "tdm":
        _run_tdm(args)
    elif args.command == "ddi":
        _run_ddi(args)
    elif args.command == "dose-adjust":
        _run_dose_adjust(args)
    elif args.command == "benchmark":
        _run_benchmark(args)
    else:
        parser.print_help()
        sys.exit(1)


def _run_predict(args: argparse.Namespace) -> None:
    """Run a single drug prediction and print results."""
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    from sisyphus.pipeline.predict import predict

    result = predict(args.smiles, args.dose, args.route)

    print(f"Drug: {result.drug_name}")
    print(f"Method: {result.method}")
    print(f"Confidence: {result.confidence}")
    print(f"Cmax: {result.pk.cmax.mean:.4f} mg/L")
    if result.pk.tmax:
        print(f"Tmax: {result.pk.tmax.mean:.2f} h")
    if result.pk.auc_0t:
        print(f"AUC: {result.pk.auc_0t.mean:.4f} mg*h/L")
    if result.pk.t_half:
        print(f"t½: {result.pk.t_half.mean:.2f} h")
    if result.engine_pk:
        print(f"  Engine Cmax: {result.engine_pk.cmax.mean:.4f} mg/L")
    if result.ml_pk:
        print(f"  ML Cmax: {result.ml_pk.cmax.mean:.4f} mg/L")
    if result.warnings:
        print(f"Warnings: {result.warnings}")


def _build_drug_and_graph(smiles: str, dose_mg: float, route: str = "oral"):
    """Shared helper: SMILES → (graph, compiled, drug)."""
    import numpy as np

    import sisyphus.engine.flux  # noqa: F401
    from sisyphus.engine.compiler import ODECompiler
    from sisyphus.graph.builder import build_from_yaml
    from sisyphus.predict.adme import predict_adme
    from sisyphus.predict.chemistry import compute_profile
    from sisyphus.predict.ivive import build_drug_on_graph

    _PHYSIOLOGY_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "physiology"
    graph = build_from_yaml(_PHYSIOLOGY_DIR / "reference_man.yaml")
    compiled = ODECompiler().compile(graph)

    profile = compute_profile(smiles)
    adme = predict_adme(profile)

    liver_enzymes = None
    if "liver" in graph.nodes and graph.nodes["liver"].enzymes:
        liver_enzymes = {
            tag: dist.mean for tag, dist in graph.nodes["liver"].enzymes.items()
        }

    drug = build_drug_on_graph(profile, adme, dose_mg, route, liver_enzymes=liver_enzymes)
    return graph, compiled, drug


def _run_simulate(args: argparse.Namespace) -> None:
    """Run multi-dose regimen simulation."""
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    import numpy as np

    from sisyphus.engine.compiler import ResolvedParams
    from sisyphus.regimen.profile import compute_steady_state_metrics
    from sisyphus.regimen.solver import solve_regimen
    from sisyphus.regimen.types import DosingRegimen

    graph, compiled, drug = _build_drug_and_graph(args.smiles, args.dose, args.route)

    if args.route == "oral":
        regimen = DosingRegimen.oral_repeated(args.dose, args.interval, args.doses)
    else:
        regimen = DosingRegimen.iv_infusion(args.dose, 0.0, args.interval, args.doses)

    rng = np.random.default_rng(42)
    params = ResolvedParams(graph.sample(rng), drug.sample(rng))

    t_total = regimen.last_dose_time_h + 48.0
    result = solve_regimen(compiled, params, regimen, t_total_h=t_total, dt_output=0.1)

    metrics = compute_steady_state_metrics(result, regimen, node="venous_blood")

    print(f"Drug: {drug.name}")
    print(f"Regimen: {args.dose:.0f} mg q{args.interval:.0f}h × {args.doses} ({args.route})")
    print(f"Solver: {'OK' if result.solver_success else 'FAILED'}")
    print()
    print(f"Css_max:              {metrics.css_max:.4f} mg/L")
    print(f"Css_min:              {metrics.css_min:.4f} mg/L")
    print(f"Accumulation ratio:   {metrics.accumulation_ratio:.3f}")
    print(f"Steady state:         {'Yes' if metrics.is_steady_state else 'No'}")
    if metrics.dose_to_steady_state > 0:
        print(f"SS reached at dose:   {metrics.dose_to_steady_state}")


def _parse_observations(obs_strings: list[str]):
    """Parse 'time:conc' strings into Observation objects."""
    from sisyphus.regimen.tdm import Observation

    observations = []
    for s in obs_strings:
        parts = s.split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid observation format {s!r}, expected 'time_h:conc_mg_L'")
        t = float(parts[0])
        c = float(parts[1])
        observations.append(Observation(time_h=t, concentration=c, cv=0.10))
    return observations


def _run_tdm(args: argparse.Namespace) -> None:
    """Run TDM Bayesian update."""
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    from sisyphus.regimen.types import DosingRegimen

    graph, compiled, drug = _build_drug_and_graph(args.smiles, args.dose, args.route)

    if args.doses == 1:
        if args.route == "oral":
            regimen = DosingRegimen.single_oral(args.dose)
        else:
            regimen = DosingRegimen.single_iv(args.dose)
    else:
        if args.interval is None:
            print("Error: --interval required for multi-dose regimens", file=sys.stderr)
            sys.exit(1)
        if args.route == "oral":
            regimen = DosingRegimen.oral_repeated(args.dose, args.interval, args.doses)
        else:
            regimen = DosingRegimen.iv_infusion(args.dose, 0.0, args.interval, args.doses)

    observations = _parse_observations(args.obs)

    if args.method == "enkf":
        from sisyphus.regimen.tdm_enkf import enkf_update

        result = enkf_update(
            compiled, graph, drug, regimen,
            observations=observations,
            n_ensemble=args.n_samples,
            seed=42,
        )
        method_label = "EnKF"
    else:
        from sisyphus.regimen.tdm import bayesian_update

        result = bayesian_update(
            compiled, graph, drug, regimen,
            observations=observations,
            n_prior=args.n_samples,
            seed=42,
        )
        method_label = "Importance Sampling"

    print(f"Drug: {drug.name}")
    print(f"Method: {method_label}")
    print(f"Observations: {len(observations)}")
    for obs in observations:
        print(f"  t={obs.time_h:.1f}h: {obs.concentration:.4f} mg/L")
    print(f"Samples: {result.n_successful}/{result.n_prior}")
    print(f"ESS: {result.ess:.1f} ({100 * result.ess / max(result.n_successful, 1):.1f}%)")
    print()
    print(f"{'':30s} {'Prior':>12s} {'Posterior':>12s}")
    print("-" * 54)
    print(f"{'Cmax (mg/L)':<30s} {result.prior_cmax.mean:>12.4f} {result.posterior_cmax.mean:>12.4f}")
    print(f"{'Cmax CV':<30s} {result.prior_cmax.cv:>11.1%} {result.posterior_cmax.cv:>11.1%}")
    for key in result.prior_params:
        print(f"{key:<30s} {result.prior_params[key]:>12.4f} {result.posterior_params[key]:>12.4f}")

    cv_red = (1.0 - result.posterior_cmax.cv / result.prior_cmax.cv) * 100 if result.prior_cmax.cv > 0 else 0.0
    print(f"\nCV reduction: {cv_red:.1f}%")


def _run_dose_adjust(args: argparse.Namespace) -> None:
    """Run MIPD dose recommendation."""
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    from sisyphus.regimen.dosing import recommend_dose
    from sisyphus.regimen.types import DosingRegimen

    graph, compiled, drug = _build_drug_and_graph(args.smiles, args.dose, args.route)

    if args.doses == 1:
        if args.route == "oral":
            regimen = DosingRegimen.single_oral(args.dose)
        else:
            regimen = DosingRegimen.single_iv(args.dose)
    else:
        if args.interval is None:
            print("Error: --interval required for multi-dose regimens", file=sys.stderr)
            sys.exit(1)
        if args.route == "oral":
            regimen = DosingRegimen.oral_repeated(args.dose, args.interval, args.doses)
        else:
            regimen = DosingRegimen.iv_infusion(args.dose, 0.0, args.interval, args.doses)

    observations = _parse_observations(args.obs)

    result = recommend_dose(
        compiled, graph, drug, regimen,
        observations=observations,
        target_css=args.target_css,
        n_prior=args.n_samples,
        seed=42,
        method=args.method,
    )

    print(f"Drug: {drug.name}")
    print(f"Current dose:     {result.current_dose_mg:.0f} mg")
    print(f"Target Css:       {result.target_css:.4f} mg/L")
    print()
    print(f"Posterior Css (current dose): {result.posterior_css_current:.4f} mg/L (CV {result.posterior_cv:.1%})")
    print(f"Scaling ratio:               {result.scaling_ratio:.2f}x")
    print()
    print(f"  Recommended dose:  {result.recommended_dose_mg:.0f} mg")
    print(f"  Predicted Css:     {result.predicted_css_recommended:.4f} mg/L")
    print()
    print(f"TDM: {result.tdm_result.n_successful}/{result.tdm_result.n_prior} samples, "
          f"ESS={result.tdm_result.ess:.0f}")


def _run_ddi(args: argparse.Namespace) -> None:
    """Run DDI prediction: victim alone vs victim + perpetrator."""
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    import numpy as np

    from sisyphus.ddi import (
        FLUCONAZOLE,
        KETOCONAZOLE,
        QUINIDINE,
        RIFAMPIN,
        apply_induction,
        apply_inhibition,
    )
    from sisyphus.engine.compiler import ODECompiler, ResolvedParams
    from sisyphus.engine.solver import solve
    from sisyphus.pk.endpoints import compute_endpoints

    graph, compiled, drug = _build_drug_and_graph(args.smiles, args.dose, args.route)

    # Resolve perpetrator
    inhibitors = {"ketoconazole": KETOCONAZOLE, "fluconazole": FLUCONAZOLE, "quinidine": QUINIDINE}
    inducers = {"rifampin": RIFAMPIN}

    if args.inhibitor and args.inducer:
        print("Error: specify --inhibitor or --inducer, not both", file=sys.stderr)
        sys.exit(1)
    if not args.inhibitor and not args.inducer:
        print("Error: specify --inhibitor or --inducer", file=sys.stderr)
        sys.exit(1)

    if args.inhibitor:
        perpetrator = inhibitors[args.inhibitor]
        ddi_graph = apply_inhibition(graph, perpetrator)
        ddi_type = f"inhibition ({perpetrator.name})"
    else:
        perpetrator = inducers[args.inducer]
        ddi_graph = apply_induction(graph, perpetrator)
        ddi_type = f"induction ({perpetrator.name})"

    def _solve_pk(g):
        import sisyphus.engine.flux  # noqa: F401
        comp = ODECompiler().compile(g)
        rng = np.random.default_rng(42)
        params = ResolvedParams(g.sample(rng), drug.sample(rng))
        y0 = np.zeros(comp.n_states)
        y0[comp.state_index[drug.administration_node]] = drug.dose_mg
        result = solve(comp, params, y0, t_span=(0, 24))
        return compute_endpoints(result)

    base_pk = _solve_pk(graph)
    ddi_pk = _solve_pk(ddi_graph)

    cmax_ratio = ddi_pk.cmax.mean / base_pk.cmax.mean if base_pk.cmax.mean > 0 else 0.0
    auc_ratio = ddi_pk.auc_0t.mean / base_pk.auc_0t.mean if base_pk.auc_0t.mean > 0 else 0.0

    print(f"Drug: {drug.name}")
    print(f"DDI: {ddi_type}")
    print()
    print(f"{'':20s} {'Alone':>12s} {'With DDI':>12s} {'Ratio':>8s}")
    print("-" * 52)
    print(f"{'Cmax (mg/L)':<20s} {base_pk.cmax.mean:>12.4f} {ddi_pk.cmax.mean:>12.4f} {cmax_ratio:>8.2f}")
    print(f"{'AUC (mg*h/L)':<20s} {base_pk.auc_0t.mean:>12.4f} {ddi_pk.auc_0t.mean:>12.4f} {auc_ratio:>8.2f}")
    if base_pk.t_half and ddi_pk.t_half:
        thalf_ratio = ddi_pk.t_half.mean / base_pk.t_half.mean if base_pk.t_half.mean > 0 else 0.0
        print(f"{'t½ (h)':<20s} {base_pk.t_half.mean:>12.2f} {ddi_pk.t_half.mean:>12.2f} {thalf_ratio:>8.2f}")


def _run_benchmark(args: argparse.Namespace) -> None:
    """Run holdout benchmark and print summary metrics."""
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    from sisyphus.validation.benchmark import run_benchmark

    result = run_benchmark(holdout_only=args.holdout, max_drugs=args.max_drugs)

    print("\nBenchmark Results:")
    print(f"  Drugs evaluated: {result.n_drugs}")
    print(f"  AAFE: {result.aafe:.3f}")
    print(f"  %2-fold: {result.pct_2fold:.1f}%")
    if result.pi_coverage_90 is not None:
        print(f"  90% PI coverage: {result.pi_coverage_90:.1%}")


if __name__ == "__main__":
    main()
