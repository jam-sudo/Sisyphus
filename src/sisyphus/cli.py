"""Sisyphus CLI entry point.

Usage::

    sisyphus predict --smiles "CC(=O)Oc1ccccc1C(=O)O" --dose 500
    sisyphus simulate --smiles "CN(C)C(=N)NC(=N)N" --dose 500 --interval 12 --doses 14
    sisyphus tdm --smiles "Clc1ccc2c(c1)..." --dose 5 --obs "1.0:0.015"
    sisyphus benchmark --development-set
"""

from __future__ import annotations

import argparse
import logging
import sys

from sisyphus.resources import get_resource_config


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Sisyphus PBPK Prediction")
    subparsers = parser.add_subparsers(dest="command")

    # predict command
    pred_parser = subparsers.add_parser("predict", help="Predict PK for a SMILES")
    pred_parser.add_argument("--smiles", required=True, help="SMILES string")
    pred_parser.add_argument("--dose", type=float, required=True, help="Dose in mg")
    pred_parser.add_argument(
        "--route", default="oral", choices=["oral"],
        help="Production Cmax is validated for oral dosing only",
    )
    pred_parser.add_argument("--verbose", "-v", action="store_true")

    # simulate command (multi-dose)
    sim_parser = subparsers.add_parser(
        "simulate", help="EXPERIMENTAL: multi-dose engine simulation"
    )
    sim_parser.add_argument("--smiles", required=True, help="SMILES string")
    sim_parser.add_argument("--dose", type=float, required=True, help="Dose per administration (mg)")  # noqa: E501
    sim_parser.add_argument("--interval", type=float, required=True, help="Dosing interval (hours)")
    sim_parser.add_argument("--doses", type=int, required=True, help="Number of doses")
    sim_parser.add_argument("--route", default="oral", choices=["oral", "iv"])
    sim_parser.add_argument("--verbose", "-v", action="store_true")

    # tdm command (Bayesian update)
    tdm_parser = subparsers.add_parser("tdm", help="EXPERIMENTAL: TDM Bayesian update")
    tdm_parser.add_argument("--smiles", required=True, help="SMILES string")
    tdm_parser.add_argument("--dose", type=float, required=True, help="Dose (mg)")
    tdm_parser.add_argument(
        "--obs", required=True, nargs="+",
        help="Observations as time_h:conc_mg_L pairs (e.g. '1.0:0.015' '6.0:0.008')",
    )
    tdm_parser.add_argument("--interval", type=float, default=None, help="Dosing interval (h), for multi-dose regimens")  # noqa: E501
    tdm_parser.add_argument("--doses", type=int, default=1, help="Number of doses administered (default: 1)")  # noqa: E501
    tdm_parser.add_argument("--route", default="oral", choices=["oral", "iv"])
    tdm_parser.add_argument("--n-samples", type=int, default=2000, help="Prior MC samples (default: 2000)")  # noqa: E501
    tdm_parser.add_argument(
        "--method",
        default="ibis",
        choices=["is", "ibis", "enkf", "sbi", "auto"],
        help=(
            "TDM method: is (importance sampling) | ibis (iterated batch IS) | "
            "enkf (Ensemble Kalman Filter) | sbi (amortized SBI, Track A) | "
            "auto (look up per-drug routing table data/sbi/method_routing.json)"
        ),
    )
    tdm_parser.add_argument(
        "--population",
        default=None,
        help="Population class for hierarchical SBI (e.g. 'adult', 'pediatric_5y'). "
             "Uses hierarchical posterior when set.",
    )
    tdm_parser.add_argument("--body-weight", type=float, default=None,
                            help="Patient body weight in kg (continuous hierarchical)")
    tdm_parser.add_argument("--age", type=float, default=None,
                            help="Patient age in years (continuous hierarchical)")
    tdm_parser.add_argument(
        "--phenotype",
        default=None,
        help="CYP phenotype spec, e.g. 'CYP2D6:PM' or '2D6:PM,2C9:IM'. "
             "Scales hepatic enzyme abundance by CPIC activity score "
             "(PM 0.1×, IM 0.5×, EM/NM 1×, RM 1.5×, UM 2×).",
    )
    tdm_parser.add_argument("--verbose", "-v", action="store_true")

    # ddi command
    ddi_parser = subparsers.add_parser(
        "ddi", help="EXPERIMENTAL: mechanistic DDI scenario"
    )
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
    da_parser = subparsers.add_parser(
        "dose-adjust", help="EXPERIMENTAL: MIPD dose recommendation from TDM"
    )
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
    da_parser.add_argument(
        "--method",
        default="ibis",
        choices=["is", "ibis", "enkf", "sbi", "auto"],
        help="TDM method (same choices as tdm command)",
    )
    da_parser.add_argument("--phenotype", default=None, help="CYP phenotype spec (e.g. 'CYP2D6:PM'). See tdm --phenotype.")  # noqa: E501
    da_parser.add_argument("--dose-min", type=float, default=None, help="Minimum allowed dose (mg). Default: 0.1× current dose.")  # noqa: E501
    da_parser.add_argument("--dose-max", type=float, default=None, help="Maximum allowed dose (mg). Default: 10× current dose.")  # noqa: E501
    da_parser.add_argument("--round-increment", type=float, default=None, help="Dose rounding increment (mg). Default: 10% of current dose magnitude.")  # noqa: E501
    da_parser.add_argument("--verbose", "-v", action="store_true")

    # benchmark command
    bench_parser = subparsers.add_parser(
        "benchmark", help="Run retrospective development benchmark"
    )
    bench_parser.add_argument(
        "--development-set",
        "--holdout",
        dest="holdout",
        action="store_true",
        help="Run the repeatedly accessed N=107 development set",
    )
    bench_parser.add_argument("--max-drugs", type=int, default=None, help="Limit number of drugs")
    bench_parser.add_argument(
        "--compute-pi",
        action="store_true",
        help=(
            "Enable Monte Carlo uncertainty propagation per drug and report "
            "empirical 90%% PI coverage. Much slower; parameter-uncertainty "
            "interval only (not empirically calibrated)."
        ),
    )
    bench_parser.add_argument(
        "--n-mc-samples",
        type=int,
        default=1000,
        help="MC samples per drug when --compute-pi is set (default 1000).",
    )
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
    print(f"Execution: {result.execution_status} ({result.resource_profile} profile)")
    print(
        "Applicability: "
        + ("structurally in scope" if result.in_applicability_domain else "flagged")
        + " (confidence is not calibrated)"
    )
    print(f"Final oral Cmax: {result.pk.cmax.mean:.4f} mg/L")
    if result.cmax_prediction and result.cmax_prediction.residual_interval_90:
        lo, hi = result.cmax_prediction.residual_interval_90
        print(
            f"  Development empirical residual 90% interval: "
            f"{lo:.4f}–{hi:.4f} mg/L"
        )
    if result.cmax_prediction and result.cmax_prediction.parameter_interval_90:
        lo, hi = result.cmax_prediction.parameter_interval_90
        print(f"  Parameter-MC 90% interval: {lo:.4f}–{hi:.4f} mg/L")
    if result.engine_simulation:
        engine = result.engine_simulation.endpoints
        print("Engine simulation endpoints (not Meta endpoints):")
        print(f"  Cmax: {engine.cmax.mean:.4f} mg/L")
        print(f"  Tmax: {engine.tmax.mean:.2f} h")
        print(f"  AUC: {engine.auc_0t.mean:.4f} mg*h/L")
        if engine.t_half:
            print(f"  t½: {engine.t_half.mean:.2f} h")
    if result.ml_pk:
        print(f"  ML Cmax: {result.ml_pk.cmax.mean:.4f} mg/L")
    if result.warnings:
        print(f"Warnings: {result.warnings}")


def _apply_phenotype(graph, phenotype_spec: str | None):
    """Apply CYP phenotype scaling to the graph's liver enzymes.

    Must be called *after* the drug has been built from the reference graph,
    so that drug.enzyme_affinity stays calibrated against reference CLint and
    the engine's runtime product (node.enzymes × affinity) reflects the
    phenotype multiplier.
    """
    if not phenotype_spec:
        return graph
    from sisyphus.predict.phenotype import apply_phenotype_to_graph, parse_phenotype_spec
    phenotypes = parse_phenotype_spec(phenotype_spec)
    return apply_phenotype_to_graph(graph, phenotypes, node="liver")


def _build_drug_and_graph(
    smiles: str,
    dose_mg: float,
    route: str = "oral",
    phenotype_spec: str | None = None,
):
    """Shared helper: SMILES → (graph, compiled, drug).

    If ``phenotype_spec`` is given (e.g. ``"CYP2D6:PM"``), the graph's
    liver enzyme abundances are scaled AFTER drug.enzyme_affinity has
    been calibrated against the reference graph. Compile is run on the
    reference graph (topology unchanged) since ResolvedParams reads the
    final enzyme values at runtime.
    """

    from sisyphus.pipeline.context import prepare_simulation_context
    from sisyphus.predict.phenotype import parse_phenotype_spec

    phenotypes = parse_phenotype_spec(phenotype_spec) if phenotype_spec else None
    context = prepare_simulation_context(
        smiles,
        dose_mg,
        route,
        phenotypes=phenotypes,
    )
    if context.phenotype_report.unsupported:
        unsupported = ", ".join(tag for tag, _ in context.phenotype_report.unsupported)
        raise ValueError(f"Unsupported phenotype tag(s): {unsupported}")
    return context.graph, context.compiled, context.drug


def _run_simulate(args: argparse.Namespace) -> None:
    """Run multi-dose regimen simulation."""
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    print(
        "EXPERIMENTAL: multi-dose simulation is not externally clinically validated.",
        file=sys.stderr,
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

    t_total = regimen.last_dose_end_h + 48.0
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


_ROUTING_TABLE_PATH = get_resource_config().data(
    "sbi", "method_routing.json", required=False
)


def _resolve_auto_method(drug_name: str) -> tuple[str, bool]:
    """Look up the per-drug routing table and return (method, sbi_reweight).

    Falls back to ``("ibis", False)`` when the table is missing or the drug
    is not listed. Expects ``data/sbi/method_routing.json`` produced by
    ``scripts/sbi_build_routing_table.py``. The optional top-level
    ``sbi_reweight`` map in that file selects drugs that should use SBI
    likelihood reweighting (P6 Option 2).
    """
    import json

    if not _ROUTING_TABLE_PATH.exists():
        return "ibis", False
    try:
        with open(_ROUTING_TABLE_PATH) as f:
            table = json.load(f)
    except Exception:
        return "ibis", False
    mapping = table.get("routes", {})
    reweight_map = table.get("sbi_reweight", {})
    lower = drug_name.lower().strip()
    method = table.get("default", "ibis")
    for key, m in mapping.items():
        if key.lower().strip() == lower:
            method = m
            break
    reweight = False
    for key, flag in reweight_map.items():
        if key.lower().strip() == lower:
            reweight = bool(flag)
            break
    return method, reweight


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
    print(
        "EXPERIMENTAL: TDM methods lack temporally external real-patient validation.",
        file=sys.stderr,
    )
    from sisyphus.regimen.types import DosingRegimen

    graph, compiled, drug = _build_drug_and_graph(
        args.smiles, args.dose, args.route,
        phenotype_spec=getattr(args, "phenotype", None),
    )
    if getattr(args, "phenotype", None):
        print(f"[phenotype] applied {args.phenotype}", file=sys.stderr)

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

    # Map CLI choice to tdm.bayesian_update method string
    cli_method = args.method
    auto_reweight = False
    if cli_method == "auto":
        resolved, auto_reweight = _resolve_auto_method(drug.name)
        suffix = " +reweight" if auto_reweight else ""
        print(f"[auto] routing {drug.name} → method={resolved}{suffix}",
              file=sys.stderr)
        cli_method = resolved
    method_map = {
        "is": ("importance_sampling", "Importance Sampling"),
        "ibis": ("ibis", "IBIS (sequential MC + MCMC)"),
        "enkf": ("enkf", "EnKF (Ensemble Kalman Filter)"),
        "sbi": ("sbi", "Amortized SBI (multi-drug NSF)"),
    }
    tdm_method, method_label = method_map[cli_method]

    from sisyphus.regimen.tdm import bayesian_update

    # Pass logp for SBI path (unused by others)
    extra_kwargs = {}
    if tdm_method == "sbi":
        try:
            from sisyphus.predict.chemistry import compute_profile
            extra_kwargs["logp_hint"] = float(compute_profile(args.smiles).logp)
        except Exception:
            pass
        if auto_reweight:
            extra_kwargs["sbi_reweight"] = True
    pop = getattr(args, "population", None)
    if pop:
        extra_kwargs["population_class"] = pop
    bw = getattr(args, "body_weight", None)
    age_val = getattr(args, "age", None)
    if bw is not None and age_val is not None:
        extra_kwargs["body_weight_kg"] = bw
        extra_kwargs["age_years"] = age_val

    result = bayesian_update(
        compiled, graph, drug, regimen,
        observations=observations,
        n_prior=args.n_samples,
        seed=42,
        method=tdm_method,
        **extra_kwargs,
    )

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
    print(f"{'Cmax (mg/L)':<30s} {result.prior_cmax.mean:>12.4f} {result.posterior_cmax.mean:>12.4f}")  # noqa: E501
    print(f"{'Cmax CV':<30s} {result.prior_cmax.cv:>11.1%} {result.posterior_cmax.cv:>11.1%}")
    for key in result.prior_params:
        print(f"{key:<30s} {result.prior_params[key]:>12.4f} {result.posterior_params[key]:>12.4f}")

    cv_red = (1.0 - result.posterior_cmax.cv / result.prior_cmax.cv) * 100 if result.prior_cmax.cv > 0 else 0.0  # noqa: E501
    print(f"\nCV reduction: {cv_red:.1f}%")


def _run_dose_adjust(args: argparse.Namespace) -> None:
    """Run MIPD dose recommendation."""
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    print(
        "EXPERIMENTAL: dose recommendations are not for clinical use.",
        file=sys.stderr,
    )
    from sisyphus.regimen.dosing import recommend_dose
    from sisyphus.regimen.types import DosingRegimen

    graph, compiled, drug = _build_drug_and_graph(
        args.smiles, args.dose, args.route,
        phenotype_spec=getattr(args, "phenotype", None),
    )
    if getattr(args, "phenotype", None):
        print(f"[phenotype] applied {args.phenotype}", file=sys.stderr)

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

    dose_range = None
    if args.dose_min is not None or args.dose_max is not None:
        dose_range = (
            args.dose_min if args.dose_min is not None else 0.1,
            args.dose_max if args.dose_max is not None else args.dose * 10,
        )

    dose_method = args.method
    if dose_method == "auto":
        dose_method, auto_reweight = _resolve_auto_method(drug.name)
        if auto_reweight and dose_method == "sbi":
            print(
                "Warning: dose-adjust does not support SBI reweight routing; "
                "using IBIS instead.",
                file=sys.stderr,
            )
            dose_method = "ibis"
        print(f"[auto] routing {drug.name} → method={dose_method}", file=sys.stderr)

    result = recommend_dose(
        compiled, graph, drug, regimen,
        observations=observations,
        target_css=args.target_css,
        dose_range=dose_range,
        round_increment=args.round_increment,
        n_prior=args.n_samples,
        seed=42,
        method=dose_method,
    )

    print(f"Drug: {drug.name}")
    print(f"Current dose:     {result.current_dose_mg:.0f} mg")
    print(f"Target Css:       {result.target_css:.4f} mg/L")
    print()
    print(f"Posterior Css (current dose): {result.posterior_css_current:.4f} mg/L (CV {result.posterior_cv:.1%})")  # noqa: E501
    print(f"Scaling ratio:               {result.scaling_ratio:.2f}x")
    print()
    print(f"  Recommended dose:  {result.recommended_dose_mg:.0f} mg")
    print(f"  Predicted Css:     {result.predicted_css_recommended:.4f} mg/L")
    print()
    print(f"TDM: {result.tdm_result.n_successful}/{result.tdm_result.n_prior} samples, "
          f"ESS={result.tdm_result.ess:.0f}")
    print(f"Inference method: {result.inference_method}")


def _run_ddi(args: argparse.Namespace) -> None:
    """Run DDI prediction: victim alone vs victim + perpetrator."""
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    print(
        "EXPERIMENTAL: DDI output is a mechanistic scenario, not a clinical prediction.",
        file=sys.stderr,
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
    print(f"{'Cmax (mg/L)':<20s} {base_pk.cmax.mean:>12.4f} {ddi_pk.cmax.mean:>12.4f} {cmax_ratio:>8.2f}")  # noqa: E501
    print(f"{'AUC (mg*h/L)':<20s} {base_pk.auc_0t.mean:>12.4f} {ddi_pk.auc_0t.mean:>12.4f} {auc_ratio:>8.2f}")  # noqa: E501
    if base_pk.t_half and ddi_pk.t_half:
        thalf_ratio = ddi_pk.t_half.mean / base_pk.t_half.mean if base_pk.t_half.mean > 0 else 0.0
        print(f"{'t½ (h)':<20s} {base_pk.t_half.mean:>12.2f} {ddi_pk.t_half.mean:>12.2f} {thalf_ratio:>8.2f}")  # noqa: E501


def _run_benchmark(args: argparse.Namespace) -> None:
    """Run the retrospective development benchmark and print metrics."""
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    from sisyphus.validation.benchmark import run_benchmark

    result = run_benchmark(
        holdout_only=args.holdout,
        max_drugs=args.max_drugs,
        compute_pi=args.compute_pi,
        n_mc_samples=args.n_mc_samples,
    )

    print("\nRetrospective development benchmark (not independent holdout):")
    print(f"  Drugs evaluated: {result.n_drugs}")
    print(f"  AAFE: {result.aafe:.3f}")
    print(f"  %2-fold: {result.pct_2fold:.1f}%")
    if result.pi_coverage_90 is not None:
        print(f"  90% PI coverage: {result.pi_coverage_90:.1%} (n={result.n_pi_computed})")
        print("    (parameter uncertainty only; not empirically calibrated)")


if __name__ == "__main__":
    main()
