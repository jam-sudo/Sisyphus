"""Sisyphus CLI entry point.

Usage::

    sisyphus predict --smiles "CC(=O)Oc1ccccc1C(=O)O" --dose 500
    sisyphus simulate --smiles "CN(C)C(=N)NC(=N)N" --dose 500 --interval 12 --doses 14
    sisyphus tdm --help
    sisyphus mipd --smiles "CCO" --dose 100 --measured-f 0.5
    sisyphus benchmark --development-set
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import TYPE_CHECKING

from sisyphus.resources import get_resource_config

if TYPE_CHECKING:
    from sisyphus.core import DrugOnGraph
    from sisyphus.regimen.tdm import Observation
    from sisyphus.regimen.types import DosingRegimen


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

    mipd_parser = subparsers.add_parser("mipd", help="Update oral PK with measured data")
    mipd_parser.add_argument("--smiles", required=True)
    mipd_parser.add_argument("--dose", required=True, type=float, help="Dose in mg")
    mipd_parser.add_argument("--measured-f", type=float, help="Measured oral bioavailability")
    mipd_parser.add_argument("--measured-cmax", type=float, help="Measured Cmax in mg/L")
    mipd_parser.add_argument("--measured-auc", type=float, help="Measured AUC in mg*h/L")
    mipd_parser.add_argument("--obs", nargs="*", default=[], help="Time_h:concentration_mg_L")
    mipd_parser.add_argument("--route", choices=("oral", "iv"), default="oral")
    mipd_parser.add_argument("--interval", type=float, help="Repeated-dose interval in hours")
    mipd_parser.add_argument("--doses", type=int, help="Number of administered doses")
    mipd_parser.add_argument("--duration", type=float, default=1.0,
                             help="IV infusion duration in hours")
    mipd_parser.add_argument("--crcl", type=float, help="Creatinine clearance in mL/min")
    mipd_parser.add_argument("--crcl-cv", type=float, default=0.0,
                             help="Measurement CV for CrCl (default: 0)")
    mipd_parser.add_argument("--body-weight", type=float, help="Body weight in kg")
    mipd_parser.add_argument("--age-years", type=float, help="Postnatal age in years")
    mipd_parser.add_argument("--pma-weeks", type=float, help="Postmenstrual age in weeks")
    mipd_parser.add_argument("--structural-latents", action="store_true",
                             help="Infer CL, V, and oral absorption rate")
    mipd_parser.add_argument("--target-trough", help="Trough window low:high in mg/L")
    mipd_parser.add_argument("--target-peak", help="Cmax window low:high in mg/L")
    mipd_parser.add_argument("--target-auc24", help="AUC24 window low:high in mg*h/L")
    mipd_parser.add_argument("--candidate-doses", type=float, nargs="+")
    mipd_parser.add_argument("--candidate-intervals", type=float, nargs="+")
    mipd_parser.add_argument("--loading-target", type=float,
                             help="Loading-dose target concentration in mg/L")
    mipd_parser.add_argument("--n-samples", type=int, default=20000)

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
    tdm_parser.add_argument("--name", help="Drug name for per-drug auto routing")
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
             "Only used by SBI; uses hierarchical posterior when set.",
    )
    tdm_parser.add_argument("--body-weight", type=float, default=None,
                            help="Patient body weight in kg (SBI only; requires --age)")
    tdm_parser.add_argument("--age", type=float, default=None,
                            help="Patient age in years (SBI only; requires --body-weight)")
    tdm_parser.add_argument(
        "--phenotype",
        default=None,
        help="CYP phenotype spec, e.g. 'CYP2D6:PM' or '2D6:PM,2C9:IM'. "
             "Supported CYP tags: 2D6, 2C9, 1A2, 3A4, 2E1, 2B6, 2C19, 3A5. "
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
    ddi_parser.add_argument("--tdi-enzyme", help="Enzyme tag for user-supplied TDI")
    ddi_parser.add_argument("--kinact", type=float, help="TDI kinact (h^-1)")
    ddi_parser.add_argument("--ki-tdi", type=float, help="TDI KI (uM)")
    ddi_parser.add_argument("--kdeg", type=float, help="Enzyme turnover kdeg (h^-1)")
    ddi_parser.add_argument(
        "--concentration", type=float, help="Perpetrator total concentration (uM)"
    )
    ddi_parser.add_argument("--fu-perpetrator", type=float, default=1.0)
    ddi_parser.add_argument("--competitive-ki", type=float, help="Optional competitive Ki (uM)")
    ddi_parser.add_argument("--verbose", "-v", action="store_true")

    # pkpd command
    pd_parser = subparsers.add_parser("pkpd", help="EXPERIMENTAL: effect-site PK/PD scenario")
    pd_parser.add_argument("--smiles", required=True, help="Drug SMILES")
    pd_parser.add_argument("--dose", type=float, required=True, help="Dose (mg)")
    pd_parser.add_argument("--route", default="oral", choices=["oral", "iv"])
    pd_parser.add_argument("--ke0", type=float, required=True, help="Effect-site rate (h^-1)")
    pd_parser.add_argument(
        "--ec50", type=float, required=True, help="Half-maximal concentration (mg/L)"
    )
    pd_parser.add_argument("--verbose", "-v", action="store_true")

    # dose-adjust command (MIPD)
    da_parser = subparsers.add_parser(
        "dose-adjust", help="EXPERIMENTAL: MIPD dose recommendation from TDM"
    )
    da_parser.add_argument("--smiles", required=True, help="SMILES string")
    da_parser.add_argument("--name", help="Drug name for per-drug auto routing")
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
    da_parser.add_argument("--round-increment", type=float, default=None, help="Dose rounding increment (mg). Default: 10%% of current dose magnitude.")  # noqa: E501
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
        help="Run the repeatedly accessed development set",
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
    elif args.command == "mipd":
        _run_mipd(args)
    elif args.command == "simulate":
        _run_simulate(args)
    elif args.command == "tdm":
        _run_tdm(args)
    elif args.command == "ddi":
        _run_ddi(args)
    elif args.command == "pkpd":
        _run_pkpd(args)
    elif args.command == "dose-adjust":
        _run_dose_adjust(args)
    elif args.command == "benchmark":
        _run_benchmark(args)
    else:
        parser.print_help()
        sys.exit(1)


def _run_mipd(args: argparse.Namespace) -> None:
    """Print the MIPD posterior from measured PK observations."""
    from sisyphus.mipd import (
        Constraint,
        Covariates,
        DoseTarget,
        MeasuredAUC,
        MeasuredCmax,
        MeasuredConc,
        MeasuredF,
        predict_posterior,
        predict_tdm,
        recommend_dose,
    )
    from sisyphus.regimen.types import DosingRegimen

    observations = []
    if args.measured_f is not None:
        observations.append(MeasuredF(args.measured_f))
    if args.measured_cmax is not None:
        observations.append(MeasuredCmax(args.measured_cmax))
    if args.measured_auc is not None:
        observations.append(MeasuredAUC(args.measured_auc))
    for value in args.obs:
        try:
            t, c = map(float, value.split(":"))
        except ValueError as exc:
            raise ValueError(f"observation must be time_h:concentration_mg_L: {value!r}") from exc
        observations.append(MeasuredConc(value=c, t=t))
    covariates = (
        Covariates(
            crcl_ml_min=args.crcl, crcl_cv=args.crcl_cv,
            body_weight_kg=args.body_weight, age_years=args.age_years,
            pma_weeks=args.pma_weeks,
        ) if any(
            x is not None for x in (args.crcl, args.body_weight, args.age_years, args.pma_weeks)
        )
        or args.crcl_cv else None
    )
    target_specs = (
        ("trough", args.target_trough),
        ("cmax", args.target_peak),
        ("auc24", args.target_auc24),
    )
    if args.route == "iv" or args.interval is not None or args.doses is not None:
        if args.interval is None or args.doses is None:
            raise ValueError("repeated-dose MIPD requires --interval and --doses")
        if args.route == "iv":
            regimen = DosingRegimen.iv_infusion(
                dose_mg=args.dose, duration_h=args.duration,
                interval_h=args.interval, n_doses=args.doses,
            )
        else:
            regimen = DosingRegimen.oral_repeated(
                dose_mg=args.dose, interval_h=args.interval, n_doses=args.doses,
            )
        if any(value is not None for _, value in target_specs):
            constraints = []
            for quantity, value in target_specs:
                if value is None:
                    continue
                try:
                    low, high = map(float, value.split(":"))
                except ValueError as exc:
                    raise ValueError(f"{quantity} target must be low:high") from exc
                constraints.append(Constraint(quantity, low=low, high=high))
            rec = recommend_dose(
                args.smiles, regimen, observations, DoseTarget(tuple(constraints)),
                covariates=covariates,
                candidate_doses_mg=tuple(args.candidate_doses or (args.dose,)),
                candidate_intervals=tuple(args.candidate_intervals or (args.interval,)),
                loading_target_mg_l=args.loading_target, n_samples=args.n_samples,
            )
            print(f"Recommended maintenance: {rec.dose_mg:.4f} mg every {rec.interval_h:g} h")
            print(f"Target attainment: {rec.attainment_prob:.3f}")
            if rec.loading_dose is not None:
                lo, hi = rec.loading_dose.ci90
                print(f"Loading dose: {rec.loading_dose.point:.4f} mg (90% CI {lo:.4f}-{hi:.4f})")
            for warning in rec.warnings:
                print(f"Warning: {warning}", file=sys.stderr)
            return
        post = predict_tdm(
            args.smiles, regimen, observations, covariates=covariates,
            n_samples=args.n_samples, structural_latents=args.structural_latents,
        )
    else:
        if any(value is not None for _, value in target_specs) or args.loading_target is not None:
            raise ValueError("dose targets require --interval and --doses")
        if args.structural_latents:
            raise ValueError("structural latents require --interval and --doses")
        post = predict_posterior(
            args.smiles, args.dose, observations,
            covariates=covariates, n_samples=args.n_samples,
        )
    print(f"Engine posterior Cmax: {post.cmax.point:.4f} mg/L")
    print(f"Engine posterior AUC: {post.auc.point:.4f} mg*h/L")
    print(f"Posterior F: {post.f.point:.4f}")
    if post.volume_l is not None:
        print(f"Posterior central V: {post.volume_l.point:.4f} L")
    if post.absorption_rate_per_h is not None:
        print(f"Posterior ka: {post.absorption_rate_per_h.point:.4f} 1/h")
    for warning in post.warnings:
        print(f"Warning: {warning}", file=sys.stderr)


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
        + ("no rule-based warning flags" if result.in_applicability_domain else "flagged")
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


def _resolve_auto_method(
    drug_name: str,
    regimen: DosingRegimen | None = None,
    observations: tuple[Observation, ...] | None = None,
    drug: DrugOnGraph | None = None,
) -> tuple[str, bool]:
    """Look up the per-drug routing table and return (method, sbi_reweight).

    Falls back to ``("ibis", False)`` when the table is missing or the drug
    is not listed. Expects ``data/sbi/method_routing.json`` produced by
    ``scripts/sbi_build_routing_table.py``. The optional top-level
    ``sbi_reweight`` map in that file selects drugs that should use SBI
    likelihood reweighting (P6 Option 2).
    """
    import json

    if not _ROUTING_TABLE_PATH.exists():
        print("Warning: routing table missing; using IBIS default", file=sys.stderr)
        return "ibis", False
    try:
        with open(_ROUTING_TABLE_PATH) as f:
            table = json.load(f)
    except (OSError, ValueError) as exc:
        print(f"Warning: routing table unreadable ({exc}); using IBIS default", file=sys.stderr)
        return "ibis", False
    mapping = table.get("routes", {})
    reweight_map = table.get("sbi_reweight", {})
    lower = drug_name.lower().strip()
    method = table.get("default", "ibis")
    for key, m in mapping.items():
        if key.lower().strip() == lower:
            method = m
            break
    else:
        print(f"Warning: {drug_name!r} not in routing table; using {method} default",
              file=sys.stderr)
        return method, False
    reweight = False
    for key, flag in reweight_map.items():
        if key.lower().strip() == lower:
            reweight = bool(flag)
            break
    if regimen is not None and observations is not None and len(observations) > 1:
        if table.get("multi_observation", {}).get("validation_status") == "pending_corrected_sbc":
            return "ibis", False
        multi = table.get("multi_observation", {}).get("drugs", {})
        for key, decision in multi.items():
            if key.lower().strip() == lower and decision.get("use_native", False):
                if drug is not None:
                    from sisyphus.regimen.tdm_sbi import _native_multi_observation_model

                    if _native_multi_observation_model(drug) is None:
                        break
                import numpy as np

                from sisyphus.sbi.multi_observation import pack_observations

                try:
                    pack_observations(
                        [(obs.time_h, obs.concentration) for obs in observations],
                        regimen, np.zeros(12),
                    )
                except ValueError:
                    break
                return "sbi", False
    return method, reweight if method == "sbi" else False


def _route_drug_name(smiles: str, name: str | None) -> str:
    """Use an explicit name or find the matching compound YAML name."""
    if name:
        return name
    import yaml
    from rdkit import Chem

    target = Chem.MolFromSmiles(smiles)
    compounds = get_resource_config().data("compounds", required=False)
    for path in sorted(compounds.glob("*.yaml")):
        data = yaml.safe_load(path.read_text())
        candidate = Chem.MolFromSmiles(data["smiles"])
        if (target is not None and candidate is not None
                and Chem.MolToSmiles(target) == Chem.MolToSmiles(candidate)):
            return str(data["name"])
    return smiles


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
        route_name = _route_drug_name(args.smiles, args.name)
        eligible_regimen = regimen if all(
            getattr(args, key, None) is None for key in ("population", "body_weight", "age")
        ) else None
        resolved, auto_reweight = _resolve_auto_method(
            route_name, eligible_regimen, observations, drug
        )
        suffix = " +reweight" if auto_reweight else ""
        print(f"[auto] routing {route_name} → method={resolved}{suffix}",
              file=sys.stderr)
        cli_method = resolved
    method_map = {
        "is": "importance_sampling",
        "ibis": "ibis",
        "enkf": "enkf",
        "sbi": "sbi",
    }
    tdm_method = method_map[cli_method]

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
    bw = getattr(args, "body_weight", None)
    age_val = getattr(args, "age", None)
    if (bw is None) != (age_val is None):
        raise ValueError("--body-weight and --age must be supplied together")
    if tdm_method != "sbi" and (pop is not None or bw is not None):
        raise ValueError("--population, --body-weight, and --age require SBI")
    if pop:
        extra_kwargs["population_class"] = pop
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
    print(f"Method: {result.method_used}")
    for warning in result.warnings:
        print(f"Warning: {warning}", file=sys.stderr)
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
        route_name = _route_drug_name(args.smiles, args.name)
        dose_method, auto_reweight = _resolve_auto_method(
            route_name, regimen, observations, drug
        )
        print(f"[auto] routing {route_name} → method={dose_method}", file=sys.stderr)
    else:
        auto_reweight = False

    result = recommend_dose(
        compiled, graph, drug, regimen,
        observations=observations,
        target_css=args.target_css,
        dose_range=dose_range,
        round_increment=args.round_increment,
        n_prior=args.n_samples,
        seed=42,
        method=dose_method,
        sbi_reweight=auto_reweight,
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
    print(f"Inference method: {result.tdm_result.method_used}")
    for warning in result.tdm_result.warnings:
        print(f"Warning: {warning}", file=sys.stderr)


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
    import math
    from dataclasses import replace

    import numpy as np

    from sisyphus.ddi import (
        FLUCONAZOLE,
        KETOCONAZOLE,
        QUINIDINE,
        RIFAMPIN,
        Inhibitor,
        apply_induction,
        apply_inhibition,
    )
    from sisyphus.engine.compiler import ODECompiler, ResolvedParams
    from sisyphus.engine.solver import solve
    from sisyphus.pk.endpoints import compute_endpoints

    # Resolve perpetrator
    inhibitors = {"ketoconazole": KETOCONAZOLE, "fluconazole": FLUCONAZOLE, "quinidine": QUINIDINE}
    inducers = {"rifampin": RIFAMPIN}

    if not math.isfinite(args.fu_perpetrator) or not 0 <= args.fu_perpetrator <= 1:
        raise ValueError("--fu-perpetrator must be between 0 and 1")
    tdi_args = (args.kinact, args.ki_tdi, args.kdeg, args.concentration)
    if args.tdi_enzyme:
        if args.inhibitor or any(value is None for value in tdi_args):
            parser_error = "--tdi-enzyme requires --kinact, --ki-tdi, --kdeg, --concentration"
            raise ValueError(parser_error + " and cannot be combined with --inhibitor")
        perpetrator = Inhibitor(
            name="user-supplied TDI",
            enzyme_ki={args.tdi_enzyme: args.competitive_ki}
            if args.competitive_ki is not None else {},
            plasma_concentration=args.concentration,
            fu_perpetrator=args.fu_perpetrator,
            mechanism="time_dependent",
            enzyme_tdi={args.tdi_enzyme: (args.kinact, args.ki_tdi, args.kdeg)},
        )
    elif any(value is not None for value in tdi_args) or args.competitive_ki is not None:
        raise ValueError("TDI parameters require --tdi-enzyme")
    elif args.inhibitor:
        perpetrator = replace(
            inhibitors[args.inhibitor], fu_perpetrator=args.fu_perpetrator
        )
    else:
        perpetrator = None

    if not perpetrator and not args.inducer:
        raise ValueError("specify --inhibitor, --tdi-enzyme, or --inducer")
    graph, _, drug = _build_drug_and_graph(args.smiles, args.dose, args.route)
    ddi_graph = graph
    ddi_types = []
    if perpetrator:
        ddi_graph = apply_inhibition(ddi_graph, perpetrator)
        ddi_types.append(f"inhibition ({perpetrator.name})")
    if args.inducer:
        inducer = replace(inducers[args.inducer], fu_perpetrator=args.fu_perpetrator)
        ddi_graph = apply_induction(ddi_graph, inducer)
        ddi_types.append(f"induction ({inducer.name})")
    ddi_type = " + ".join(ddi_types)

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


def _run_pkpd(args: argparse.Namespace) -> None:
    """Simulate a dose and post-process its effect-site response."""
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    print("EXPERIMENTAL: PK/PD parameters require external validation.", file=sys.stderr)
    import numpy as np

    import sisyphus.engine.flux  # noqa: F401
    from sisyphus.engine.compiler import ResolvedParams
    from sisyphus.engine.solver import solve
    from sisyphus.pkpd import PDModel, compute_effect

    graph, compiled, drug = _build_drug_and_graph(args.smiles, args.dose, args.route)
    rng = np.random.default_rng(42)
    params = ResolvedParams(graph.sample(rng), drug.sample(rng))
    y0 = np.zeros(compiled.n_states)
    y0[compiled.state_index[drug.administration_node]] = drug.dose_mg
    result = solve(compiled, params, y0, t_span=(0, 24))
    effect = compute_effect(result, PDModel(ke0=args.ke0, ec50=args.ec50))
    print(f"Drug: {drug.name}")
    print(f"Peak effect: {effect.emax_achieved:.4f} at {effect.temax:.2f} h")


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
