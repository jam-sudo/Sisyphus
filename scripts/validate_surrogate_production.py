#!/usr/bin/env python3
"""Validate the neural surrogate against the full scipy engine on real
production drugs after the Track D1 ``params_to_features_single`` fix.

For each drug in ``data/sbi/validation_drug_set.json`` (or a CLI-specified
subset), run:
  1. Full scipy engine single-dose oral simulation → Cmax_engine
  2. Surrogate ensemble prediction on the same DrugOnGraph → Cmax_surrogate
Compute fold error, relative error, and pass/fail gate.

Output: ``data/validation/surrogate_production_accuracy.json``

Gate (mirroring surrogate training validation):
  - ≥ 80 % of drugs within 30 % relative error (fold 1.3)
  - R² ≥ 0.5 on log10(Cmax)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("val_surrogate")


def _build_drug_and_graph(smiles: str, dose_mg: float):
    import sisyphus.engine.flux  # noqa: F401
    from sisyphus.engine.compiler import ODECompiler
    from sisyphus.graph.builder import build_from_yaml
    from sisyphus.predict.adme import predict_adme
    from sisyphus.predict.chemistry import compute_profile
    from sisyphus.predict.ivive import build_drug_on_graph

    graph = build_from_yaml(ROOT / "data" / "physiology" / "reference_man.yaml")
    compiled = ODECompiler().compile(graph)
    profile = compute_profile(smiles)
    adme = predict_adme(profile)
    liver_enzymes = None
    if "liver" in graph.nodes and graph.nodes["liver"].enzymes:
        liver_enzymes = {
            tag: dist.mean for tag, dist in graph.nodes["liver"].enzymes.items()
        }
    drug = build_drug_on_graph(
        profile, adme, dose_mg, "oral", liver_enzymes=liver_enzymes
    )
    return graph, compiled, drug, float(profile.logp)


def _engine_cmax(compiled, graph, drug) -> float:
    """Run full scipy engine for a single drug, return max venous_blood Cmax."""
    from sisyphus.engine.compiler import ResolvedParams
    from sisyphus.engine.solver import solve

    rng = np.random.default_rng(0)
    rg = graph.sample(rng)
    rd = drug.sample(rng)
    params = ResolvedParams(rg, rd)
    y0 = np.zeros(compiled.n_states)
    y0[compiled.state_index[drug.administration_node]] = drug.dose_mg
    result = solve(compiled, params, y0, t_span=(0.0, 48.0))
    if not result.solver_success or "venous_blood" not in result.concentrations:
        return float("nan")
    return float(np.max(result.concentrations["venous_blood"]))


def _surrogate_cmax(compiled, graph, drug, models, scaler_mean, scaler_std) -> tuple[float, float]:
    """Run surrogate ensemble for a single drug, return (cmax_mean, ensemble_std)."""
    from sisyphus.engine.compiler import ResolvedParams
    from sisyphus.ml.surrogate import (
        features_in_distribution,
        params_to_features_single,
        predict_surrogate_ensemble,
    )

    rng = np.random.default_rng(0)
    rg = graph.sample(rng)
    rd = drug.sample(rng)
    params = ResolvedParams(rg, rd)
    feats = params_to_features_single(params)
    ok, offenders = features_in_distribution(feats)

    pred_log_cd, uncertainty = predict_surrogate_ensemble(
        models,
        feats[None, :],
        np.asarray(scaler_mean),
        np.asarray(scaler_std),
    )
    cmax = float(10 ** pred_log_cd[0] * drug.dose_mg)
    return cmax, float(uncertainty[0]), ok, offenders


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--validation-set", type=Path,
                   default=ROOT / "data" / "sbi" / "validation_drug_set.json")
    p.add_argument("--out", type=Path,
                   default=ROOT / "data" / "validation" / "surrogate_production_accuracy.json")
    p.add_argument("--model-dir", type=Path,
                   default=ROOT / "models" / "surrogate")
    p.add_argument("--n-ensemble", type=int, default=5)
    p.add_argument("--gate-within-pct", type=float, default=0.30,
                   help="pass gate: fraction within this relative error")
    p.add_argument("--gate-min-pass", type=float, default=0.80,
                   help="minimum fraction of drugs passing gate-within-pct")
    args = p.parse_args()

    from sisyphus.ml.surrogate import load_surrogate_ensemble

    log.info("Loading surrogate ensemble (%d models)", args.n_ensemble)
    models = load_surrogate_ensemble(args.model_dir, n_ensemble=args.n_ensemble)
    scaler = np.load(args.model_dir / "scaler.npz")
    scaler_mean, scaler_std = scaler["mean"], scaler["std"]

    with open(args.validation_set) as f:
        val = json.load(f)
    drugs = val["drugs"]
    log.info("Validating on %d drugs", len(drugs))

    report: dict = {
        "validation_set": str(args.validation_set),
        "model_dir": str(args.model_dir),
        "n_ensemble": args.n_ensemble,
        "gate_within_pct": args.gate_within_pct,
        "drugs": [],
    }

    engine_log = []
    surrogate_log = []
    rel_errors = []
    fold_errors = []
    ood_count = 0
    pass_count = 0

    t_total = time.time()
    for i, d in enumerate(drugs):
        name = d["name"]
        smi = d["smiles"]
        dose = float(d["dose_mg"])

        t0 = time.time()
        graph, compiled, drug, logp = _build_drug_and_graph(smi, dose)

        t_engine_start = time.time()
        cmax_engine = _engine_cmax(compiled, graph, drug)
        engine_time = time.time() - t_engine_start

        t_sur_start = time.time()
        cmax_sur, sur_std, in_dist, offenders = _surrogate_cmax(
            compiled, graph, drug, models, scaler_mean, scaler_std
        )
        sur_time = time.time() - t_sur_start

        if not (np.isfinite(cmax_engine) and cmax_engine > 0
                and np.isfinite(cmax_sur) and cmax_sur > 0):
            log.warning("[%d/%d] %s: NaN/zero Cmax (eng=%.4f sur=%.4f)",
                        i + 1, len(drugs), name, cmax_engine, cmax_sur)
            drug_report = {
                "name": name, "dose_mg": dose,
                "cmax_engine": cmax_engine, "cmax_surrogate": cmax_sur,
                "in_distribution": in_dist, "offenders": offenders,
                "ok": False,
            }
            report["drugs"].append(drug_report)
            continue

        engine_log.append(np.log10(cmax_engine))
        surrogate_log.append(np.log10(cmax_sur))
        rel_err = (cmax_sur - cmax_engine) / cmax_engine
        rel_errors.append(rel_err)
        fold = max(cmax_sur, cmax_engine) / min(cmax_sur, cmax_engine)
        fold_errors.append(fold)
        passed = abs(rel_err) <= args.gate_within_pct
        if passed:
            pass_count += 1
        if not in_dist:
            ood_count += 1

        log.info(
            "[%d/%d] %-15s  eng=%.4f (t=%.2fs)  sur=%.4f (t=%.3fs)  "
            "rel_err=%+.1f%%  fold=%.2f  in_dist=%s  pass=%s",
            i + 1, len(drugs), name, cmax_engine, engine_time,
            cmax_sur, sur_time, rel_err * 100, fold, in_dist, passed,
        )
        report["drugs"].append({
            "name": name, "dose_mg": dose,
            "cmax_engine": cmax_engine,
            "cmax_surrogate": cmax_sur,
            "engine_time_s": engine_time,
            "surrogate_time_s": sur_time,
            "relative_error": rel_err,
            "fold_error": fold,
            "ensemble_std_log10": sur_std,
            "in_distribution": in_dist,
            "offenders": offenders,
            "passed_gate": passed,
            "ok": True,
        })

    n_ok = len(rel_errors)
    if n_ok == 0:
        log.error("No successful drug pairs — cannot compute gate")
        sys.exit(1)

    rel_arr = np.array(rel_errors)
    fold_arr = np.array(fold_errors)
    eng_arr = np.array(engine_log)
    sur_arr = np.array(surrogate_log)

    ss_res = np.sum((sur_arr - eng_arr) ** 2)
    ss_tot = np.sum((eng_arr - np.mean(eng_arr)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    pass_frac = pass_count / n_ok
    gate_passed = pass_frac >= args.gate_min_pass

    report["summary"] = {
        "n_total": len(drugs),
        "n_ok": n_ok,
        "pass_count": pass_count,
        "pass_fraction": float(pass_frac),
        "ood_count": ood_count,
        "mean_abs_rel_error": float(np.mean(np.abs(rel_arr))),
        "median_abs_rel_error": float(np.median(np.abs(rel_arr))),
        "mean_fold_error": float(np.mean(fold_arr)),
        "r2_log10": float(r2),
        "gate_passed": bool(gate_passed),
        "gate_within_pct": args.gate_within_pct,
        "gate_min_pass": args.gate_min_pass,
        "total_wall_seconds": float(time.time() - t_total),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    log.info("Wrote %s", args.out)
    log.info(
        "SUMMARY: %d/%d pass (%.0f%%), mean |rel_err|=%.1f%%, R²=%.3f, OOD=%d, gate=%s",
        pass_count, n_ok, pass_frac * 100,
        np.mean(np.abs(rel_arr)) * 100, r2, ood_count,
        "PASS" if gate_passed else "FAIL",
    )


if __name__ == "__main__":
    main()
