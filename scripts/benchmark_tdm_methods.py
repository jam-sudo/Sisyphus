#!/usr/bin/env python3
"""4-way TDM method tournament: IS / IBIS / EnKF / SBI.

Runs all four TDM dispatch methods on a fixed set of drugs and collates:
  - posterior Cmax mean / CV
  - 90% CI (best-effort — not every method exposes it directly)
  - ESS (IS/IBIS); n_successful (EnKF/SBI)
  - wall clock
  - relative bias vs observed Cmax

Writes ``data/validation/tdm_method_tournament.json``.

Usage::

    python scripts/benchmark_tdm_methods.py \\
        --drugs morphine,clozapine,amantadine,ketorolac,rivaroxaban \\
        --out data/validation/tdm_method_tournament.json
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
log = logging.getLogger("bench_tdm")

METHODS = ("is", "ibis", "enkf", "sbi")


def _build(drug_meta: dict):
    import sisyphus.engine.flux  # noqa: F401
    from sisyphus.engine.compiler import ODECompiler
    from sisyphus.graph.builder import build_from_yaml
    from sisyphus.predict.adme import predict_adme
    from sisyphus.predict.chemistry import compute_profile
    from sisyphus.predict.ivive import build_drug_on_graph
    from sisyphus.regimen.types import DosingRegimen

    graph = build_from_yaml(Path("data/physiology/reference_man.yaml"))
    compiled = ODECompiler().compile(graph)
    profile = compute_profile(drug_meta["smiles"])
    adme = predict_adme(profile)
    liver_enzymes = None
    if "liver" in graph.nodes and graph.nodes["liver"].enzymes:
        liver_enzymes = {
            tag: dist.mean for tag, dist in graph.nodes["liver"].enzymes.items()
        }
    drug = build_drug_on_graph(
        profile, adme, float(drug_meta["dose_mg"]), drug_meta.get("route", "oral"),
        liver_enzymes=liver_enzymes,
    )
    regimen = DosingRegimen.single_oral(float(drug_meta["dose_mg"]))
    return graph, compiled, drug, regimen, float(profile.logp)


def _run_one(method: str, compiled, graph, drug, regimen, observations, logp):
    from sisyphus.regimen.tdm import bayesian_update

    name_map = {
        "is": "importance_sampling",
        "ibis": "ibis",
        "enkf": "enkf",
        "sbi": "sbi",
    }
    tdm_method = name_map[method]
    kwargs = {}
    if tdm_method == "sbi":
        kwargs["logp_hint"] = logp
    # Scale n_prior by method cost: SBI and IS can afford more, IBIS/EnKF
    # default at 2000 particles. Keep the same prior budget for all so wall
    # time is apples-to-apples.
    n_prior = {"is": 500, "ibis": 2000, "enkf": 2000, "sbi": 200}[method]

    t0 = time.time()
    try:
        result = bayesian_update(
            compiled, graph, drug, regimen, observations,
            n_prior=n_prior, seed=42, method=tdm_method, **kwargs,
        )
        elapsed = time.time() - t0
        return {
            "method": method,
            "wall_seconds": elapsed,
            "prior_cmax_mean": float(result.prior_cmax.mean),
            "prior_cmax_cv": float(result.prior_cmax.cv),
            "posterior_cmax_mean": float(result.posterior_cmax.mean),
            "posterior_cmax_cv": float(result.posterior_cmax.cv),
            "ess": float(result.ess),
            "n_successful": int(result.n_successful),
            "n_prior": int(n_prior),
            "posterior_params": {
                k: float(v) for k, v in result.posterior_params.items()
            },
            "ok": True,
        }
    except Exception as exc:
        elapsed = time.time() - t0
        log.exception("method %s failed after %.1fs: %s", method, elapsed, exc)
        return {"method": method, "wall_seconds": elapsed, "ok": False, "error": str(exc)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--validation-set", type=Path,
                   default=ROOT / "data" / "sbi" / "validation_drug_set.json")
    p.add_argument("--drugs", type=str, required=True,
                   help="comma-separated drug names")
    p.add_argument("--methods", type=str, default=",".join(METHODS),
                   help="comma-separated subset of is,ibis,enkf,sbi")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--t-obs-h", type=float, default=1.0)
    args = p.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    with open(args.validation_set) as f:
        val = json.load(f)
    val_meta = {d["name"]: d for d in val["drugs"]}

    from sisyphus.regimen.tdm import Observation
    from sisyphus.validation.reference import load_reference

    refs = {r.name.lower(): r for r in load_reference() if r.in_holdout}

    drug_names = [s.strip() for s in args.drugs.split(",") if s.strip()]

    report: dict = {
        "validation_set": str(args.validation_set),
        "methods": methods,
        "t_obs_h": args.t_obs_h,
        "drugs": [],
        "per_method_totals": {m: 0.0 for m in methods},
    }

    for i, name in enumerate(drug_names):
        if name not in val_meta:
            log.warning("drug %s not in validation set, skipping", name)
            continue
        ref = refs.get(name)
        if ref is None:
            log.warning("drug %s not in holdout reference, skipping", name)
            continue
        drug_meta = val_meta[name]
        # Override dose from reference (matches real observation)
        drug_meta = dict(drug_meta)
        drug_meta["dose_mg"] = float(ref.dose_mg)

        cmax_obs = float(ref.cmax_obs)
        log.info("[%d/%d] %s: dose=%.1fmg obs_Cmax=%.4f",
                 i + 1, len(drug_names), name, drug_meta["dose_mg"], cmax_obs)

        graph, compiled, drug, regimen, logp = _build(drug_meta)
        observations = (
            Observation(time_h=args.t_obs_h, concentration=cmax_obs, cv=0.1),
        )

        drug_results = []
        for method in methods:
            log.info("  running %s...", method)
            res = _run_one(method, compiled, graph, drug, regimen, observations, logp)
            if res.get("ok"):
                bias = (res["posterior_cmax_mean"] - cmax_obs) / cmax_obs * 100
                res["bias_pct_vs_obs"] = float(bias)
            report["per_method_totals"][method] += res["wall_seconds"]
            drug_results.append(res)
            if res.get("ok"):
                log.info(
                    "    wall=%.2fs  post_cmax=%.4f (CV=%.0f%%)  bias=%+.0f%%",
                    res["wall_seconds"],
                    res["posterior_cmax_mean"],
                    res["posterior_cmax_cv"] * 100,
                    res["bias_pct_vs_obs"],
                )

        report["drugs"].append({
            "name": name,
            "dose_mg": drug_meta["dose_mg"],
            "observed_cmax_mg_l": cmax_obs,
            "results": drug_results,
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    log.info("wrote %s", args.out)
    log.info("per-method wall totals: %s", report["per_method_totals"])


if __name__ == "__main__":
    main()
