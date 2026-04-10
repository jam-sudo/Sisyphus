#!/usr/bin/env python3
"""Multi-drug amortized SBI vs IBIS comparison.

For each target drug in a provided validation set (subset), run:
  (a) Amortized posterior query on the multi-drug NSF network
  (b) IBIS with MCMC rejuvenation on the full scipy engine

Aggregate speedup, posterior mean agreement, and 90% CI comparison
into a single JSON report.

Usage::

    python scripts/sbi_compare_ibis_multi_drug.py \\
        --posterior models/sbi/multi_drug_train_nsf.pt \\
        --validation-set data/sbi/validation_drug_set.json \\
        --drugs morphine,clozapine,rivaroxaban,diclofenac \\
        --ibis-particles 2000 \\
        --out data/validation/sbi_vs_ibis_multi_drug.json
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
log = logging.getLogger("sbi_vs_ibis_md")

from sisyphus.sbi.amortizer import load_result  # noqa: E402
from sisyphus.sbi.multi_drug import (  # noqa: E402
    DrugSpec,
    MultiDrugSimulator,
    extract_drug_features,
)
from sisyphus.validation.reference import load_reference  # noqa: E402


def _posterior_stats(samples: np.ndarray) -> dict[str, float]:
    mean = float(np.mean(samples))
    std = float(np.std(samples))
    cv = std / abs(mean) if abs(mean) > 1e-12 else float("nan")
    lo, hi = np.quantile(samples, [0.05, 0.95])
    return {
        "mean": mean,
        "std": std,
        "cv": cv,
        "ci90_lo": float(lo),
        "ci90_hi": float(hi),
    }


def _run_amortized(
    posterior, x_obs: float, drug_features: np.ndarray, n_samples: int = 5000
) -> tuple[np.ndarray, float]:
    import torch

    vec = np.concatenate([[float(x_obs)], drug_features]).astype(np.float32)
    x_t = torch.as_tensor(vec[None, :], dtype=torch.float32)
    t0 = time.time()
    samples = posterior.sample((n_samples,), x=x_t, show_progress_bars=False)
    elapsed = time.time() - t0
    return samples.detach().cpu().numpy(), elapsed


def _run_ibis_for_drug(
    smiles: str,
    dose_mg: float,
    cmax_obs: float,
    t_obs_h: float,
    n_particles: int,
    seed: int = 42,
) -> tuple[dict, float]:
    import sisyphus.engine.flux  # noqa: F401
    from sisyphus.engine.compiler import ODECompiler
    from sisyphus.graph.builder import build_from_yaml
    from sisyphus.predict.adme import predict_adme
    from sisyphus.predict.chemistry import compute_profile
    from sisyphus.predict.ivive import build_drug_on_graph
    from sisyphus.regimen.tdm import Observation
    from sisyphus.regimen.tdm_ibis import ibis_update
    from sisyphus.regimen.types import DosingRegimen

    graph = build_from_yaml(Path("data/physiology/reference_man.yaml"))
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
    regimen = DosingRegimen.single_oral(dose_mg)
    observations = (
        Observation(time_h=float(t_obs_h), concentration=float(cmax_obs), cv=0.1),
    )

    t0 = time.time()
    ibis_res = ibis_update(
        compiled,
        graph,
        drug,
        regimen,
        observations,
        n_particles=n_particles,
        seed=seed,
    )
    elapsed = time.time() - t0
    summary = {
        "prior_cmax_mean": float(ibis_res.prior_cmax.mean),
        "prior_cmax_cv": float(ibis_res.prior_cmax.cv),
        "posterior_cmax_mean": float(ibis_res.posterior_cmax.mean),
        "posterior_cmax_cv": float(ibis_res.posterior_cmax.cv),
        "cmax_ci_90": list(ibis_res.cmax_ci_90),
        "n_particles": int(ibis_res.n_particles),
        "n_successful": int(ibis_res.n_successful),
        "ess_last": float(ibis_res.ess_history[-1]) if ibis_res.ess_history else 0.0,
        "posterior_params": {
            k: float(v) for k, v in ibis_res.posterior_params.items()
        },
    }
    return summary, elapsed


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--posterior", type=Path, required=True)
    p.add_argument(
        "--validation-set",
        type=Path,
        default=ROOT / "data" / "sbi" / "validation_drug_set.json",
    )
    p.add_argument("--drugs", type=str, required=True,
                   help="comma-separated drug names")
    p.add_argument("--n-samples", type=int, default=5000)
    p.add_argument("--ibis-particles", type=int, default=2000)
    p.add_argument("--t-obs-h", type=float, default=1.0)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    log.info("Loading posterior: %s", args.posterior)
    result = load_result(args.posterior)
    import torch
    aux_path = args.posterior.with_suffix(".aux.pt")
    feat_mean = feat_std = None
    if aux_path.exists():
        aux = torch.load(aux_path, weights_only=False)
        feat_mean = np.asarray(aux["feat_mean"], dtype=np.float64)
        feat_std = np.asarray(aux["feat_std"], dtype=np.float64)
        log.info("  loaded feature standardizer")

    with open(args.validation_set) as f:
        val_data = json.load(f)
    val_meta = {d["name"]: d for d in val_data["drugs"]}

    refs = {r.name.lower(): r for r in load_reference() if r.in_holdout}

    target_names = [s.strip() for s in args.drugs.split(",") if s.strip()]
    report: dict = {
        "posterior": str(args.posterior),
        "ibis_particles": args.ibis_particles,
        "n_sbi_samples": args.n_samples,
        "t_obs_h": args.t_obs_h,
        "drugs": [],
    }

    total_ibis = 0.0
    total_sbi = 0.0

    for i, name in enumerate(target_names):
        if name not in val_meta:
            log.warning("drug %s not in validation set, skipping", name)
            continue
        meta = val_meta[name]
        ref = refs.get(name)
        if ref is None:
            log.warning("drug %s not in holdout reference, skipping", name)
            continue
        cmax_obs = float(ref.cmax_obs)
        dose_mg = float(ref.dose_mg)
        x_obs = float(np.log10(cmax_obs))
        log.info(
            "[%d/%d] %s: observed Cmax=%.4f mg/L (log10=%.3f) @ dose=%.1fmg",
            i + 1, len(target_names), name, cmax_obs, x_obs, dose_mg,
        )

        # Build per-drug simulator to get features + run posterior predictive.
        spec = DrugSpec(name=name, smiles=meta["smiles"], dose_mg=dose_mg)
        bundle = MultiDrugSimulator.for_single(spec)
        sim = bundle.simulators[name]
        feats = extract_drug_features(sim, logp=float(meta.get("logp", 2.0)))
        feats_std = (feats - feat_mean) / feat_std if feat_mean is not None else feats

        # Amortized posterior query
        samples, sbi_time = _run_amortized(
            result.posterior, x_obs, feats_std, n_samples=args.n_samples
        )
        total_sbi += sbi_time

        # Posterior predictive Cmax (via engine)
        rng = np.random.default_rng(12345 + i)
        idx = rng.choice(samples.shape[0], size=50, replace=False)
        pp_log10 = np.zeros(len(idx))
        for j, k in enumerate(idx):
            pp_log10[j] = sim.simulate_single(samples[k].astype(np.float64), seed=50000 + 100 * i + j)
        pp_log10 = pp_log10[np.isfinite(pp_log10)]
        sbi_pp_cmax_mean = float(10 ** np.mean(pp_log10)) if len(pp_log10) else float("nan")

        # IBIS (gold standard)
        log.info("  running IBIS (%d particles)...", args.ibis_particles)
        ibis_summary, ibis_time = _run_ibis_for_drug(
            smiles=meta["smiles"],
            dose_mg=dose_mg,
            cmax_obs=cmax_obs,
            t_obs_h=args.t_obs_h,
            n_particles=args.ibis_particles,
        )
        total_ibis += ibis_time

        speedup = ibis_time / max(sbi_time, 1e-9)
        drug_report = {
            "name": name,
            "dose_mg": dose_mg,
            "observed_cmax_mg_l": cmax_obs,
            "observed_log10_cmax": x_obs,
            "sbi": {
                "wall_seconds": sbi_time,
                "n_samples": int(samples.shape[0]),
                "theta_stats": {
                    "log10_clint_shift": _posterior_stats(samples[:, 0]),
                    "fup": _posterior_stats(samples[:, 1]),
                    "log10_peff_shift": _posterior_stats(samples[:, 2]),
                },
                "posterior_predictive_cmax_mg_l": sbi_pp_cmax_mean,
            },
            "ibis": {
                **ibis_summary,
                "wall_seconds": ibis_time,
            },
            "speedup": float(speedup),
        }
        report["drugs"].append(drug_report)
        log.info(
            "  SBI=%.3fs  IBIS=%.1fs  speedup=%.1fx  "
            "sbi_pp_cmax=%.4f  ibis_post_cmax=%.4f",
            sbi_time, ibis_time, speedup, sbi_pp_cmax_mean,
            ibis_summary["posterior_cmax_mean"],
        )

    report["total_sbi_seconds"] = total_sbi
    report["total_ibis_seconds"] = total_ibis
    report["cumulative_speedup"] = float(total_ibis / max(total_sbi, 1e-9))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    log.info("Wrote %s", args.out)
    log.info(
        "Cumulative: SBI=%.3fs  IBIS=%.1fs  speedup=%.1fx",
        total_sbi, total_ibis, report["cumulative_speedup"],
    )


if __name__ == "__main__":
    main()
