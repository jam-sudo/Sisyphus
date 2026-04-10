#!/usr/bin/env python3
"""Real-data comparison: amortized SBI posterior vs IBIS on morphine.

For a single observed Cmax (from the holdout reference), run:
  (a) the amortized NPE posterior via one forward pass + sampling
  (b) the IBIS (Iterated Batch Importance Sampling) posterior — the
      established TDM gold standard in Sisyphus.

Compare posterior statistics (mean, CV, 90% CI) over CLint, fup, Peff and
measure wall-clock speed.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("sbi_vs_ibis")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sisyphus.sbi.amortizer import load_result  # noqa: E402
from sisyphus.sbi.simulator import EngineSimulator  # noqa: E402
from sisyphus.validation.reference import load_reference  # noqa: E402

MORPHINE_SMILES = "CN1CCC23C4C1CC5=C2C(=C(C=C5)O)OC3C(C=C4)O"
MORPHINE_DOSE_MG = 30.0


def _posterior_stats(samples: np.ndarray) -> dict[str, float]:
    mean = float(np.mean(samples))
    std = float(np.std(samples))
    cv = std / abs(mean) if abs(mean) > 1e-12 else float("nan")
    lo, hi = np.quantile(samples, [0.05, 0.95])
    return {"mean": mean, "std": std, "cv": cv, "ci90_lo": float(lo), "ci90_hi": float(hi)}


def run_amortized(posterior_path: Path, x_obs: float, n_samples: int = 5000):
    result = load_result(posterior_path)
    import torch

    x_t = torch.tensor([[float(x_obs)]], dtype=torch.float32)
    t0 = time.time()
    samples_t = result.posterior.sample((n_samples,), x=x_t, show_progress_bars=False)
    elapsed = time.time() - t0
    samples = samples_t.detach().cpu().numpy()
    return samples, elapsed


def run_ibis(observed_cmax_mg_l: float, n_particles: int = 2000):
    """Run IBIS TDM on morphine with a single observation at t_peak."""
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
    profile = compute_profile(MORPHINE_SMILES)
    adme = predict_adme(profile)
    liver_enzymes = None
    if "liver" in graph.nodes and graph.nodes["liver"].enzymes:
        liver_enzymes = {
            tag: dist.mean for tag, dist in graph.nodes["liver"].enzymes.items()
        }
    drug = build_drug_on_graph(profile, adme, MORPHINE_DOSE_MG, "oral", liver_enzymes=liver_enzymes)

    regimen = DosingRegimen.single_oral(MORPHINE_DOSE_MG)
    # Observation at t=1.0h (near oral Cmax peak for morphine)
    observations = (Observation(time_h=1.0, concentration=observed_cmax_mg_l, cv=0.1),)

    t0 = time.time()
    ibis_res = ibis_update(
        compiled,
        graph,
        drug,
        regimen,
        observations,
        n_particles=n_particles,
        seed=42,
    )
    elapsed = time.time() - t0
    return ibis_res, elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--posterior", type=Path, default=ROOT / "models" / "sbi" / "morphine_posterior.pt"
    )
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--ibis-particles", type=int, default=2000)
    parser.add_argument(
        "--out", type=Path, default=ROOT / "data" / "validation" / "sbi_vs_ibis_morphine.json"
    )
    args = parser.parse_args()

    refs = [r for r in load_reference() if r.in_holdout and r.name.lower() == "morphine"]
    if not refs:
        log.error("morphine not found in holdout reference")
        sys.exit(1)
    ref = refs[0]
    cmax_obs = float(ref.cmax_obs)
    x_obs = float(np.log10(cmax_obs))

    log.info(
        "Morphine observation: Cmax=%.4f mg/L (log10=%.3f), dose=%.0f mg oral",
        cmax_obs,
        x_obs,
        ref.dose_mg,
    )

    log.info("Running amortized posterior...")
    samples, sbi_time = run_amortized(args.posterior, x_obs, n_samples=args.n_samples)
    log.info("  done in %.3fs (N=%d samples)", sbi_time, samples.shape[0])

    log.info("Running IBIS (gold standard)...")
    ibis_res, ibis_time = run_ibis(cmax_obs, n_particles=args.ibis_particles)
    log.info("  done in %.3fs (N=%d particles)", ibis_time, args.ibis_particles)

    # SBI posterior stats per theta dim
    sbi_stats = {
        "log10_clint_shift": _posterior_stats(samples[:, 0]),
        "fup": _posterior_stats(samples[:, 1]),
        "log10_peff_shift": _posterior_stats(samples[:, 2]),
    }

    # SBI posterior predictive Cmax via simulator
    log.info("Running SBI posterior predictive (100 theta samples × 1 sim each)...")
    sim = EngineSimulator.for_drug(MORPHINE_SMILES, MORPHINE_DOSE_MG, route="oral")
    pp_idx = np.random.default_rng(7).choice(samples.shape[0], size=100, replace=False)
    pp_log10_cmax = np.zeros(len(pp_idx))
    for i, k in enumerate(pp_idx):
        pp_log10_cmax[i] = sim.simulate_single(samples[k].astype(np.float64), seed=1000 + i)
    pp_log10_cmax = pp_log10_cmax[np.isfinite(pp_log10_cmax)]
    sbi_posterior_cmax_mean = float(10 ** np.mean(pp_log10_cmax))
    sbi_posterior_cmax_log10 = float(np.mean(pp_log10_cmax))

    # IBIS summary
    ibis_summary = {
        "prior_cmax_mean": float(ibis_res.prior_cmax.mean),
        "prior_cmax_cv": float(ibis_res.prior_cmax.cv),
        "posterior_cmax_mean": float(ibis_res.posterior_cmax.mean),
        "posterior_cmax_cv": float(ibis_res.posterior_cmax.cv),
        "cmax_ci_90": list(ibis_res.cmax_ci_90),
        "n_particles": int(ibis_res.n_particles),
        "n_successful": int(ibis_res.n_successful),
        "ess_last": float(ibis_res.ess_history[-1]) if ibis_res.ess_history else 0.0,
        "prior_params": {k: float(v) for k, v in ibis_res.prior_params.items()},
        "posterior_params": {k: float(v) for k, v in ibis_res.posterior_params.items()},
    }

    print("\n" + "=" * 70)
    print("MORPHINE TDM POSTERIOR: amortized SBI vs IBIS")
    print("=" * 70)
    print(f"Observed Cmax: {cmax_obs:.4f} mg/L  (log10 = {x_obs:.3f})")
    print()
    print(f"IBIS   ({ibis_time*1000:7.1f} ms): posterior Cmax mean = {ibis_summary['posterior_cmax_mean']:.4f} mg/L, CV = {ibis_summary['posterior_cmax_cv']*100:.1f}%")
    print(f"         CI90 = [{ibis_summary['cmax_ci_90'][0]:.4f}, {ibis_summary['cmax_ci_90'][1]:.4f}]  ESS = {ibis_summary['ess_last']:.1f}")
    print()
    print(f"SBI    ({sbi_time*1000:7.1f} ms): posterior Cmax mean = {sbi_posterior_cmax_mean:.4f} mg/L")
    print(f"         N posterior samples = {samples.shape[0]}")
    print()
    print(f"Speedup: {ibis_time/sbi_time:.1f}x")
    print()
    print("SBI posterior over theta:")
    for name, stats in sbi_stats.items():
        print(f"  {name:22s}: mean={stats['mean']:+7.3f} CV={stats['cv']:.2f}  CI90=[{stats['ci90_lo']:+7.3f}, {stats['ci90_hi']:+7.3f}]")

    report = {
        "drug": "morphine",
        "dose_mg": MORPHINE_DOSE_MG,
        "route": "oral",
        "observed_cmax_mg_l": cmax_obs,
        "observed_log10_cmax": x_obs,
        "sbi": {
            "theta_stats": sbi_stats,
            "posterior_cmax_predictive_mean_mg_l": sbi_posterior_cmax_mean,
            "posterior_cmax_predictive_log10": sbi_posterior_cmax_log10,
            "wall_seconds": sbi_time,
            "n_samples": int(samples.shape[0]),
        },
        "ibis": {
            **ibis_summary,
            "wall_seconds": ibis_time,
        },
        "speedup_sbi_over_ibis": float(ibis_time / sbi_time),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    log.info("Wrote %s", args.out)


if __name__ == "__main__":
    main()
