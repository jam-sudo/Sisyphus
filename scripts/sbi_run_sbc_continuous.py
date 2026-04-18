#!/usr/bin/env python3
"""Per-(population, drug) SBC gate for the continuous hierarchical amortizer.

Validates the continuous (body_weight, age) conditioned NPE on a 4-point
age/BW grid × validation drugs. For each (population, drug) pair:

    1. Build physiology graph via ``generate_physiology(bw, age)``.
    2. Compile ODE + build drug (adult reference features).
    3. Draw n_calibration thetas from prior, simulate log10(Cmax) per draw.
    4. Pack 15D observation = [log10_cmax, drug_features_std(12), log_bw_norm, log_age_norm]
       then standardize 14 non-cmax dims with feat_mean/feat_std from aux.
    5. Sample posterior, compute ranks + coverage.
    6. KS uniformity test + coverage deviation.

Gate (per pair):
    - All KS p-values > 0.01
    - All coverage levels within 10pp of nominal
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
log = logging.getLogger("sbi_sbc_continuous")

from sisyphus.sbi.multi_drug import (  # noqa: E402
    extract_drug_features,
    load_drug_specs_from_json,
)
from sisyphus.sbi.physiology_generator import generate_physiology  # noqa: E402
from sisyphus.sbi.priors import build_box_prior  # noqa: E402
from sisyphus.sbi.simulator import EngineSimulator, apply_theta_to_drug  # noqa: E402


_ADULT_PHYS = ROOT / "data" / "physiology" / "reference_man.yaml"
_DEFAULT_POPS = [
    {"name": "adult", "bw": 70.0, "age": 35.0},
    {"name": "elderly", "bw": 65.0, "age": 75.0},
    {"name": "pediatric_5y", "bw": 18.0, "age": 5.0},
    {"name": "infant_1y", "bw": 8.0, "age": 1.0},
]


def _simulate_cmax(graph, compiled, nominal_drug, nominal_peff, theta, seed):
    import sisyphus.engine.flux  # noqa: F401
    from sisyphus.engine.compiler import ResolvedParams
    from sisyphus.engine.solver import solve

    overridden = apply_theta_to_drug(nominal_drug, theta, nominal_peff)
    rng = np.random.default_rng(seed)
    rg = graph.sample(rng)
    rd = overridden.sample(rng)
    params = ResolvedParams(rg, rd)
    y0 = np.zeros(compiled.n_states)
    y0[compiled.state_index[overridden.administration_node]] = overridden.dose_mg
    try:
        result = solve(compiled, params, y0, t_span=(0, 48.0))
        if result.solver_success and "venous_blood" in result.concentrations:
            cmax = float(result.concentrations["venous_blood"].max())
            if cmax > 0 and np.isfinite(cmax):
                noise = rng.normal(0.0, 0.0414)
                return float(np.log10(cmax) + noise)
    except Exception:
        pass
    return -20.0


def _coverage_and_ranks(thetas_true, posterior, xs_packed, levels, n_posterior=500):
    import torch

    n_cal, d_theta = thetas_true.shape
    ranks = np.zeros((n_cal, d_theta), dtype=np.int64)
    cov = {level: np.zeros(d_theta, dtype=np.int64) for level in levels}
    for i in range(n_cal):
        x_t = torch.as_tensor(xs_packed[i:i + 1], dtype=torch.float32)
        samples = posterior.sample((n_posterior,), x=x_t, show_progress_bars=False)
        s = samples.detach().cpu().numpy()
        for d in range(d_theta):
            ranks[i, d] = int(np.sum(s[:, d] < thetas_true[i, d]))
        for level in levels:
            lo_q = (100 - level) / 2 / 100.0
            hi_q = 1.0 - lo_q
            lo = np.quantile(s, lo_q, axis=0)
            hi = np.quantile(s, hi_q, axis=0)
            inside = (thetas_true[i] >= lo) & (thetas_true[i] <= hi)
            cov[level] += inside.astype(np.int64)
    return ranks, cov


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--posterior", type=Path, required=True)
    p.add_argument("--validation-set", type=Path,
                   default=ROOT / "data" / "sbi" / "validation_drug_set.json")
    p.add_argument("--n-calibration", type=int, default=50)
    p.add_argument("--n-posterior", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--levels", type=str, default="50,80,90,95")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--ks-threshold", type=float, default=0.01)
    p.add_argument("--coverage-tol", type=float, default=0.10)
    p.add_argument("--populations", type=str, default="",
                   help="JSON list of {name, bw, age}; empty→default 4-pop grid")
    args = p.parse_args()

    levels = tuple(int(x) for x in args.levels.split(","))

    log.info("Loading posterior: %s", args.posterior)
    import torch
    payload = torch.load(args.posterior, weights_only=False)
    posterior_obj = payload["posterior"]
    aux_path = args.posterior.with_suffix(".aux.pt")
    if not aux_path.exists():
        log.error("Missing aux file %s — cannot standardize features", aux_path)
        return 1
    aux = torch.load(aux_path, weights_only=False)
    feat_mean = np.asarray(aux["feat_mean"], dtype=np.float64)  # (14,)
    feat_std = np.asarray(aux["feat_std"], dtype=np.float64)
    if feat_mean.shape[0] != 14:
        log.error("Expected 14D feat standardizer, got %d", feat_mean.shape[0])
        return 1
    log.info("  feat standardizer: 14D (drug_features(12) + log_bw_norm + log_age_norm)")

    if args.populations:
        pops = json.loads(args.populations)
    else:
        pops = _DEFAULT_POPS
    log.info("Populations (%d): %s", len(pops),
             [f"{p['name']}({p['bw']}kg/{p['age']}y)" for p in pops])

    prior = build_box_prior()
    specs = load_drug_specs_from_json(args.validation_set)
    log.info("Validation drugs (%d): %s", len(specs), [s.name for s in specs])

    with open(args.validation_set) as f:
        val_meta = {d["name"]: d for d in json.load(f)["drugs"]}

    # Precompute drug features once per drug (adult reference graph)
    import sisyphus.engine.flux  # noqa: F401
    from sisyphus.engine.compiler import ODECompiler
    from sisyphus.predict.adme import predict_adme
    from sisyphus.predict.chemistry import compute_profile
    from sisyphus.predict.ivive import build_drug_on_graph

    drug_feat_cache: dict[str, np.ndarray] = {}
    for spec in specs:
        adult_sim = EngineSimulator.for_drug(
            smiles=spec.smiles, dose_mg=spec.dose_mg, route=spec.route,
            physiology_yaml=_ADULT_PHYS,
        )
        profile_for_feat = compute_profile(spec.smiles)
        logp_hint = val_meta.get(spec.name, {}).get("logp")
        drug_feat_cache[spec.name] = extract_drug_features(
            adult_sim,
            logp=float(logp_hint) if logp_hint is not None else float(profile_for_feat.logp),
        )

    from scipy import stats as scistats

    out: dict = {
        "posterior": str(args.posterior),
        "validation_set": str(args.validation_set),
        "populations": pops,
        "n_calibration": args.n_calibration,
        "n_posterior": args.n_posterior,
        "levels": list(levels),
        "seed": args.seed,
        "ks_threshold": args.ks_threshold,
        "coverage_tol": args.coverage_tol,
        "pairs": [],
    }

    pass_count = 0
    total_pairs = len(specs) * len(pops)
    pair_idx = 0
    t_all = time.time()

    for pop in pops:
        pop_name = pop["name"]
        bw = float(pop["bw"])
        age = float(pop["age"])
        log_bw_norm = float(np.log10(bw / 70.0))
        log_age_norm = float(np.log10(max(age, 0.5)))

        for spec in specs:
            pair_idx += 1
            t0 = time.time()
            log.info("[sbc-cont] [%d/%d] %s / %s", pair_idx, total_pairs, pop_name, spec.name)

            graph = generate_physiology(bw, age)
            compiled = ODECompiler().compile(graph)
            profile = compute_profile(spec.smiles)
            adme = predict_adme(profile)
            liver_enzymes = None
            if "liver" in graph.nodes and graph.nodes["liver"].enzymes:
                liver_enzymes = {
                    tag: dist.mean for tag, dist in graph.nodes["liver"].enzymes.items()
                }
            drug = build_drug_on_graph(
                profile, adme, spec.dose_mg, spec.route,
                liver_enzymes=liver_enzymes,
            )
            nominal_peff = float(adme.peff.mean)

            feats = drug_feat_cache[spec.name]
            feat14 = np.concatenate([feats, [log_bw_norm, log_age_norm]])
            feat14_std = (feat14 - feat_mean) / feat_std

            rng = np.random.default_rng(args.seed + pair_idx * 1000)
            thetas_true = prior.sample_numpy(args.n_calibration, rng)

            xs = np.zeros((args.n_calibration, 1), dtype=np.float64)
            for j in range(args.n_calibration):
                xs[j, 0] = _simulate_cmax(
                    graph, compiled, drug, nominal_peff,
                    thetas_true[j], seed=args.seed + 7777 * pair_idx + j,
                )

            xs_packed = np.concatenate(
                [xs, np.tile(feat14_std[None, :], (args.n_calibration, 1))],
                axis=1,
            )  # (n_cal, 15)

            ranks, cov = _coverage_and_ranks(
                thetas_true, posterior_obj, xs_packed, levels, args.n_posterior,
            )

            # KS uniformity
            n_post = args.n_posterior
            uniform_ranks = ranks.astype(np.float64) / n_post
            ks_p = [float(scistats.kstest(uniform_ranks[:, d], "uniform").pvalue)
                    for d in range(ranks.shape[1])]

            # Coverage deviations
            cov_summary: dict[str, dict] = {}
            max_cov_dev = 0.0
            for level in levels:
                nominal = level / 100.0
                emp_per_dim = cov[level] / args.n_calibration
                emp_mean = float(emp_per_dim.mean())
                dev = float(abs(emp_mean - nominal))
                max_cov_dev = max(max_cov_dev, dev)
                cov_summary[str(level)] = {
                    "nominal": nominal,
                    "empirical_mean": emp_mean,
                    "deviation": dev,
                    "per_dim": emp_per_dim.tolist(),
                }

            ks_pass = all(pv > args.ks_threshold for pv in ks_p)
            cov_pass = max_cov_dev <= args.coverage_tol
            pair_pass = ks_pass and cov_pass
            if pair_pass:
                pass_count += 1

            dt = time.time() - t0
            log.info(
                "  KS p=%s cov_dev=%.3f %s (%.1fs)",
                [f"{pv:.3f}" for pv in ks_p], max_cov_dev,
                "PASS" if pair_pass else "FAIL", dt,
            )

            out["pairs"].append({
                "population": pop_name,
                "bw": bw,
                "age": age,
                "drug": spec.name,
                "ks_pvalues": ks_p,
                "coverage": cov_summary,
                "max_coverage_deviation": max_cov_dev,
                "ks_pass": ks_pass,
                "coverage_pass": cov_pass,
                "pass": pair_pass,
                "wall_s": dt,
            })

    out["summary"] = {
        "total_pairs": total_pairs,
        "pass": pass_count,
        "pass_rate": pass_count / total_pairs if total_pairs else 0.0,
        "wall_s": time.time() - t_all,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    log.info("Saved %s", args.out)
    log.info(
        "Gate: %d/%d pairs pass (%.1f%%)",
        pass_count, total_pairs, 100 * pass_count / total_pairs if total_pairs else 0,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
