#!/usr/bin/env python3
"""Train and gate a two-latent MIPD SNPE model from engine CL grids.

Input is a JSON list of {"smiles": ..., "dose_mg": ...} oral compounds. The last
20% of compounds are held out. No model is written unless held-out coverage and
agreement with the grid SIR reference both pass.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import numpy as np
from scipy.stats import kstest

from sisyphus.descriptors import compute_features
from sisyphus.mipd.amortizer import NeuralAmortizer, _neural_context
from sisyphus.mipd.clgrid import CLGrid, CLGridForward, CLPrior, MeasuredConc, sir_posterior_2d
from sisyphus.mipd.core import FPrior, MeasuredAUC, MeasuredCmax, MeasuredF
from sisyphus.mipd.grid import build_cl_grid
from sisyphus.regimen.types import DosingRegimen
from sisyphus.sbi.amortizer import save_result, train_npe

log = logging.getLogger(__name__)


def _draw_pair(forward, f_prior, cl_prior, features, regimen, rng):
    f = float(f_prior.sample(1, rng)[0])
    s = float(cl_prior.sample(1, rng)[0])
    state = forward(np.array([f]), np.array([s]))
    mode = int(rng.integers(0, 5))
    if mode == 0:
        obs = (MeasuredF(min(float(rng.lognormal(math.log(f), 0.15)), 1.0), cv=0.15),)
    elif mode == 1:
        obs = (MeasuredCmax(float(rng.lognormal(math.log(state["cmax"][0]), 0.30))),)
    elif mode == 2:
        obs = (MeasuredAUC(float(rng.lognormal(math.log(state["auc"][0]), 0.30))),)
    else:
        times = (4.0,) if mode == 3 else (4.0, 16.0)
        obs = tuple(
            MeasuredConc(
                float(rng.lognormal(math.log(state["conc_at"](t)[0]), 0.25)), t
            ) for t in times
        )
    x = _neural_context(f_prior, cl_prior, forward, obs, features, regimen)
    return np.array([f, s], dtype=np.float32), x, obs


def _setup(spec, n_grid, cache_dir=None, *, generate=False):
    smiles = spec["smiles"]
    dose = float(spec["dose_mg"])
    if dose <= 0:
        raise ValueError("dose_mg must be positive")
    cache_path = cache_dir / f"{spec['_index']}.npz" if cache_dir is not None else None
    if generate or cache_path is None:
        grid = build_cl_grid(smiles, dose, n_grid=n_grid, s_range=(0.05, 20.0))
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                cache_path, smiles=smiles, dose_mg=dose,
                s_grid=grid.s_grid, t_grid=grid.t_grid,
                conc=grid.conc, cmax=grid.cmax, auc=grid.auc, f_engine=grid.f_engine,
            )
    else:
        if not cache_path.exists():
            raise FileNotFoundError(f"missing engine grid: {cache_path}")
        with np.load(cache_path) as data:
            if str(data["smiles"]) != smiles or float(data["dose_mg"]) != dose:
                raise ValueError(f"engine grid input mismatch in {cache_path}")
            grid = CLGrid(**{name: data[name] for name in (
                "s_grid", "t_grid", "conc", "cmax", "auc", "f_engine"
            )})
        if len(grid.s_grid) != n_grid:
            raise ValueError(f"grid resolution mismatch in {cache_path}")
    forward = CLGridForward(grid)
    f_prior = FPrior(float(np.interp(1.0, grid.s_grid, grid.f_engine)))
    # The shared feature function's final nine coordinates are RDKit descriptors.
    return (
        forward, f_prior, CLPrior(), compute_features(smiles)[-9:],
        DosingRegimen.single_oral(dose),
    )


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--samples-per-compound", type=int, default=1000)
    parser.add_argument("--n-grid", type=int, default=13)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--generate-index", type=int)
    parser.add_argument("--n-calibration", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    specs = json.loads(args.manifest.read_text())
    for index, spec in enumerate(specs):
        spec["_index"] = index
    if len(specs) < 5 or args.samples_per_compound < 1 or args.n_calibration < 1:
        raise ValueError("manifest needs at least five compounds and positive sample counts")
    if len({spec["smiles"] for spec in specs}) != len(specs):
        raise ValueError("manifest contains duplicate compounds")
    if args.generate_index is not None:
        if args.cache_dir is None or not 0 <= args.generate_index < len(specs):
            raise ValueError("--generate-index needs --cache-dir and a valid manifest index")
        _setup(specs[args.generate_index], args.n_grid, args.cache_dir, generate=True)
        return
    if args.out is None or args.report is None:
        raise ValueError("training requires --out and --report")
    rng = np.random.default_rng(args.seed)
    rng.shuffle(specs)
    n_train = len(specs) - max(1, len(specs) // 5)
    train, heldout = specs[:n_train], specs[n_train:]
    theta_rows, x_rows = [], []
    for i, spec in enumerate(train, 1):
        log.info("Training grid %d/%d: %s", i, len(train), spec["smiles"])
        setup = _setup(spec, args.n_grid, args.cache_dir)
        for _ in range(args.samples_per_compound):
            theta, x, _ = _draw_pair(*setup, rng)
            theta_rows.append(theta)
            x_rows.append(x)
    result = train_npe(
        np.stack(theta_rows), np.stack(x_rows),
        np.array([1e-6, 0.05]), np.array([1.0, 20.0]),
        density_estimator="nsf", embedding_net_hidden=64,
        max_num_epochs=args.epochs, stop_after_epochs=15, seed=args.seed,
    )
    neural = NeuralAmortizer(result)
    coverage = np.zeros(2)
    agreement = np.zeros(2)
    ranks = []
    cases = 0
    for i, spec in enumerate(heldout, 1):
        log.info("Held-out grid %d/%d: %s", i, len(heldout), spec["smiles"])
        forward, f_prior, cl_prior, features, regimen = _setup(
            spec, args.n_grid, args.cache_dir
        )
        for _ in range(args.n_calibration):
            truth, _, obs = _draw_pair(forward, f_prior, cl_prior, features, regimen, rng)
            post = neural.posterior(
                f_prior, cl_prior, forward, obs, molecular_features=features,
                regimen=regimen, n_samples=500, rng=rng,
            )
            sir = sir_posterior_2d(
                f_prior, cl_prior, forward, obs, n_samples=5000, rng=rng
            )
            neural_samples = (post.f.samples, post.cl_scale.samples)
            sir_samples = (sir.f.samples, sir.cl_scale.samples)
            ranks.append([float(np.mean(draws < value)) for draws, value in zip(
                neural_samples, truth
            )])
            for dim in range(2):
                lo, hi = np.quantile(neural_samples[dim], [0.05, 0.95])
                coverage[dim] += lo <= truth[dim] <= hi
                median_error = abs(np.median(neural_samples[dim]) - np.median(sir_samples[dim]))
                # The pooled Monte Carlo standard error of two sample means is a
                # conservative scale for checking posterior location agreement.
                mc_se = math.sqrt(
                    np.var(neural_samples[dim]) / len(neural_samples[dim])
                    + np.var(sir_samples[dim]) / len(sir_samples[dim])
                )
                agreement[dim] += median_error <= 3 * mc_se
            cases += 1
    coverage /= cases
    agreement /= cases
    ks_p = np.array([kstest(np.asarray(ranks)[:, dim], "uniform").pvalue for dim in range(2)])
    passed = bool(
        np.all(abs(coverage - 0.9) <= 0.1)
        and np.all(agreement >= 0.95)
        and np.all(ks_p > 0.01)
    )
    report = {
        "train_compounds": len(train), "heldout_compounds": len(heldout),
        "train_pairs": len(theta_rows), "heldout_cases": cases,
        "coverage_90_f_clint": coverage.tolist(),
        "sbc_ks_p_f_clint": ks_p.tolist(),
        "sir_agreement_fraction_f_clint": agreement.tolist(),
        "gate_passed": passed,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    log.info("Held-out gate: %s", report)
    if passed:
        save_result(result, args.out)
        log.info("Validated model saved to %s", args.out)
    else:
        log.warning("Gate failed; model was not saved")
        raise SystemExit(2)


if __name__ == "__main__":
    _main()
