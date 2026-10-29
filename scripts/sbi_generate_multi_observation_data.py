"""Generate time-aware multi-observation SBI pairs with the current regimen engine."""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sisyphus.regimen.types import DosingRegimen  # noqa: E402
from sisyphus.sbi.multi_drug import extract_drug_features, load_drug_specs_from_json  # noqa: E402
from sisyphus.sbi.multi_observation import (  # noqa: E402
    DOSING_INTERVALS_H,
    MAX_DOSE_COUNT,
    MAX_OBSERVATION_COUNT,
    OBSERVATION_OFFSETS_H,
    _simulate_context,
)
from sisyphus.sbi.priors import build_box_prior  # noqa: E402
from sisyphus.sbi.simulator import EngineSimulator  # noqa: E402

log = logging.getLogger(__name__)

def _worker(job: tuple) -> dict:
    spec, n_theta, seed, logp = job
    sim = EngineSimulator.for_drug(spec.smiles, spec.dose_mg, spec.route)
    features = extract_drug_features(sim, logp=logp)
    rng = np.random.default_rng(seed)
    prior = build_box_prior()
    theta_all = prior.sample_numpy(n_theta, rng)
    rows = []
    thetas = []
    start = time.monotonic()
    for theta in theta_all:
        n_doses = int(rng.integers(1, MAX_DOSE_COUNT + 1))
        interval = float(rng.choice(DOSING_INTERVALS_H))
        regimen = DosingRegimen.oral_repeated(spec.dose_mg, interval, n_doses)
        n_obs = int(rng.integers(1, MAX_OBSERVATION_COUNT + 1))
        offsets = sorted(rng.choice(OBSERVATION_OFFSETS_H, size=n_obs, replace=False))
        times = [regimen.last_dose_time_h + float(offset) for offset in offsets]
        context = _simulate_context(sim, theta, rng, regimen, times, features)
        if context is not None:
            rows.append(context)
            thetas.append(theta)
    return {
        "name": spec.name,
        "theta": np.asarray(thetas, dtype=np.float32).reshape(-1, 3),
        "x": np.asarray(rows, dtype=np.float32).reshape(-1, 24),
        "requested": n_theta,
        "elapsed_s": time.monotonic() - start,
    }


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drug-set", type=Path, default=ROOT / "data/sbi/train_drug_set.json")
    parser.add_argument("--theta-per-drug", type=int, default=500)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-index", type=int, default=0)
    parser.add_argument("--chunk-count", type=int, default=1)
    parser.add_argument("--out", type=Path, default=ROOT / "data/sbi/multi_observation_train.npz")
    args = parser.parse_args()
    specs = load_drug_specs_from_json(args.drug_set)
    if not 0 <= args.chunk_index < args.chunk_count:
        parser.error("--chunk-index must be between 0 and --chunk-count - 1")
    logp_by_name = {
        entry["name"]: float(entry.get("logp", 2.0))
        for entry in json.loads(args.drug_set.read_text())["drugs"]
    }
    jobs = [
        (spec, args.theta_per_drug, args.seed + i * 100_000, logp_by_name[spec.name])
        for i, spec in list(enumerate(specs))[args.chunk_index::args.chunk_count]
    ]
    start = time.monotonic()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        results = list(pool.imap_unordered(_worker, jobs))
    theta = np.concatenate([r["theta"] for r in results])
    x = np.concatenate([r["x"] for r in results])
    if len(theta) == 0:
        raise RuntimeError("No successful multi-observation simulations")
    prior = build_box_prior()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, theta=theta, multi_obs_x=x, prior_low=prior.low, prior_high=prior.high)
    meta = {
        "requested": len(jobs) * args.theta_per_drug,
        "valid": len(theta),
        "wall_s": time.monotonic() - start,
        "per_drug": [
            {key: value for key, value in result.items() if key not in ("theta", "x")}
            for result in results
        ],
    }
    args.out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    log.info(
        "Wrote %s: %d/%d rows in %.1fs", args.out, len(theta), meta["requested"], meta["wall_s"]
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    _main()
