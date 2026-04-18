"""Generate training data for continuous hierarchical SBI.

Per-(drug, population) checkpointing: after each pair completes, partial
npz is flushed to disk.  On restart, existing pairs are detected and skipped.

Usage:
    python scripts/sbi_generate_continuous_data.py \
        --n-theta 500 --n-pops 10 --out data/sbi/continuous_train.npz
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sisyphus.sbi.multi_drug import (
    DrugSpec,
    extract_drug_features,
    load_drug_specs_from_json,
)
from sisyphus.sbi.physiology_generator import generate_physiology
from sisyphus.sbi.priors import PRIOR_HIGH, PRIOR_LOW
from sisyphus.sbi.simulator import apply_theta_to_drug

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_DRUG_SET = ROOT / "data" / "sbi" / "train_drug_set.json"
_ADULT_PHYS = ROOT / "data" / "physiology" / "reference_man.yaml"


def sample_bw_for_age(rng: np.random.Generator, age: float) -> float:
    if age < 1:
        mean, sd = 8.0, 2.0
    elif age < 3:
        mean, sd = 13.0, 3.0
    elif age < 6:
        mean, sd = 18.0, 4.0
    elif age < 10:
        mean, sd = 28.0, 6.0
    elif age < 15:
        mean, sd = 45.0, 10.0
    elif age < 20:
        mean, sd = 65.0, 12.0
    elif age < 70:
        mean, sd = 75.0, 15.0
    else:
        mean, sd = 65.0, 12.0
    return float(np.clip(rng.normal(mean, sd), 3.0, 150.0))


def _pair_rng(seed: int, drug_idx: int, pop_idx: int) -> np.random.Generator:
    """Deterministic RNG per (drug, pop) pair — enables resume without stream drift."""
    s = (int(seed) * 1_000_003 + int(drug_idx) * 1009 + int(pop_idx)) & 0xFFFFFFFF
    return np.random.default_rng(s)


def _atomic_save(path: Path, **arrays) -> None:
    tmp = path.parent / (path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate continuous hierarchical SBI training data")
    parser.add_argument("--n-theta", type=int, default=1000)
    parser.add_argument("--n-pops", type=int, default=20)
    parser.add_argument("--drug-set", type=str, default=str(_DRUG_SET))
    parser.add_argument("--out", type=str, default="data/sbi/continuous_train.npz")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-every", type=int, default=1,
                        help="Save partial npz every N pairs (default 1)")
    args = parser.parse_args()

    specs = load_drug_specs_from_json(Path(args.drug_set))
    logger.info("Loaded %d drugs", len(specs))

    # Populations: deterministic from seed alone
    rng_pops = np.random.default_rng(args.seed)
    ages = rng_pops.uniform(0.5, 85.0, size=args.n_pops)
    bws = np.array([sample_bw_for_age(rng_pops, a) for a in ages])
    logger.info(
        "Sampled %d populations: age %.1f-%.1f, BW %.1f-%.1f",
        args.n_pops, ages.min(), ages.max(), bws.min(), bws.max(),
    )

    prior_low = np.array(PRIOR_LOW, dtype=np.float64)
    prior_high = np.array(PRIOR_HIGH, dtype=np.float64)

    import sisyphus.engine.flux  # noqa: F401
    from sisyphus.engine.compiler import ODECompiler, ResolvedParams
    from sisyphus.engine.solver import solve
    from sisyphus.graph.builder import build_from_yaml
    from sisyphus.predict.adme import predict_adme
    from sisyphus.predict.chemistry import compute_profile
    from sisyphus.predict.ivive import build_drug_on_graph
    from sisyphus.sbi.simulator import EngineSimulator

    adult_graph = build_from_yaml(_ADULT_PHYS)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = out_path.with_suffix(".partial.npz")

    # Resume state
    done_pairs: set[tuple[str, int]] = set()
    all_theta: list = []
    all_logcmax: list = []
    all_feat: list = []
    all_bw: list = []
    all_age: list = []
    all_drug: list = []
    all_pop_idx: list = []

    if partial_path.exists():
        try:
            p = np.load(partial_path, allow_pickle=True)
            all_theta.append(p["theta"])
            all_logcmax.append(p["x"])
            all_feat.append(p["drug_features"])
            all_bw.append(p["bw"])
            all_age.append(p["age"])
            all_drug.extend(list(p["drug_names"]))
            all_pop_idx.append(p["pop_idx"])
            for name, pi in zip(p["drug_names"], p["pop_idx"]):
                done_pairs.add((str(name), int(pi)))
            logger.info("Resumed from %s: %d pairs already done", partial_path, len(done_pairs))
        except Exception as exc:
            logger.warning("Failed to resume from partial: %s — starting fresh", exc)
            done_pairs = set()
            all_theta, all_logcmax, all_feat = [], [], []
            all_bw, all_age, all_drug, all_pop_idx = [], [], [], []

    t0 = time.time()
    total_pairs = len(specs) * args.n_pops
    pairs_done_this_run = 0
    pairs_skipped = len(done_pairs)

    for si, spec in enumerate(specs):
        adult_sim = None
        profile = None
        feats = None

        for pi in range(args.n_pops):
            if (spec.name, pi) in done_pairs:
                continue

            if adult_sim is None:
                logger.info("[%d/%d] %s", si + 1, len(specs), spec.name)
                adult_sim = EngineSimulator.for_drug(
                    smiles=spec.smiles,
                    dose_mg=spec.dose_mg,
                    route=spec.route,
                    physiology_yaml=_ADULT_PHYS,
                )
                profile = compute_profile(spec.smiles)
                feats = extract_drug_features(adult_sim, logp=float(profile.logp))

            bw, age = float(bws[pi]), float(ages[pi])
            graph = generate_physiology(bw, age)

            compiled = ODECompiler().compile(graph)
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

            rng_pair = _pair_rng(args.seed, si, pi)
            thetas = rng_pair.uniform(prior_low, prior_high, size=(args.n_theta, 3))
            logcmax_batch = np.full(args.n_theta, np.nan)

            for ti in range(args.n_theta):
                overridden = apply_theta_to_drug(drug, thetas[ti], nominal_peff)
                seed_i = int(rng_pair.integers(0, 2**31))
                rng_i = np.random.default_rng(seed_i)
                rg = graph.sample(rng_i)
                rd = overridden.sample(rng_i)
                params = ResolvedParams(rg, rd)
                y0 = np.zeros(compiled.n_states)
                y0[compiled.state_index[drug.administration_node]] = drug.dose_mg
                try:
                    result = solve(compiled, params, y0, t_span=(0, 48.0))
                    if result.solver_success and "venous_blood" in result.concentrations:
                        cmax = float(result.concentrations["venous_blood"].max())
                        if cmax > 0 and np.isfinite(cmax):
                            noise = rng_i.normal(0.0, 0.0414)
                            logcmax_batch[ti] = np.log10(cmax) + noise
                except Exception:
                    pass

            all_theta.append(thetas)
            all_logcmax.append(logcmax_batch[:, None])
            all_feat.append(np.tile(feats[None, :], (args.n_theta, 1)))
            all_bw.append(np.full(args.n_theta, bw))
            all_age.append(np.full(args.n_theta, age))
            all_drug.extend([spec.name] * args.n_theta)
            all_pop_idx.append(np.full(args.n_theta, pi, dtype=np.int32))

            pairs_done_this_run += 1
            done_pairs.add((spec.name, pi))

            if pairs_done_this_run % args.checkpoint_every == 0:
                _atomic_save(
                    partial_path,
                    theta=np.concatenate(all_theta),
                    x=np.concatenate(all_logcmax),
                    drug_features=np.concatenate(all_feat),
                    bw=np.concatenate(all_bw),
                    age=np.concatenate(all_age),
                    drug_names=np.array(all_drug),
                    pop_idx=np.concatenate(all_pop_idx),
                )

            elapsed = time.time() - t0
            total_done = pairs_done_this_run + pairs_skipped
            rate = pairs_done_this_run / elapsed if elapsed > 0 else 0
            remaining = total_pairs - total_done
            eta_min = remaining / rate / 60 if rate > 0 else 0
            if pairs_done_this_run % 5 == 0 or pairs_done_this_run == 1:
                logger.info(
                    "  pair %d/%d (run rate %.2f/s, ETA %.0fm)",
                    total_done, total_pairs, rate, eta_min,
                )

    # Final save — atomic rename of partial → final
    _atomic_save(
        partial_path,
        theta=np.concatenate(all_theta),
        x=np.concatenate(all_logcmax),
        drug_features=np.concatenate(all_feat),
        bw=np.concatenate(all_bw),
        age=np.concatenate(all_age),
        drug_names=np.array(all_drug),
        pop_idx=np.concatenate(all_pop_idx),
    )
    os.replace(partial_path, out_path)
    elapsed = time.time() - t0
    logger.info("Saved %s (%.1f min, %d new pairs, %d resumed)",
                out_path, elapsed / 60, pairs_done_this_run, pairs_skipped)
    return 0


if __name__ == "__main__":
    sys.exit(main())
