"""Generate training data for continuous hierarchical SBI.

Usage:
    python scripts/sbi_generate_continuous_data.py \
        --n-theta 1000 --n-pops 20 --out data/sbi/continuous_train.npz
"""

from __future__ import annotations

import argparse
import logging
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate continuous hierarchical SBI training data")
    parser.add_argument("--n-theta", type=int, default=1000)
    parser.add_argument("--n-pops", type=int, default=20)
    parser.add_argument("--drug-set", type=str, default=str(_DRUG_SET))
    parser.add_argument("--out", type=str, default="data/sbi/continuous_train.npz")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    specs = load_drug_specs_from_json(Path(args.drug_set))
    logger.info("Loaded %d drugs", len(specs))

    ages = rng.uniform(0.5, 85.0, size=args.n_pops)
    bws = np.array([sample_bw_for_age(rng, a) for a in ages])
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

    all_theta, all_logcmax, all_feat = [], [], []
    all_bw, all_age, all_drug = [], [], []

    t0 = time.time()
    total_sims = len(specs) * args.n_pops * args.n_theta
    done = 0

    for si, spec in enumerate(specs):
        logger.info("[%d/%d] %s", si + 1, len(specs), spec.name)
        adult_sim = EngineSimulator.for_drug(
            smiles=spec.smiles,
            dose_mg=spec.dose_mg,
            route=spec.route,
            physiology_yaml=_ADULT_PHYS,
        )
        profile = compute_profile(spec.smiles)
        feats = extract_drug_features(adult_sim, logp=float(profile.logp))

        for pi in range(args.n_pops):
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

            thetas = rng.uniform(prior_low, prior_high, size=(args.n_theta, 3))
            logcmax_batch = np.full(args.n_theta, np.nan)

            for ti in range(args.n_theta):
                overridden = apply_theta_to_drug(drug, thetas[ti], nominal_peff)
                seed_i = int(rng.integers(0, 2**31))
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

                done += 1
                if done % 1000 == 0:
                    elapsed = time.time() - t0
                    rate = done / elapsed if elapsed > 0 else 0
                    eta = (total_sims - done) / rate / 60 if rate > 0 else 0
                    logger.info(
                        "  %d/%d sims (%.1f/s, ETA %.0fm)",
                        done, total_sims, rate, eta,
                    )

            all_theta.append(thetas)
            all_logcmax.append(logcmax_batch[:, None])
            all_feat.append(np.tile(feats[None, :], (args.n_theta, 1)))
            all_bw.append(np.full(args.n_theta, bw))
            all_age.append(np.full(args.n_theta, age))
            all_drug.extend([spec.name] * args.n_theta)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        theta=np.concatenate(all_theta),
        x=np.concatenate(all_logcmax),
        drug_features=np.concatenate(all_feat),
        bw=np.concatenate(all_bw),
        age=np.concatenate(all_age),
        drug_names=np.array(all_drug),
    )
    elapsed = time.time() - t0
    logger.info("Saved %s (%.1f min)", out_path, elapsed / 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
