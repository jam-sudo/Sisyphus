#!/usr/bin/env python3
"""Phase 2A chain benchmark: 3 IVIVE chains × holdout drugs.

Chains = 3 Kp methods × well-stirred CLh:
  A: R&R + well-stirred          (baseline)
  B: Poulin-Theil + well-stirred
  C: R&R+BZ + well-stirred

The parallel-tube CLh chains (former D/E/F) are removed: an unexpanded
``parallel_tube`` edge now fails loud at compile and must be axially expanded
first (``graph.axial.expand_axial``), but single-organ axial expansion requires
a *perfusion-only* organ (flow + clearance edges only). In reference_man the
only well_stirred clearance organ is ``gut_wall``, which also carries 8
absorption edges, so ``expand_axial`` correctly raises ``NotImplementedError``
on it. Rather than hack around the scope guard, the PT chains are dropped; the
``expand_axial`` call below stays wired (a no-op for A/B/C) so any future
perfusion-only parallel_tube organ is handled.

Output:
  1. Per-chain engine AAFE
  2. Per-chain × compound_type AAFE
  3. Oracle chain selector AAFE
  4. Geometric median ensemble AAFE
  5. Non-base chain comparison
"""

from __future__ import annotations

import copy
import dataclasses
import logging
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

PHYSIOLOGY_DIR = ROOT / "data" / "physiology"

CHAINS = [
    {"name": "A: R&R/WS",    "kp": "rodgers_rowland", "clh": "well_stirred"},
    {"name": "B: PT/WS",     "kp": "poulin_theil",    "clh": "well_stirred"},
    {"name": "C: BZ/WS",     "kp": "berezhkovskiy",   "clh": "well_stirred"},
    # Parallel-tube CLh chains (D/E/F) removed — see module docstring. An
    # unexpanded parallel_tube edge fails loud at compile, and axial expansion
    # of reference_man's gut_wall is out of scope (it has absorption edges).
]


def swap_clearance_model(graph, model: str):
    """Return a deep copy of graph with well_stirred edges replaced by model."""
    from sisyphus.graph.types import ClearanceEdge
    g2 = copy.deepcopy(graph)
    g2.edges = [
        dataclasses.replace(e, model=model)
        if isinstance(e, ClearanceEdge) and e.model == "well_stirred"
        else e
        for e in g2.edges
    ]
    return g2


def run_engine_for_drug(ref, chain, base_graph):
    """Run engine for one drug + one chain. Returns engine Cmax or None."""
    import sisyphus.engine.flux  # noqa: F401
    from sisyphus.engine.compiler import ODECompiler, ResolvedParams
    from sisyphus.engine.solver import solve
    from sisyphus.pk.endpoints import compute_endpoints
    from sisyphus.predict.adme import predict_adme
    from sisyphus.predict.chemistry import compute_profile
    from sisyphus.predict.ivive import build_drug_on_graph

    profile = compute_profile(ref.smiles)
    adme = predict_adme(profile)

    # Kp method via predict layer
    graph = base_graph
    if "liver" in graph.nodes and graph.nodes["liver"].enzymes:
        liver_enzymes = {tag: d.mean for tag, d in graph.nodes["liver"].enzymes.items()}
    else:
        liver_enzymes = None

    drug = build_drug_on_graph(
        profile, adme, ref.dose_mg, ref.route,
        liver_enzymes=liver_enzymes,
        kp_method=chain["kp"],
    )

    # CLh method via graph edge model
    if chain["clh"] != "well_stirred":
        graph = swap_clearance_model(graph, chain["clh"])

    # Expand any parallel_tube organ into N serial well_stirred tanks before
    # compile (an unexpanded parallel_tube edge fails loud at compile). No-op
    # for the current well_stirred chains (A/B/C); kept wired so a future
    # perfusion-only parallel_tube organ would compile to genuine axial
    # parallel-tube extraction (default N=10) instead of raising.
    from sisyphus.graph.axial import expand_axial
    graph = expand_axial(graph)

    compiler = ODECompiler()
    compiled = compiler.compile(graph)

    rng = np.random.default_rng(42)
    realized_graph = graph.sample(rng)
    realized_drug = drug.sample(rng)
    params = ResolvedParams(realized_graph, realized_drug)

    y0 = np.zeros(compiled.n_states)
    admin_idx = compiled.state_index[drug.administration_node]
    y0[admin_idx] = drug.dose_mg

    sim = solve(compiled, params, y0, t_span=(0, 24))
    if sim.solver_success:
        pk = compute_endpoints(sim)
        return pk.cmax.mean, profile.compound_type
    return None, profile.compound_type


def geometric_median_1d(values: list[float]) -> float:
    """Geometric median of positive values = exp(median(log(values)))."""
    logs = sorted(math.log(max(v, 1e-30)) for v in values)
    n = len(logs)
    if n == 0:
        return 0.0
    if n % 2 == 1:
        return math.exp(logs[n // 2])
    return math.exp((logs[n // 2 - 1] + logs[n // 2]) / 2)


def aafe(pred: np.ndarray, obs: np.ndarray) -> float:
    mask = (pred > 0) & (obs > 0)
    if not np.any(mask):
        return float("inf")
    return float(10 ** np.mean(np.abs(np.log10(pred[mask] / obs[mask]))))


def main():
    from sisyphus.graph.builder import build_from_yaml
    from sisyphus.validation.reference import load_reference

    refs = [r for r in load_reference() if r.in_holdout]
    print(f"Holdout drugs: {len(refs)}")

    base_graph = build_from_yaml(PHYSIOLOGY_DIR / "reference_man.yaml")

    # results[chain_name][drug_name] = {"cmax": float, "type": str, "obs": float}
    results: dict[str, dict[str, dict]] = {c["name"]: {} for c in CHAINS}

    for i, ref in enumerate(refs):
        print(f"  [{i+1}/{len(refs)}] {ref.name:<25s}", end="", flush=True)
        for chain in CHAINS:
            try:
                cmax, ctype = run_engine_for_drug(ref, chain, base_graph)
                results[chain["name"]][ref.name] = {
                    "cmax": cmax if cmax else 0.0,
                    "type": ctype,
                    "obs": ref.cmax_obs,
                }
            except Exception as e:
                results[chain["name"]][ref.name] = {
                    "cmax": 0.0, "type": "?", "obs": ref.cmax_obs
                }
        # Print chain A cmax for progress
        a_cmax = results[CHAINS[0]["name"]].get(ref.name, {}).get("cmax", 0)
        print(f"  A={a_cmax:.4f}  obs={ref.cmax_obs:.4f}")

    # ═══════════════════════════════════════════════════════════════
    # ANALYSIS
    # ═══════════════════════════════════════════════════════════════

    # 1. Per-chain AAFE
    print(f"\n{'='*80}")
    print("1. PER-CHAIN ENGINE AAFE")
    print(f"{'='*80}")
    print(f"{'Chain':<20s} {'N':>4s} {'AAFE':>8s} {'%2-fold':>8s} {'%3-fold':>8s}")
    print("-" * 50)

    chain_aafs = {}
    for cname in [c["name"] for c in CHAINS]:
        drugs = results[cname]
        pred = np.array([d["cmax"] for d in drugs.values()])
        obs = np.array([d["obs"] for d in drugs.values()])
        valid = (pred > 0) & (obs > 0)
        if np.any(valid):
            a = aafe(pred[valid], obs[valid])
            ratios = pred[valid] / obs[valid]
            p2 = np.sum((ratios >= 0.5) & (ratios <= 2.0)) / len(ratios) * 100
            p3 = np.sum((ratios >= 1/3) & (ratios <= 3.0)) / len(ratios) * 100
            chain_aafs[cname] = a
            print(f"{cname:<20s} {np.sum(valid):4d} {a:8.3f} {p2:7.1f}% {p3:7.1f}%")

    # 2. Per-chain × compound_type AAFE
    print(f"\n{'='*80}")
    print("2. PER-CHAIN × COMPOUND_TYPE AAFE")
    print(f"{'='*80}")
    types = ["base", "acid", "neutral", "zwitterion"]
    header = f"{'Chain':<20s}" + "".join(f"{'  '+t:>12s}" for t in types)
    print(header)
    print("-" * (20 + 12 * len(types)))

    for cname in [c["name"] for c in CHAINS]:
        drugs = results[cname]
        row = f"{cname:<20s}"
        for ctype in types:
            sub = {k: v for k, v in drugs.items() if v["type"] == ctype}
            if sub:
                p = np.array([v["cmax"] for v in sub.values()])
                o = np.array([v["obs"] for v in sub.values()])
                valid = (p > 0) & (o > 0)
                if np.any(valid):
                    a = aafe(p[valid], o[valid])
                    row += f"  {a:8.3f}({np.sum(valid):2d})"
                else:
                    row += f"  {'N/A':>11s}"
            else:
                row += f"  {'---':>11s}"
        print(row)

    # 3. Oracle chain selector
    print(f"\n{'='*80}")
    print("3. ORACLE CHAIN SELECTOR")
    print(f"{'='*80}")

    drug_names = list(results[CHAINS[0]["name"]].keys())
    oracle_errors = []
    chain_win_counts = {c["name"]: 0 for c in CHAINS}

    for dname in drug_names:
        obs = results[CHAINS[0]["name"]][dname]["obs"]
        best_err = float("inf")
        best_chain = ""
        for cname in [c["name"] for c in CHAINS]:
            cmax = results[cname][dname]["cmax"]
            if cmax > 0 and obs > 0:
                err = abs(math.log10(cmax / obs))
                if err < best_err:
                    best_err = err
                    best_chain = cname
        if best_err < float("inf"):
            oracle_errors.append(best_err)
            chain_win_counts[best_chain] += 1

    oracle_aafe = 10 ** np.mean(oracle_errors) if oracle_errors else float("inf")
    print(f"Oracle chain selector AAFE: {oracle_aafe:.3f}")
    print(f"\nChain win counts:")
    for cname, count in sorted(chain_win_counts.items(), key=lambda x: -x[1]):
        print(f"  {cname:<20s}: {count:3d}/{len(drug_names)} ({count/len(drug_names)*100:.0f}%)")

    # 4. Geometric median ensemble
    print(f"\n{'='*80}")
    print("4. GEOMETRIC MEDIAN ENSEMBLE")
    print(f"{'='*80}")

    ensemble_pred = []
    ensemble_obs = []
    for dname in drug_names:
        obs = results[CHAINS[0]["name"]][dname]["obs"]
        chain_cmax = []
        for cname in [c["name"] for c in CHAINS]:
            cmax = results[cname][dname]["cmax"]
            if cmax > 0:
                chain_cmax.append(cmax)
        if chain_cmax and obs > 0:
            ensemble_pred.append(geometric_median_1d(chain_cmax))
            ensemble_obs.append(obs)

    ep = np.array(ensemble_pred)
    eo = np.array(ensemble_obs)
    ens_aafe = aafe(ep, eo)
    ratios = ep / eo
    p2 = np.sum((ratios >= 0.5) & (ratios <= 2.0)) / len(ratios) * 100
    p3 = np.sum((ratios >= 1/3) & (ratios <= 3.0)) / len(ratios) * 100
    print(f"Geometric median ensemble: AAFE={ens_aafe:.3f}, %2-fold={p2:.1f}%, %3-fold={p3:.1f}%")

    # 5. Non-base comparison
    print(f"\n{'='*80}")
    print("5. NON-BASE DRUGS COMPARISON")
    print(f"{'='*80}")
    print(f"{'Chain':<20s} {'N':>4s} {'AAFE':>8s}  vs Chain A")
    print("-" * 50)

    chain_a_name = CHAINS[0]["name"]
    for cname in [c["name"] for c in CHAINS]:
        drugs = results[cname]
        sub = {k: v for k, v in drugs.items() if v["type"] != "base"}
        p = np.array([v["cmax"] for v in sub.values()])
        o = np.array([v["obs"] for v in sub.values()])
        valid = (p > 0) & (o > 0)
        if np.any(valid):
            a = aafe(p[valid], o[valid])
            # Compare to chain A
            sub_a = {k: v for k, v in results[chain_a_name].items() if v["type"] != "base"}
            pa = np.array([v["cmax"] for v in sub_a.values()])
            oa = np.array([v["obs"] for v in sub_a.values()])
            va = (pa > 0) & (oa > 0)
            a_aafe = aafe(pa[va], oa[va])
            delta = (a - a_aafe) / a_aafe * 100
            print(f"{cname:<20s} {np.sum(valid):4d} {a:8.3f}  {delta:+.1f}%")

    # Non-base ensemble
    nb_pred, nb_obs = [], []
    for dname in drug_names:
        if results[CHAINS[0]["name"]][dname]["type"] == "base":
            continue
        obs = results[CHAINS[0]["name"]][dname]["obs"]
        chain_cmax = [results[c["name"]][dname]["cmax"] for c in CHAINS
                      if results[c["name"]][dname]["cmax"] > 0]
        if chain_cmax and obs > 0:
            nb_pred.append(geometric_median_1d(chain_cmax))
            nb_obs.append(obs)
    if nb_pred:
        nbp, nbo = np.array(nb_pred), np.array(nb_obs)
        print(f"\nNon-base ensemble:   {len(nbp):4d} {aafe(nbp, nbo):8.3f}")


if __name__ == "__main__":
    main()
