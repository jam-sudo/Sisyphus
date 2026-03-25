#!/usr/bin/env python3
"""UGT fm sensitivity test.

For each UGT-annotated holdout drug, test fm_ugt at multiple levels
and measure engine Cmax change relative to baseline (CYP-only).

The test modifies enzyme_affinity at runtime:
- Scale all CYP affinities by (1 - fm_ugt)
- Add UGT affinity entries at fm_ugt of total CLint
- Add UGT abundances to graph nodes

Engine changes: 0 lines. The engine treats UGT tags identically to CYP tags.
"""

from __future__ import annotations

import copy
import logging
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# UGT-annotated holdout drugs from mechanism audit
_UGT_DRUGS: dict[str, list[str]] = {
    "morphine": ["UGT2B7", "UGT1A1", "UGT1A8", "UGT2B15", "UGT2B4", "UGT1A3"],
    "codeine": ["UGT2B7", "UGT2B4"],
    "phenytoin": ["UGT1A1", "UGT1A6", "UGT1A9", "UGT1A4"],
    "indomethacin": ["UGT1A9", "UGT1A1", "UGT2B7"],
    "diclofenac": ["UGT2B7", "UGT1A3", "UGT1A9", "UGT2B4"],
    "losartan": ["UGT1A1", "UGT1A3", "UGT1A10", "UGT2B7"],
    "carbamazepine": ["UGT2B7"],
    "zonisamide": ["UGT1A1"],
    "clozapine": ["UGT1A4"],
    "tamoxifen": ["UGT1A10", "UGT2B7"],
    "metronidazole": ["UGT1A1"],
    "bexagliflozin": ["UGT1A9", "UGT1A1"],
    "glasdegib": ["UGT1A9"],
    "febuxostat": ["UGT1A1", "UGT1A3", "UGT1A9", "UGT2B7"],
}

# UGT abundance (pmol, total organ) — from proteomics literature
# Achour et al. 2014 (Drug Metab Dispos), Fallon et al. 2013
# pmol/mg microsomal protein × 45 mg protein/g liver × 1500 g liver
# Using a single representative UGT tag for simplicity in this test
# All UGT isoforms get equal abundance for now
_UGT_LIVER_ABUNDANCE: dict[str, float] = {
    "UGT2B7": 25.0 * 45 * 1500,     # 1,687,500 pmol — Achour et al. 2014
    "UGT1A1": 18.0 * 45 * 1500,     # 1,215,000 pmol
    "UGT1A4": 10.0 * 45 * 1500,     # 675,000 pmol
    "UGT1A9": 15.0 * 45 * 1500,     # 1,012,500 pmol
    "UGT1A3": 8.0 * 45 * 1500,      # 540,000 pmol
    "UGT1A6": 12.0 * 45 * 1500,     # 810,000 pmol
    "UGT1A8": 3.0 * 45 * 1500,      # 202,500 pmol — low in liver
    "UGT1A10": 2.0 * 45 * 1500,     # 135,000 pmol — mostly gut
    "UGT2B4": 15.0 * 45 * 1500,     # 1,012,500 pmol
    "UGT2B15": 10.0 * 45 * 1500,    # 675,000 pmol
}

# Gut UGT abundance — generally lower than liver
# Paine et al. 2006, Harbourt et al. 2012
# Gut CYP3A4 in YAML: 21,224,338 pmol
_UGT_GUT_ABUNDANCE: dict[str, float] = {
    "UGT2B7": 5.0 * 45 * 300,       # ~67,500 pmol (gut is ~300g tissue)
    "UGT1A1": 8.0 * 45 * 300,       # ~108,000 pmol
    "UGT1A4": 2.0 * 45 * 300,       # ~27,000 pmol
    "UGT1A9": 3.0 * 45 * 300,       # ~40,500 pmol
    "UGT1A3": 2.0 * 45 * 300,       # ~27,000 pmol
    "UGT1A6": 3.0 * 45 * 300,       # ~40,500 pmol
    "UGT1A8": 6.0 * 45 * 300,       # ~81,000 pmol — higher in gut
    "UGT1A10": 10.0 * 45 * 300,     # ~135,000 pmol — primarily intestinal
    "UGT2B4": 3.0 * 45 * 300,       # ~40,500 pmol
    "UGT2B15": 2.0 * 45 * 300,      # ~27,000 pmol
}

FM_UGT_LEVELS = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]

# IVIVE scaling constant (same as reference_man.yaml)
_IVIVE_SCALING = 6e-5


def add_ugt_to_graph(graph):
    """Add UGT enzyme abundances to liver and gut_wall nodes."""
    from sisyphus.core import Distribution
    from sisyphus.graph.types import Node

    for node_name, ugt_abundances in [("liver", _UGT_LIVER_ABUNDANCE), ("gut_wall", _UGT_GUT_ABUNDANCE)]:
        if node_name not in graph.nodes:
            continue
        old_node = graph.nodes[node_name]
        new_enzymes = dict(old_node.enzymes)
        for tag, abundance in ugt_abundances.items():
            if tag not in new_enzymes:
                new_enzymes[tag] = Distribution(mean=abundance, cv=0.40)
        # Create new node with updated enzymes
        new_node = Node(
            name=old_node.name,
            node_type=old_node.node_type,
            volume=old_node.volume,
            composition=old_node.composition,
            enzymes=new_enzymes,
            transporters=old_node.transporters,
            ivive_scaling=old_node.ivive_scaling,
            lookup_name=old_node.lookup_name,
        )
        graph.nodes[node_name] = new_node


def modify_drug_for_ugt(drug, fm_ugt: float, ugt_isoforms: list[str]):
    """Create a new DrugOnGraph with UGT fm redistribution.

    - Scale all existing CYP affinities by (1 - fm_ugt)
    - Add UGT affinities for the specified isoforms
    """
    from sisyphus.core import Distribution, DrugOnGraph

    if fm_ugt == 0.0:
        return drug  # Baseline: no UGT

    # Back-calculate total hepatic CLint from existing CYP affinities
    # CLint_enzyme = abundance × affinity × ivive_scaling
    # Total CLint = Σ(CLint_enzyme)
    total_clint_l_per_h = 0.0
    for tag, aff in drug.enzyme_affinity.items():
        # Get liver abundance for this enzyme
        liver_abund = _UGT_LIVER_ABUNDANCE.get(tag)
        if liver_abund is None:
            # It's a CYP — get from hardcoded liver abundances
            from sisyphus.predict.ivive import _LIVER_ENZYME_ABUNDANCE
            liver_abund = _LIVER_ENZYME_ABUNDANCE.get(tag, 1.0)
        total_clint_l_per_h += liver_abund * aff.mean * _IVIVE_SCALING

    # Scale existing CYP affinities by (1 - fm_ugt)
    new_affinity: dict[str, Distribution] = {}
    for tag, aff in drug.enzyme_affinity.items():
        new_affinity[tag] = Distribution(mean=aff.mean * (1 - fm_ugt), cv=aff.cv)

    # Add UGT affinities: distribute fm_ugt equally among isoforms
    fm_per_ugt = fm_ugt / len(ugt_isoforms)
    for ugt_tag in ugt_isoforms:
        abundance = _UGT_LIVER_ABUNDANCE.get(ugt_tag, 675_000.0)
        # affinity = (CLint_total * fm_per_ugt) / (abundance * ivive_scaling)
        ugt_affinity = (total_clint_l_per_h * fm_per_ugt) / (abundance * _IVIVE_SCALING)
        new_affinity[ugt_tag] = Distribution(mean=max(ugt_affinity, 0.0), cv=0.40)

    return DrugOnGraph(
        name=drug.name,
        smiles=drug.smiles,
        dose_mg=drug.dose_mg,
        route=drug.route,
        administration_node=drug.administration_node,
        mw=drug.mw,
        pka=drug.pka,
        compound_type=drug.compound_type,
        fup=drug.fup,
        rbp=drug.rbp,
        kp_method=drug.kp_method,
        kp_overrides=drug.kp_overrides,
        peff=drug.peff,
        solubility=drug.solubility,
        enzyme_affinity=new_affinity,
        renal_clearance=drug.renal_clearance,
        particle_radius_um=drug.particle_radius_um,
        ps_overrides=drug.ps_overrides,
    )


def run_engine(drug, graph, compiled):
    """Run single deterministic engine simulation, return Cmax."""
    from sisyphus.engine.compiler import ResolvedParams
    from sisyphus.engine.solver import solve
    from sisyphus.pk.endpoints import compute_endpoints

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
        return pk.cmax.mean
    return 0.0


def main():
    import sisyphus.engine.flux  # noqa: F401 — register flux specs
    from sisyphus.engine.compiler import ODECompiler
    from sisyphus.graph.builder import build_from_yaml
    from sisyphus.predict.adme import predict_adme
    from sisyphus.predict.chemistry import compute_profile
    from sisyphus.predict.ivive import build_drug_on_graph
    from sisyphus.validation.reference import load_reference

    physiology_dir = _ROOT / "data" / "physiology"

    # Load reference data for UGT drugs
    refs = load_reference()
    holdout_refs = {r.name: r for r in refs if r.in_holdout}

    # Build graph with UGT abundances
    print("Building graph with UGT abundances...", file=sys.stderr)
    graph = build_from_yaml(physiology_dir / "reference_man.yaml")
    add_ugt_to_graph(graph)

    # Compile (topology unchanged by adding enzyme tags — only abundances change)
    compiler = ODECompiler()
    compiled = compiler.compile(graph)

    # Get liver enzyme abundances for IVIVE
    liver_enzymes: dict[str, float] = {
        tag: dist.mean for tag, dist in graph.nodes["liver"].enzymes.items()
    }

    # Header
    print("Drug\tType\tCmax_obs\t" + "\t".join(f"fm_ugt={fm}" for fm in FM_UGT_LEVELS))
    print("Drug\tType\tCmax_obs\t" + "\t".join(f"fold_{fm}" for fm in FM_UGT_LEVELS), file=sys.stderr)

    results = {}

    for drug_name, ugt_isoforms in _UGT_DRUGS.items():
        ref = holdout_refs.get(drug_name)
        if ref is None:
            print(f"  {drug_name}: not in holdout, skipping", file=sys.stderr)
            continue

        # Build base drug
        profile = compute_profile(ref.smiles)
        adme = predict_adme(profile)
        base_drug = build_drug_on_graph(
            profile, adme, ref.dose_mg, ref.route, liver_enzymes=liver_enzymes
        )

        cmax_values = []
        fold_values = []

        for fm_ugt in FM_UGT_LEVELS:
            modified_drug = modify_drug_for_ugt(base_drug, fm_ugt, ugt_isoforms)
            cmax = run_engine(modified_drug, graph, compiled)
            cmax_values.append(cmax)
            fold = cmax / ref.cmax_obs if ref.cmax_obs > 0 and cmax > 0 else 0.0
            fold_values.append(fold)

        results[drug_name] = (profile.compound_type, ref.cmax_obs, cmax_values, fold_values)

        cmax_str = "\t".join(f"{c:.4f}" for c in cmax_values)
        fold_str = "\t".join(f"{f:.2f}" for f in fold_values)
        print(f"{drug_name}\t{profile.compound_type}\t{ref.cmax_obs:.4f}\t{cmax_str}")
        print(f"  {drug_name}: {fold_str}", file=sys.stderr)

    # Summary
    print("\n\n# ── SUMMARY ──", file=sys.stderr)

    for fm_ugt in FM_UGT_LEVELS:
        idx = FM_UGT_LEVELS.index(fm_ugt)
        folds = [results[d][3][idx] for d in results if results[d][3][idx] > 0]
        if folds:
            afes = np.power(10, np.mean(np.abs(np.log10(folds))))
            print(f"fm_ugt={fm_ugt}: AAFE={afes:.3f} (N={len(folds)})", file=sys.stderr)

    # Best fm_ugt per drug
    print("\n# ── BEST fm_ugt PER DRUG ──", file=sys.stderr)
    for drug_name in results:
        ctype, cmax_obs, cmax_vals, fold_vals = results[drug_name]
        # Find fm_ugt with fold closest to 1.0
        best_idx = min(range(len(fold_vals)), key=lambda i: abs(np.log10(fold_vals[i])) if fold_vals[i] > 0 else 99)
        base_fold = fold_vals[0]
        best_fold = fold_vals[best_idx]
        best_fm = FM_UGT_LEVELS[best_idx]
        improved = "IMPROVED" if abs(np.log10(best_fold)) < abs(np.log10(base_fold)) else "WORSE/SAME"
        print(
            f"  {drug_name}: base_fold={base_fold:.2f} best_fold={best_fold:.2f} "
            f"at fm_ugt={best_fm} ({improved}) type={ctype}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
