"""Liver-zonation invariance probe (Phase-0).

Demonstrates that axial first-pass extraction is invariant to the spatial distribution
of a hepatic enzyme (total preserved): ΔE(N) = E_zonated - E_uniform -> 0 as N grows
(plug-flow convergence). Harness-isolated; reuses the synthetic-engine helpers from
scripts/validate_pgx_cmax_v2b.py. No predict()/reference_man.yaml/holdout change.
"""
from __future__ import annotations

import dataclasses as _dc
import importlib.util
import re
from pathlib import Path

from sisyphus.core import Distribution
from sisyphus.graph.body import BodyGraph
from sisyphus.validation.pgx_metrics import plugflow_E_linear, zonation_weights  # noqa: F401

_ROOT = Path(__file__).resolve().parent.parent
_HARNESS = _ROOT / "scripts" / "validate_pgx_cmax_v2b.py"


def _load_harness():
    spec = importlib.util.spec_from_file_location("v2b_harness", _HARNESS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


h = _load_harness()  # exposes _axial_graph, _engine_e_h, _SYNTHETIC_GENE_ABUND, ...


def _liver_subtanks(graph):
    """Liver sub-tanks ordered inlet->outlet by the __ax{i} index."""
    subs = [n for n in graph.nodes.values() if (n.lookup_name or n.name) == "liver"]
    return sorted(subs, key=lambda nd: int(re.search(r"__ax(\d+)$", nd.name).group(1)))


def apply_zonation(graph, gene_tag: str, weights: list[float]):
    """Return a new graph with the gene's abundance redistributed across liver sub-tanks
    by `weights` (sum=1), TOTAL PRESERVED: abundance_i = total * weights[i]."""
    subs = _liver_subtanks(graph)
    if len(subs) != len(weights):
        raise ValueError(f"{len(weights)} weights for {len(subs)} sub-tanks")
    total = sum(nd.enzymes[gene_tag].mean for nd in subs)
    new_nodes = dict(graph.nodes)
    for nd, w in zip(subs, weights):
        old = nd.enzymes[gene_tag]
        new_enz = dict(nd.enzymes)
        new_enz[gene_tag] = Distribution(total * w, old.cv, old.dist_type)
        new_nodes[nd.name] = _dc.replace(nd, enzymes=new_enz)
    g2 = BodyGraph()
    g2.nodes = new_nodes
    g2.edges = list(graph.edges)
    g2.global_params = dict(graph.global_params)
    return g2


def delta_E(gene_tag, fm, n_sub, ratio, direction, cltot, fup, mw, km_mgl=None,
            dose_mg=100.0, kp=3.0, peff=20.0):
    """E_zonated - E_uniform at sub-tank count n_sub. abund=_SYNTHETIC_GENE_ABUND so the
    drug affinity matches the (total-preserved) per-tank abundances. km_mgl=None => linear."""
    abund = h._SYNTHETIC_GENE_ABUND
    g_uni = h._axial_graph(gene_tag, n_sub=n_sub)
    e_uni = h._engine_e_h(g_uni, gene_tag, fm, cltot, abund, peff, kp, fup, dose_mg, mw, km_mgl)
    w = zonation_weights(n_sub, ratio, direction)
    g_zon = apply_zonation(g_uni, gene_tag, w)
    e_zon = h._engine_e_h(g_zon, gene_tag, fm, cltot, abund, peff, kp, fup, dose_mg, mw, km_mgl)
    return e_zon - e_uni, e_uni, e_zon
