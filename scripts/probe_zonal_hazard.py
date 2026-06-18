"""Zonal reactive-metabolite hazard probe (Bridge B / B1, Phase-0).

Computes a per-zone reactive-metabolite hazard as a POST-PROCESSOR on the axial
parent concentration profile (the reactive metabolite is NOT an engine species).
Demonstrates: hazard localizes by zonation; bulk parent PK is invariant to that
zonation (DE-50) while per-zone hazard is not; a saturable-detox dose-threshold with
zone-specificity (acetaminophen pattern). Harness-isolated; reuses the synthetic-engine
helpers from scripts/validate_pgx_cmax_v2b.py. No predict()/reference_man.yaml/holdout.
"""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import numpy as np

from sisyphus.validation.pgx_metrics import zonal_hazard, zonation_weights  # noqa: F401

_ROOT = Path(__file__).resolve().parent.parent
_HARNESS = _ROOT / "scripts" / "validate_pgx_cmax_v2b.py"


def _load_harness():
    spec = importlib.util.spec_from_file_location("v2b_harness", _HARNESS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


h = _load_harness()


def _subtank_names(graph):
    subs = [n.name for n in graph.nodes.values() if (n.lookup_name or n.name) == "liver"]
    return sorted(subs, key=lambda nm: int(re.search(r"__ax(\d+)$", nm).group(1)))


def _parent_profile_by_zone(gene_tag, fm, n_sub, cltot, fup, mw, km_mgl, dose_mg=100.0,
                            kp=3.0, peff=20.0):
    """Solve a single oral dose on the synthetic axial liver; return
    (c_u_by_zone, time): per-sub-tank UNBOUND parent conc arrays (C_u = fup*c_node,
    matching _peak_liver_cu) ordered inlet(ax1)->outlet(axN), and the time grid."""
    from sisyphus.engine.compiler import ODECompiler, ResolvedParams
    from sisyphus.engine.solver import solve
    g = h._axial_graph(gene_tag, n_sub=n_sub)
    abund = h._SYNTHETIC_GENE_ABUND
    drug = h._sat_drug(gene_tag, fm, cltot, abund, peff, kp, km_mgl, fup, dose_mg, mw)
    rg, rd = g.realize_means(), drug.realize_means()
    compiled = ODECompiler().compile(rg)
    params = ResolvedParams(rg, rd)
    y0 = np.zeros(compiled.n_states)
    y0[compiled.state_index[drug.administration_node]] = drug.dose_mg
    res = solve(compiled, params, y0, t_span=(0.0, float(h._T_EVAL[-1])), t_eval=h._T_EVAL)
    names = _subtank_names(g)
    c_u_by_zone = [fup * np.asarray(res.concentrations[nm]) for nm in names]
    return c_u_by_zone, np.asarray(res.time_h)


def zone_hazard_profile(gene_tag, fm, n_sub, cltot, fup, mw, km_mgl, dose_mg,
                        bio_direction, bio_ratio, detox_direction, detox_ratio,
                        vmax_bio_total, vmax_detox_total, km_bio):
    """Per-zone hazard for given bioactivation- and detox-zonation (independent),
    totals preserved. Returns per-zone hazard (inlet->outlet)."""
    c_by_zone, time = _parent_profile_by_zone(gene_tag, fm, n_sub, cltot, fup, mw,
                                              km_mgl, dose_mg)
    wbio = zonation_weights(n_sub, bio_ratio, bio_direction)
    wdet = zonation_weights(n_sub, detox_ratio, detox_direction)
    vmax_bio = [vmax_bio_total * w for w in wbio]
    vmax_detox = [vmax_detox_total * w for w in wdet]
    return zonal_hazard(c_by_zone, vmax_bio, km_bio, vmax_detox, time)


def bulk_E(gene_tag, fm, n_sub, cltot, fup, mw, km_mgl, bio_direction, bio_ratio):
    """Bulk parent extraction with the bioactivation enzyme zonated (G2 invariance arm)."""
    from scripts.probe_liver_zonation import apply_zonation
    g = h._axial_graph(gene_tag, n_sub=n_sub)
    g = apply_zonation(g, gene_tag, zonation_weights(n_sub, bio_ratio, bio_direction))
    return h._engine_e_h(g, gene_tag, fm, cltot, h._SYNTHETIC_GENE_ABUND, 20.0, 3.0,
                         fup, 100.0, mw, km_mgl)
