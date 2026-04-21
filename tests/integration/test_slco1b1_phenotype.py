"""Integration test for SLCO1B1 phenotype effect on hepatic uptake.

Verifies that scaling OATP1B1 abundance down (SLCO1B1 PM) increases
plasma Cmax for pravastatin (less hepatic extraction). Clinical data
(Niemi 2006, Pasanen 2007) report ~60-100% pravastatin AUC increase in
SLCO1B1 *5/*15 homozygous Poor Function carriers.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

import sisyphus.engine.flux  # noqa: F401 -- register flux specs
from sisyphus.engine.compiler import ODECompiler, ResolvedParams
from sisyphus.engine.solver import solve
from sisyphus.graph.builder import build_from_yaml
from sisyphus.predict.adme import predict_adme
from sisyphus.predict.chemistry import compute_profile
from sisyphus.predict.ivive import build_drug_on_graph
from sisyphus.predict.phenotype import apply_phenotype_to_graph
from sisyphus.predict.transporter_db import (
    load_hepatic_ecm_params,
    load_oatp1b1_kinetics,
)

_PHYS = pathlib.Path("data/physiology/reference_man.yaml")
_PRAVASTATIN = (
    "CC[C@@H](C)C(=O)O[C@@H]1C[C@H](C=C2[C@@H]1CC[C@H]"
    "([C@@H]2CC[C@H](C[C@H](CC(=O)O)O)O)C)O"
)


def _cmax(graph, drug, t_end: float = 8.0) -> float:
    rng = np.random.default_rng(42)
    realized_graph = graph.sample(rng)
    realized_drug = drug.sample(rng)
    compiler = ODECompiler()
    compiled = compiler.compile(realized_graph)
    params = ResolvedParams(realized_graph, realized_drug)
    y0 = np.zeros(compiled.n_states)
    admin_idx = compiled.state_index[drug.administration_node]
    y0[admin_idx] = drug.dose_mg
    result = solve(compiled, params, y0, t_span=(0, t_end))
    if not result.solver_success:
        pytest.fail("solver did not converge")
    return float(np.max(result.concentrations["venous_blood"]))


@pytest.mark.slow
def test_slco1b1_phenotype_runs_and_graph_scales():
    """Engine runs end-to-end with SLCO1B1 phenotype applied; graph-level
    scaling is verified.

    Post 2026-04-20 ECM migration, hepatic uptake is no longer flow-limited
    (abundance 5.0e5 under ECM vs 1.0e11 under the old MM-uptake model).
    PM (0.1× abundance) now produces clinically meaningful Cmax increase.
    See ``test_slco1b1_pm_increases_pravastatin_cmax`` for the directional gate.
    """
    g_em = build_from_yaml(_PHYS)
    g_pm = apply_phenotype_to_graph(g_em, {"SLCO1B1": "PM"})

    # Graph-level scaling contract.
    assert g_pm.nodes["liver"].transporters["OATP1B1"].mean == pytest.approx(
        g_em.nodes["liver"].transporters["OATP1B1"].mean * 0.10,
        rel=1e-6,
    )

    profile = compute_profile(_PRAVASTATIN)
    adme = predict_adme(profile)
    liver_enzymes = {
        tag: dist.mean for tag, dist in g_em.nodes["liver"].enzymes.items()
    }
    drug = build_drug_on_graph(
        profile, adme, dose_mg=40.0, route="oral",
        liver_enzymes=liver_enzymes,
        transporter_kinetics=load_oatp1b1_kinetics("pravastatin"),
    )

    cmax_em = _cmax(g_em, drug)
    cmax_pm = _cmax(g_pm, drug)

    # End-to-end sanity: both solves produce finite, positive Cmax.
    assert np.isfinite(cmax_em) and cmax_em > 0
    assert np.isfinite(cmax_pm) and cmax_pm > 0


def test_slco1b1_em_is_noop_on_transporter():
    """EM phenotype (1.0× scale) must leave transporter unchanged."""
    g_em = build_from_yaml(_PHYS)
    g_em_applied = apply_phenotype_to_graph(g_em, {"SLCO1B1": "EM"})
    assert g_em_applied.nodes["liver"].transporters["OATP1B1"].mean == (
        g_em.nodes["liver"].transporters["OATP1B1"].mean
    )


@pytest.mark.slow
def test_slco1b1_pm_increases_pravastatin_cmax():
    """PM phenotype → pravastatin Cmax ≥ 1.3× EM (Niemi 2006 directional).

    Under ECM (post 2026-04-20), OATP1B1 is no longer flow-limited —
    scaling abundance 0.1× (PM) moves PS_active from ≈503 L/h to ≈50 L/h,
    producing measurable Cmax change. Clinical AUC +60-100% in PM
    translates to ~1.3-1.5× Cmax.
    """
    graph = build_from_yaml(_PHYS)
    profile = compute_profile(_PRAVASTATIN)
    adme = predict_adme(profile)
    liver_enzymes = {
        tag: d.mean for tag, d in graph.nodes["liver"].enzymes.items()
    }
    drug = build_drug_on_graph(
        profile, adme, dose_mg=40.0, route="oral",
        liver_enzymes=liver_enzymes,
        transporter_kinetics=load_oatp1b1_kinetics("pravastatin"),
        hepatic_ecm_params=load_hepatic_ecm_params("pravastatin"),
    )
    graph_em = apply_phenotype_to_graph(graph, {"SLCO1B1": "EM"})
    graph_pm = apply_phenotype_to_graph(graph, {"SLCO1B1": "PM"})

    cmax_em = _cmax(graph_em, drug)
    cmax_pm = _cmax(graph_pm, drug)
    ratio = cmax_pm / cmax_em
    assert ratio >= 1.3, (
        f"SLCO1B1 PM gate failed: PM/EM = {ratio:.2f} < 1.3. "
        f"EM Cmax={cmax_em:.4f}, PM Cmax={cmax_pm:.4f}"
    )
