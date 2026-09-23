"""V3: verify solve() injects t_min_h into t_eval when provided."""

from __future__ import annotations

import numpy as np
import pytest

from sisyphus.engine.compiler import ODECompiler, ResolvedParams
from sisyphus.engine.solver import _IV_CMAX_DELAY_H, solve, solve_mc
from sisyphus.engine.uncertainty import UncertaintyEngine
from sisyphus.graph.builder import build_from_yaml
from sisyphus.pipeline.context import prepare_simulation_context
from sisyphus.pk.endpoints import compute_endpoints
from sisyphus.predict.adme import predict_adme
from sisyphus.predict.chemistry import compute_profile
from sisyphus.predict.ivive import build_drug_on_graph

_PHYS = "data/physiology/reference_man.yaml"


def _setup(route: str = "iv"):
    graph = build_from_yaml(_PHYS)
    profile = compute_profile("CCCCC(=O)N([C@@H](C(C)C)C(=O)O)Cc1ccc(-c2ccccc2-c2nnn[nH]2)cc1")  # valsartan  # noqa: E501
    adme = predict_adme(profile)
    drug = build_drug_on_graph(profile, adme, dose_mg=20.0, route=route)
    compiler = ODECompiler()
    compiled = compiler.compile(graph)
    rng = np.random.default_rng(42)
    params = ResolvedParams(graph.sample(rng), drug.sample(rng))
    y0 = np.zeros(compiled.n_states)
    y0[compiled.state_index[drug.administration_node]] = drug.dose_mg
    return compiled, params, y0


def test_iv_cmax_delay_constant_is_five_minutes():
    assert _IV_CMAX_DELAY_H == pytest.approx(5.0 / 60.0)


def test_solve_t_min_h_injects_anchor_point():
    compiled, params, y0 = _setup("iv")
    result = solve(compiled, params, y0, t_span=(0.0, 24.0), t_min_h=_IV_CMAX_DELAY_H)
    # t_min_h must appear as an exact time point in the grid.
    assert np.any(np.isclose(result.time_h, _IV_CMAX_DELAY_H))


def test_iv_auc_resolves_distribution_before_cmax_anchor():
    # Caffeine redistributes rapidly: a single 0-to-5-minute trapezoid used
    # to overestimate IV AUC by more than 25%, corrupting oral/IV F.
    context = prepare_simulation_context("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", 100.0, "iv")
    y0 = np.zeros(context.compiled.n_states)
    y0[context.compiled.state_index["venous_blood"]] = 100.0
    fine_grid = np.r_[
        np.linspace(0, _IV_CMAX_DELAY_H, 1001, endpoint=False),
        np.linspace(_IV_CMAX_DELAY_H, 24, 499),
    ]
    reference = solve(context.compiled, context.params, y0, (0, 24), t_eval=fine_grid)
    reference_auc = compute_endpoints(reference).auc_0t.mean
    default = solve(context.compiled, context.params, y0, (0, 24), t_min_h=_IV_CMAX_DELAY_H)
    default_auc = compute_endpoints(default).auc_0t.mean
    _, _, mc_auc, ok = solve_mc(
        context.compiled, context.params, y0, (0, 24), t_min_h=_IV_CMAX_DELAY_H
    )
    assert ok
    assert default_auc == pytest.approx(reference_auc, rel=0.01)
    assert mc_auc == pytest.approx(reference_auc, rel=0.01)


def test_solve_t_min_h_zero_is_backward_compatible():
    compiled, params, y0 = _setup("oral")
    result_default = solve(compiled, params, y0, t_span=(0.0, 24.0))
    result_zero = solve(compiled, params, y0, t_span=(0.0, 24.0), t_min_h=0.0)
    np.testing.assert_allclose(result_default.time_h, result_zero.time_h)


def test_solve_mc_iv_cmax_excludes_t0_spike():
    """Zero-width PI root cause: IV Cmax at t=0 = dose/V_venous (deterministic).
    With t_min_h > 0, the windowed max must be < the t=0 value."""
    compiled, params, y0 = _setup("iv")
    # Unfiltered (V2 behavior): Cmax should equal dose/V_venous = 20/3.7 ≈ 5.405
    cmax_v2, _, _, ok_v2 = solve_mc(compiled, params, y0, t_span=(0.0, 24.0))
    assert ok_v2
    assert cmax_v2 == pytest.approx(20.0 / 3.7, rel=1e-3)
    # Windowed (V3 behavior): Cmax must be strictly less than t=0 value.
    cmax_v3, tmax_v3, _, ok_v3 = solve_mc(
        compiled, params, y0, t_span=(0.0, 24.0), t_min_h=_IV_CMAX_DELAY_H
    )
    assert ok_v3
    assert cmax_v3 < cmax_v2
    assert tmax_v3 >= _IV_CMAX_DELAY_H


def test_solve_mc_t_min_h_zero_is_backward_compatible():
    compiled, params, y0 = _setup("oral")
    cmax_default, tmax_default, auc_default, ok_d = solve_mc(
        compiled, params, y0, t_span=(0.0, 24.0)
    )
    cmax_zero, tmax_zero, auc_zero, ok_z = solve_mc(
        compiled, params, y0, t_span=(0.0, 24.0), t_min_h=0.0
    )
    assert ok_d and ok_z
    assert cmax_default == pytest.approx(cmax_zero, rel=1e-6)
    assert tmax_default == pytest.approx(tmax_zero, rel=1e-6)
    assert auc_default == pytest.approx(auc_zero, rel=1e-6)


def test_propagate_fast_t_min_h_matches_solve_mc():
    """propagate_fast with t_min_h must produce non-degenerate, decreased Cmax
    median vs the V2 (t_min_h=0) path for an IV bolus."""
    compiled, params, y0 = _setup("iv")
    graph = build_from_yaml(_PHYS)
    profile = compute_profile("CCCCC(=O)N([C@@H](C(C)C)C(=O)O)Cc1ccc(-c2ccccc2-c2nnn[nH]2)cc1")
    adme = predict_adme(profile)
    drug = build_drug_on_graph(profile, adme, dose_mg=20.0, route="iv")

    ue = UncertaintyEngine()
    mc_v2 = ue.propagate_fast(
        compiled=compiled, graph=graph, drug=drug,
        n_samples=50, seed=42, t_span=(0.0, 24.0),
    )
    mc_v3 = ue.propagate_fast(
        compiled=compiled, graph=graph, drug=drug,
        n_samples=50, seed=42, t_span=(0.0, 24.0),
        t_min_h=_IV_CMAX_DELAY_H,
    )
    # V3 must yield strictly smaller median and non-degenerate PI (low < high).
    assert mc_v3.pk.cmax.mean < mc_v2.pk.cmax.mean
    low_v3, high_v3 = mc_v3.cmax_90ci
    assert high_v3 > low_v3  # non-degenerate
