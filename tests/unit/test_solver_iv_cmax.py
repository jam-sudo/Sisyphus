"""V3: verify solve() injects t_min_h into t_eval when provided."""

from __future__ import annotations

import numpy as np
import pytest

from sisyphus.engine.compiler import ODECompiler, ResolvedParams
from sisyphus.engine.solver import _IV_CMAX_DELAY_H, solve
from sisyphus.graph.builder import build_from_yaml
from sisyphus.predict.adme import predict_adme
from sisyphus.predict.chemistry import compute_profile
from sisyphus.predict.ivive import build_drug_on_graph

_PHYS = "data/physiology/reference_man.yaml"


def _setup(route: str = "iv"):
    graph = build_from_yaml(_PHYS)
    profile = compute_profile("CCCCC(=O)N([C@@H](C(C)C)C(=O)O)Cc1ccc(-c2ccccc2-c2nnn[nH]2)cc1")  # valsartan
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


def test_solve_t_min_h_zero_is_backward_compatible():
    compiled, params, y0 = _setup("oral")
    result_default = solve(compiled, params, y0, t_span=(0.0, 24.0))
    result_zero = solve(compiled, params, y0, t_span=(0.0, 24.0), t_min_h=0.0)
    np.testing.assert_allclose(result_default.time_h, result_zero.time_h)
