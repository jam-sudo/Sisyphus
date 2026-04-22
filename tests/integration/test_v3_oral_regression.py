"""V3 regression guard: oral drugs must produce identical results to V2.

V3 is route-conditional — when route == "oral", t_min_h = 0.0 and the engine
code path is structurally unchanged. This test pins that invariant.
"""

from __future__ import annotations

import numpy as np
import pytest

from sisyphus.engine.compiler import ODECompiler, ResolvedParams
from sisyphus.engine.solver import solve, solve_mc
from sisyphus.graph.builder import build_from_yaml
from sisyphus.predict.adme import predict_adme
from sisyphus.predict.chemistry import compute_profile
from sisyphus.predict.ivive import build_drug_on_graph


def _setup_oral(dose_mg: float = 20.0):
    graph = build_from_yaml("data/physiology/reference_man.yaml")
    # Caffeine — well-behaved oral reference.
    profile = compute_profile("Cn1cnc2c1c(=O)n(C)c(=O)n2C")
    adme = predict_adme(profile)
    drug = build_drug_on_graph(profile, adme, dose_mg=dose_mg, route="oral")
    compiler = ODECompiler()
    compiled = compiler.compile(graph)
    rng = np.random.default_rng(42)
    params = ResolvedParams(graph.sample(rng), drug.sample(rng))
    y0 = np.zeros(compiled.n_states)
    y0[compiled.state_index[drug.administration_node]] = drug.dose_mg
    return compiled, params, y0


def test_oral_solve_v3_equals_v2():
    compiled, params, y0 = _setup_oral()
    sim_v2 = solve(compiled, params, y0, t_span=(0.0, 24.0))
    sim_v3 = solve(compiled, params, y0, t_span=(0.0, 24.0), t_min_h=0.0)
    np.testing.assert_allclose(sim_v2.time_h, sim_v3.time_h)
    np.testing.assert_allclose(
        sim_v2.concentrations["venous_blood"],
        sim_v3.concentrations["venous_blood"],
    )


def test_oral_solve_mc_v3_equals_v2():
    compiled, params, y0 = _setup_oral()
    cmax_v2, tmax_v2, auc_v2, ok_v2 = solve_mc(compiled, params, y0, t_span=(0.0, 24.0))
    cmax_v3, tmax_v3, auc_v3, ok_v3 = solve_mc(
        compiled, params, y0, t_span=(0.0, 24.0), t_min_h=0.0
    )
    assert ok_v2 and ok_v3
    assert cmax_v2 == pytest.approx(cmax_v3, rel=1e-6)
    assert tmax_v2 == pytest.approx(tmax_v3, rel=1e-6)
    assert auc_v2 == pytest.approx(auc_v3, rel=1e-6)
