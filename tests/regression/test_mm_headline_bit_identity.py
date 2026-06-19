"""v2.2a headline-isolation guard (stack-independent).

A drug with empty ``enzyme_km`` takes the verbatim LINEAR ``well_stirred`` branch; the same
drug with an astronomically large ``Km`` takes the SATURABLE branch in its linear limit
(``1/(1+C_u/Km) → 1``). They must agree to full float tolerance on ANY numerics stack — this
locks that the guarded fork does not perturb the linear path and that empty ``enzyme_km`` routes
to it. A regression that broke the linear branch (or mis-routed the fork) would make the two
diverge.

NOTE: an *exact absolute* Cmax pin is deliberately NOT used — per-drug Cmax is not bit-identical
across BLAS/numerics stacks (see the README "Holdout benchmark reproducibility note"), so a macOS-
recorded float fails on CI Linux. The absolute cross-stack headline (2.731) is guarded by the
stack-tolerant holdout cache pin ``test_cached_holdout_aafe_is_2p731``; this test guards the
v2.2a fork locally and portably.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import sisyphus.engine.flux  # noqa: F401
from sisyphus.core import Distribution, DrugOnGraph
from sisyphus.engine.compiler import ODECompiler, ResolvedParams
from sisyphus.engine.solver import solve
from sisyphus.graph.builder import build_from_yaml


def _cmax(graph, enzyme_km):
    d = DrugOnGraph(
        name="caf", smiles="C", dose_mg=100.0, route="oral",
        administration_node="stomach_lumen", mw=194.0, pka=None, compound_type="base",
        fup=Distribution(0.65, 0), rbp=Distribution(1.0, 0), kp_method="rodgers_rowland",
        kp_overrides={}, peff=Distribution(5.0, 0), solubility=Distribution(1000.0, 0),
        enzyme_affinity={"CYP1A2": Distribution(1.0e-3, 0)},
        renal_clearance=Distribution(0.0, 0),
        enzyme_km=enzyme_km,
    ).realize_means()
    c = ODECompiler().compile(graph)
    p = ResolvedParams(graph, d)
    y0 = np.zeros(c.n_states)
    y0[c.state_index["stomach_lumen"]] = d.dose_mg
    r = solve(c, p, y0, t_span=(0.0, 200.0))
    return float(r.concentrations["venous_blood"].max())


def test_empty_enzyme_km_linear_branch_matches_saturable_limit():
    g = build_from_yaml(Path("data/physiology/reference_man.yaml")).realize_means()
    linear = _cmax(g, {})                                        # empty ⇒ verbatim linear branch
    sat_limit = _cmax(g, {"CYP1A2": Distribution(1.0e12, 0)})    # huge Km ⇒ saturable→linear limit
    assert linear > 0.0
    assert sat_limit == pytest.approx(linear, rel=1e-9)
