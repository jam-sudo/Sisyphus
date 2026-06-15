from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import sisyphus.engine.flux  # noqa: F401  -- register flux specs
from sisyphus.core import Distribution, DrugOnGraph
from sisyphus.engine.compiler import ODECompiler, ResolvedParams
from sisyphus.engine.solver import solve
from sisyphus.graph.builder import build_from_yaml

_YAML = Path("data/physiology/reference_man.yaml")


def _ws_graph():
    g = build_from_yaml(_YAML)
    g.edges[:] = [
        replace(e, model="well_stirred")
        if getattr(e, "source", None) == "liver" and getattr(e, "model", None) == "extended"
        else e
        for e in g.edges
    ]
    g.nodes["liver"].enzymes["CYP2D6"] = Distribution(1.0e6, 0.0)
    return g


def _drug(dose, enzyme_km=None):
    return DrugOnGraph(
        name="syn", smiles="C", dose_mg=dose, route="intravenous",
        administration_node="venous_blood", mw=300.0, pka=None, compound_type="neutral",
        fup=Distribution(0.3, 0.0), rbp=Distribution(1.0, 0.0),
        kp_method="provided",
        kp_overrides={t: Distribution(3.0, 0.0) for t in
                      ["adipose", "bone", "brain", "gut", "heart", "kidney", "liver",
                       "lung", "muscle", "skin", "spleen"]},
        peff=Distribution(5.0, 0.0), solubility=Distribution(1000.0, 0.0),
        enzyme_affinity={"CYP2D6": Distribution(2.0e-3, 0.0)},
        renal_clearance=Distribution(0.0, 0.0),
        enzyme_km=enzyme_km or {},
    )


def _auc(graph, drug, t_end=10000.0):
    # The synthetic CYP2D6-only drug is very-low-extraction (fu*CLint << Q_liver),
    # so it clears slowly; integrate well past elimination to capture the full AUC
    # (and the saturation signal, which lives in the terminal tail). At 400h most of
    # the AUC is still un-eliminated, masking the dose-nonlinearity.
    rg, rd = graph.realize_means(), drug.realize_means()
    compiled = ODECompiler().compile(rg)
    params = ResolvedParams(rg, rd)
    y0 = np.zeros(compiled.n_states)
    y0[compiled.state_index[drug.administration_node]] = drug.dose_mg
    res = solve(compiled, params, y0, t_span=(0.0, t_end))
    conc, t = res.concentrations["venous_blood"], res.time_h
    trapz = getattr(np, "trapezoid", np.trapz)
    return float(trapz(conc, t))


def test_mm_rate_oracle_matches_formula():
    g = _ws_graph().realize_means()
    drug = _drug(100.0, {"CYP2D6": Distribution(0.05, 0.0)}).realize_means()
    params = ResolvedParams(g, drug)
    compiled = ODECompiler().compile(g)
    spec = next(s for s in compiled.flux_specs
                if getattr(s, "model", None) == "well_stirred"
                and getattr(s, "source_name", None) == "liver")
    y = np.zeros(compiled.n_states)
    y[spec.source_idx] = 50.0
    dydt = np.zeros(compiled.n_states)
    spec.apply(0.0, y, dydt, params)
    fup = params.drug_param("fup")
    if params.node_param("liver", "fu_correction_applicable") > 0:
        fup = fup * params.drug_param("fu_correction_liver")
    v = params.node_param("liver", "volume")
    kp = params.drug_kp("liver")
    c_plasma = y[spec.source_idx] / (v * kp)
    c_u = fup * c_plasma
    ivive = params.node_param("liver", "ivive_scaling")
    clint = 0.0
    for tag, ab in params.node_enzymes("liver").items():
        af = params.drug_enzyme_affinity(tag)
        if af > 0 and ab > 0:
            clint += ab * af * ivive / (1.0 + c_u / params.drug_enzyme_km(tag))
    expected = -(fup * clint) * c_plasma
    assert dydt[spec.source_idx] == pytest.approx(expected, rel=1e-9)


def test_linear_limit_matches_no_km():
    g = _ws_graph()
    auc_linear = _auc(g, _drug(100.0))
    auc_huge_km = _auc(g, _drug(100.0, {"CYP2D6": Distribution(1.0e12, 0.0)}))
    assert auc_huge_km == pytest.approx(auc_linear, rel=1e-6)


def test_dose_proportionality_breaks_under_saturation():
    g = _ws_graph()
    km = {"CYP2D6": Distribution(0.05, 0.0)}
    auc_1x = _auc(g, _drug(100.0, km))
    auc_2x = _auc(g, _drug(200.0, km))
    assert auc_2x / auc_1x > 2.05
    lin_1x = _auc(g, _drug(100.0))
    lin_2x = _auc(g, _drug(200.0))
    assert lin_2x / lin_1x == pytest.approx(2.0, rel=1e-3)
