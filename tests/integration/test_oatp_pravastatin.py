"""Integration tests for OATP1B1 hepatic uptake.

Depends on:
- data/physiology/reference_man.yaml having liver.transporters.OATP1B1
- data/transporters/oatp1b1.json with pravastatin entry
- predict/ivive.py supporting transporter_kinetics kwarg
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
from sisyphus.predict.transporter_db import (
    load_hepatic_ecm_params,
    load_oatp1b1_kinetics,
)

_PHYS = pathlib.Path("data/physiology/reference_man.yaml")
_PRAVASTATIN = (
    "CC[C@@H](C)C(=O)O[C@@H]1C[C@H](C=C2[C@@H]1CC[C@H]"
    "([C@@H]2CC[C@H](C[C@H](CC(=O)O)O)O)C)O"
)


def _simulate_cmax(drug, graph, t_end: float = 24.0) -> float:
    """Run a deterministic ODE solve and return venous_blood Cmax."""
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
def test_pravastatin_cmax_moves_with_oatp():
    """Engine Cmax for pravastatin decreases when OATP1B1 kinetics AND
    ECM parameters (biliary clearance + sinusoidal PS) are active vs off.

    Under ECM (post 2026-04-20), OATP kinetics alone do not change Cmax
    because the PS_passive/PS_eff defaults (1e6 L/h) collapse the model
    to well-stirred algebraically. Meaningful hepatic extraction reduction
    requires supplying ``hepatic_ecm_params`` (pravastatin: PS_passive=0.8,
    PS_eff=0.8, CL_int_bile=45). In the "on" arm both are loaded; in the
    "off" arm both are None → default well-stirred path.
    """
    graph = build_from_yaml(_PHYS)
    profile = compute_profile(_PRAVASTATIN)
    adme = predict_adme(profile)
    liver_enzymes = {
        tag: dist.mean for tag, dist in graph.nodes["liver"].enzymes.items()
    }

    drug_off = build_drug_on_graph(
        profile, adme, dose_mg=40.0, route="oral",
        liver_enzymes=liver_enzymes,
        transporter_kinetics=None,
        hepatic_ecm_params=None,
    )
    drug_on = build_drug_on_graph(
        profile, adme, dose_mg=40.0, route="oral",
        liver_enzymes=liver_enzymes,
        transporter_kinetics=load_oatp1b1_kinetics("pravastatin"),
        hepatic_ecm_params=load_hepatic_ecm_params("pravastatin"),
    )

    cmax_off = _simulate_cmax(drug_off, graph)
    cmax_on = _simulate_cmax(drug_on, graph)

    print(f"\npravastatin: cmax_off={cmax_off:.4f}, cmax_on={cmax_on:.4f}, "
          f"ratio={cmax_on / cmax_off:.3f}")

    ratio = cmax_on / cmax_off
    assert ratio < 0.95, (
        f"expected OATP1B1 uptake + ECM params to reduce Cmax meaningfully, "
        f"got cmax_off={cmax_off:.4f}, cmax_on={cmax_on:.4f}, ratio={ratio:.3f}"
    )


@pytest.mark.slow
def test_non_oatp_drug_unaffected_by_oatp_wiring():
    """Morphine (no OATP1B1 substrate) → transporter_kinetics empty → MM inactive."""
    graph = build_from_yaml(_PHYS)
    morphine_smiles = "CN1CCC23C4C1CC5=C2C(=C(C=C5)O)OC3C(C=C4)O"
    profile = compute_profile(morphine_smiles)
    adme = predict_adme(profile)
    liver_enzymes = {
        tag: dist.mean for tag, dist in graph.nodes["liver"].enzymes.items()
    }
    kinetics = load_oatp1b1_kinetics("morphine")
    drug = build_drug_on_graph(
        profile, adme, dose_mg=30.0, route="oral",
        liver_enzymes=liver_enzymes,
        transporter_kinetics=kinetics,
    )
    assert drug.transporter_kinetics == {}, "morphine should not have OATP kinetics"

    rng = np.random.default_rng(42)
    realized_graph = graph.sample(rng)
    realized_drug = drug.sample(rng)

    compiler = ODECompiler()
    compiled = compiler.compile(realized_graph)
    params = ResolvedParams(realized_graph, realized_drug)

    y0 = np.zeros(compiled.n_states)
    admin_idx = compiled.state_index[drug.administration_node]
    y0[admin_idx] = drug.dose_mg

    result = solve(compiled, params, y0, t_span=(0, 24.0))
    assert result.solver_success
    assert np.max(result.concentrations["venous_blood"]) > 0
