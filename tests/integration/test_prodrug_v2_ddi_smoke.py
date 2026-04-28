"""Smoke test: halving CES1 abundance approximately halves remdesivir activation.

Probes the active metabolite (GS-441524) pool concentration under two
graphs that differ only in CES1 abundance. The active production rate is
proportional to CES1 abundance via the well-stirred enzyme flux, so
halving CES1 should reduce active Cmax meaningfully (~2x lower).
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from sisyphus.core import Distribution
from sisyphus.engine.compiler import ODECompiler, ResolvedParams
from sisyphus.graph.body import BodyGraph
from sisyphus.graph.builder import augment_for_active_species, build_from_yaml
from sisyphus.predict.adme import predict_adme
from sisyphus.predict.chemistry import compute_profile
from sisyphus.predict.ivive import build_drug_on_graph

# Register flux specs.
import sisyphus.engine.flux  # noqa: F401

REMDESIVIR_SMILES = (
    "CCC(CC)COC(=O)[C@H](C)N[P@](=O)"
    "(OC[C@H]1O[C@@](C#N)(c2ccc3c(N)ncnn23)[C@H](O)[C@@H]1O)Oc1ccccc1"
)


def _scaled_ces1_graph(scale: float) -> BodyGraph:
    yaml_path = Path(__file__).resolve().parent.parent.parent / "data" / "physiology" / "reference_man.yaml"
    g = build_from_yaml(yaml_path)
    if scale != 1.0:
        for node_name, node in list(g.nodes.items()):
            if "CES1" in node.enzymes:
                old = node.enzymes["CES1"]
                new_enzymes = dict(node.enzymes)
                new_enzymes["CES1"] = Distribution(
                    mean=old.mean * scale,
                    cv=old.cv,
                    dist_type=old.dist_type,
                    correlation_group=old.correlation_group,
                )
                g.nodes[node_name] = dataclasses.replace(node, enzymes=new_enzymes)
    return g


def _build_remdesivir_drug(graph: BodyGraph):
    profile = compute_profile(REMDESIVIR_SMILES)
    adme = predict_adme(profile)
    liver_enzymes = {tag: dist.mean for tag, dist in graph.nodes["liver"].enzymes.items()}
    return build_drug_on_graph(
        profile, adme, dose_mg=200.0, route="iv", liver_enzymes=liver_enzymes,
    )


def _solve_active_cmax(graph: BodyGraph, drug) -> float:
    g_aug = augment_for_active_species(graph, drug, observation_node="venous_blood")
    compiler = ODECompiler()
    compiled = compiler.compile(g_aug)

    rng = np.random.default_rng(42)
    realized_graph = g_aug.sample(rng)
    realized_drug = drug.sample(rng)
    params = ResolvedParams(realized_graph, realized_drug)
    rhs = compiled.make_rhs(params)

    y0 = np.zeros(compiled.n_states)
    y0[compiled.state_index[drug.administration_node]] = drug.dose_mg

    sol = solve_ivp(
        rhs, (0.0, 24.0), y0, method="LSODA",
        t_eval=np.linspace(0.0, 24.0, 200),
        rtol=1e-8, atol=1e-10,
    )
    assert sol.success, f"solver failed: {sol.message}"

    active_idx = compiled.state_index["venous_blood_active"]
    v_active = drug.active_metabolite.Vd_L.mean
    c_active = sol.y[active_idx] / v_active
    return float(c_active.max())


def test_halving_ces1_reduces_remdesivir_activation():
    """Halving CES1 should reduce active Cmax meaningfully (>=30% drop, <70% drop)."""
    g_full = _scaled_ces1_graph(1.0)
    drug_full = _build_remdesivir_drug(g_full)
    cmax_full = _solve_active_cmax(g_full, drug_full)

    g_half = _scaled_ces1_graph(0.5)
    drug_half = _build_remdesivir_drug(g_half)
    cmax_half = _solve_active_cmax(g_half, drug_half)

    ratio = cmax_half / cmax_full
    assert ratio < 0.7, (
        f"Halving CES1 should reduce active Cmax meaningfully; "
        f"ratio={ratio:.3f} (full={cmax_full:.4g}, half={cmax_half:.4g})"
    )
    assert ratio > 0.3, f"ratio={ratio:.3f} seems too aggressive"
