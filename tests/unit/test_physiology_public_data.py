"""Public-source adult physiology presets and phenotype coverage."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from sisyphus.compounds import load_compound
from sisyphus.engine.compiler import ODECompiler, ResolvedParams
from sisyphus.engine.solver import solve
from sisyphus.engine.uncertainty import UncertaintyEngine
from sisyphus.graph.presets import reference_man, reference_woman
from sisyphus.graph.types import FlowEdge
from sisyphus.predict.phenotype import apply_phenotype_to_graph


def test_blood_pool_cv_keeps_reference_means() -> None:
    graph = reference_man()
    for name, mean in (("venous_blood", 3.7), ("arterial_blood", 1.5)):
        volume = graph.nodes[name].volume
        assert volume.mean == mean
        assert volume.cv == 0.12
        draws = [volume.sample(np.random.default_rng(seed)) for seed in range(20)]
        assert min(draws) < mean < max(draws)


def test_blood_pool_cv_changes_mc_interval() -> None:
    fixed = reference_man().realize_means()
    variable = reference_man().realize_means()
    for name in ("venous_blood", "arterial_blood"):
        variable.nodes[name] = replace(
            variable.nodes[name], volume=reference_man().nodes[name].volume
        )
    drug = load_compound(Path("data/compounds/midazolam.yaml")).realize_means()
    compiled = ODECompiler().compile(fixed)
    engine = UncertaintyEngine()
    fixed_ci = engine.propagate_fast(compiled, fixed, drug, n_samples=30, seed=13).cmax_90ci
    variable_ci = engine.propagate_fast(compiled, variable, drug, n_samples=30, seed=13).cmax_90ci
    assert fixed_ci[0] == fixed_ci[1]
    assert variable_ci[0] < variable_ci[1]


def test_reference_woman_builds_and_conserves_flow() -> None:
    graph = reference_woman()
    man = reference_man()
    assert len(graph.nodes) == 34
    assert graph.validate() == []
    assert graph.global_params["body_weight"].mean == 60.0
    assert graph.global_params["cardiac_output"].mean == 354.0
    assert graph.nodes["liver"].volume.mean == 1.40
    assert sum(graph.nodes[name].volume.mean for name in (
        "venous_blood", "arterial_blood", "portal_vein"
    )) == pytest.approx(3.9)
    liver_return = next(
        edge for edge in graph.edges
        if isinstance(edge, FlowEdge)
        and edge.source == "liver" and edge.target == "venous_blood"
    )
    assert liver_return.flow_rate.mean == pytest.approx(0.27 * 354.0)
    for node, ratio in (("liver", 1400 / 1500), ("gut_wall", 1.10 / 1.03),
                        ("kidney", 0.275 / 0.31)):
        for tag, abundance in man.nodes[node].enzymes.items():
            assert graph.nodes[node].enzymes[tag].mean == pytest.approx(
                abundance.mean * ratio, rel=1e-9
            )


def test_reference_woman_simulates_with_mass_balance() -> None:
    graph = reference_woman().realize_means()
    drug = load_compound(Path("data/compounds/midazolam.yaml")).realize_means()
    compiled = ODECompiler().compile(graph)
    y0 = np.zeros(compiled.n_states)
    y0[compiled.state_index[drug.administration_node]] = drug.dose_mg
    result = solve(compiled, ResolvedParams(graph, drug), y0, (0.0, 24.0))
    assert result.solver_success
    assert result.mass_balance_error < 1e-6


@pytest.mark.parametrize("tag,mean", (
    ("CYP2B6", 457650),
    ("CYP2C19", 231525),
    ("CYP3A5", 582525),
))
def test_new_cyp_tags_scale_hepatic_abundance(tag: str, mean: float) -> None:
    graph = reference_man()
    assert graph.nodes["liver"].enzymes[tag].mean == mean
    scaled, report = apply_phenotype_to_graph(
        graph, {tag: "PM"}, return_report=True
    )
    assert report.applied == ((tag, "PM"),)
    assert report.unsupported == ()
    assert scaled.nodes["liver"].enzymes[tag].mean == pytest.approx(mean * 0.1)
    assert scaled.nodes["liver"].enzymes[tag].cv == graph.nodes["liver"].enzymes[tag].cv
