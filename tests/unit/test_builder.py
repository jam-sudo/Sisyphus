"""Tests for YAML builder and compound loader."""

from pathlib import Path

import numpy as np
import pytest

from sisyphus.compounds import load_compound
from sisyphus.core import DrugOnGraph
from sisyphus.engine.compiler import ODECompiler, ResolvedParams
from sisyphus.engine.solver import solve
from sisyphus.graph.body import BodyGraph
from sisyphus.graph.builder import build_from_yaml, merge_overlay
from sisyphus.graph.types import ActiveTransportEdge, FlowEdge


class TestBuilder:
    @pytest.mark.parametrize("overlay,node", [
        ("sc_overlay.yaml", "sc_depot"),
        ("tumor_overlay.yaml", "tumor"),
    ])
    def test_merge_overlay(self, overlay, node):
        base = build_from_yaml(Path("data/physiology/reference_man.yaml"))
        merged = merge_overlay(base, Path("data/physiology") / overlay)
        assert node in merged.nodes
        assert node not in base.nodes
        assert merged.validate() == []
        if node == "tumor":
            rest_flow = next(
                e for e in merged.edges
                if isinstance(e, FlowEdge) and e.source == "arterial_blood" and e.target == "rest"
            )
            assert rest_flow.flow_rate.mean == pytest.approx(0.064 * 390.0)

    def test_sc_yaml_compound_simulates_with_closed_mass_balance(self):
        graph = merge_overlay(
            build_from_yaml(Path("data/physiology/reference_man.yaml")),
            Path("data/physiology/sc_overlay.yaml"),
        ).realize_means()
        drug = load_compound(Path("data/compounds/midazolam_sc.yaml")).realize_means()
        compiled = ODECompiler().compile(graph)
        y0 = np.zeros(compiled.n_states)
        y0[compiled.state_index[drug.administration_node]] = drug.dose_mg
        result = solve(compiled, ResolvedParams(graph, drug), y0, (0.0, 24.0))
        assert result.solver_success
        assert result.mass_balance_error < 1e-6

    def test_yaml_efflux_direction(self, tmp_path):
        path = tmp_path / "efflux.yaml"
        path.write_text(
            "global_params: {cardiac_output: 1}\n"
            "nodes:\n"
            "  - {name: tissue, type: organ, volume: 1}\n"
            "  - {name: lumen, type: lumen, volume: 1}\n"
            "edges:\n"
            "  - {source: tissue, target: lumen, type: active_transport, direction: efflux}\n"
        )
        graph = build_from_yaml(path)
        assert isinstance(graph.edges[0], ActiveTransportEdge)
        assert graph.edges[0].direction == "efflux"
        path.write_text(path.read_text().replace("efflux", "backwards"))
        with pytest.raises(ValueError, match="Unknown active transport direction"):
            build_from_yaml(path)

    def test_build_reference_man(self):
        g = build_from_yaml(Path("data/physiology/reference_man.yaml"))
        assert isinstance(g, BodyGraph)
        assert len(g.nodes) == 34
        assert "liver" in g.nodes
        assert "venous_blood" in g.nodes
        assert g.nodes["liver"].enzymes  # has enzymes

    def test_flow_conservation(self):
        g = build_from_yaml(Path("data/physiology/reference_man.yaml"))
        errors = g.validate()
        assert errors == [], f"Validation errors: {errors}"

    def test_cardiac_output_in_global_params(self):
        g = build_from_yaml(Path("data/physiology/reference_man.yaml"))
        assert "cardiac_output" in g.global_params
        co = g.global_params["cardiac_output"]
        assert co.mean == pytest.approx(390.0)

    def test_flow_rates_absolute(self):
        """Flow rates should be absolute L/h, not fractions."""
        g = build_from_yaml(Path("data/physiology/reference_man.yaml"))
        # Find brain inflow edge
        brain_inflow = [e for e in g.edges if e.target == "brain" and e.source == "arterial_blood"]
        assert len(brain_inflow) == 1
        assert brain_inflow[0].flow_rate.mean == pytest.approx(0.12 * 390.0, rel=1e-3)

    def test_return_flow_inferred(self):
        """Return edges should have same flow rate as incoming arterial edge."""
        g = build_from_yaml(Path("data/physiology/reference_man.yaml"))
        brain_in = [e for e in g.edges if e.target == "brain" and e.source == "arterial_blood"][0]
        brain_out = [e for e in g.edges if e.source == "brain" and e.target == "venous_blood"][0]
        assert brain_out.flow_rate.mean == pytest.approx(brain_in.flow_rate.mean)

    def test_liver_total_inflow(self):
        """Liver inflow = HA (0.065*CO) + portal vein (0.19*CO)."""
        g = build_from_yaml(Path("data/physiology/reference_man.yaml"))
        liver_inflows = [e for e in g.edges if e.target == "liver" and hasattr(e, "flow_rate")]
        total = sum(e.flow_rate.mean for e in liver_inflows)
        assert total == pytest.approx(0.255 * 390.0, rel=1e-3)

    def test_gut_wall_has_enzymes(self):
        g = build_from_yaml(Path("data/physiology/reference_man.yaml"))
        assert "CYP3A4" in g.nodes["gut_wall"].enzymes

    def test_reference_man_liver_uses_extended_clearance(self):
        """Liver clearance edge should be model=extended after ECM migration."""
        from sisyphus.graph.types import ClearanceEdge

        g = build_from_yaml(Path("data/physiology/reference_man.yaml"))
        liver_clearance = [
            e for e in g.edges
            if isinstance(e, ClearanceEdge) and e.source == "liver"
        ]
        assert len(liver_clearance) == 1
        assert liver_clearance[0].model == "extended"
        assert liver_clearance[0].bile_target == "duodenum_lumen"

    def test_clearance_bile_target_must_exist(self, tmp_path):
        path = tmp_path / "bile.yaml"
        path.write_text(
            "global_params: {cardiac_output: 1}\n"
            "nodes:\n"
            "  - {name: tissue, type: organ, volume: 1}\n"
            "  - {name: sink, type: sink, volume: 1}\n"
            "edges:\n"
            "  - {source: tissue, target: sink, type: clearance, "
            "model: extended, bile_target: missing}\n"
        )
        with pytest.raises(ValueError, match="bile target node 'missing' not found"):
            build_from_yaml(path)

    def test_reference_man_has_no_active_transport_edges(self):
        """active_transport edges removed; ECM flux replaces MM uptake."""
        from sisyphus.graph.types import ActiveTransportEdge

        g = build_from_yaml(Path("data/physiology/reference_man.yaml"))
        at_edges = [e for e in g.edges if isinstance(e, ActiveTransportEdge)]
        assert len(at_edges) == 0


class TestCompoundLoader:
    def test_load_midazolam(self):
        drug = load_compound(Path("data/compounds/midazolam.yaml"))
        assert isinstance(drug, DrugOnGraph)
        assert drug.name == "Midazolam"
        assert drug.dose_mg == 2.0
        assert drug.fup.mean == pytest.approx(0.032)
        assert "CYP3A4" in drug.enzyme_affinity

    def test_load_caffeine(self):
        drug = load_compound(Path("data/compounds/caffeine.yaml"))
        assert drug.fup.mean == pytest.approx(0.65)
        assert drug.compound_type == "neutral"
        assert "CYP1A2" in drug.enzyme_affinity

    def test_kp_overrides_are_distributions(self):
        drug = load_compound(Path("data/compounds/midazolam.yaml"))
        assert drug.kp_overrides["liver"].mean == pytest.approx(5.87)
        assert drug.kp_overrides["liver"].cv == 0.0
