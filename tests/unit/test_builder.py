"""Tests for YAML builder and compound loader."""

from pathlib import Path

import pytest

from sisyphus.compounds import load_compound
from sisyphus.core import DrugOnGraph
from sisyphus.graph.body import BodyGraph
from sisyphus.graph.builder import build_from_yaml


class TestBuilder:
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
