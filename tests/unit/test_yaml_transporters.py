"""Unit tests for YAML parsing of transporters and active_transport edges."""

from __future__ import annotations

from pathlib import Path

import pytest

from sisyphus.graph.builder import build_from_yaml
from sisyphus.graph.types import ActiveTransportEdge


def _write_yaml(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "test.yaml"
    p.write_text(body)
    return p


def test_node_transporters_parsed(tmp_path: Path):
    yaml_body = """
global_params:
  cardiac_output: 390.0
nodes:
  - name: venous_blood
    type: blood_pool
    volume: 4.0
  - name: arterial_blood
    type: blood_pool
    volume: 1.5
  - name: liver
    type: organ
    volume: 1.8
    composition: {fn: 0.0348, fp: 0.0252, fw: 0.751, pH: 7.0}
    transporters:
      OATP1B1: 1.8e11
      OATP1B3: {mean: 5.0e10, cv: 0.2}
    ivive_scaling: 6.0e-5
edges:
  - {source: arterial_blood, target: liver, type: flow, flow_fraction: 1.0}
  - {source: liver, target: venous_blood, type: flow}
  - {source: venous_blood, target: arterial_blood, type: flow, flow_fraction: 1.0}
"""
    graph = build_from_yaml(_write_yaml(tmp_path, yaml_body))
    liver = graph.nodes["liver"]

    assert "OATP1B1" in liver.transporters
    assert liver.transporters["OATP1B1"].mean == pytest.approx(1.8e11)
    assert liver.transporters["OATP1B1"].cv == 0.0
    assert "OATP1B3" in liver.transporters
    assert liver.transporters["OATP1B3"].mean == pytest.approx(5.0e10)
    assert liver.transporters["OATP1B3"].cv == pytest.approx(0.2)


def test_node_without_transporters_defaults_empty(tmp_path: Path):
    yaml_body = """
global_params:
  cardiac_output: 390.0
nodes:
  - name: venous_blood
    type: blood_pool
    volume: 4.0
  - name: arterial_blood
    type: blood_pool
    volume: 1.5
  - name: brain
    type: organ
    volume: 1.45
    composition: {fn: 0.0391, fp: 0.0550, fw: 0.620, pH: 7.0}
edges:
  - {source: arterial_blood, target: brain, type: flow, flow_fraction: 1.0}
  - {source: brain, target: venous_blood, type: flow}
  - {source: venous_blood, target: arterial_blood, type: flow, flow_fraction: 1.0}
"""
    graph = build_from_yaml(_write_yaml(tmp_path, yaml_body))
    assert graph.nodes["brain"].transporters == {}


def test_active_transport_edge_parsed(tmp_path: Path):
    yaml_body = """
global_params:
  cardiac_output: 390.0
nodes:
  - name: arterial_blood
    type: blood_pool
    volume: 1.5
  - name: venous_blood
    type: blood_pool
    volume: 4.0
  - name: portal_vein
    type: blood_pool
    volume: 0.05
  - name: liver
    type: organ
    volume: 1.8
    composition: {fn: 0.0348, fp: 0.0252, fw: 0.751, pH: 7.0}
    transporters:
      OATP1B1: 1.8e11
    ivive_scaling: 6.0e-5
edges:
  - {source: arterial_blood, target: liver, type: flow, flow_fraction: 0.5}
  - {source: arterial_blood, target: portal_vein, type: flow, flow_fraction: 0.5}
  - {source: portal_vein, target: liver, type: flow}
  - {source: liver, target: venous_blood, type: flow}
  - {source: venous_blood, target: arterial_blood, type: flow, flow_fraction: 1.0}
  - {source: portal_vein, target: liver, type: active_transport}
  - {source: arterial_blood, target: liver, type: active_transport}
"""
    graph = build_from_yaml(_write_yaml(tmp_path, yaml_body))

    transport_edges = [e for e in graph.edges if isinstance(e, ActiveTransportEdge)]
    assert len(transport_edges) == 2
    srcs = {e.source for e in transport_edges}
    tgts = {e.target for e in transport_edges}
    assert srcs == {"portal_vein", "arterial_blood"}
    assert tgts == {"liver"}
