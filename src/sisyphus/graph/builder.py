"""YAML -> BodyGraph constructor.

Reads physiology YAML files, constructs Node and Edge instances, and
assembles a validated BodyGraph.  Flow fractions in YAML are converted
to absolute flow rates (L/h) using ``cardiac_output``.

The builder enforces flow conservation at build time -- invalid topology
never reaches the engine.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from sisyphus.core import Distribution
from sisyphus.graph.body import BodyGraph
from sisyphus.graph.types import (
    AbsorptionEdge,
    ActiveTransportEdge,
    ClearanceEdge,
    DiffusionEdge,
    FlowEdge,
    Node,
    TissueComposition,
    TransitEdge,
)


def build_from_yaml(path: Path) -> BodyGraph:
    """Parse a physiology YAML file and return a validated BodyGraph.

    The YAML schema supports:
    - ``global_params``: graph-level parameters (cardiac_output, etc.)
    - ``nodes``: compartment definitions with volumes, composition, enzymes
    - ``edges``: transport connections with type-specific parameters

    Flow fractions are converted to absolute flow rates by multiplying
    with ``cardiac_output``.

    After construction, ``BodyGraph.validate()`` is called.  If any
    violations are found, ``ValueError`` is raised.

    Args:
        path: Path to the YAML file.

    Returns:
        A validated BodyGraph.

    Raises:
        ValueError: If the YAML is malformed or flow conservation fails.
    """
    with open(path) as f:
        data = yaml.safe_load(f)

    graph = BodyGraph()

    # --- Parse global_params ---
    raw_globals = data.get("global_params", {})
    for key, val in raw_globals.items():
        graph.global_params[key] = _parse_distribution(val)

    cardiac_output = graph.global_params["cardiac_output"].mean

    # --- Parse nodes ---
    for node_spec in data.get("nodes", []):
        node = _build_node(node_spec)
        graph.add_node(node)

    # --- Parse edges (two-pass for flow inference) ---
    raw_edges = data.get("edges", [])

    # Track arterial inflow rates by target node name for return-flow inference.
    # Key: node name that receives arterial inflow, Value: absolute flow rate.
    arterial_inflow: dict[str, float] = {}

    # First pass: process all edges with explicit flow parameters.
    deferred_flow_edges: list[dict[str, Any]] = []

    for edge_spec in raw_edges:
        edge_type = edge_spec["type"]

        if edge_type == "flow":
            if "flow_fraction" in edge_spec:
                # Explicit flow_fraction -> convert to absolute rate.
                flow_rate = edge_spec["flow_fraction"] * cardiac_output
                edge = FlowEdge(
                    source=edge_spec["source"],
                    target=edge_spec["target"],
                    flow_rate=Distribution(flow_rate),
                )
                graph.add_edge(edge)

                # Track arterial inflow for return-flow inference.
                if edge_spec["source"] == "arterial_blood":
                    arterial_inflow[edge_spec["target"]] = flow_rate
            else:
                # Defer: needs return-flow inference.
                deferred_flow_edges.append(edge_spec)

        elif edge_type == "diffusion":
            edge = DiffusionEdge(
                source=edge_spec["source"],
                target=edge_spec["target"],
                ps_product=_parse_distribution(edge_spec.get("ps_product", 0.0)),
            )
            graph.add_edge(edge)

        elif edge_type == "transit":
            edge = TransitEdge(
                source=edge_spec["source"],
                target=edge_spec["target"],
                transit_rate=_parse_distribution(edge_spec.get("transit_rate", 0.0)),
            )
            graph.add_edge(edge)

        elif edge_type == "absorption":
            edge = AbsorptionEdge(
                source=edge_spec["source"],
                target=edge_spec["target"],
                ka_fraction=_parse_distribution(edge_spec.get("ka_fraction", 1.0)),
            )
            graph.add_edge(edge)

        elif edge_type == "clearance":
            edge = ClearanceEdge(
                source=edge_spec["source"],
                target=edge_spec["target"],
                model=edge_spec.get("model", "well_stirred"),
            )
            graph.add_edge(edge)

        elif edge_type == "active_transport":
            edge = ActiveTransportEdge(
                source=edge_spec["source"],
                target=edge_spec["target"],
            )
            graph.add_edge(edge)

        else:
            raise ValueError(f"Unknown edge type: {edge_type!r}")

    # Second pass: infer flow rates for deferred flow edges.
    # For return flows (organ -> venous_blood or organ -> portal_vein),
    # use the arterial inflow to that organ.
    for edge_spec in deferred_flow_edges:
        source = edge_spec["source"]

        # Look up the arterial inflow to the source node.
        if source in arterial_inflow:
            flow_rate = arterial_inflow[source]
        else:
            raise ValueError(
                f"Cannot infer flow rate for edge {source!r} -> {edge_spec['target']!r}: "
                f"no arterial inflow found for node {source!r}"
            )

        edge = FlowEdge(
            source=source,
            target=edge_spec["target"],
            flow_rate=Distribution(flow_rate),
        )
        graph.add_edge(edge)

    # --- Validate ---
    errors = graph.validate()
    if errors:
        raise ValueError(f"Graph validation failed: {errors}")

    return graph


def merge_overlay(base: BodyGraph, overlay_path: Path) -> BodyGraph:
    """Merge an overlay YAML into a base BodyGraph.

    Overlays can add nodes, add edges, and override existing edge
    parameters (e.g. adjusting flow fractions when adding a new organ).

    Args:
        base: The base BodyGraph to extend.
        overlay_path: Path to the overlay YAML.

    Returns:
        A new validated BodyGraph with the overlay applied.

    Raises:
        ValueError: If the merged graph fails validation.
    """
    raise NotImplementedError


def _build_node(spec: dict[str, Any]) -> Node:
    """Construct a Node from a YAML node specification."""
    # Volume: can be a bare float or a distribution dict.
    volume = _parse_distribution(spec["volume"])

    # Composition: optional.
    composition = None
    if "composition" in spec:
        c = spec["composition"]
        composition = TissueComposition(
            fn=c["fn"],
            fp=c["fp"],
            fw=c["fw"],
            pH=c["pH"],
        )

    # Enzymes: optional dict of enzyme tag -> abundance.
    enzymes: dict[str, Distribution] = {}
    for tag, val in spec.get("enzymes", {}).items():
        enzymes[tag] = _parse_distribution(val)

    # Transporters: optional dict of transporter tag -> abundance.
    # Mirrors the enzymes parser. Supports bare scalars and {mean, cv} dicts.
    transporters: dict[str, Distribution] = {}
    for tag, val in spec.get("transporters", {}).items():
        transporters[tag] = _parse_distribution(val)

    # IVIVE scaling: optional float, default 0.0.
    ivive_scaling = float(spec.get("ivive_scaling", 0.0))

    # lookup_name: for Kp/PS parameter resolution. Set explicitly in YAML
    # or defaults to the node name itself. This keeps the engine identity-blind —
    # no string manipulation needed to resolve "adipose_tissue" → "adipose".
    lookup_name = spec.get("lookup_name", spec["name"])

    return Node(
        name=spec["name"],
        node_type=spec["type"],
        volume=volume,
        composition=composition,
        enzymes=enzymes,
        transporters=transporters,
        ivive_scaling=ivive_scaling,
        lookup_name=lookup_name,
    )


def _parse_distribution(raw: Any) -> Distribution:
    """Convert a YAML distribution spec to a Distribution instance.

    Supports formats:
    - ``{mean: 390, cv: 0.10}`` -> ``Distribution(mean=390, cv=0.10)``
    - ``{mean: 390}`` -> ``Distribution(mean=390, cv=0.0)``
    - ``{mean: 390, cv: 0.10, correlation_group: "liver_achour2021"}``
        -> ``Distribution(mean=390, cv=0.10, correlation_group="liver_achour2021")``
    - bare float/int -> ``Distribution(mean=value, cv=0.0)``
    """
    if isinstance(raw, dict):
        mean = float(raw["mean"])
        cv = float(raw.get("cv", 0.0))
        correlation_group = raw.get("correlation_group")
        if correlation_group is not None:
            correlation_group = str(correlation_group)
        return Distribution(mean=mean, cv=cv, correlation_group=correlation_group)
    # bare numeric value
    return Distribution(mean=float(raw), cv=0.0)
