"""BodyGraph — the typed directed multi-graph representing the human body.

A BodyGraph is built from YAML (via ``builder.py``) or programmatically,
then fed to the engine's ODE compiler.  The engine reads topology and
parameters from the graph but never inspects node/edge identities.

Imports only from ``sisyphus.core`` and ``sisyphus.graph.types``.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from sisyphus.core import Distribution
from sisyphus.graph.types import (
    AbsorptionEdge,
    DiffusionEdge,
    Edge,
    FlowEdge,
    Node,
    OneCompartmentEliminationEdge,
    ProdrugActivationEdge,
    TransitEdge,
)

# Node types that are exempt from flow conservation checks.
_FLOW_EXEMPT_NODE_TYPES = frozenset({"sink", "lumen"})

# Relative tolerance for flow conservation: |in - out| / max(in, out) < tol
_FLOW_TOLERANCE = 1e-6


class BodyGraph:
    """Typed directed multi-graph of the human body.

    Nodes are compartments (organs, blood pools, lumens, sinks).
    Edges are transport pathways (flow, diffusion, transit, absorption,
    clearance).  Multiple edge types may connect the same pair of nodes.

    Attributes:
        nodes: Mapping of node name → Node.
        edges: Ordered list of directed edges.
        global_params: Graph-level parameters (e.g. cardiac_output)
            stored as Distributions.
    """

    def __init__(self) -> None:
        self.nodes: dict[str, Node] = {}
        self.edges: list[Edge] = []
        self.global_params: dict[str, Distribution] = {}

    # -- Mutation ----------------------------------------------------------

    def add_node(self, node: Node) -> None:
        """Register a node.  Raises ``ValueError`` on duplicate name."""
        if node.name in self.nodes:
            raise ValueError(f"duplicate node name: {node.name!r}")
        self.nodes[node.name] = node

    def add_edge(self, edge: Edge) -> None:
        """Add a directed edge.  Source and target must already exist."""
        if edge.source not in self.nodes:
            raise ValueError(f"source node {edge.source!r} not found in graph")
        if edge.target not in self.nodes:
            raise ValueError(f"target node {edge.target!r} not found in graph")
        self.edges.append(edge)

    def remove_node(self, name: str) -> None:
        """Remove a node and all edges that reference it."""
        if name not in self.nodes:
            raise ValueError(f"node {name!r} not found in graph")
        del self.nodes[name]
        self.edges = [e for e in self.edges if e.source != name and e.target != name]

    # -- Validation --------------------------------------------------------

    def validate(self) -> list[str]:
        """Check physical constraints and return a list of violations.

        Checks include:
        - All edge endpoints reference existing nodes.
        - Flow conservation: for FlowEdge edges, total inflow ≈ total outflow
          per node (within tolerance), skipping sink and lumen node types.

        Returns an empty list if the graph is valid.
        """
        errors: list[str] = []

        # Check that all edge endpoints reference existing nodes.
        for edge in self.edges:
            if edge.source not in self.nodes:
                errors.append(f"edge {edge.source!r} → {edge.target!r}: source node not found")
            if edge.target not in self.nodes:
                errors.append(f"edge {edge.source!r} → {edge.target!r}: target node not found")

        # Flow conservation: sum of FlowEdge inflows ≈ sum of FlowEdge outflows per node.
        inflow: dict[str, float] = {name: 0.0 for name in self.nodes}
        outflow: dict[str, float] = {name: 0.0 for name in self.nodes}

        for edge in self.edges:
            if isinstance(edge, FlowEdge):
                rate = edge.flow_rate.mean
                if edge.source in outflow:
                    outflow[edge.source] += rate
                if edge.target in inflow:
                    inflow[edge.target] += rate

        for name, node in self.nodes.items():
            if node.node_type in _FLOW_EXEMPT_NODE_TYPES:
                continue
            total_in = inflow[name]
            total_out = outflow[name]
            if total_in == 0.0 and total_out == 0.0:
                # No flow edges touch this node — not a flow conservation error.
                continue
            denom = max(total_in, total_out)
            if denom > 0.0 and abs(total_in - total_out) / denom > _FLOW_TOLERANCE:
                errors.append(
                    f"flow imbalance at node {name!r}: "
                    f"inflow={total_in:.6g}, outflow={total_out:.6g}"
                )

        return errors

    # -- Sampling ----------------------------------------------------------

    def sample(self, rng: np.random.Generator) -> BodyGraph:
        """Sample all Distributions to produce a realized (point-value) copy.

        Returns a new ``BodyGraph`` where every Distribution is replaced
        by ``Distribution(mean=sampled_value, cv=0.0)``.  Used by the
        MC uncertainty engine.
        """
        g2 = BodyGraph()

        # Sample nodes.
        for name, node in self.nodes.items():
            sampled_volume = Distribution(node.volume.sample(rng), cv=0.0)
            sampled_enzymes = {
                tag: Distribution(dist.sample(rng), cv=0.0) for tag, dist in node.enzymes.items()
            }
            sampled_transporters = {
                tag: Distribution(dist.sample(rng), cv=0.0)
                for tag, dist in node.transporters.items()
            }
            new_node = dataclasses.replace(
                node,
                volume=sampled_volume,
                enzymes=sampled_enzymes,
                transporters=sampled_transporters,
            )
            g2.nodes[name] = new_node

        # Sample edges.
        for edge in self.edges:
            if isinstance(edge, FlowEdge):
                new_edge = dataclasses.replace(
                    edge,
                    flow_rate=Distribution(edge.flow_rate.sample(rng), cv=0.0),
                )
            elif isinstance(edge, DiffusionEdge):
                new_edge = dataclasses.replace(
                    edge,
                    ps_product=Distribution(edge.ps_product.sample(rng), cv=0.0),
                )
            elif isinstance(edge, TransitEdge):
                new_edge = dataclasses.replace(
                    edge,
                    transit_rate=Distribution(edge.transit_rate.sample(rng), cv=0.0),
                )
            elif isinstance(edge, AbsorptionEdge):
                new_edge = dataclasses.replace(
                    edge,
                    ka_fraction=Distribution(edge.ka_fraction.sample(rng), cv=0.0),
                )
            elif isinstance(edge, ProdrugActivationEdge):
                # v2: enzyme_tags is a frozenset (compile-time constant, no resampling);
                # conversion_yield is the only Distribution to resample.
                new_edge = dataclasses.replace(
                    edge,
                    conversion_yield=Distribution(edge.conversion_yield.sample(rng), cv=0.0),
                )
            elif isinstance(edge, OneCompartmentEliminationEdge):
                new_edge = dataclasses.replace(
                    edge,
                    cl_per_h=Distribution(edge.cl_per_h.sample(rng), cv=0.0),
                    vd_l=Distribution(edge.vd_l.sample(rng), cv=0.0),
                )
            else:
                # ClearanceEdge and unknown edge types have no Distribution fields.
                new_edge = edge
            g2.edges.append(new_edge)

        # Sample global_params.
        g2.global_params = {
            key: Distribution(dist.sample(rng), cv=0.0) for key, dist in self.global_params.items()
        }

        return g2

    def realize_means(self) -> BodyGraph:
        """Deterministic realization using means (RNG-independent).

        Returns a new ``BodyGraph`` where every Distribution is replaced
        by ``Distribution(mean=mean, cv=0.0)`` — using the existing mean
        directly, NOT a stochastic sample. Used by the deterministic
        prediction path (``n_mc_samples=0``) to eliminate seed-dependent
        RNG-order coupling.

        Mirrors :meth:`sample` but skips RNG entirely: adding a new
        Distribution to physiology YAML cannot shift downstream realized
        values for non-related drugs.
        """
        g2 = BodyGraph()

        # Realize nodes.
        for name, node in self.nodes.items():
            mean_volume = Distribution(node.volume.mean, cv=0.0)
            mean_enzymes = {
                tag: Distribution(dist.mean, cv=0.0) for tag, dist in node.enzymes.items()
            }
            mean_transporters = {
                tag: Distribution(dist.mean, cv=0.0)
                for tag, dist in node.transporters.items()
            }
            new_node = dataclasses.replace(
                node,
                volume=mean_volume,
                enzymes=mean_enzymes,
                transporters=mean_transporters,
            )
            g2.nodes[name] = new_node

        # Realize edges.
        for edge in self.edges:
            if isinstance(edge, FlowEdge):
                new_edge = dataclasses.replace(
                    edge,
                    flow_rate=Distribution(edge.flow_rate.mean, cv=0.0),
                )
            elif isinstance(edge, DiffusionEdge):
                new_edge = dataclasses.replace(
                    edge,
                    ps_product=Distribution(edge.ps_product.mean, cv=0.0),
                )
            elif isinstance(edge, TransitEdge):
                new_edge = dataclasses.replace(
                    edge,
                    transit_rate=Distribution(edge.transit_rate.mean, cv=0.0),
                )
            elif isinstance(edge, AbsorptionEdge):
                new_edge = dataclasses.replace(
                    edge,
                    ka_fraction=Distribution(edge.ka_fraction.mean, cv=0.0),
                )
            elif isinstance(edge, ProdrugActivationEdge):
                new_edge = dataclasses.replace(
                    edge,
                    conversion_yield=Distribution(edge.conversion_yield.mean, cv=0.0),
                )
            elif isinstance(edge, OneCompartmentEliminationEdge):
                new_edge = dataclasses.replace(
                    edge,
                    cl_per_h=Distribution(edge.cl_per_h.mean, cv=0.0),
                    vd_l=Distribution(edge.vd_l.mean, cv=0.0),
                )
            else:
                new_edge = edge
            g2.edges.append(new_edge)

        # Realize global_params.
        g2.global_params = {
            key: Distribution(dist.mean, cv=0.0) for key, dist in self.global_params.items()
        }

        return g2

    # -- Introspection -----------------------------------------------------

    def node_names(self) -> list[str]:
        """Return sorted list of node names."""
        return sorted(self.nodes.keys())

    def edge_count(self) -> int:
        """Total number of edges."""
        return len(self.edges)

    def __repr__(self) -> str:
        return f"BodyGraph(nodes={len(self.nodes)}, edges={len(self.edges)})"
