"""Axial sub-compartment expansion (WS-3).

Rewrites each ``parallel_tube`` clearance organ into N serial well-stirred
sub-tanks. N serial well-stirred tanks (each CLint/N, volume/N, full flow Q)
converge to the parallel-tube extraction ``1 - exp(-fu_b·CLint/Q)``. The engine
compiles only well_stirred tanks, so it is unchanged (invariant #8). Identity-
blind: sub-tanks carry ``lookup_name`` = parent so Kp/PS resolve to the parent
organ; the engine never matches a literal name.
"""
from __future__ import annotations

import dataclasses

from sisyphus.core import Distribution
from sisyphus.graph.body import BodyGraph
from sisyphus.graph.types import ClearanceEdge, FlowEdge

# Default discretization when parallel_tube is requested without an explicit N.
# Numerical convergence parameter (<~2% from analytic E_PT at typical CLint),
# NOT tuned to Cmax loss (invariant #8).
_DEFAULT_AXIAL_N = 10

_PERFUSION_OK_EDGE_TYPES = frozenset({"flow", "clearance"})


def _scale(dists: dict[str, Distribution], n: int) -> dict[str, Distribution]:
    return {tag: Distribution(d.mean / n, cv=d.cv) for tag, d in dists.items()}


def expand_axial(graph: BodyGraph) -> BodyGraph:
    """Expand every ``parallel_tube`` clearance organ into N serial well-stirred tanks.

    Returns the *same* graph object unchanged when no ``parallel_tube`` edge
    exists (production path → bit-identical). Otherwise returns a new BodyGraph.

    Raises ``NotImplementedError`` if an organ tagged for expansion has any edge
    other than flow-in / flow-out / clearance.
    """
    organs = sorted({
        e.source for e in graph.edges
        if isinstance(e, ClearanceEdge) and e.model == "parallel_tube"
    })
    if not organs:
        return graph

    new = BodyGraph()
    new.global_params = dict(graph.global_params)

    organ_set = set(organs)
    # Carry over every node that is NOT being expanded.
    for name, node in graph.nodes.items():
        if name not in organ_set:
            new.add_node(node)

    # Pre-validate scope + create tanks for each expanded organ.
    tanks_by_organ: dict[str, list[str]] = {}
    for organ in organs:
        node = graph.nodes[organ]
        touching = [e for e in graph.edges if e.source == organ or e.target == organ]
        for e in touching:
            if e.edge_type not in _PERFUSION_OK_EDGE_TYPES:
                raise NotImplementedError(
                    f"axial expansion supports a perfusion organ with only "
                    f"flow/clearance edges; organ {organ!r} has a {e.edge_type!r} "
                    f"edge. Expand a perfusion organ, or model this differently."
                )
        n = node.axial_subcompartments if node.axial_subcompartments >= 2 else _DEFAULT_AXIAL_N
        names = [f"{organ}__ax{i}" for i in range(1, n + 1)]
        tanks_by_organ[organ] = names
        for tname in names:
            new.add_node(dataclasses.replace(
                node,
                name=tname,
                volume=Distribution(node.volume.mean / n, cv=node.volume.cv),
                enzymes=_scale(node.enzymes, n),
                transporters=_scale(node.transporters, n),
                lookup_name=node.lookup_name or organ,
                axial_subcompartments=1,
            ))

    # Rewrite edges.
    for e in graph.edges:
        if isinstance(e, ClearanceEdge) and e.source in organ_set and e.model == "parallel_tube":
            # Replicate as N well_stirred clearance edges, one per tank.
            for tname in tanks_by_organ[e.source]:
                new.add_edge(ClearanceEdge(source=tname, target=e.target, model="well_stirred"))
            continue
        if isinstance(e, FlowEdge) and e.target in organ_set:
            new.add_edge(dataclasses.replace(e, target=tanks_by_organ[e.target][0]))
            continue
        if isinstance(e, FlowEdge) and e.source in organ_set:
            new.add_edge(dataclasses.replace(e, source=tanks_by_organ[e.source][-1]))
            continue
        if e.source in organ_set or e.target in organ_set:
            continue  # any other organ-touching edge was rejected above
        new.add_edge(e)

    # Internal full-Q series edges per organ.
    for organ, names in tanks_by_organ.items():
        q_total = sum(
            e.flow_rate.mean for e in graph.edges
            if isinstance(e, FlowEdge) and e.target == organ
        )
        for a, b in zip(names, names[1:]):
            new.add_edge(FlowEdge(source=a, target=b, flow_rate=Distribution(q_total)))

    return new
