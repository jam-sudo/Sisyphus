"""Graph element types — nodes, edges, and tissue composition.

These are the building blocks of a BodyGraph.  The engine operates on
these types by *type* (node_type, edge_type), never by *identity*
(node name, enzyme name).

Imports only from ``sisyphus.core`` — no cross-layer dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sisyphus.core import Distribution

# ---------------------------------------------------------------------------
# Tissue composition (for Kp estimation)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TissueComposition:
    """Fractional tissue composition used for tissue:plasma partition
    coefficient (Kp) estimation via Rodgers & Rowland or Berezhkovskiy.

    All fractions are dimensionless (volume fraction of wet tissue weight).

    Note: kept as bare floats (not Distribution) because Kp sensitivity
    to tissue composition fractions is low relative to fup/CLint
    uncertainty.  This is a conscious exception to Invariant 2.
    """

    fn: float  # neutral lipid fraction
    fp: float  # phospholipid fraction
    fw: float  # water fraction
    pH: float  # intracellular pH


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Node:
    """A compartment in the body graph.

    Frozen after construction — do not attempt to mutate.

    Attributes:
        name: Unique identifier (e.g. ``"liver"``, ``"brain"``).
            The engine never inspects this string — it is used only for
            graph topology and result labelling.
        node_type: Structural role.  One of ``"organ"``,
            ``"barrier_organ"``, ``"blood_pool"``, ``"lumen"``, ``"sink"``.
        volume: Compartment volume (L) as a Distribution.
        composition: Tissue fractions for Kp calculation.  ``None`` for
            non-tissue nodes (blood pools, lumens, sinks).
        enzymes: Mapping of enzyme tag → abundance Distribution.
            Tags are arbitrary strings matched against
            ``DrugOnGraph.enzyme_affinity``.
        transporters: Mapping of transporter tag → abundance Distribution.
        ivive_scaling: Scaling factor for enzyme-mediated clearance.
            Converts abundance × affinity (µL/min) to CLint (L/h).
            Typically MPPGL × organ_weight × 60 / 1e6 for enzyme-bearing
            nodes, 0.0 for nodes without enzymes.
    """

    name: str
    # "organ" | "barrier_organ" | "blood_pool" | "lumen" | "sink"
    node_type: str
    volume: Distribution
    composition: TissueComposition | None = None
    enzymes: dict[str, Distribution] = field(default_factory=dict)
    transporters: dict[str, Distribution] = field(default_factory=dict)
    ivive_scaling: float = 0.0
    lookup_name: str = ""  # base organ name for Kp/PS lookup (e.g. "adipose" for "adipose_tissue")


# ---------------------------------------------------------------------------
# Edge hierarchy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Edge:
    """Base class for all directed edges in the body graph.

    Frozen after construction — do not attempt to mutate.
    Each edge connects a source node to a target node and carries a
    type tag that the engine uses to dispatch the correct flux function.
    """

    source: str
    target: str
    edge_type: str


@dataclass(frozen=True)
class FlowEdge(Edge):
    """Convective transport (blood flow, lymphatic flow).

    Flux: ``dA_target/dt += Q × C_source − Q × C_out``
    """

    edge_type: str = field(default="flow", init=False)
    flow_rate: Distribution = field(default_factory=lambda: Distribution(0.0))


@dataclass(frozen=True)
class DiffusionEdge(Edge):
    """Permeability–surface-area limited transfer.

    Flux proportional to PS product × concentration gradient.
    """

    edge_type: str = field(default="diffusion", init=False)
    ps_product: Distribution = field(default_factory=lambda: Distribution(0.0))


@dataclass(frozen=True)
class TransitEdge(Edge):
    """First-order transit between GI lumen segments.

    Flux: ``rate = k_transit × A_source``
    """

    edge_type: str = field(default="transit", init=False)
    transit_rate: Distribution = field(default_factory=lambda: Distribution(0.0))


@dataclass(frozen=True)
class AbsorptionEdge(Edge):
    """Drug absorption from lumen to tissue.

    Absorption rate depends on drug properties (Peff, solubility)
    resolved at simulation time.  ``ka_fraction`` is the segment-specific
    absorption scaling factor (0.0 = no absorption, 1.0 = full).
    """

    edge_type: str = field(default="absorption", init=False)
    ka_fraction: Distribution = field(default_factory=lambda: Distribution(1.0))


@dataclass(frozen=True)
class ClearanceEdge(Edge):
    """Elimination edge.

    Rate is computed from the node's enzyme abundances, the drug's
    enzyme affinities, and the clearance model.

    ``model`` selects the mathematical formulation:
        - ``"well_stirred"`` — hepatic well-stirred model
        - ``"gfr_filtration"`` — glomerular filtration (renal)
    """

    edge_type: str = field(default="clearance", init=False)
    model: str = "well_stirred"
