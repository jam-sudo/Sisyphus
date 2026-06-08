"""Graph element types — nodes, edges, and tissue composition.

These are the building blocks of a BodyGraph.  The engine operates on
these types by *type* (node_type, edge_type), never by *identity*
(node name, enzyme name).

Imports only from ``sisyphus.core`` — no cross-layer dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sisyphus.core import Distribution, TissueComposition

# Re-export TissueComposition for backward compatibility (defined in core.py).
__all__ = ["TissueComposition"]


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
    # B-11: 1.0 if hepatic intracellular fu correction applies at this node.
    fu_correction_applicable: float = 0.0


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
        - ``"parallel_tube"`` — parallel-tube extraction
        - ``"gfr_filtration"`` — glomerular filtration (renal)
        - ``"extended"`` — Extended Clearance Model (ECM): active + passive
          uptake, passive efflux, metabolism, biliary clearance via QSSA.
    """

    edge_type: str = field(default="clearance", init=False)
    model: str = "well_stirred"


@dataclass(frozen=True)
class ActiveTransportEdge(Edge):
    """Active transporter-mediated mass transfer (efflux or uptake).

    Full Michaelis-Menten: rate = abundance × Jmax × C / (Km + C).
    Direction is determined by edge source → target:
    - Gut efflux: gut_wall → lumen (P-gp: tissue → lumen)
    - Hepatic uptake: liver_blood → liver (OATP: blood → tissue)

    Identity-blind: engine matches node.transporters[tag] with
    drug.transporter_kinetics[tag], never inspecting tag names.
    """

    edge_type: str = field(default="active_transport", init=False)
    # WS-5: "uptake" → transporter at TARGET (e.g. hepatic OATP, blood→liver);
    # "efflux" → transporter at SOURCE (e.g. gut P-gp, gut_wall→lumen). Driving
    # (substrate) concentration is the SOURCE in both cases.
    direction: str = "uptake"


@dataclass(frozen=True)
class ProdrugActivationEdge(Edge):
    """Mass transfer: parent drug → active metabolite via enzyme catalysis.

    v2 (2026-04-27): conversion is well-stirred extraction at flow-through
    nodes (replaces v1's kinetic 1st-order). Drug declares which enzymes
    catalyze the conversion via ``enzyme_tags``; engine computes CLint from
    node enzyme abundance × drug.enzyme_affinity_for_conversion[tag].

    Mass routing: source loses parent (mg); target gains active (mg)
    scaled by mw_active/mw_parent × conversion_yield.

    Identity-blind: engine matches by edge_type and tag strings only.
    """

    edge_type: str = field(default="prodrug_activation", init=False)
    enzyme_tags: frozenset[str] = field(default_factory=frozenset)
    conversion_yield: Distribution = field(default_factory=lambda: Distribution(1.0))
    mw_parent: float = 0.0
    mw_active: float = 0.0


@dataclass(frozen=True)
class OneCompartmentEliminationEdge(Edge):
    """Aggregate 1st-order elimination from a 1-compartment plasma node.

    Used for active metabolite clearance where literature reports total
    plasma CL (not enzyme-level decomposition). Rate = (CL/Vd) × A_source.
    Mass accumulates at target sink node for mass-balance audit.
    """

    edge_type: str = field(default="one_compartment_elimination", init=False)
    cl_per_h: Distribution = field(default_factory=lambda: Distribution(0.0))
    vd_l: Distribution = field(default_factory=lambda: Distribution(1.0))
