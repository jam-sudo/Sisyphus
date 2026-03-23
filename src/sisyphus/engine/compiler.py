"""ODE Compiler — graph topology → ODE skeleton.

Compile once, parameterize many: the compiler analyses graph topology
to produce a ``CompiledODE`` skeleton.  This skeleton is reused across
MC samples — only the parameter values change, not the structure.

Imports from ``sisyphus.graph`` and ``sisyphus.core`` only.
The compiler is **identity-blind**: it never inspects node names,
enzyme names, or drug names.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from sisyphus.engine.flux import FLUX_REGISTRY
from sisyphus.graph.body import BodyGraph
from sisyphus.graph.types import (
    AbsorptionEdge,
    ClearanceEdge,
    DiffusionEdge,
    FlowEdge,
    TransitEdge,
)

if TYPE_CHECKING:
    from sisyphus.core import DrugOnGraph


# ---------------------------------------------------------------------------
# ResolvedParams — point-value parameter set injected into RHS
# ---------------------------------------------------------------------------


class ResolvedParams:
    """Resolved (sampled) parameters for a single ODE evaluation.

    Created from a realized BodyGraph + realized DrugOnGraph where all
    Distributions have been sampled to point values.  Provides fast
    lookup methods used by FluxSpec implementations.
    """

    def __init__(self, graph: BodyGraph, drug: DrugOnGraph) -> None:
        # Pre-compute lookups for fast access during RHS evaluation
        self._volumes = {name: node.volume.mean for name, node in graph.nodes.items()}
        self._enzymes = {
            name: {tag: d.mean for tag, d in node.enzymes.items()}
            for name, node in graph.nodes.items()
        }
        self._ivive_scaling = {name: node.ivive_scaling for name, node in graph.nodes.items()}
        self._drug = drug
        self._kp = self._build_kp_map(graph, drug)
        self._inflows = self._compute_inflows(graph)
        self._edge_params = self._build_edge_params(graph)
        self._global_params = {k: d.mean for k, d in graph.global_params.items()}

    def node_param(self, node_name: str, param: str) -> float:
        """Look up a scalar node parameter (e.g. ``"volume"``)."""
        if param == "volume":
            return self._volumes[node_name]
        if param == "ivive_scaling":
            return self._ivive_scaling.get(node_name, 0.0)
        raise KeyError(f"Unknown node param: {param}")

    def edge_param(self, edge_id: int, param: str) -> float:
        """Look up a scalar edge parameter (e.g. ``"flow_rate"``)."""
        return self._edge_params[edge_id][param]

    def node_enzymes(self, node_name: str) -> dict[str, float]:
        """Return ``{enzyme_tag: abundance}`` for a node."""
        return self._enzymes.get(node_name, {})

    def drug_param(self, param: str) -> float:
        """Look up a scalar drug parameter (e.g. ``"fup"``, ``"rbp"``)."""
        if param == "fup":
            return self._drug.fup.mean
        if param == "rbp":
            return self._drug.rbp.mean
        if param == "renal_clearance":
            return self._drug.renal_clearance.mean
        if param == "dose_mg":
            return self._drug.dose_mg
        if param == "peff":
            return self._drug.peff.mean
        if param == "particle_radius_um":
            return self._drug.particle_radius_um
        raise KeyError(f"Unknown drug param: {param}")

    def drug_kp(self, node_name: str) -> float:
        """Return the drug's tissue:plasma partition coefficient for a node."""
        return self._kp.get(node_name, 1.0)

    def drug_enzyme_affinity(self, tag: str) -> float:
        """Return the drug's intrinsic clearance per unit enzyme for *tag*."""
        if tag in self._drug.enzyme_affinity:
            return self._drug.enzyme_affinity[tag].mean
        return 0.0

    def drug_ps(self, node_name: str) -> float:
        """Return drug-specific PS for a node, or 0.0 if not overridden."""
        if node_name in self._drug.ps_overrides:
            return self._drug.ps_overrides[node_name].mean
        return 0.0

    def total_inflow(self, node_name: str) -> float:
        """Sum of all flow-type inflows to a node (L/h)."""
        return self._inflows.get(node_name, 0.0)

    def global_param(self, param: str) -> float:
        """Look up a global graph parameter (e.g. ``"cardiac_output"``)."""
        return self._global_params.get(param, 0.0)

    def _build_kp_map(self, graph: BodyGraph, drug: DrugOnGraph) -> dict[str, float]:
        """Build node_name -> Kp mapping."""
        kp: dict[str, float] = {}
        for name, node in graph.nodes.items():
            if node.node_type in ("blood_pool", "lumen", "sink"):
                kp[name] = 1.0
            elif name in drug.kp_overrides:
                kp[name] = drug.kp_overrides[name].mean
            else:
                # Check if this is a permeability-limited tissue node
                # e.g., "adipose_tissue" — check if base name "adipose" is in kp_overrides
                base = name.replace("_tissue", "")
                if base in drug.kp_overrides:
                    kp[name] = drug.kp_overrides[base].mean
                else:
                    kp[name] = 1.0  # default
        return kp

    def _compute_inflows(self, graph: BodyGraph) -> dict[str, float]:
        """Sum flow-type inflows per node."""
        inflows: dict[str, float] = {name: 0.0 for name in graph.nodes}
        for edge in graph.edges:
            if isinstance(edge, FlowEdge):
                inflows[edge.target] += edge.flow_rate.mean
        return inflows

    def _build_edge_params(self, graph: BodyGraph) -> dict[int, dict[str, float]]:
        """Build edge_id -> {param: value} mapping."""
        edge_params: dict[int, dict[str, float]] = {}
        for i, edge in enumerate(graph.edges):
            params: dict[str, float] = {}
            if isinstance(edge, FlowEdge):
                params["flow_rate"] = edge.flow_rate.mean
            elif isinstance(edge, DiffusionEdge):
                params["ps_product"] = edge.ps_product.mean
            elif isinstance(edge, TransitEdge):
                params["transit_rate"] = edge.transit_rate.mean
            elif isinstance(edge, AbsorptionEdge):
                params["ka_fraction"] = edge.ka_fraction.mean
            elif isinstance(edge, ClearanceEdge):
                params["model"] = edge.model  # type: ignore[assignment]
            edge_params[i] = params
        return edge_params


# ---------------------------------------------------------------------------
# CompiledODE — reusable ODE skeleton
# ---------------------------------------------------------------------------


class CompiledODE:
    """ODE skeleton compiled from graph topology.

    The skeleton fixes state indexing and edge -> flux mappings.
    ``make_rhs`` injects resolved parameters to produce a callable
    RHS function for the ODE solver.

    Attributes:
        n_states: Number of ODE state variables.
        state_index: Mapping of node name -> state vector index.
        flux_specs: List of FluxSpec instances, one per edge.
    """

    def __init__(self) -> None:
        self.n_states: int = 0
        self.state_index: dict[str, int] = {}
        self.flux_specs: list = []

    def make_rhs(self, params: ResolvedParams) -> Callable[[float, np.ndarray], np.ndarray]:
        """Bind resolved parameters to produce the ODE right-hand side.

        Args:
            params: Point-value parameters for this MC sample.

        Returns:
            A function ``rhs(t, y) -> dydt`` suitable for SciPy's
            ``solve_ivp``.
        """
        specs = self.flux_specs  # local ref for speed
        n = self.n_states

        def rhs(t: float, y: np.ndarray) -> np.ndarray:
            dydt = np.zeros(n)
            for spec in specs:
                spec.apply(t, y, dydt, params)
            return dydt

        return rhs


# ---------------------------------------------------------------------------
# ODECompiler — BodyGraph -> CompiledODE
# ---------------------------------------------------------------------------


class ODECompiler:
    """Compiles a BodyGraph into a reusable ODE skeleton.

    The compiler walks the graph once, assigns state indices, and maps
    each edge to its corresponding FluxSpec via the flux registry.
    The result is a ``CompiledODE`` that can be parameterized repeatedly
    for MC sampling without re-traversing the graph.
    """

    def compile(self, graph: BodyGraph) -> CompiledODE:
        """Analyse graph topology and produce an ODE skeleton.

        This step is independent of drug parameters and distribution
        samples.  It fixes:
        - State variable indexing (one per node).
        - Edge -> FluxSpec mapping (via ``FLUX_REGISTRY``).

        Args:
            graph: A validated BodyGraph.

        Returns:
            A ``CompiledODE`` ready for ``make_rhs()`` calls.
        """
        compiled = CompiledODE()

        # Assign state indices — one per node, sorted alphabetically by name.
        for i, name in enumerate(sorted(graph.nodes.keys())):
            compiled.state_index[name] = i
        compiled.n_states = len(graph.nodes)

        # Map each edge to its FluxSpec via the flux registry.
        for edge_id, edge in enumerate(graph.edges):
            if edge.edge_type not in FLUX_REGISTRY:
                raise ValueError(
                    f"No FluxSpec registered for edge type: {edge.edge_type!r}"
                )
            flux_cls = FLUX_REGISTRY[edge.edge_type]
            spec = flux_cls.from_edge(edge_id, edge, compiled.state_index)
            compiled.flux_specs.append(spec)

        return compiled
