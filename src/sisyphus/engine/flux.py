"""Flux registry and base FluxSpec.

Each edge type in the body graph dispatches to a FluxSpec subclass that
knows how to compute the mass transfer rate for that transport mechanism.

The registry pattern: edge_type string → FluxSpec class.  The ODE compiler
looks up the registry to map edges to flux functions at compile time.

**Identity-blind:** FluxSpec implementations operate on indices and
parameter lookups, never on node/enzyme/drug names.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sisyphus.engine.compiler import ResolvedParams


# ---------------------------------------------------------------------------
# Flux registry
# ---------------------------------------------------------------------------

FLUX_REGISTRY: dict[str, type[FluxSpec]] = {}
"""Global mapping of edge_type → FluxSpec subclass.

Populated by the ``@register_flux`` decorator.  The ODE compiler
queries this to resolve edge types at compile time.
"""


def register_flux(edge_type: str):
    """Decorator that registers a FluxSpec subclass for an edge type.

    Usage::

        @register_flux("flow")
        class FlowFluxSpec(FluxSpec):
            ...
    """

    def decorator(cls: type[FluxSpec]) -> type[FluxSpec]:
        FLUX_REGISTRY[edge_type] = cls
        return cls

    return decorator


# ---------------------------------------------------------------------------
# FluxSpec — abstract base
# ---------------------------------------------------------------------------


class FluxSpec(ABC):
    """Abstract base for flux computations.

    A FluxSpec is instantiated once per edge during ODE compilation.
    It stores the source/target state indices and any static edge
    metadata.  At evaluation time, ``apply()`` reads the current state
    vector and resolved parameters, then writes mass transfer rates
    into ``dydt``.

    Subclasses must implement ``apply`` and the ``from_edge`` classmethod.
    """

    def __init__(
        self,
        edge_id: int,
        source_idx: int,
        target_idx: int,
        source_name: str = "",
        target_name: str = "",
    ) -> None:
        self.edge_id = edge_id
        self.source_idx = source_idx
        self.target_idx = target_idx
        self.source_name = source_name
        self.target_name = target_name

    @classmethod
    @abstractmethod
    def from_edge(cls, edge_id: int, edge, state_index: dict[str, int]) -> FluxSpec:
        """Construct a FluxSpec from an Edge and the compiled state index.

        Args:
            edge_id: Ordinal position of this edge in the graph.
            edge: The Edge instance (FlowEdge, ClearanceEdge, etc.).
            state_index: Mapping of node name → state vector index.

        Returns:
            A configured FluxSpec ready for ``apply()`` calls.
        """
        ...

    @abstractmethod
    def apply(
        self,
        t: float,
        y: np.ndarray,
        dydt: np.ndarray,
        params: ResolvedParams,
    ) -> None:
        """Compute flux and write mass transfer rates into ``dydt``.

        This method is called once per edge per RHS evaluation.
        It must be fast — no allocations, no lookups beyond
        ``params`` accessors.

        Args:
            t: Current time (h).
            y: State vector (amounts in mg).
            dydt: Derivative vector (mutated in place).
            params: Resolved parameters for this MC sample.
        """
        ...


# ---------------------------------------------------------------------------
# Concrete FluxSpec implementations
# ---------------------------------------------------------------------------


@register_flux("flow")
class FlowFluxSpec(FluxSpec):
    """Convective transport: blood flow carries drug between compartments.

    For tissue nodes (Kp != 1): outflow concentration is corrected by Kp
    and RBP.  ``C_out = A * RBP / (V * Kp)``

    For blood pool nodes (Kp = 1): ``C_out = A / V`` (simple concentration).
    """

    @classmethod
    def from_edge(cls, edge_id: int, edge, state_index: dict[str, int]) -> FlowFluxSpec:
        return cls(
            edge_id,
            state_index[edge.source],
            state_index[edge.target],
            edge.source,
            edge.target,
        )

    def apply(
        self,
        t: float,
        y: np.ndarray,
        dydt: np.ndarray,
        params: ResolvedParams,
    ) -> None:
        q = params.edge_param(self.edge_id, "flow_rate")
        v_source = params.node_param(self.source_name, "volume")

        kp = params.drug_kp(self.source_name)
        rbp = params.drug_param("rbp")

        # Outflow concentration from source
        # For tissue: C_out = A * RBP / (V * Kp)
        # For blood: Kp=1, RBP cancels or =1, so C_out = A / V
        c_out = y[self.source_idx] * rbp / (v_source * kp) if v_source > 0 else 0.0

        flux = q * c_out
        dydt[self.source_idx] -= flux
        dydt[self.target_idx] += flux


@register_flux("clearance")
class ClearanceFluxSpec(FluxSpec):
    """Enzyme-mediated clearance (well-stirred) or renal filtration (GFR).

    Well-stirred model::

        CLint_organ = sum(abundance_i * affinity_i) * ivive_scaling
        CL_organ = (Q * fup * CLint) / (Q + fup * CLint)
        rate = CL_organ * C_in

    GFR filtration::

        rate = renal_clearance * C_plasma
    """

    def __init__(
        self,
        edge_id: int,
        source_idx: int,
        target_idx: int,
        source_name: str,
        target_name: str,
        model: str,
    ) -> None:
        super().__init__(edge_id, source_idx, target_idx, source_name, target_name)
        self.model = model

    @classmethod
    def from_edge(cls, edge_id: int, edge, state_index: dict[str, int]) -> ClearanceFluxSpec:
        return cls(
            edge_id,
            state_index[edge.source],
            state_index[edge.target],
            edge.source,
            edge.target,
            edge.model,
        )

    def apply(
        self,
        t: float,
        y: np.ndarray,
        dydt: np.ndarray,
        params: ResolvedParams,
    ) -> None:
        if self.model == "well_stirred":
            # Compute organ-level CLint from enzyme abundances x drug affinities
            clint_organ = 0.0
            ivive = params.node_param(self.source_name, "ivive_scaling")
            for tag, abundance in params.node_enzymes(self.source_name).items():
                affinity = params.drug_enzyme_affinity(tag)
                if affinity > 0 and abundance > 0:
                    clint_organ += abundance * affinity * ivive

            if clint_organ <= 0:
                return  # No metabolism at this node

            fup = params.drug_param("fup")
            q = params.total_inflow(self.source_name)

            # Well-stirred: CL = (Q * fup * CLint) / (Q + fup * CLint)
            denom = q + fup * clint_organ
            if denom < 1e-12:
                return
            clh = (q * fup * clint_organ) / denom

            # Concentration leaving the organ
            v = params.node_param(self.source_name, "volume")
            kp = params.drug_kp(self.source_name)
            rbp = params.drug_param("rbp")
            c_out = y[self.source_idx] * rbp / (v * kp) if v > 0 else 0.0

            rate = clh * c_out

        elif self.model == "parallel_tube":
            # Compute organ-level CLint (same as well-stirred)
            clint_organ = 0.0
            ivive = params.node_param(self.source_name, "ivive_scaling")
            for tag, abundance in params.node_enzymes(self.source_name).items():
                affinity = params.drug_enzyme_affinity(tag)
                if affinity > 0 and abundance > 0:
                    clint_organ += abundance * affinity * ivive

            if clint_organ <= 0:
                return

            fup = params.drug_param("fup")
            q = params.total_inflow(self.source_name)

            # Parallel-tube: CL = Q × (1 - e^(-fup × CLint / Q))
            if q < 1e-12:
                return
            exponent = -fup * clint_organ / q
            # Clamp exponent to avoid overflow for very high CLint
            exponent = max(exponent, -50.0)
            clh = q * (1.0 - np.exp(exponent))

            v = params.node_param(self.source_name, "volume")
            kp = params.drug_kp(self.source_name)
            rbp = params.drug_param("rbp")
            c_out = y[self.source_idx] * rbp / (v * kp) if v > 0 else 0.0

            rate = clh * c_out

        elif self.model == "gfr_filtration":
            renal_cl = params.drug_param("renal_clearance")
            if renal_cl <= 0:
                return
            v = params.node_param(self.source_name, "volume")
            kp = params.drug_kp(self.source_name)
            rbp = params.drug_param("rbp")
            c_plasma = y[self.source_idx] * rbp / (v * kp) if v > 0 else 0.0
            rate = renal_cl * c_plasma

        elif self.model == "extended":
            # Extended Clearance Model (ECM) — QSSA-closed hepatocyte.
            # See docs/superpowers/specs/2026-04-20-oatp-ecm-hepatic-clearance-design.md
            src = self.source_name
            ivive = params.node_param(src, "ivive_scaling")

            # PS_active from transporters at the source (identity-blind iteration)
            ps_active = 0.0
            for tag, abundance in params.node_transporters(src).items():
                jmax = params.drug_transporter_jmax(tag)
                km = params.drug_transporter_km(tag)
                if jmax <= 0 or km <= 0 or abundance <= 0:
                    continue
                ps_active += abundance * jmax / km
            ps_active *= ivive

            ps_passive = params.drug_param("ps_passive")
            ps_eff = params.drug_param("ps_eff")
            cl_int_bile = params.drug_param("cl_int_bile")

            ps_inf = ps_active + ps_passive

            # Metabolism — same pattern as well_stirred (organ-blind)
            cl_int_metab = 0.0
            for tag, abundance in params.node_enzymes(src).items():
                affinity = params.drug_enzyme_affinity(tag)
                if affinity > 0 and abundance > 0:
                    cl_int_metab += abundance * affinity * ivive
            cl_int_h = cl_int_metab + cl_int_bile

            fup = params.drug_param("fup")
            q = params.total_inflow(src)

            num = q * fup * ps_inf * cl_int_h
            den = q * (ps_eff + cl_int_h) + fup * ps_inf * cl_int_h
            if den < 1e-12:
                return
            clh = num / den

            v = params.node_param(src, "volume")
            kp = params.drug_kp(src)
            rbp = params.drug_param("rbp")
            c_out = y[self.source_idx] * rbp / (v * kp) if v > 0 else 0.0
            rate = clh * c_out

        else:
            return

        dydt[self.source_idx] -= rate
        dydt[self.target_idx] += rate


@register_flux("transit")
class TransitFluxSpec(FluxSpec):
    """First-order transit: ``rate = k_transit * A_source``."""

    @classmethod
    def from_edge(cls, edge_id: int, edge, state_index: dict[str, int]) -> TransitFluxSpec:
        return cls(
            edge_id,
            state_index[edge.source],
            state_index[edge.target],
            edge.source,
            edge.target,
        )

    def apply(
        self,
        t: float,
        y: np.ndarray,
        dydt: np.ndarray,
        params: ResolvedParams,
    ) -> None:
        kt = params.edge_param(self.edge_id, "transit_rate")
        amount = y[self.source_idx]
        rate = kt * amount
        dydt[self.source_idx] -= rate
        dydt[self.target_idx] += rate


@register_flux("absorption")
class AbsorptionFluxSpec(FluxSpec):
    """Drug absorption from lumen to tissue.

    ``ka = 2.88 * Peff * ka_fraction / particle_radius_um``

    The 2.88 comes from calibration of the effective permeability
    model to observed absorption rates.
    """

    @classmethod
    def from_edge(cls, edge_id: int, edge, state_index: dict[str, int]) -> AbsorptionFluxSpec:
        return cls(
            edge_id,
            state_index[edge.source],
            state_index[edge.target],
            edge.source,
            edge.target,
        )

    def apply(
        self,
        t: float,
        y: np.ndarray,
        dydt: np.ndarray,
        params: ResolvedParams,
    ) -> None:
        peff = params.drug_param("peff")
        ka_frac = params.edge_param(self.edge_id, "ka_fraction")
        radius = params.drug_param("particle_radius_um")

        if ka_frac <= 0 or peff <= 0:
            return

        # ka = 2.88 * Peff * ka_fraction / radius (h^-1)
        ka = 2.88 * peff * ka_frac / radius

        rate = ka * y[self.source_idx]
        dydt[self.source_idx] -= rate
        dydt[self.target_idx] += rate


@register_flux("diffusion")
class DiffusionFluxSpec(FluxSpec):
    """PS-limited exchange between vascular and tissue compartments.

    Uses unbound concentrations::

        cu_vasc = fup * C_vasc / RBP
        cu_tissue = fup * C_tissue / Kp
        flux = PS * (cu_vasc - cu_tissue)

    PS comes from: drug ps_overrides (if available) or edge ps_product.
    """

    @classmethod
    def from_edge(cls, edge_id: int, edge, state_index: dict[str, int]) -> DiffusionFluxSpec:
        return cls(
            edge_id,
            state_index[edge.source],
            state_index[edge.target],
            edge.source,
            edge.target,
        )

    def apply(
        self,
        t: float,
        y: np.ndarray,
        dydt: np.ndarray,
        params: ResolvedParams,
    ) -> None:
        fup = params.drug_param("fup")
        rbp = params.drug_param("rbp")

        # PS: prefer drug-specific override (resolved via lookup_name),
        # fall back to edge parameter.  No string manipulation — identity-blind.
        ps = params.drug_ps(self.source_name)
        if ps <= 0:
            ps = params.edge_param(self.edge_id, "ps_product")
        if ps <= 0:
            return

        v_vasc = params.node_param(self.source_name, "volume")
        v_tissue = params.node_param(self.target_name, "volume")
        kp = params.drug_kp(self.target_name)

        # Unbound concentrations
        c_vasc = y[self.source_idx] / v_vasc if v_vasc > 0 else 0.0
        c_tissue = y[self.target_idx] / v_tissue if v_tissue > 0 else 0.0

        cu_vasc = fup * c_vasc / rbp if rbp > 0 else 0.0
        cu_tissue = fup * c_tissue / kp if kp > 0 else 0.0

        flux = ps * (cu_vasc - cu_tissue)
        dydt[self.source_idx] -= flux
        dydt[self.target_idx] += flux


@register_flux("active_transport")
class ActiveTransportFluxSpec(FluxSpec):
    """Transporter-mediated efflux or uptake. Full Michaelis-Menten.

    For each transporter tag present in both node.transporters and
    drug.transporter_kinetics::

        C_µM = (A / V) * 1000 / MW        # mg/L → µM
        rate = abundance × Jmax × C_µM / (Km + C_µM)

    Direction is source → target (edge direction):
    - Gut efflux: gut_wall → lumen (P-gp: tissue → lumen, reduces Fg)
    - Hepatic uptake: blood → liver (OATP: increases clearance)

    Identity-blind: engine matches tags, never inspects names.

    Unit conversion note: Km is in µM, concentration is computed in mg/L.
    MW is needed to convert mg/L → µM: C_µM = C_mg_L × 1000 / MW.
    """

    @classmethod
    def from_edge(cls, edge_id: int, edge, state_index: dict[str, int]) -> ActiveTransportFluxSpec:
        return cls(
            edge_id,
            state_index[edge.source],
            state_index[edge.target],
            edge.source,
            edge.target,
        )

    def apply(
        self,
        t: float,
        y: np.ndarray,
        dydt: np.ndarray,
        params: ResolvedParams,
    ) -> None:
        mw = params.drug_mw()
        if mw <= 0:
            return

        v_source = params.node_param(self.source_name, "volume")
        if v_source <= 0:
            return

        # Source concentration in µM
        c_mg_l = y[self.source_idx] / v_source
        c_um = c_mg_l * 1000.0 / mw

        if c_um <= 0:
            return

        total_rate = 0.0
        node_transporters = params.node_transporters(self.target_name)

        for tag, abundance in node_transporters.items():
            jmax = params.drug_transporter_jmax(tag)
            km = params.drug_transporter_km(tag)

            if jmax <= 0 or km <= 0 or abundance <= 0:
                continue

            # Full Michaelis-Menten (saturable)
            rate = abundance * jmax * c_um / (km + c_um)
            total_rate += rate

        if total_rate <= 0:
            return

        # Convert from µM·volume/time units back to mg/time
        # rate is in arbitrary units scaled by abundance, jmax, and concentration
        # The IVIVE scaling factor handles unit conversion
        ivive = params.node_param(self.target_name, "ivive_scaling")
        mass_rate = total_rate * ivive

        dydt[self.source_idx] -= mass_rate
        dydt[self.target_idx] += mass_rate
