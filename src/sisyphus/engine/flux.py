"""Flux registry and base FluxSpec.

Each edge type in the body graph dispatches to a FluxSpec subclass that
knows how to compute the mass transfer rate for that transport mechanism.

The registry pattern: edge_type string → FluxSpec class.  The ODE compiler
looks up the registry to map edges to flux functions at compile time.

**Identity-blind:** FluxSpec implementations operate on indices and
parameter lookups, never on node/enzyme/drug names.
"""

from __future__ import annotations

import math
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
        # RBP-2: the convective edge carries WHOLE BLOOD. A blood-pool node's A/V
        # is already blood (gate RBP to 1); a tissue node's blood outlet is
        # A*RBP/(V*Kp). (Spec 2026-06-04-rbp-concentration-basis-design.md.)
        rbp = 1.0 if params.is_blood_pool(self.source_name) else params.drug_param("rbp")

        # Outflow blood concentration from source
        #   tissue:     C_out = A * RBP / (V * Kp)
        #   blood pool: C_out = A / V    (RBP gated to 1)
        c_out = y[self.source_idx] * rbp / (v_source * kp) if v_source > 0 else 0.0

        flux = q * c_out
        dydt[self.source_idx] -= flux
        dydt[self.target_idx] += flux


@register_flux("clearance")
class ClearanceFluxSpec(FluxSpec):
    """Hepatic / renal clearance flux. Four models supported:

    - ``well_stirred``  — classical WS with organ-level CLint (default)
    - ``parallel_tube`` — NOT a single-tank flux; expanded into N serial
      well_stirred sub-tanks at build time by ``graph.axial.expand_axial``
      (true axial-gradient parallel-tube).
    - ``gfr_filtration`` — renal filtration (``CL_renal × C_plasma``)
    - ``extended``      — ECM: QSSA-closed hepatocyte with active + passive
      uptake, passive efflux, metabolism, biliary clearance. See
      ``docs/superpowers/specs/2026-04-20-oatp-ecm-hepatic-clearance-design.md``.

    Hepatic fu correction (B-11): at a node flagged ``fu_correction_applicable``,
    ``well_stirred`` replaces ``fup`` with ``fup × fu_correction_liver`` (an
    empirical fu_inc/fu_plasma uptake correction). Because ``parallel_tube`` is
    expanded into ``well_stirred`` sub-tanks at build time, the correction
    applies to each sub-tank through that same branch.
    ``extended`` and ``gfr_filtration`` intentionally do NOT — the
    ECM models concentrative hepatic uptake explicitly (PS_active/PS_inf/PS_eff)
    so the WS-style factor would double-count it, and GFR is a plasma sink with
    no uptake term. The production liver edge is ``extended``; pipeline.predict
    warns if a curated non-1.0 value lands on such an edge.

    Well-stirred model (FLUX-1: intrinsic clearance on the compartment outlet)::

        CLint_organ = sum(abundance_i * affinity_i) * ivive_scaling
        rate = (fup * CLint_organ) * c_out        # NOT the whole-organ CL_h

    The node is a perfusion compartment: a separate convective FlowEdge carries
    Q * c_out out of it, so the flow limitation emerges from the ODE and the
    realized hepatic extraction is fup*CLint/(Q + fup*CLint) -> 1.0. Applying the
    whole-organ CL_h = Q*fup*CLint/(Q+fup*CLint) to c_out here would double-count
    the flow term and cap extraction at 0.5 (see flux.py FLUX-1 comments).

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
        if edge.model == "parallel_tube":
            raise ValueError(
                "ClearanceEdge model='parallel_tube' must be expanded into axial "
                "sub-compartments before compile (graph.axial.expand_axial); it is "
                "not a single-tank flux. Set Node.axial_subcompartments and call "
                "expand_axial(graph) before ODECompiler().compile(graph)."
            )
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
            # B-11: hepatic intracellular fu correction at flagged nodes.
            if params.node_param(self.source_name, "fu_correction_applicable") > 0:
                fup = fup * params.drug_param("fu_correction_liver")

            # FLUX-1: apply the *intrinsic* (flow-unlimited) clearance to c_out.
            # The node is a perfusion compartment with a separate convective
            # Q·c_out outflow edge, so the flow limitation emerges from the ODE
            # and the realized extraction is fup·CLint/(Q+fup·CLint) → 1.0.
            # Applying the whole-organ CL_h (which already embeds Q) here would
            # double-count the flow term and cap extraction at 0.5.
            cl_intrinsic = fup * clint_organ

            # RBP-2: the metabolic sink acts on unbound PLASMA (fup·CLint·C_plasma).
            # The separate convective Q·c_out edge supplies the flow limitation, so
            # the realized extraction emerges E = fup·CLint/(Q·RBP + fup·CLint) =
            # fu_b·CLint/(Q+fu_b·CLint), fu_b = fup/RBP (canonical well-stirred).
            v = params.node_param(self.source_name, "volume")
            kp = params.drug_kp(self.source_name)
            c_plasma = y[self.source_idx] / (v * kp) if v > 0 else 0.0

            rate = cl_intrinsic * c_plasma

        elif self.model == "gfr_filtration":
            renal_cl = params.drug_param("renal_clearance")
            if renal_cl <= 0:
                return
            # RBP-2: renal_cl = GFR·fup is plasma-basis → filter PLASMA A/(V·Kp),
            # not blood. The convective edge supplies flow limitation.
            v = params.node_param(self.source_name, "volume")
            kp = params.drug_kp(self.source_name)
            c_plasma = y[self.source_idx] / (v * kp) if v > 0 else 0.0
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
            # NOTE: the extended ECM intentionally does NOT apply the B-11
            # fu_correction_liver here (unlike well_stirred/parallel_tube). It is
            # an empirical fu_inc/fu_plasma factor for concentrative hepatic
            # uptake, which the ECM already models mechanistically via
            # PS_active/PS_inf/PS_eff — applying it on top would double-count.
            # pipeline.predict warns if a non-1.0 value is curated for a drug
            # whose flagged node clears via this model.

            # FLUX-1: intrinsic hepatic clearance (flow-unlimited). At QSSA the
            # hepatocyte removal per unit blood concentration is
            #   CL_int,hep = fup·PS_inf·CL_int_h / (PS_eff + CL_int_h).
            # The separate convective Q·c_out outflow edge supplies the flow
            # limitation, so the realized extraction is the well-stirred wrap
            # CL_int,hep/(Q+CL_int,hep) → 1.0 at high CLint. The prior whole-organ
            # form embedded Q in num+den and, applied to c_out, double-counted
            # flow, capping extraction at 0.5.
            den = ps_eff + cl_int_h
            if den < 1e-12:
                return
            cl_intrinsic = fup * ps_inf * cl_int_h / den

            # RBP-2: plasma-basis sink (CL_int,hep·C_plasma); the convective edge
            # supplies flow limitation (see well_stirred).
            v = params.node_param(src, "volume")
            kp = params.drug_kp(src)
            c_plasma = y[self.source_idx] / (v * kp) if v > 0 else 0.0
            rate = cl_intrinsic * c_plasma

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


# ---------------------------------------------------------------------------
# Paracellular absorption physics (Renkin pore-sieving)
#
# A second, parallel absorption route to the transcellular AbsorptionFluxSpec.
# Models pore-restricted, charge-selective permeation through intestinal tight
# junctions — the dominant route for small hydrophilic drugs (cimetidine,
# metformin, atenolol class) that have poor transcellular permeability yet
# absorb well in vivo.
#
# Peff_para [x10^-4 cm/s] = P_scale * F_renkin(r_mol / R_pore) * E_charge
#
# All constants are EXTERNAL human-intestinal-physiology / in-vitro anchors,
# NEVER fit to holdout Cmax (Invariants #5/#8):
#   - r_mol(MW): Avdeef 2010, Pharm Res 27:480 (PMID 20069445), refined
#     Stokes-Einstein hydrodynamic radius.
#   - F_renkin: Renkin 1954, J Gen Physiol 38:225 (PMID 13211998), steric x
#     Faxen hydrodynamic hindrance.
#   - R_pore, P_scale, E_charge: supplied by the graph (global_params), sourced
#     from Linnankoski 2010 (PMID 19827099) / Sugano 2002 / Kataoka 2011
#     (pore radius), an atenolol human-jejunal Peff anchor (Dahlgren 2016,
#     PMID 27504798) for the lumped scale, and Adson 1994 (DOI
#     10.1002/jps.2600831103) / Tam-Velicky-Dryfe for the cation-selective
#     tight-junction charge factor.
# ---------------------------------------------------------------------------

# Refined Stokes-Einstein constants (Avdeef 2010).
_PARA_KB_J_PER_K = 1.380649e-23  # Boltzmann constant
_PARA_BODY_TEMP_K = 310.15  # 37 C
_PARA_WATER_VISCOSITY_PA_S = 0.6913e-3  # water at 37 C


def _hydrodynamic_radius_angstrom(mw: float) -> float:
    """Drug hydrodynamic radius (Angstrom) from molecular weight.

    Avdeef 2010 (PMID 20069445) refined Stokes-Einstein:
        D_aq(37C) = 9.9e-5 * MW^-0.453   [cm^2/s]
        r_SE      = kB*T / (6*pi*eta*D_aq)
        r_HYD     = (0.92 + 21.8/MW) * r_SE
    """
    if mw <= 0:
        return 0.0
    d_aq_cm2_s = 9.9e-5 * mw ** (-0.453)
    d_aq_m2_s = d_aq_cm2_s * 1e-4
    r_se_m = _PARA_KB_J_PER_K * _PARA_BODY_TEMP_K / (
        6.0 * math.pi * _PARA_WATER_VISCOSITY_PA_S * d_aq_m2_s
    )
    return (0.92 + 21.8 / mw) * r_se_m * 1e10  # m -> Angstrom


def _renkin_sieving(lam: float) -> float:
    """Renkin (1954) molecular sieving factor for a solute of relative radius
    ``lam = r_molecule / r_pore``.

    F(lam) = (1-lam)^2 * (1 - 2.104*lam + 2.09*lam^3 - 0.95*lam^5),  lam < 1
           = 0,                                                       lam >= 1

    (1-lam)^2 is the steric partition; the polynomial is the Faxen
    hydrodynamic wall-drag correction.  Coefficients are exact analytic
    constants, not fitted.
    """
    if lam <= 0.0 or lam >= 1.0:
        return 0.0
    f = (1.0 - lam) ** 2 * (1.0 - 2.104 * lam + 2.09 * lam**3 - 0.95 * lam**5)
    return f if f > 0.0 else 0.0


def _paracellular_charge_factor(
    compound_type: str, cation_factor: float, anion_factor: float
) -> float:
    """Electrostatic factor for the cation-selective tight junction.

    Cations (protonated bases) are enhanced, anions (deprotonated acids)
    hindered, neutrals/zwitterions unaffected.  Keyed on the drug's charge
    class — a type descriptor, never a drug identity.
    """
    if compound_type == "base":
        return cation_factor
    if compound_type == "acid":
        return anion_factor
    return 1.0  # neutral, zwitterion


@register_flux("paracellular_absorption")
class ParacellularAbsorptionFluxSpec(FluxSpec):
    """Paracellular (tight-junction pore) absorption from lumen to tissue.

    Parallel to the transcellular ``AbsorptionFluxSpec``: applies the same
    gut surface-to-volume / dissolution geometry (``2.88 * ka_frac / radius``)
    but to the paracellular permeability ``Peff_para`` computed from the drug's
    molecular size (Renkin sieving) and charge class against pore physiology
    carried in ``global_params``.  Independent of the drug's transcellular
    ``peff`` — a low-permeability hydrophilic drug still absorbs through pores.

    Inert (no flux) when the pore physiology globals are absent, when the
    molecule is larger than the pore (Renkin -> 0), or when ``ka_fraction`` is
    zero.
    """

    @classmethod
    def from_edge(
        cls, edge_id: int, edge, state_index: dict[str, int]
    ) -> ParacellularAbsorptionFluxSpec:
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
        ka_frac = params.edge_param(self.edge_id, "ka_fraction")
        if ka_frac <= 0:
            return

        r_pore = params.global_param("paracellular_pore_radius_angstrom")
        p_scale = params.global_param("paracellular_permeability_scale")
        if r_pore <= 0 or p_scale <= 0:
            return  # paracellular physiology not configured -> inert

        r_mol = _hydrodynamic_radius_angstrom(params.drug_mw())
        f_sieve = _renkin_sieving(r_mol / r_pore)
        if f_sieve <= 0:
            return  # molecule too large for the pore

        # Charge factors default to 1.0 (no electrostatic effect) if a graph
        # omits them; compound_type is a charge-class descriptor, not identity.
        cation = params.global_param("paracellular_charge_factor_cation") or 1.0
        anion = params.global_param("paracellular_charge_factor_anion") or 1.0
        e_charge = _paracellular_charge_factor(params._drug.compound_type, cation, anion)

        # Peff_para [x10^-4 cm/s]; ka uses the same gut geometry as the
        # transcellular flux (2.88 * ka_frac / particle_radius_um, h^-1).
        peff_para = p_scale * f_sieve * e_charge
        radius = params.drug_param("particle_radius_um")
        ka_para = 2.88 * peff_para * ka_frac / radius

        rate = ka_para * y[self.source_idx]
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

    Direction is source → target (edge direction). Intended use cases:
    - Gut efflux (P-gp, BCRP at gut_wall → lumen): reduces Fg.
    - Renal secretion (OAT/OCT at proximal tubule → urine): increases CL_renal.
    - BBB efflux (P-gp at brain vascular → brain tissue).

    **Hepatic OATP uptake is NOT handled here anymore.** Since the
    2026-04-20 ECM migration, hepatic active uptake is folded into
    ``ClearanceFluxSpec(model="extended")`` (QSSA-closed hepatocyte).
    No YAML currently instantiates ``ActiveTransportFluxSpec`` in the
    reference physiology; the class is retained for future
    non-hepatic-uptake applications.

    Identity-blind: engine matches tags, never inspects names.

    Unit conversion note: Km is in µM, concentration is computed in mg/L.
    MW is needed to convert mg/L → µM: C_µM = C_mg_L × 1000 / MW.
    """

    def __init__(
        self,
        edge_id: int,
        source_idx: int,
        target_idx: int,
        source_name: str,
        target_name: str,
        direction: str = "uptake",
    ) -> None:
        super().__init__(edge_id, source_idx, target_idx, source_name, target_name)
        self.direction = direction

    @classmethod
    def from_edge(cls, edge_id: int, edge, state_index: dict[str, int]) -> ActiveTransportFluxSpec:
        return cls(
            edge_id,
            state_index[edge.source],
            state_index[edge.target],
            edge.source,
            edge.target,
            getattr(edge, "direction", "uptake"),
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

        # WS-5: transporter sits at the target for uptake, the source for efflux.
        # The driving (substrate) concentration is always the source (computed above).
        transporter_node = self.target_name if self.direction == "uptake" else self.source_name

        total_rate = 0.0
        node_transporters = params.node_transporters(transporter_node)

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
        ivive = params.node_param(transporter_node, "ivive_scaling")
        mass_rate = total_rate * ivive

        dydt[self.source_idx] -= mass_rate
        dydt[self.target_idx] += mass_rate


@register_flux("prodrug_activation")
class ProdrugActivationFluxSpec(FluxSpec):
    """Mass transfer via well-stirred enzyme catalysis: parent → active.

    v2 (2026-04-27): well-stirred extraction at flow-through nodes.
    Mirrors ClearanceFluxSpec(model="well_stirred") math but routes flux
    to the active species pool (not a sink), with MW × yield scaling.

    FLUX-1: the conversion site is a perfusion compartment with a separate
    convective Q·c_out outflow edge, so the intrinsic clearance is applied::

        CLint_node = Σ_tag (abundance[tag] × affinity_for_conversion[tag]) × ivive
        rate_parent = (fup × CLint_node) × c_out      # NOT the whole-organ CL_h
        rate_active = rate_parent × (mw_active/mw_parent) × conversion_yield

    The flow limitation emerges from the convective edge (extraction → 1.0);
    the old whole-organ form Q·fup·CLint/(Q+fup·CLint) double-counted flow.

    Identity-blind: engine iterates enzyme_tags only.
    """

    def __init__(
        self,
        edge_id: int,
        source_idx: int,
        target_idx: int,
        source_name: str,
        target_name: str,
        enzyme_tags: frozenset[str],
        mw_ratio: float,
    ) -> None:
        super().__init__(edge_id, source_idx, target_idx, source_name, target_name)
        self.enzyme_tags = enzyme_tags
        self.mw_ratio = mw_ratio

    @classmethod
    def from_edge(
        cls, edge_id: int, edge, state_index: dict[str, int]
    ) -> ProdrugActivationFluxSpec:
        if edge.mw_parent <= 0:
            raise ValueError(
                f"ProdrugActivationEdge mw_parent must be positive, got {edge.mw_parent}"
            )
        return cls(
            edge_id=edge_id,
            source_idx=state_index[edge.source],
            target_idx=state_index[edge.target],
            source_name=edge.source,
            target_name=edge.target,
            enzyme_tags=edge.enzyme_tags,
            mw_ratio=edge.mw_active / edge.mw_parent,
        )

    def apply(
        self,
        t: float,
        y: np.ndarray,
        dydt: np.ndarray,
        params: ResolvedParams,
    ) -> None:
        # Compute per-node CLint from enzyme abundance × drug affinity × ivive
        clint_organ = 0.0
        ivive = params.node_param(self.source_name, "ivive_scaling")
        node_enzymes = params.node_enzymes(self.source_name)
        for tag in self.enzyme_tags:
            abundance = node_enzymes.get(tag, 0.0)
            affinity = params.drug_enzyme_affinity_for_conversion(tag)
            if affinity > 0 and abundance > 0:
                clint_organ += abundance * affinity * ivive

        if clint_organ <= 0:
            return  # No catalysis at this node for this drug

        fup = params.drug_param("fup")
        # B-11: hepatic intracellular fu correction at flagged nodes.
        if params.node_param(self.source_name, "fu_correction_applicable") > 0:
            fup = fup * params.drug_param("fu_correction_liver")

        # FLUX-1: conversion sites are perfusion compartments (liver/gut_wall)
        # with a separate convective Q·c_out outflow edge, so the activation
        # sink must use the intrinsic clearance fup·CLint. The flow limitation
        # emerges from the convective edge; applying the whole-organ CL_h here
        # would double-count Q and cap the activated fraction at 0.5.
        cl_intrinsic = fup * clint_organ

        # RBP-2: plasma-basis activation sink (fup·CLint·C_plasma); the convective
        # edge supplies flow limitation (see well_stirred / ClearanceFluxSpec).
        v = params.node_param(self.source_name, "volume")
        kp = params.drug_kp(self.source_name)
        c_plasma = y[self.source_idx] / (v * kp) if v > 0 else 0.0

        rate_parent = cl_intrinsic * c_plasma
        y_frac = params.edge_param(self.edge_id, "conversion_yield")
        rate_active = rate_parent * self.mw_ratio * y_frac

        dydt[self.source_idx] -= rate_parent
        dydt[self.target_idx] += rate_active


@register_flux("one_compartment_elimination")
class OneCompartmentEliminationFluxSpec(FluxSpec):
    """Aggregate 1st-order elimination: rate = (CL/Vd) × A_source.

    Mass-conserving: source loses mass; target (sink-type node) gains
    it for mass-balance audit. Used for active metabolite clearance
    where literature reports plasma CL and Vd directly (no enzyme
    decomposition).
    """

    @classmethod
    def from_edge(
        cls, edge_id: int, edge, state_index: dict[str, int]
    ) -> OneCompartmentEliminationFluxSpec:
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
        cl = params.edge_param(self.edge_id, "cl_per_h")
        vd = params.edge_param(self.edge_id, "vd_l")
        if vd <= 0:
            return
        rate = (cl / vd) * y[self.source_idx]
        dydt[self.source_idx] -= rate
        dydt[self.target_idx] += rate
