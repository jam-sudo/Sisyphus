"""Pure-JAX RHS function builder for the ODE engine.

Builds a JIT-compilable right-hand side function from the compiled ODE
topology.  At **compile time**, static index arrays are extracted from
``CompiledODE.flux_specs`` (which edges are flow, which are clearance,
etc.).  At **runtime**, all fluxes are computed functionally with
vectorized operations -- no Python loops, no dict lookups, no
if/else branching.

The resulting ``rhs(t, y, params)`` is compatible with ``jax.jit``,
``jax.grad``, and ``jax.vmap``.

Lazy-imports JAX so the module can be imported even when JAX is not
installed (the rest of the engine still works with SciPy).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

try:
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except ImportError:  # pragma: no cover
    JAX_AVAILABLE = False
    jnp = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from sisyphus.engine.compiler import CompiledODE
    from sisyphus.engine.params_jax import JaxParams


# ---------------------------------------------------------------------------
# Internal: flux-type detection helpers
# ---------------------------------------------------------------------------

# Import concrete FluxSpec subclasses for isinstance checks.
# These are only needed at build time (not at JAX runtime).
from sisyphus.engine.flux import (
    AbsorptionFluxSpec,
    ActiveTransportFluxSpec,
    ClearanceFluxSpec,
    DiffusionFluxSpec,
    FlowFluxSpec,
    TransitFluxSpec,
)


# ---------------------------------------------------------------------------
# make_jax_rhs — the public API
# ---------------------------------------------------------------------------


def make_jax_rhs(
    compiled: "CompiledODE",
) -> Callable:
    """Build a pure-JAX RHS from compiled ODE topology.

    Extracts static topology (edge index arrays per flux type) at build
    time.  Returns a function ``rhs(t, y, params: JaxParams) -> dydt``
    that uses only JAX operations and is compatible with ``jax.jit``,
    ``jax.grad``, and ``jax.vmap``.

    Args:
        compiled: A ``CompiledODE`` compiled from a ``BodyGraph``.

    Returns:
        A callable ``rhs(t, y, params) -> dydt`` where ``y`` and ``dydt``
        are ``jnp.ndarray`` of shape ``(n_states,)`` and ``params`` is a
        ``JaxParams``.

    Raises:
        RuntimeError: If JAX is not installed.
    """
    if not JAX_AVAILABLE:
        raise RuntimeError(
            "JAX is not installed. Install with: pip install jax jaxlib"
        )

    n_states = compiled.n_states

    # -- Extract static topology per flux type (Python, not JAX) ------------
    # Each list collects (source_idx, target_idx, edge_id) for that flux type.

    flow_src = []
    flow_tgt = []
    flow_eid = []

    # Clearance: separate by model type
    cl_ws_src = []  # well_stirred source indices
    cl_ws_tgt = []
    cl_pt_src = []  # parallel_tube source indices
    cl_pt_tgt = []
    cl_gfr_src = []  # gfr_filtration source indices
    cl_gfr_tgt = []

    transit_src = []
    transit_tgt = []
    transit_eid = []

    absorption_src = []
    absorption_tgt = []
    absorption_eid = []

    diffusion_src = []
    diffusion_tgt = []
    diffusion_eid = []

    transport_src = []
    transport_tgt = []

    for spec in compiled.flux_specs:
        if isinstance(spec, FlowFluxSpec):
            flow_src.append(spec.source_idx)
            flow_tgt.append(spec.target_idx)
            flow_eid.append(spec.edge_id)

        elif isinstance(spec, ClearanceFluxSpec):
            if spec.model == "well_stirred":
                cl_ws_src.append(spec.source_idx)
                cl_ws_tgt.append(spec.target_idx)
            elif spec.model == "parallel_tube":
                cl_pt_src.append(spec.source_idx)
                cl_pt_tgt.append(spec.target_idx)
            elif spec.model == "gfr_filtration":
                cl_gfr_src.append(spec.source_idx)
                cl_gfr_tgt.append(spec.target_idx)

        elif isinstance(spec, TransitFluxSpec):
            transit_src.append(spec.source_idx)
            transit_tgt.append(spec.target_idx)
            transit_eid.append(spec.edge_id)

        elif isinstance(spec, AbsorptionFluxSpec):
            absorption_src.append(spec.source_idx)
            absorption_tgt.append(spec.target_idx)
            absorption_eid.append(spec.edge_id)

        elif isinstance(spec, DiffusionFluxSpec):
            diffusion_src.append(spec.source_idx)
            diffusion_tgt.append(spec.target_idx)
            diffusion_eid.append(spec.edge_id)

        elif isinstance(spec, ActiveTransportFluxSpec):
            transport_src.append(spec.source_idx)
            transport_tgt.append(spec.target_idx)

    # Convert to JAX arrays (static, traced as constants by jit)
    _flow_src = jnp.array(flow_src, dtype=jnp.int32)
    _flow_tgt = jnp.array(flow_tgt, dtype=jnp.int32)
    _flow_eid = jnp.array(flow_eid, dtype=jnp.int32)

    _cl_ws_src = jnp.array(cl_ws_src, dtype=jnp.int32)
    _cl_ws_tgt = jnp.array(cl_ws_tgt, dtype=jnp.int32)
    _cl_pt_src = jnp.array(cl_pt_src, dtype=jnp.int32)
    _cl_pt_tgt = jnp.array(cl_pt_tgt, dtype=jnp.int32)
    _cl_gfr_src = jnp.array(cl_gfr_src, dtype=jnp.int32)
    _cl_gfr_tgt = jnp.array(cl_gfr_tgt, dtype=jnp.int32)

    _transit_src = jnp.array(transit_src, dtype=jnp.int32)
    _transit_tgt = jnp.array(transit_tgt, dtype=jnp.int32)
    _transit_eid = jnp.array(transit_eid, dtype=jnp.int32)

    _absorption_src = jnp.array(absorption_src, dtype=jnp.int32)
    _absorption_tgt = jnp.array(absorption_tgt, dtype=jnp.int32)
    _absorption_eid = jnp.array(absorption_eid, dtype=jnp.int32)

    _diffusion_src = jnp.array(diffusion_src, dtype=jnp.int32)
    _diffusion_tgt = jnp.array(diffusion_tgt, dtype=jnp.int32)
    _diffusion_eid = jnp.array(diffusion_eid, dtype=jnp.int32)

    _transport_src = jnp.array(transport_src, dtype=jnp.int32)
    _transport_tgt = jnp.array(transport_tgt, dtype=jnp.int32)

    # -- Boolean flags for which flux types are present ---------------------
    has_flow = len(flow_src) > 0
    has_cl_ws = len(cl_ws_src) > 0
    has_cl_pt = len(cl_pt_src) > 0
    has_cl_gfr = len(cl_gfr_src) > 0
    has_transit = len(transit_src) > 0
    has_absorption = len(absorption_src) > 0
    has_diffusion = len(diffusion_src) > 0
    has_transport = len(transport_src) > 0

    # -- The RHS function ---------------------------------------------------

    def rhs(t: float, y: "jnp.ndarray", params: "JaxParams") -> "jnp.ndarray":
        """Pure-JAX ODE right-hand side.

        All operations use ``jnp`` -- no Python control flow on data,
        no in-place mutation.  Uses ``jnp.where`` instead of ``if/else``
        and ``.at[].add()`` for scatter-add.

        Args:
            t: Current time (h).
            y: State vector, shape ``(n_states,)`` -- amounts in mg.
            params: A ``JaxParams`` instance.

        Returns:
            Derivative vector ``dydt``, shape ``(n_states,)``.
        """
        dydt = jnp.zeros(n_states, dtype=jnp.float64)

        # Shorthand for frequently used params
        fup = params.drug_fup
        rbp = params.drug_rbp

        # ---------------------------------------------------------------
        # 1. Flow fluxes (convective transport)
        #    C_out = A[src] * RBP / (V[src] * Kp[src])   (tissue)
        #    C_out = A[src] / V[src]                      (blood pool, Kp=1)
        #    flux = Q * C_out
        # ---------------------------------------------------------------
        if has_flow:
            v_src = params.node_volumes[_flow_src]
            kp_src = params.node_kp[_flow_src]
            q = params.edge_flow_rates[_flow_eid]

            # C_out: safe division (avoid /0 when volume is zero)
            c_out = jnp.where(
                v_src > 0.0,
                y[_flow_src] * rbp / (v_src * kp_src),
                0.0,
            )
            flow_flux = q * c_out

            dydt = dydt.at[_flow_src].add(-flow_flux)
            dydt = dydt.at[_flow_tgt].add(flow_flux)

        # ---------------------------------------------------------------
        # 2. Clearance fluxes
        # ---------------------------------------------------------------

        # 2a. Well-stirred model
        #     CL = (Q * fup * CLint) / (Q + fup * CLint)
        #     C_out = A[src] * RBP / (V[src] * Kp[src])
        #     rate = CL * C_out
        if has_cl_ws:
            clint = params.node_clint_organ[_cl_ws_src]
            q_ws = params.node_total_inflow[_cl_ws_src]
            v_ws = params.node_volumes[_cl_ws_src]
            kp_ws = params.node_kp[_cl_ws_src]

            denom_ws = q_ws + fup * clint
            # Safe division: if denom < epsilon, CL = 0
            clh_ws = jnp.where(
                denom_ws > 1e-12,
                (q_ws * fup * clint) / denom_ws,
                0.0,
            )
            c_out_ws = jnp.where(
                v_ws > 0.0,
                y[_cl_ws_src] * rbp / (v_ws * kp_ws),
                0.0,
            )
            rate_ws = clh_ws * c_out_ws

            dydt = dydt.at[_cl_ws_src].add(-rate_ws)
            dydt = dydt.at[_cl_ws_tgt].add(rate_ws)

        # 2b. Parallel-tube model
        #     CL = Q * (1 - exp(-fup * CLint / Q))
        #     C_out same as well-stirred
        if has_cl_pt:
            clint_pt = params.node_clint_organ[_cl_pt_src]
            q_pt = params.node_total_inflow[_cl_pt_src]
            v_pt = params.node_volumes[_cl_pt_src]
            kp_pt = params.node_kp[_cl_pt_src]

            # Exponent with clamp to avoid overflow for very high CLint
            exponent = jnp.where(
                q_pt > 1e-12,
                -fup * clint_pt / q_pt,
                0.0,
            )
            exponent = jnp.clip(exponent, -50.0, 0.0)

            clh_pt = jnp.where(
                q_pt > 1e-12,
                q_pt * (1.0 - jnp.exp(exponent)),
                0.0,
            )
            c_out_pt = jnp.where(
                v_pt > 0.0,
                y[_cl_pt_src] * rbp / (v_pt * kp_pt),
                0.0,
            )
            rate_pt = clh_pt * c_out_pt

            dydt = dydt.at[_cl_pt_src].add(-rate_pt)
            dydt = dydt.at[_cl_pt_tgt].add(rate_pt)

        # 2c. GFR filtration
        #     rate = renal_cl * C_plasma
        #     C_plasma = A[src] * RBP / (V[src] * Kp[src])
        if has_cl_gfr:
            renal_cl = params.drug_renal_cl
            v_gfr = params.node_volumes[_cl_gfr_src]
            kp_gfr = params.node_kp[_cl_gfr_src]

            c_plasma_gfr = jnp.where(
                v_gfr > 0.0,
                y[_cl_gfr_src] * rbp / (v_gfr * kp_gfr),
                0.0,
            )
            rate_gfr = renal_cl * c_plasma_gfr

            dydt = dydt.at[_cl_gfr_src].add(-rate_gfr)
            dydt = dydt.at[_cl_gfr_tgt].add(rate_gfr)

        # ---------------------------------------------------------------
        # 3. Transit fluxes (first-order)
        #    rate = k_transit * A[src]
        # ---------------------------------------------------------------
        if has_transit:
            kt = params.edge_transit_rates[_transit_eid]
            transit_flux = kt * y[_transit_src]

            dydt = dydt.at[_transit_src].add(-transit_flux)
            dydt = dydt.at[_transit_tgt].add(transit_flux)

        # ---------------------------------------------------------------
        # 4. Absorption fluxes
        #    ka = 2.88 * Peff * ka_fraction / particle_radius
        #    rate = ka * A[src]
        # ---------------------------------------------------------------
        if has_absorption:
            ka_frac = params.edge_ka_fractions[_absorption_eid]
            peff = params.drug_peff
            radius = params.drug_particle_radius

            # ka = 2.88 * Peff * ka_fraction / radius
            ka = jnp.where(
                (ka_frac > 0.0) & (peff > 0.0) & (radius > 0.0),
                2.88 * peff * ka_frac / radius,
                0.0,
            )
            absorption_flux = ka * y[_absorption_src]

            dydt = dydt.at[_absorption_src].add(-absorption_flux)
            dydt = dydt.at[_absorption_tgt].add(absorption_flux)

        # ---------------------------------------------------------------
        # 5. Diffusion fluxes (PS-limited)
        #    cu_vasc = fup * C_vasc / RBP
        #    cu_tissue = fup * C_tissue / Kp
        #    flux = PS * (cu_vasc - cu_tissue)
        # ---------------------------------------------------------------
        if has_diffusion:
            ps = params.edge_ps_products[_diffusion_eid]

            v_vasc = params.node_volumes[_diffusion_src]
            v_tissue = params.node_volumes[_diffusion_tgt]
            kp_tgt = params.node_kp[_diffusion_tgt]

            c_vasc = jnp.where(v_vasc > 0.0, y[_diffusion_src] / v_vasc, 0.0)
            c_tissue = jnp.where(v_tissue > 0.0, y[_diffusion_tgt] / v_tissue, 0.0)

            cu_vasc = jnp.where(rbp > 0.0, fup * c_vasc / rbp, 0.0)
            cu_tissue = jnp.where(kp_tgt > 0.0, fup * c_tissue / kp_tgt, 0.0)

            diff_flux = jnp.where(ps > 0.0, ps * (cu_vasc - cu_tissue), 0.0)

            dydt = dydt.at[_diffusion_src].add(-diff_flux)
            dydt = dydt.at[_diffusion_tgt].add(diff_flux)

        # ---------------------------------------------------------------
        # 6. Active transport (Michaelis-Menten)
        #    C_uM = (A[src] / V[src]) * 1000 / MW
        #    rate = Vmax_total * C_uM / (Km_eff + C_uM) * ivive
        #
        #    Pre-computed per node:
        #      Vmax_total = sum(abundance_i * jmax_i)
        #      Km_eff = weighted average Km
        # ---------------------------------------------------------------
        if has_transport:
            mw = params.drug_mw
            v_src_t = params.node_volumes[_transport_src]
            vmax = params.node_transport_vmax[_transport_src]
            km = params.node_transport_km[_transport_src]
            ivive_t = params.node_ivive_scaling[_transport_src]

            # Concentration in uM
            c_mg_l = jnp.where(v_src_t > 0.0, y[_transport_src] / v_src_t, 0.0)
            c_um = jnp.where(mw > 0.0, c_mg_l * 1000.0 / mw, 0.0)

            # Michaelis-Menten: Vmax * C / (Km + C)
            mm_rate = jnp.where(
                (vmax > 0.0) & (km > 0.0) & (c_um > 0.0),
                vmax * c_um / (km + c_um),
                0.0,
            )
            # IVIVE scaling converts to mass rate (mg/h)
            mass_rate = mm_rate * ivive_t

            dydt = dydt.at[_transport_src].add(-mass_rate)
            dydt = dydt.at[_transport_tgt].add(mass_rate)

        return dydt

    return rhs
