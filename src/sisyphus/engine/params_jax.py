"""JAX-compatible parameter container for the ODE engine.

Flattens ``ResolvedParams`` (Python dicts with method dispatch) into
flat ``jnp.ndarray`` fields that are compatible with ``jax.jit``,
``jax.grad``, and ``jax.vmap``.

Called once per MC sample to produce a ``JaxParams`` instance.
All dict lookups, enzyme aggregation, and IVIVE scaling happen at
construction time -- the RHS never touches Python dicts.

Lazy-imports JAX so the module can be imported even when JAX is not
installed (the rest of the engine still works with SciPy).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

try:
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except ImportError:  # pragma: no cover
    JAX_AVAILABLE = False
    jnp = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from sisyphus.engine.compiler import CompiledODE, ResolvedParams


# ---------------------------------------------------------------------------
# JaxParams — flat numeric container
# ---------------------------------------------------------------------------


@dataclass
class JaxParams:
    """Flat, JAX-compatible parameter set for one ODE evaluation.

    Every field is either a ``jnp.ndarray`` or a Python float.
    No dicts, no strings, no method dispatch -- pure numerics.

    Constructed via :func:`resolve_to_jax`, never by hand.
    """

    # -- Per-node arrays (indexed by state_index order) ---------------------
    node_volumes: jnp.ndarray  # (n_nodes,)
    node_kp: jnp.ndarray  # (n_nodes,)
    node_is_blood: jnp.ndarray  # (n_nodes,) 1.0 if blood_pool, else 0.0

    # -- Scalar drug properties ---------------------------------------------
    drug_fup: float
    drug_rbp: float
    drug_mw: float
    drug_renal_cl: float
    drug_peff: float
    drug_particle_radius: float

    # -- Per-edge arrays (indexed by edge_id) -------------------------------
    edge_flow_rates: jnp.ndarray  # (n_edges,)
    edge_transit_rates: jnp.ndarray  # (n_edges,)
    edge_ka_fractions: jnp.ndarray  # (n_edges,)
    edge_ps_products: jnp.ndarray  # (n_edges,)

    # -- Pre-computed per-node clearance ------------------------------------
    node_clint_organ: jnp.ndarray  # (n_nodes,) total CLint for well-stirred/PT
    node_total_inflow: jnp.ndarray  # (n_nodes,) sum of flow inflows (L/h)
    node_ivive_scaling: jnp.ndarray  # (n_nodes,)

    # -- Pre-computed per-node active transport -----------------------------
    node_transport_vmax: jnp.ndarray  # (n_nodes,) sum(abundance * jmax)
    node_transport_km: jnp.ndarray  # (n_nodes,) effective Km per node


# Register JaxParams as a JAX pytree so it can be passed through jit/vmap.
if JAX_AVAILABLE:
    def _jaxparams_flatten(p):
        children = (
            p.node_volumes, p.node_kp, p.node_is_blood,
            p.drug_fup, p.drug_rbp, p.drug_mw, p.drug_renal_cl,
            p.drug_peff, p.drug_particle_radius,
            p.edge_flow_rates, p.edge_transit_rates,
            p.edge_ka_fractions, p.edge_ps_products,
            p.node_clint_organ, p.node_total_inflow, p.node_ivive_scaling,
            p.node_transport_vmax, p.node_transport_km,
        )
        return children, None

    def _jaxparams_unflatten(aux, children):
        return JaxParams(*children)

    jax.tree_util.register_pytree_node(
        JaxParams, _jaxparams_flatten, _jaxparams_unflatten,
    )


# ---------------------------------------------------------------------------
# resolve_to_jax — conversion from ResolvedParams
# ---------------------------------------------------------------------------


def resolve_to_jax(compiled: CompiledODE, params: ResolvedParams) -> JaxParams:
    """Convert a ``ResolvedParams`` to a ``JaxParams``.

    Called once per MC sample.  All dict lookups, enzyme aggregation,
    and IVIVE scaling happen here so the RHS is pure numerics.

    Args:
        compiled: The compiled ODE skeleton (provides topology).
        params: Point-value resolved parameters for this MC sample.

    Returns:
        A ``JaxParams`` ready for the JAX RHS function.

    Raises:
        RuntimeError: If JAX is not installed.
    """
    if not JAX_AVAILABLE:
        raise RuntimeError(
            "JAX is not installed. Install with: pip install jax jaxlib"
        )

    n_nodes = compiled.n_states  # noqa: F841
    n_edges = len(compiled.flux_specs)  # noqa: F841

    # -- Build the sorted node name list (same order as state_index) --------
    # state_index was built from sorted(graph.nodes.keys())
    node_names_sorted = sorted(compiled.state_index.keys(), key=lambda n: compiled.state_index[n])

    # -- Per-node arrays ----------------------------------------------------
    volumes = []
    kps = []
    is_blood = []
    ivive_scaling = []
    total_inflow = []
    clint_organ = []
    transport_vmax = []
    transport_km = []

    for name in node_names_sorted:
        volumes.append(params.node_param(name, "volume"))
        kps.append(params.drug_kp(name))
        is_blood.append(1.0 if params.is_blood_pool(name) else 0.0)
        ivive_scaling.append(params.node_param(name, "ivive_scaling"))
        total_inflow.append(params.total_inflow(name))

        # Pre-compute organ-level CLint = sum(abundance_i * affinity_i * ivive)
        node_ivive = params.node_param(name, "ivive_scaling")
        enzymes = params.node_enzymes(name)
        cl_sum = 0.0
        for tag, abundance in enzymes.items():
            affinity = params.drug_enzyme_affinity(tag)
            if affinity > 0 and abundance > 0:
                cl_sum += abundance * affinity * node_ivive
        clint_organ.append(cl_sum)

        # Pre-compute active transport: aggregate Vmax and effective Km
        # Vmax_total = sum(abundance_i * jmax_i)
        # Effective Km: weighted harmonic mean, or simple average if only one.
        # For Michaelis-Menten with multiple transporters summed:
        #   total_rate = sum_i( abundance_i * jmax_i * C / (km_i + C) )
        # We cannot collapse to single Vmax/Km exactly when Km differs.
        # Approximation: store aggregate Vmax and abundance-weighted Km.
        transporters = params.node_transporters(name)
        vmax_sum = 0.0
        km_weighted_num = 0.0  # sum(abundance * jmax * km)
        vmax_weight = 0.0  # sum(abundance * jmax) -- same as vmax_sum
        for tag, abundance in transporters.items():
            j = params.drug_transporter_jmax(tag)
            km = params.drug_transporter_km(tag)
            if j > 0 and km > 0 and abundance > 0:
                contrib = abundance * j
                vmax_sum += contrib
                km_weighted_num += contrib * km
                vmax_weight += contrib

        transport_vmax.append(vmax_sum)
        # Weighted average Km (weight = abundance * jmax).
        # Falls back to 1.0 if no transporters (never used since vmax=0).
        transport_km.append(km_weighted_num / vmax_weight if vmax_weight > 0 else 1.0)

    # -- Per-edge arrays ----------------------------------------------------
    flow_rates = []
    transit_rates = []
    ka_fractions = []
    ps_products = []

    for spec in compiled.flux_specs:
        edge_id = spec.edge_id

        # Flow
        try:
            flow_rates.append(params.edge_param(edge_id, "flow_rate"))
        except KeyError:
            flow_rates.append(0.0)

        # Transit
        try:
            transit_rates.append(params.edge_param(edge_id, "transit_rate"))
        except KeyError:
            transit_rates.append(0.0)

        # Absorption
        try:
            ka_fractions.append(params.edge_param(edge_id, "ka_fraction"))
        except KeyError:
            ka_fractions.append(0.0)

        # Diffusion (PS product)
        # For diffusion edges, prefer drug-specific PS override, fall back to edge param
        try:
            ps = params.drug_ps(spec.source_name)
            if ps <= 0:
                ps = params.edge_param(edge_id, "ps_product")
            ps_products.append(ps)
        except KeyError:
            ps_products.append(0.0)

    # -- Scalar drug properties ---------------------------------------------
    drug_fup = params.drug_param("fup")
    drug_rbp = params.drug_param("rbp")
    drug_mw = params.drug_mw()
    drug_renal_cl = params.drug_param("renal_clearance")
    drug_peff = params.drug_param("peff")
    drug_particle_radius = params.drug_param("particle_radius_um")

    return JaxParams(
        node_volumes=jnp.array(volumes, dtype=jnp.float64),
        node_kp=jnp.array(kps, dtype=jnp.float64),
        node_is_blood=jnp.array(is_blood, dtype=jnp.float64),
        drug_fup=drug_fup,
        drug_rbp=drug_rbp,
        drug_mw=drug_mw,
        drug_renal_cl=drug_renal_cl,
        drug_peff=drug_peff,
        drug_particle_radius=drug_particle_radius,
        edge_flow_rates=jnp.array(flow_rates, dtype=jnp.float64),
        edge_transit_rates=jnp.array(transit_rates, dtype=jnp.float64),
        edge_ka_fractions=jnp.array(ka_fractions, dtype=jnp.float64),
        edge_ps_products=jnp.array(ps_products, dtype=jnp.float64),
        node_clint_organ=jnp.array(clint_organ, dtype=jnp.float64),
        node_total_inflow=jnp.array(total_inflow, dtype=jnp.float64),
        node_ivive_scaling=jnp.array(ivive_scaling, dtype=jnp.float64),
        node_transport_vmax=jnp.array(transport_vmax, dtype=jnp.float64),
        node_transport_km=jnp.array(transport_km, dtype=jnp.float64),
    )
