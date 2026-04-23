"""Parametric physiology generator for Sisyphus.

Generates a BodyGraph for any (body_weight_kg, age_years) by scaling the
ICRP Reference Man YAML template using:
  - Organ volumes: linear BW scaling (V × BW/70)
  - Cardiac output: allometric (CO × (BW/70)^0.75) + aging decline after 40
  - Flow edges: proportional to scaled cardiac output
  - Enzyme abundances: maturation (Hines 2008) × BW/70 scaling × aging decline
  - Transporter abundances: BW/70 only (no ontogeny data available)
  - body_weight global param: set to actual BW

Sources:
  - Volume scaling: ICRP 2002
  - Allometric CO scaling: West 1997 (exponent 0.75)
  - Cardiac output aging decline: Hogben 1971, Lindsay 1996
  - Enzyme ontogeny: Hines 2008 (t_half = 1.0-3.0 years for maturation)
  - Enzyme aging decline: Tanaka 1998, Schmucker 2005
"""

from __future__ import annotations

import dataclasses
import logging
import math
from pathlib import Path

import numpy as np

from sisyphus.core import Distribution
from sisyphus.graph.body import BodyGraph
from sisyphus.graph.builder import build_from_yaml
from sisyphus.graph.types import DiffusionEdge, FlowEdge
from sisyphus.physiology import correlation_registry

logger = logging.getLogger(__name__)

_DEFAULT_YAML = Path("data/physiology/reference_man.yaml")

# ---------------------------------------------------------------------------
# Enzyme maturation / aging parameters
# ---------------------------------------------------------------------------

ENZYME_PARAMS: dict[str, dict[str, float]] = {
    "CYP3A4": {"t_half": 2.0, "decline_frac": 0.20},
    "CYP2D6": {"t_half": 1.4, "decline_frac": 0.10},
    "CYP1A2": {"t_half": 2.8, "decline_frac": 0.25},
    "CYP2C9": {"t_half": 1.7, "decline_frac": 0.20},
    "CYP2E1": {"t_half": 1.2, "decline_frac": 0.15},
    "UGT2B7": {"t_half": 1.0, "decline_frac": 0.15},
    "UGT1A1": {"t_half": 3.0, "decline_frac": 0.15},
    "UGT1A4": {"t_half": 2.0, "decline_frac": 0.15},
    "UGT1A9": {"t_half": 1.5, "decline_frac": 0.15},
}


# ---------------------------------------------------------------------------
# Core scaling functions
# ---------------------------------------------------------------------------


def enzyme_factor(age: float, bw: float, t_half: float, decline_frac: float) -> float:
    """Return the multiplicative scale factor for an enzyme abundance.

    Combines three effects:
    1. Ontogeny / maturation: 1 - exp(-0.693 × age / t_half)
       (Hill-type: reaches ~50% of adult level at age = t_half years)
    2. Body-weight scaling: bw / 70  (linear organ-weight proportionality)
    3. Aging decline (age > 40): linear decline from 1.0 at age 40 to
       (1 - decline_frac) at age 85, floored at 0.5.

    The reference YAML abundances represent a fully mature 70 kg adult
    (age ≈ 30).  Calling enzyme_factor(30, 70, ...) returns ~1.0, so
    new_abundance = ref_abundance × enzyme_factor(...) is correct.

    Args:
        age: Age in years (float, >0).
        bw: Body weight in kg.
        t_half: Maturation half-life in years (from Hines 2008).
        decline_frac: Fractional decline from age 40→85 (from Tanaka 1998).

    Returns:
        Multiplicative scale factor (dimensionless, always > 0).
    """
    maturation = 1.0 - math.exp(-0.693 * age / t_half)
    bw_scale = bw / 70.0
    if age <= 40:
        return maturation * bw_scale
    aging = 1.0 - decline_frac * (age - 40) / 45.0
    aging = max(aging, 0.5)
    return maturation * aging * bw_scale


def _flow_aging_factor(age: float) -> float:
    """Cardiac output aging reduction factor.

    1.0 for age ≤ 40, then linear decline to 0.65 at age 85, floored at 0.5.

    Source: Lindsay 1996 (CO declines ~35% between age 40 and 85).
    """
    if age <= 40:
        return 1.0
    return max(1.0 - 0.35 * (age - 40) / 45, 0.5)


def _gfr_aging_factor(age: float) -> float:
    """Glomerular filtration rate aging reduction factor.

    1.0 for age ≤ 40, then linear decline, floored at 0.4.

    Source: Lindeman 1985 (~1% per year after 40).
    """
    if age <= 40:
        return 1.0
    return max(1.0 - (age - 40) / 125, 0.4)


# ---------------------------------------------------------------------------
# Correlated abundance resampler
# ---------------------------------------------------------------------------


def _resample_correlated_abundances(
    graph: BodyGraph, rng: np.random.Generator
) -> BodyGraph:
    """Replace every grouped Distribution in the graph with a sampled draw.

    For each node, Distributions (in enzymes and transporters) that share a
    correlation_group are collected and sampled jointly from their registered
    multivariate-lognormal spec. Sampled values replace the original means;
    the new Distributions have cv=0 and correlation_group=None, representing
    "one realized individual" at this MC iteration.

    Distributions with correlation_group=None are left unchanged.
    """
    g = BodyGraph()
    g.edges = list(graph.edges)
    g.global_params = dict(graph.global_params)

    for name, node in graph.nodes.items():
        # Collect items by group for this node
        groups: dict[str, list[tuple[str, str, Distribution]]] = {}
        for tag, d in node.enzymes.items():
            if d.correlation_group:
                groups.setdefault(d.correlation_group, []).append(("enzyme", tag, d))
        for tag, d in node.transporters.items():
            if d.correlation_group:
                groups.setdefault(d.correlation_group, []).append(("transporter", tag, d))

        new_enzymes = dict(node.enzymes)
        new_transporters = dict(node.transporters)

        for group_name, entries in groups.items():
            spec = correlation_registry.get(group_name)
            if spec is None:
                raise KeyError(
                    f"Node {name!r} references correlation_group "
                    f"{group_name!r} but no such group is registered. "
                    f"Did you call load_from_json() on the appropriate JSON?"
                )
            # Index by tag so we can reorder to spec.members
            by_tag = {tag: (kind, d) for (kind, tag, d) in entries}
            missing = set(spec.members) - set(by_tag.keys())
            if missing:
                raise KeyError(
                    f"Node {name!r} group {group_name!r} is missing members "
                    f"{sorted(missing)} required by registered spec."
                )
            means = np.array([by_tag[m][1].mean for m in spec.members])
            cvs = np.array([by_tag[m][1].cv for m in spec.members])
            sampled = correlation_registry.sample_correlated(
                means, cvs, spec.log_corr_matrix, rng
            )
            for i, member_tag in enumerate(spec.members):
                kind, _old = by_tag[member_tag]
                new_d = Distribution(
                    mean=float(sampled[i]), cv=0.0, correlation_group=None
                )
                if kind == "enzyme":
                    new_enzymes[member_tag] = new_d
                else:
                    new_transporters[member_tag] = new_d

        g.nodes[name] = dataclasses.replace(
            node, enzymes=new_enzymes, transporters=new_transporters
        )

    return g


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------


def generate_physiology(
    body_weight_kg: float,
    age_years: float,
    base_yaml: Path | None = None,
    rng: np.random.Generator | None = None,
) -> BodyGraph:
    """Generate a BodyGraph for arbitrary (body_weight_kg, age_years).

    Loads the reference adult YAML as a template and overrides all
    physiology parameters according to allometric and ontogenic scaling.

    Scaling rules (all relative to the 70 kg ICRP Reference Man):
      - Node volumes          : × (BW / 70)
      - Cardiac output        : × (BW / 70)^0.75 × _flow_aging_factor(age)
      - FlowEdge flow rates   : × (BW / 70)^0.75 × _flow_aging_factor(age)
      - DiffusionEdge ps_product : × (BW / 70)   (surface-area proportional)
      - Enzyme abundances     : × enzyme_factor(age, bw, t_half, decline_frac)
      - Transporter abundances: × (BW / 70)       (no ontogeny data)
      - body_weight param     : set to body_weight_kg
      - cardiac_output param  : set to scaled value

    Note: The function builds the graph by direct dict assignment rather than
    add_node/add_edge to allow mutation of a fully-valid template. Validation
    is not called on the output (flow conservation is preserved by scaling all
    flow rates uniformly).

    When ``rng`` is provided, Distributions that carry a ``correlation_group``
    tag are jointly resampled from their registered multivariate-lognormal
    specification via ``_resample_correlated_abundances``.  The resulting graph
    represents one realized individual: all grouped Distributions are replaced
    with collapsed ``Distribution(mean=sampled, cv=0, correlation_group=None)``
    values.  When ``rng=None`` (default), the deterministic mean-field graph
    is returned unchanged.

    Args:
        body_weight_kg: Target body weight in kg (> 0).
        age_years: Target age in years (> 0).
        base_yaml: Optional path to the reference YAML. Defaults to
            data/physiology/reference_man.yaml.
        rng: Optional numpy random Generator for correlated enzyme/transporter
            abundance sampling.  Pass ``np.random.default_rng(seed)`` to
            enable inter-individual variability.  ``None`` preserves the
            prior deterministic behavior.

    Returns:
        A BodyGraph with all parameters scaled to the requested population.
        If ``rng`` is provided, grouped abundances are sampled from their
        correlated lognormal prior.
    """
    ref = build_from_yaml(base_yaml or _DEFAULT_YAML)

    bw_ratio = body_weight_kg / 70.0
    co_scale = bw_ratio**0.75 * _flow_aging_factor(age_years)

    g = BodyGraph()

    # --- Scale nodes ---
    for name, node in ref.nodes.items():
        new_vol = Distribution(node.volume.mean * bw_ratio, cv=node.volume.cv)

        new_enzymes: dict[str, Distribution] = {}
        for tag, enz in node.enzymes.items():
            params = ENZYME_PARAMS.get(tag)
            if params:
                ef = enzyme_factor(
                    age_years, body_weight_kg, params["t_half"], params["decline_frac"]
                )
                new_enzymes[tag] = Distribution(enz.mean * ef, cv=enz.cv)
            else:
                # No ontogeny data for this enzyme tag — BW-scale only
                new_enzymes[tag] = Distribution(enz.mean * bw_ratio, cv=enz.cv)

        new_transporters: dict[str, Distribution] = {
            tag: Distribution(tr.mean * bw_ratio, cv=tr.cv)
            for tag, tr in node.transporters.items()
        }

        g.nodes[name] = dataclasses.replace(
            node,
            volume=new_vol,
            enzymes=new_enzymes,
            transporters=new_transporters,
        )

    # --- Scale edges ---
    for edge in ref.edges:
        if isinstance(edge, FlowEdge):
            g.edges.append(
                dataclasses.replace(
                    edge,
                    flow_rate=Distribution(edge.flow_rate.mean * co_scale, cv=edge.flow_rate.cv),
                )
            )
        elif isinstance(edge, DiffusionEdge):
            # PS product scales with membrane surface area ~ BW
            g.edges.append(
                dataclasses.replace(
                    edge,
                    ps_product=Distribution(edge.ps_product.mean * bw_ratio, cv=edge.ps_product.cv),
                )
            )
        else:
            # ClearanceEdge, TransitEdge, AbsorptionEdge, ActiveTransportEdge:
            # no numeric scaling needed (rates are derived from node parameters at solve time)
            g.edges.append(edge)

    # --- Scale global_params ---
    ref_co = ref.global_params["cardiac_output"].mean
    g.global_params = dict(ref.global_params)
    g.global_params["cardiac_output"] = Distribution(
        ref_co * co_scale, cv=ref.global_params["cardiac_output"].cv
    )
    g.global_params["body_weight"] = Distribution(body_weight_kg, cv=0.0)

    if rng is not None:
        g = _resample_correlated_abundances(g, rng)

    logger.debug(
        "generate_physiology: BW=%.1f kg, age=%.1f y → CO=%.1f L/h, "
        "bw_ratio=%.4f, co_scale=%.4f",
        body_weight_kg,
        age_years,
        g.global_params["cardiac_output"].mean,
        bw_ratio,
        co_scale,
    )

    return g
