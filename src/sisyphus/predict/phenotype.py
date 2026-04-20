"""CYP / transporter pharmacogenomic phenotype scaling.

Applies patient-level phenotype metadata (CYP2D6 PM, SLCO1B1 PM, etc.) to
the body graph by scaling the hepatic enzyme OR transporter abundance.
The engine multiplies abundance × drug.enzyme_affinity (or
transporter MM Jmax scales with abundance) automatically, so no engine
changes are needed.

Scale factors follow CPIC activity score conventions:
    PM  (Poor Metabolizer / Poor Function)    — 0.10×
    IM  (Intermediate)                        — 0.50×
    EM  (Extensive / Normal Function)         — 1.00×
    UM  (Ultra-rapid / Increased Function)    — 2.00×

Supported enzymes: CYP2D6, CYP2C9, CYP2C19, CYP3A5, CYP1A2, CYP2B6.
Supported transporters: SLCO1B1 (→ OATP1B1 protein). For SLCO1B1,
Cooper-DeHoff 2022 CPIC labels map as: Normal Function → EM, Decreased
Function (*1/*5, *1/*15) → IM, Poor Function (*5/*5, *15/*15) → PM,
Increased Function (rare *14/*14) → UM. PM/IM labels are used directly
for a unified CLI, while tag ``SLCO1B1`` is aliased to graph
``transporters["OATP1B1"]``.

Source: CPIC guidelines (https://cpicpgx.org/), Caudle 2017,
Cooper-DeHoff 2022 (SLCO1B1).
"""

from __future__ import annotations

import logging
from dataclasses import replace

from sisyphus.core import Distribution
from sisyphus.graph.body import BodyGraph

logger = logging.getLogger(__name__)

# Phenotype → activity multiplier. Applied to enzyme or transporter abundance.
PHENOTYPE_SCALES: dict[str, float] = {
    "PM": 0.10,
    "IM": 0.50,
    "EM": 1.00,
    "NM": 1.00,  # Normal metabolizer (CPIC synonym for EM)
    "UM": 2.00,
    "RM": 1.50,  # Rapid metabolizer (intermediate between EM and UM)
}

# Gene symbol → transporter protein tag used in graph.nodes[*].transporters.
# CPIC uses gene names (SLCO1B1) while the graph uses the protein product
# (OATP1B1). Canonical gene→protein mapping.
TRANSPORTER_ALIASES: dict[str, str] = {
    "SLCO1B1": "OATP1B1",
}


def parse_phenotype_spec(spec: str) -> dict[str, str]:
    """Parse a CLI phenotype string into {gene_or_enzyme: phenotype}.

    Accepts CYP enzyme tags (CYP2D6) or transporter gene tags (SLCO1B1).
    Short-form CYP numbers (``2D6``) are auto-prefixed; transporter gene
    names pass through as-is.

    Examples:
        "CYP2D6:PM"                  → {"CYP2D6": "PM"}
        "CYP2D6:PM,CYP3A5:EM"        → {"CYP2D6": "PM", "CYP3A5": "EM"}
        "2D6:pm,2c9:im"              → {"CYP2D6": "PM", "CYP2C9": "IM"}
        "SLCO1B1:PM"                 → {"SLCO1B1": "PM"}
        "CYP2D6:PM,SLCO1B1:IM"       → {"CYP2D6": "PM", "SLCO1B1": "IM"}
    """
    out: dict[str, str] = {}
    if not spec:
        return out
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(
                f"Invalid phenotype entry {part!r}, expected 'CYP<gene>:<phenotype>' "
                f"or '<TRANSPORTER_GENE>:<phenotype>'"
            )
        tag, phenotype = part.split(":", 1)
        tag = tag.strip().upper()
        # CYP short-form auto-prefix (2D6 → CYP2D6). Transporter gene tags
        # (SLCO1B1, ABCB1, ...) and full CYP names pass through unchanged.
        if tag not in TRANSPORTER_ALIASES and not tag.startswith("CYP") and tag[:1].isdigit():
            tag = "CYP" + tag
        phenotype = phenotype.strip().upper()
        if phenotype not in PHENOTYPE_SCALES:
            raise ValueError(
                f"Unknown phenotype {phenotype!r} for {tag}. "
                f"Valid: {sorted(PHENOTYPE_SCALES)}"
            )
        out[tag] = phenotype
    return out


def apply_phenotype_to_graph(
    graph: BodyGraph,
    phenotypes: dict[str, str],
    node: str = "liver",
) -> BodyGraph:
    """Return a new BodyGraph with enzyme/transporter abundances scaled.

    Args:
        graph: Input body graph (unchanged).
        phenotypes: {tag: phenotype_code} from ``parse_phenotype_spec``.
            Tag may be a CYP enzyme (``CYP2D6``) or transporter gene
            (``SLCO1B1``, aliased to graph transporter ``OATP1B1``).
        node: Which node to scale. Default "liver". Enzymes and
            transporters are both sourced from ``graph.nodes[node]``.

    Returns:
        New BodyGraph with scaled Distribution for matched enzymes /
        transporters. The CV is preserved so MC sampling still captures
        population variability *on top of* the phenotype.
    """
    if not phenotypes:
        return graph
    if node not in graph.nodes:
        logger.warning("phenotype: node %r not in graph, skipping", node)
        return graph

    target = graph.nodes[node]
    target_enzymes = target.enzymes
    target_transporters = getattr(target, "transporters", {}) or {}
    new_enzymes: dict[str, Distribution] = dict(target_enzymes)
    new_transporters: dict[str, Distribution] = dict(target_transporters)
    applied: list[str] = []
    unknown: list[str] = []

    for tag, phenotype in phenotypes.items():
        scale = PHENOTYPE_SCALES[phenotype]
        transporter_tag = TRANSPORTER_ALIASES.get(tag)
        if transporter_tag is not None:
            if transporter_tag not in target_transporters:
                unknown.append(f"{tag}→{transporter_tag}")
                continue
            old = target_transporters[transporter_tag]
            new_transporters[transporter_tag] = Distribution(
                mean=old.mean * scale,
                cv=old.cv,
                dist_type=old.dist_type,
            )
            applied.append(f"{tag}:{phenotype}({scale}×)→transporter")
        else:
            if tag not in target_enzymes:
                unknown.append(tag)
                continue
            old = target_enzymes[tag]
            new_enzymes[tag] = Distribution(
                mean=old.mean * scale,
                cv=old.cv,
                dist_type=old.dist_type,
            )
            applied.append(f"{tag}:{phenotype}({scale}×)")

    if unknown:
        available = sorted(list(target_enzymes) + [f"(transporter){t}" for t in target_transporters])
        logger.warning(
            "phenotype: tags %s not found in %s (available: %s)",
            unknown, node, available,
        )
    if applied:
        logger.info("phenotype: applied %s at %s", ", ".join(applied), node)

    new_node = replace(target, enzymes=new_enzymes, transporters=new_transporters)
    new_graph = BodyGraph()
    new_graph.nodes = dict(graph.nodes)
    new_graph.nodes[node] = new_node
    new_graph.edges = list(graph.edges)
    new_graph.global_params = dict(graph.global_params)
    return new_graph
