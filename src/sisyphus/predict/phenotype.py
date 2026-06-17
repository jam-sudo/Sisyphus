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
from sisyphus.graph.types import Node

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

# CPIC NAT2 uses acetylator nomenclature (SA/IA/RA) while Sisyphus uses the
# unified PM/IM/EM scheme. The mapping is gene-conditional: "RA" (Rapid
# Acetylator) means wild-type / normal NAT2 activity (≡ EM, 1.00×) and must
# not be confused with the CYP "RM" (Rapid Metabolizer, 1.50×). Applied only
# when the tag is NAT2.
# Source: Relling 2020 CPIC NAT2 / isoniazid guideline (CPT 108:1265).
NAT2_ACETYLATOR_ALIASES: dict[str, str] = {
    "SA": "PM",
    "IA": "IM",
    "RA": "EM",
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
        "NAT2:SA"                    → {"NAT2": "PM"}   # CPIC acetylator alias
        "NAT2:RA"                    → {"NAT2": "EM"}
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
        if tag == "NAT2" and phenotype in NAT2_ACETYLATOR_ALIASES:
            phenotype = NAT2_ACETYLATOR_ALIASES[phenotype]
        if phenotype not in PHENOTYPE_SCALES:
            raise ValueError(
                f"Unknown phenotype {phenotype!r} for {tag}. "
                f"Valid: {sorted(PHENOTYPE_SCALES)}"
            )
        out[tag] = phenotype
    return out


def _scale_node(
    target: Node,
    phenotypes: dict[str, str],
    phenotype_scale_overrides: dict[str, float] | None,
) -> tuple[Node, list[str], list[str]]:
    """Scale one node's enzyme/transporter abundances by the phenotype factors.

    Pure: returns ``(new_node, applied, unknown)`` and does not log.
    """
    target_enzymes = target.enzymes
    target_transporters = getattr(target, "transporters", {}) or {}
    new_enzymes: dict[str, Distribution] = dict(target_enzymes)
    new_transporters: dict[str, Distribution] = dict(target_transporters)
    applied: list[str] = []
    unknown: list[str] = []

    for tag, phenotype in phenotypes.items():
        scale = PHENOTYPE_SCALES[phenotype]
        if phenotype_scale_overrides is not None and tag in phenotype_scale_overrides:
            override_scale = phenotype_scale_overrides[tag]
            if override_scale < 0:
                raise ValueError(
                    f"phenotype_scale_overrides[{tag!r}]={override_scale} is negative"
                )
            scale = override_scale
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

    new_node = replace(target, enzymes=new_enzymes, transporters=new_transporters)
    return new_node, applied, unknown


def apply_phenotype_to_graph(
    graph: BodyGraph,
    phenotypes: dict[str, str],
    node: str = "liver",
    phenotype_scale_overrides: dict[str, float] | None = None,
) -> BodyGraph:
    """Return a new BodyGraph with enzyme/transporter abundances scaled.

    Args:
        graph: Input body graph (unchanged).
        phenotypes: {tag: phenotype_code} from ``parse_phenotype_spec``.
            Tag may be a CYP enzyme (``CYP2D6``) or transporter gene
            (``SLCO1B1``, aliased to graph transporter ``OATP1B1``).
        node: Which organ to scale, by identity. A node matches if its
            ``name`` equals ``node`` OR its ``lookup_name`` equals ``node``.
            On a normal graph this is the single literal node (e.g. "liver").
            On an axially-expanded graph (``graph.axial.expand_axial``) it is
            every sub-tank ``liver__ax{i}`` (which carry ``lookup_name="liver"``),
            so the organ total is scaled correctly. Default "liver".
        phenotype_scale_overrides: Optional ``{gene: effective_scale}`` dict.
            When provided AND a gene matches a key in ``phenotypes``, the
            override value replaces ``PHENOTYPE_SCALES[phenotype]`` for that
            gene's effect on the matched node's enzyme/transporter abundance.
            Values are caller-supplied and caller-justified — Sisyphus does
            not endorse specific values. Negative values raise ValueError.
            Default ``None`` preserves current behavior.

    Returns:
        New BodyGraph with scaled Distribution for matched enzymes /
        transporters. The CV is preserved so MC sampling still captures
        population variability *on top of* the phenotype.
    """
    if not phenotypes:
        return graph

    targets = [
        n for n in graph.nodes.values()
        if n.name == node or (n.lookup_name or "") == node
    ]
    if not targets:
        logger.warning("phenotype: node %r not in graph, skipping", node)
        return graph

    if phenotype_scale_overrides is not None:
        for tag, phenotype in phenotypes.items():
            if tag in phenotype_scale_overrides:
                override_scale = phenotype_scale_overrides[tag]
                if override_scale >= 0:
                    logger.info(
                        "phenotype: override %s default scale %.3f -> %.3f",
                        tag, PHENOTYPE_SCALES[phenotype], override_scale,
                    )

    new_nodes = dict(graph.nodes)
    all_applied: list[str] = []
    all_unknown: list[str] = []
    last_target = targets[-1]
    for target in targets:
        new_node, applied, unknown = _scale_node(
            target, phenotypes, phenotype_scale_overrides
        )
        new_nodes[target.name] = new_node
        all_applied.extend(applied)
        all_unknown.extend(unknown)

    if phenotype_scale_overrides:
        unused_overrides = sorted(set(phenotype_scale_overrides) - set(phenotypes))
        if unused_overrides:
            logger.info(
                "phenotype: overrides for %s not in phenotypes dict, ignored",
                unused_overrides,
            )
    if all_unknown:
        available = sorted(
            list(last_target.enzymes)
            + [f"(transporter){t}" for t in (getattr(last_target, "transporters", {}) or {})]
        )
        logger.warning(
            "phenotype: tags %s not found in %s (available: %s)",
            sorted(set(all_unknown)), node, available,
        )
    if all_applied:
        logger.info(
            "phenotype: applied %s at %s (%d node(s))",
            ", ".join(sorted(set(all_applied))), node, len(targets),
        )

    new_graph = BodyGraph()
    new_graph.nodes = new_nodes
    new_graph.edges = list(graph.edges)
    new_graph.global_params = dict(graph.global_params)
    return new_graph
