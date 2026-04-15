"""CYP pharmacogenomic phenotype scaling.

Applies patient-level phenotype metadata (CYP2D6 PM, CYP2C9 IM, etc.) to
the body graph by scaling the hepatic enzyme abundance. The engine does
the multiplication with drug.enzyme_affinity automatically, so no engine
changes are needed.

Scale factors follow CPIC activity score conventions:
    PM  (Poor Metabolizer)      — 0.10×  (two no-function alleles)
    IM  (Intermediate)          — 0.50×  (one reduced-function allele)
    EM  (Extensive / Normal)    — 1.00×  (reference)
    UM  (Ultra-rapid)           — 2.00×  (gene duplication)

Supported enzymes: CYP2D6, CYP2C9, CYP2C19, CYP3A5, CYP1A2, CYP2B6.
The set matches enzymes present in reference_man.yaml; unknown enzymes
are skipped with a warning.

Source: CPIC guidelines (https://cpicpgx.org/) and Caudle 2017.
"""

from __future__ import annotations

import logging
from dataclasses import replace

from sisyphus.core import Distribution
from sisyphus.graph.body import BodyGraph

logger = logging.getLogger(__name__)

# Phenotype → activity multiplier. Applied to enzyme abundance (pmol).
PHENOTYPE_SCALES: dict[str, float] = {
    "PM": 0.10,
    "IM": 0.50,
    "EM": 1.00,
    "NM": 1.00,  # Normal metabolizer (CPIC synonym for EM)
    "UM": 2.00,
    "RM": 1.50,  # Rapid metabolizer (intermediate between EM and UM)
}


def parse_phenotype_spec(spec: str) -> dict[str, str]:
    """Parse a CLI phenotype string into {enzyme: phenotype}.

    Examples:
        "CYP2D6:PM"                  → {"CYP2D6": "PM"}
        "CYP2D6:PM,CYP3A5:EM"        → {"CYP2D6": "PM", "CYP3A5": "EM"}
        "2D6:pm,2c9:im"              → {"CYP2D6": "PM", "CYP2C9": "IM"}
    """
    out: dict[str, str] = {}
    if not spec:
        return out
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Invalid phenotype entry {part!r}, expected 'CYP<gene>:<phenotype>'")
        enzyme, phenotype = part.split(":", 1)
        enzyme = enzyme.strip().upper()
        if not enzyme.startswith("CYP"):
            enzyme = "CYP" + enzyme
        phenotype = phenotype.strip().upper()
        if phenotype not in PHENOTYPE_SCALES:
            raise ValueError(
                f"Unknown phenotype {phenotype!r} for {enzyme}. "
                f"Valid: {sorted(PHENOTYPE_SCALES)}"
            )
        out[enzyme] = phenotype
    return out


def apply_phenotype_to_graph(
    graph: BodyGraph,
    phenotypes: dict[str, str],
    node: str = "liver",
) -> BodyGraph:
    """Return a new BodyGraph with enzyme abundances scaled by phenotype.

    Args:
        graph: Input body graph (unchanged).
        phenotypes: {enzyme_tag: phenotype_code} from ``parse_phenotype_spec``.
        node: Which node's enzymes to scale. Default "liver" (hepatic only,
            which is where ~95% of oxidative metabolism occurs). For gut
            effects (CYP3A4), pass node="small_intestine_wall" as well.

    Returns:
        New BodyGraph with scaled Distribution for matched enzymes. Other
        enzymes and nodes are unchanged. The CV is preserved so MC sampling
        still captures population variability *on top of* the phenotype.
    """
    if not phenotypes:
        return graph
    if node not in graph.nodes:
        logger.warning("phenotype: node %r not in graph, skipping", node)
        return graph

    target_enzymes = graph.nodes[node].enzymes
    new_enzymes: dict[str, Distribution] = dict(target_enzymes)
    applied: list[str] = []
    unknown: list[str] = []

    for enzyme, phenotype in phenotypes.items():
        if enzyme not in target_enzymes:
            unknown.append(enzyme)
            continue
        scale = PHENOTYPE_SCALES[phenotype]
        old = target_enzymes[enzyme]
        new_enzymes[enzyme] = Distribution(
            mean=old.mean * scale,
            cv=old.cv,
            dist_type=old.dist_type,
        )
        applied.append(f"{enzyme}:{phenotype}({scale}×)")

    if unknown:
        logger.warning(
            "phenotype: enzymes %s not found in %s (available: %s)",
            unknown, node, sorted(target_enzymes),
        )
    if applied:
        logger.info("phenotype: applied %s at %s", ", ".join(applied), node)

    new_node = replace(graph.nodes[node], enzymes=new_enzymes)
    # BodyGraph is a mutable container, not a dataclass. Build a shallow copy.
    new_graph = BodyGraph()
    new_graph.nodes = dict(graph.nodes)
    new_graph.nodes[node] = new_node
    new_graph.edges = list(graph.edges)
    new_graph.global_params = dict(graph.global_params)
    return new_graph
