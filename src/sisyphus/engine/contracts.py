"""Engine-level contract validation.

Cross-edge / cross-node checks a single per-edge FluxSpec cannot make (FluxSpecs
are identity-blind and see only their own edge). Invoked by the engine's solve
orchestrators (``uncertainty``) and the production pipeline before integration.
Does NOT touch ``compiler.py`` / ``solver.py`` (invariant #8).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from sisyphus.graph.types import ClearanceEdge, ProdrugActivationEdge

if TYPE_CHECKING:
    from sisyphus.graph.body import BodyGraph

# Clearance models that do NOT apply the B-11 fu_correction_liver factor
# (extended ECM models hepatic uptake explicitly; gfr is a plasma sink).
_FU_CORRECTION_DROP_MODELS = frozenset({"extended", "gfr_filtration"})
# Clearance models that DO apply it (well-stirred family).
# parallel_tube is expanded to well_stirred tanks before compile (graph.axial),
# so the honoring clearance model at runtime is well_stirred only.
_FU_CORRECTION_HONORING_MODELS = frozenset({"well_stirred"})


def flagged_nodes_without_honoring_flux(graph: BodyGraph) -> list[str]:
    """Return flagged nodes whose fu_correction would be *entirely* dropped.

    A node is returned iff: it is flagged ``fu_correction_applicable > 0``; it is
    the source of a ClearanceEdge with a drop-model (extended / gfr_filtration);
    and NO ClearanceEdge with a honoring model (well_stirred / parallel_tube) and
    NO ProdrugActivationEdge also originate from it.

    Identity-blind: inspects node flags + edge types/models only, never names.
    """
    flagged = {
        name for name, node in graph.nodes.items()
        if node.fu_correction_applicable > 0
    }
    if not flagged:
        return []

    has_drop: set[str] = set()
    has_honoring: set[str] = set()
    for edge in graph.edges:
        if isinstance(edge, ClearanceEdge) and edge.source in flagged:
            if edge.model in _FU_CORRECTION_DROP_MODELS:
                has_drop.add(edge.source)
            elif edge.model in _FU_CORRECTION_HONORING_MODELS:
                has_honoring.add(edge.source)
        elif isinstance(edge, ProdrugActivationEdge) and edge.source in flagged:
            has_honoring.add(edge.source)

    return sorted(has_drop - has_honoring)


def assert_fu_correction_honored(graph: BodyGraph, fu_correction_liver_mean: float) -> None:
    """Raise ``ValueError`` if a non-identity fu_correction_liver is entirely dropped.

    No-op when the value is the identity ``1.0`` (the production default), so the
    headline path stays bit-identical. Check the *mean* (sample-independent),
    never a per-MC-sample realized value.
    """
    if fu_correction_liver_mean == 1.0:
        return
    offenders = flagged_nodes_without_honoring_flux(graph)
    if offenders:
        raise ValueError(
            f"fu_correction_liver={fu_correction_liver_mean:.3g} is entirely "
            f"dropped at flagged node(s) {offenders}: their clearance uses a "
            f"model that does not apply it (extended ECM / gfr_filtration) and no "
            f"well_stirred/parallel_tube clearance or prodrug_activation edge at "
            f"the node applies it either. Remove the curated value, switch the "
            f"node's clearance model, or model uptake via transporter params."
        )
