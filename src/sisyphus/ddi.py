"""Drug-drug interaction modeling.

DDI is implemented by modifying enzyme abundances in the BodyGraph
before ODE compilation.  The engine sees adjusted abundances and
computes clearance accordingly -- zero engine modifications needed.

Competitive inhibition::

    abundance_effective = abundance_base / (1 + [I] / Ki)

Enzyme induction::

    abundance_effective = abundance_base * (1 + Emax * [I] / (EC50 + [I]))

This approach is consistent with the "identity-blind engine" invariant:
the engine never knows DDI is happening.  It just sees different enzyme
abundances and computes clearance as usual.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass

from sisyphus.core import Distribution
from sisyphus.graph.body import BodyGraph

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DDI perpetrator types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Inhibitor:
    """CYP enzyme inhibitor specification.

    Attributes:
        name: Inhibitor drug name.
        enzyme_ki: Mapping of enzyme tag -> Ki (uM).
            e.g., {"CYP3A4": 0.015} for ketoconazole (potent 3A4 inhibitor).
        plasma_concentration: Steady-state **total** plasma Cmax (uM).
            Preset values use total (not unbound) concentrations for
            simplicity, consistent with FDA DDI guidance basic models.
            Used to compute [I]/Ki ratio.
        fu_perpetrator: Fraction unbound for the perpetrator drug.
            Defaults to 1.0 (total = unbound assumption).  Set to
            the actual fu when unbound concentrations are needed for
            more refined predictions.
        mechanism: Inhibition mechanism.  Currently only "competitive"
            is implemented.  "time_dependent" is a future extension.
    """

    name: str
    enzyme_ki: dict[str, float]  # enzyme_tag -> Ki in uM
    plasma_concentration: float  # [I] at steady state, uM (total Cmax)
    fu_perpetrator: float = 1.0  # fraction unbound (1.0 = total ≈ unbound)
    mechanism: str = "competitive"  # "competitive" | "time_dependent"


@dataclass(frozen=True)
class Inducer:
    """CYP enzyme inducer specification.

    Attributes:
        name: Inducer drug name.
        enzyme_emax: Mapping of enzyme tag -> Emax (fold induction above baseline).
        enzyme_ec50: Mapping of enzyme tag -> EC50 (uM).
        plasma_concentration: Steady-state **total** plasma Cmax (uM).
            Preset values use total (not unbound) concentrations.
        fu_perpetrator: Fraction unbound for the perpetrator drug.
            Defaults to 1.0 (total = unbound assumption).
    """

    name: str
    enzyme_emax: dict[str, float]  # enzyme_tag -> Emax (fold increase)
    enzyme_ec50: dict[str, float]  # enzyme_tag -> EC50 (uM)
    plasma_concentration: float  # [I] at steady state, uM (total Cmax)
    fu_perpetrator: float = 1.0  # fraction unbound (1.0 = total ≈ unbound)


# ---------------------------------------------------------------------------
# Graph modification functions
# ---------------------------------------------------------------------------


def apply_inhibition(graph: BodyGraph, inhibitor: Inhibitor) -> BodyGraph:
    """Apply competitive CYP inhibition to a BodyGraph.

    For each node with enzymes, reduces affected enzyme abundances by::

        abundance_new = abundance_base / (1 + [I] / Ki)

    Returns a NEW BodyGraph (does not mutate the original).
    Engine sees reduced enzyme abundances -> lower clearance -> higher Cmax.

    Args:
        graph: Base BodyGraph.
        inhibitor: Inhibitor specification.

    Returns:
        Modified BodyGraph with reduced enzyme abundances.
    """
    if inhibitor.mechanism != "competitive":
        raise ValueError(
            f"Unsupported inhibition mechanism: {inhibitor.mechanism!r}. "
            f"Only 'competitive' is currently implemented."
        )

    new_graph = BodyGraph()
    new_graph.global_params = dict(graph.global_params)

    for name, node in graph.nodes.items():
        if not node.enzymes:
            new_graph.nodes[name] = node
            continue

        new_enzymes: dict[str, Distribution] = {}
        modified = False

        for tag, abundance in node.enzymes.items():
            if tag in inhibitor.enzyme_ki:
                ki = inhibitor.enzyme_ki[tag]
                if ki <= 0:
                    raise ValueError(f"Ki must be positive for enzyme {tag!r}, got {ki}")
                ratio = inhibitor.plasma_concentration / ki
                # Competitive inhibition: effective = base / (1 + [I]/Ki)
                factor = 1.0 / (1.0 + ratio)
                new_enzymes[tag] = Distribution(
                    mean=abundance.mean * factor,
                    cv=abundance.cv,
                )
                modified = True
                logger.debug(
                    "%s: %s abundance %.0f -> %.0f (factor=%.4f, [I]/Ki=%.1f)",
                    name,
                    tag,
                    abundance.mean,
                    abundance.mean * factor,
                    factor,
                    ratio,
                )
            else:
                new_enzymes[tag] = abundance

        if modified:
            new_node = dataclasses.replace(node, enzymes=new_enzymes)
            new_graph.nodes[name] = new_node
        else:
            new_graph.nodes[name] = node

    new_graph.edges = list(graph.edges)
    return new_graph


def apply_induction(graph: BodyGraph, inducer: Inducer) -> BodyGraph:
    """Apply CYP enzyme induction to a BodyGraph.

    For each node with affected enzymes, increases abundances by::

        abundance_new = abundance_base * (1 + Emax * [I] / (EC50 + [I]))

    Returns a NEW BodyGraph (does not mutate the original).
    Engine sees increased enzyme abundances -> higher clearance -> lower Cmax.

    Args:
        graph: Base BodyGraph.
        inducer: Inducer specification.

    Returns:
        Modified BodyGraph with increased enzyme abundances.
    """
    new_graph = BodyGraph()
    new_graph.global_params = dict(graph.global_params)

    for name, node in graph.nodes.items():
        if not node.enzymes:
            new_graph.nodes[name] = node
            continue

        new_enzymes: dict[str, Distribution] = {}
        modified = False

        for tag, abundance in node.enzymes.items():
            if tag in inducer.enzyme_emax and tag in inducer.enzyme_ec50:
                emax = inducer.enzyme_emax[tag]
                ec50 = inducer.enzyme_ec50[tag]
                conc = inducer.plasma_concentration

                if ec50 <= 0:
                    raise ValueError(f"EC50 must be positive for enzyme {tag!r}, got {ec50}")

                # Emax model induction
                fold = 1.0 + emax * conc / (ec50 + conc)
                new_enzymes[tag] = Distribution(
                    mean=abundance.mean * fold,
                    cv=abundance.cv,
                )
                modified = True
                logger.debug(
                    "%s: %s abundance %.0f -> %.0f (fold=%.2f)",
                    name,
                    tag,
                    abundance.mean,
                    abundance.mean * fold,
                    fold,
                )
            else:
                new_enzymes[tag] = abundance

        if modified:
            new_node = dataclasses.replace(node, enzymes=new_enzymes)
            new_graph.nodes[name] = new_node
        else:
            new_graph.nodes[name] = node

    new_graph.edges = list(graph.edges)
    return new_graph


# ---------------------------------------------------------------------------
# Preset inhibitors and inducers (clinical DDI reference values)
# ---------------------------------------------------------------------------

# Sources:
#   FDA DDI Guidance (2020), Table 6: Clinical Index Inhibitors/Inducers
#   Fahmi et al. (2009) Drug Metab Dispos 37:1658-1666
#   Obach et al. (2006) Drug Metab Dispos 34:1-8

KETOCONAZOLE = Inhibitor(
    name="ketoconazole",
    enzyme_ki={"CYP3A4": 0.015},  # Ki = 15 nM = 0.015 uM (potent)
    plasma_concentration=5.0,  # ~5 uM at 200 mg dose (Cmax ~10 ug/mL, MW=531)
)

FLUCONAZOLE = Inhibitor(
    name="fluconazole",
    enzyme_ki={"CYP2C9": 7.0, "CYP3A4": 10.0},  # moderate inhibitor
    plasma_concentration=25.0,  # ~25 uM at 200 mg dose (Cmax ~8 ug/mL, MW=306)
)

QUINIDINE = Inhibitor(
    name="quinidine",
    enzyme_ki={"CYP2D6": 0.06},  # Ki = 60 nM = 0.06 uM (potent 2D6 inhibitor)
    plasma_concentration=3.0,  # ~3 uM
)

RIFAMPIN = Inducer(
    name="rifampin",
    enzyme_emax={"CYP3A4": 8.0, "CYP2C9": 3.0},  # up to 8x induction for 3A4
    enzyme_ec50={"CYP3A4": 0.3, "CYP2C9": 1.0},  # uM
    plasma_concentration=10.0,  # ~10 uM at 600 mg dose (Cmax ~8 ug/mL, MW=823)
)
