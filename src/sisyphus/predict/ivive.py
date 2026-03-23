"""In Vitro to In Vivo Extrapolation.

Translates ADME properties into DrugOnGraph parameters.
This is where global CLint is decomposed into per-enzyme affinities
based on known metabolic pathway fractions.
"""

from __future__ import annotations

from sisyphus.core import DrugOnGraph
from sisyphus.predict.adme import ADMEProperties
from sisyphus.predict.chemistry import MolecularProfile


def build_drug_on_graph(
    profile: MolecularProfile,
    adme: ADMEProperties,
    dose_mg: float,
    route: str = "oral",
) -> DrugOnGraph:
    """Construct a DrugOnGraph from predicted properties.

    Converts global CLint into per-enzyme affinities using
    CYP contribution fractions.  Maps absorption parameters.
    Sets administration_node based on route.

    Args:
        profile: Molecular profile (physicochemical properties).
        adme: Predicted ADME properties.
        dose_mg: Dose in mg.
        route: Administration route (``"oral"`` or ``"iv"``).

    Returns:
        A fully parameterized DrugOnGraph ready for the engine.
    """
    raise NotImplementedError
