"""ADME property prediction.

Predicts absorption, distribution, metabolism, and excretion
properties from molecular descriptors.  All outputs are
Distributions carrying prediction uncertainty.
"""

from __future__ import annotations

from dataclasses import dataclass

from sisyphus.core import Distribution
from sisyphus.predict.chemistry import MolecularProfile


@dataclass(frozen=True)
class ADMEProperties:
    """Predicted ADME properties, all as Distributions.

    Attributes:
        fup: Fraction unbound in plasma.
        clint: Intrinsic clearance (µL/min/10⁶ cells).
        peff: Effective permeability (×10⁻⁴ cm/s).
        solubility: Aqueous solubility (mg/mL).
        vdss: Volume of distribution at steady state (L/kg).
        rbp: Blood:plasma ratio.
    """

    fup: Distribution
    clint: Distribution
    peff: Distribution
    solubility: Distribution
    vdss: Distribution
    rbp: Distribution


def predict_adme(profile: MolecularProfile) -> ADMEProperties:
    """Predict ADME properties from molecular profile.

    Uses trained ML models (XGBoost) with prediction intervals
    derived from model uncertainty + training data variability.

    Args:
        profile: MolecularProfile from chemistry module.

    Returns:
        ADMEProperties with uncertainty.
    """
    raise NotImplementedError
