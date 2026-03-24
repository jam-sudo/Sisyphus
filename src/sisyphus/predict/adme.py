"""ADME property prediction.

Predicts absorption, distribution, metabolism, and excretion
properties from molecular descriptors.  All outputs are
Distributions carrying prediction uncertainty.

Uses trained XGBoost models for fup, CLint, RBP, VDss.
Uses heuristic relationships for Peff and solubility.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xgboost as xgb

from sisyphus.core import Distribution
from sisyphus.descriptors import compute_features
from sisyphus.predict.chemistry import MolecularProfile

logger = logging.getLogger(__name__)

# Resolve model directory relative to this source file:
# src/sisyphus/predict/adme.py -> ../../../../models/adme
_MODEL_DIR = Path(__file__).resolve().parent.parent.parent.parent / "models" / "adme"

# ---------------------------------------------------------------------------
# Default CVs for prediction uncertainty (from Omega conformal intervals)
# ---------------------------------------------------------------------------
_FUP_CV = 0.5  # fup prediction has ~50% CV
_CLINT_CV = 1.0  # CLint prediction is the weakest link (R²=0.24)
_PEFF_CV = 0.4  # permeability
_SOLUBILITY_CV = 0.5  # solubility
_RBP_CV = 0.3  # blood:plasma ratio
_VDSS_CV = 0.5  # volume of distribution


@dataclass(frozen=True)
class ADMEProperties:
    """Predicted ADME properties, all as Distributions.

    Attributes:
        fup: Fraction unbound in plasma.
        clint: Intrinsic clearance (uL/min/10^6 cells).
        peff: Effective permeability (x10^-4 cm/s).
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


# ---------------------------------------------------------------------------
# Model cache
# ---------------------------------------------------------------------------
_model_cache: dict[str, xgb.XGBRegressor] = {}


def _load_model(filename: str) -> xgb.XGBRegressor:
    """Load an XGBoost model from the models/adme directory, with caching."""
    if filename not in _model_cache:
        path = _MODEL_DIR / filename
        if not path.exists():
            raise FileNotFoundError(
                f"ADME model not found: {path}. Expected models in {_MODEL_DIR}"
            )
        model = xgb.XGBRegressor()
        model.load_model(str(path))
        _model_cache[filename] = model
        logger.debug("Loaded ADME model: %s", path)
    return _model_cache[filename]


# ---------------------------------------------------------------------------
# Individual ADME predictors
# ---------------------------------------------------------------------------


def _predict_fup(features: np.ndarray) -> Distribution:
    """Predict fraction unbound in plasma from 2057-element feature vector.

    Model predicts log10(fup).  Output clamped to [0.001, 1.0].
    """
    model = _load_model("xgboost_fup.json")
    log_fup = float(model.predict(features)[0])
    fup = float(np.clip(10**log_fup, 0.001, 1.0))
    return Distribution(mean=fup, cv=_FUP_CV)


def _predict_clint(features: np.ndarray) -> Distribution:
    """Predict hepatocyte intrinsic clearance from 2057-element feature vector.

    Model predicts log10(CLint) in uL/min/10^6 cells.
    Output floored at 0.1 uL/min/10^6 cells.
    """
    model = _load_model("xgboost_clint.json")
    log_clint = float(model.predict(features)[0])
    clint = max(10**log_clint, 0.1)
    return Distribution(mean=clint, cv=_CLINT_CV)


def _predict_rbp(features: np.ndarray) -> Distribution:
    """Predict blood:plasma ratio from Morgan FP only (2048 features).

    Model predicts log10(RBP).  Output clamped to [0.5, 3.0].
    RBP prediction is poor (R^2=-0.08 on 50 compounds).  Predictions
    far from 1.0 are reset to 1.0 as a safety measure.
    """
    model = _load_model("xgboost_rbp.json")
    log_rbp = float(model.predict(features)[0])
    rbp = float(np.clip(10**log_rbp, 0.5, 3.0))
    # RBP prediction is poor — default to 1.0 if unreasonable
    if abs(rbp - 1.0) > 0.5:
        logger.warning("RBP prediction %.2f far from 1.0, defaulting to 1.0", rbp)
        rbp = 1.0
    return Distribution(mean=rbp, cv=_RBP_CV)


def _predict_vdss(features: np.ndarray) -> Distribution:
    """Predict volume of distribution at steady state from 2057-element feature vector.

    Model predicts log10(VDss) in L/kg.  Output floored at 0.01 L/kg.
    """
    model = _load_model("xgboost_vdss.json")
    log_vdss = float(model.predict(features)[0])
    vdss = max(10**log_vdss, 0.01)
    return Distribution(mean=vdss, cv=_VDSS_CV)


def _estimate_peff(profile: MolecularProfile) -> Distribution:
    """Estimate effective permeability from LogP using sigmoidal relationship.

    Peff (x10^-4 cm/s) ~ 10^(0.4 * logP - 0.4) for passive diffusion.
    Capped to [0.1, 50.0].

    Source: Sun et al. (2004) AAPS PharmSciTech, adapted.
    """
    peff = 10 ** (0.4 * profile.logp - 0.4)
    peff = float(np.clip(peff, 0.1, 50.0))
    return Distribution(mean=peff, cv=_PEFF_CV)


def _estimate_solubility(profile: MolecularProfile) -> Distribution:
    """Estimate aqueous solubility from LogP.

    log10(S_mg_mL) ~ -logP + 0.5  (rough Yalkowsky-type estimate).
    Capped to [0.001, 100.0] mg/mL.

    Source: Yalkowsky & Valvani (1980), simplified.
    """
    log_s = -profile.logp + 0.5
    s = max(10**log_s, 0.001)
    s = min(s, 100.0)
    return Distribution(mean=float(s), cv=_SOLUBILITY_CV)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def predict_adme(profile: MolecularProfile) -> ADMEProperties:
    """Predict ADME properties from molecular profile.

    Uses trained XGBoost models (fup, CLint, RBP, VDss) with prediction
    intervals derived from conformal calibration.  Peff and solubility
    are estimated from logP heuristics (no XGBoost model available).

    Args:
        profile: MolecularProfile from chemistry module.

    Returns:
        ADMEProperties with uncertainty (all fields are Distributions).
    """
    features = compute_features(profile.smiles)
    features_2d = features.reshape(1, -1)

    # fup: DrugBank measured → XGBoost fallback
    from sisyphus.predict.drugbank import drugbank_lookup
    db = drugbank_lookup()
    db_fup = db.get_fup(profile.smiles)
    if db_fup is not None and 0.001 <= db_fup <= 1.0:
        xgb_fup = _predict_fup(features_2d)
        if xgb_fup.mean > 0 and (db_fup / xgb_fup.mean > 5.0 or xgb_fup.mean / db_fup > 5.0):
            logger.warning(
                "DrugBank fup (%.3f) disagrees with XGBoost (%.3f) by >5x, using XGBoost",
                db_fup, xgb_fup.mean,
            )
            fup = xgb_fup
        else:
            fup = Distribution(mean=db_fup, cv=0.20)
            logger.info("Using DrugBank measured fup=%.3f", db_fup)
    else:
        fup = _predict_fup(features_2d)

    clint = _predict_clint(features_2d)
    rbp = _predict_rbp(features_2d[:, :2048])  # RBP model uses only Morgan FP
    vdss = _predict_vdss(features_2d)

    # Heuristic estimates (no trained models)
    peff = _estimate_peff(profile)
    solubility = _estimate_solubility(profile)

    logger.info(
        "ADME predicted: fup=%.3f, CLint=%.1f uL/min/10^6, Peff=%.2f, "
        "solubility=%.3f mg/mL, VDss=%.2f L/kg, RBP=%.2f",
        fup.mean,
        clint.mean,
        peff.mean,
        solubility.mean,
        vdss.mean,
        rbp.mean,
    )

    return ADMEProperties(
        fup=fup,
        clint=clint,
        peff=peff,
        solubility=solubility,
        vdss=vdss,
        rbp=rbp,
    )
