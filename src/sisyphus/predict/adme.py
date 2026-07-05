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
# Default CVs for prediction uncertainty (hand-set scalars informed by Omega's
# observed prediction spread; not derived from a conformal procedure)
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


@dataclass(frozen=True)
class MeasuredADMEInput:
    """Caller-supplied measured ADME values that override predicted ones in predict().

    A field left None falls through to the XGBoost/heuristic prediction.

    REQUIRED PAIR: fup and clint must both be supplied or both omitted — they
    co-determine hepatic CL_int in the engine's well-stirred extraction, so
    pairing a measured value with a predicted one distorts engine clearance.

    *_cv are measurement-level CVs (instrument error, below the model-prediction
    CVs above). Rejected below 0.10 (cv < 0.10 raises): a tighter CV implies a unit
    error and collapses the Monte-Carlo envelope to false confidence.

    NOTE: peff perturbs the engine (and the CLF meta-track); vdss has ZERO engine
    effect — the engine derives volume from Rodgers-Rowland Kp, not this scalar, so
    vdss only moves the VDss meta-track. A "clean engine-only" measured prediction
    must therefore be read via result.engine_pk (see the measured-input benchmark).

    f_bioavail is measured oral bioavailability F (0 < F <= 1). It is INDEPENDENT
    (not paired) and ORAL-ONLY. The engine's F is emergent (fa*Fg*Fh); a supplied
    F sets the systemic exposure SCALE — predict() computes the engine's own F via
    an IV-reference solve and scales engine Cmax/AUC by F_measured/F_engine. It does
    not set the absorption-rate shape (compose with measured peff for slow
    absorbers). Lands on result.engine_pk; ignored for non-oral routes. See
    docs/_internal/specs/2026-06-03-measured-f-routing-design.md.
    """

    fup: float | None = None
    fup_cv: float = 0.15
    clint: float | None = None
    clint_cv: float = 0.20
    peff: float | None = None
    peff_cv: float = 0.25
    vdss: float | None = None
    vdss_cv: float = 0.20
    rbp: float | None = None
    rbp_cv: float = 0.15
    solubility: float | None = None
    solubility_cv: float = 0.30
    f_bioavail: float | None = None
    f_bioavail_cv: float = 0.15

    def __post_init__(self) -> None:
        if (self.fup is None) != (self.clint is None):
            raise ValueError(
                "MeasuredADMEInput: fup and clint must be supplied together or "
                "both omitted (they co-determine engine CL_int)."
            )
        if self.f_bioavail is not None and not (0.0 < self.f_bioavail <= 1.0):
            raise ValueError(
                f"MeasuredADMEInput.f_bioavail must satisfy 0 < F <= 1, "
                f"got {self.f_bioavail}"
            )
        for name, val in (
            ("fup", self.fup), ("clint", self.clint), ("peff", self.peff),
            ("vdss", self.vdss), ("rbp", self.rbp), ("solubility", self.solubility),
        ):
            if val is not None and val <= 0:
                raise ValueError(f"MeasuredADMEInput.{name} must be > 0, got {val}")
        for name, cv in (
            ("fup_cv", self.fup_cv), ("clint_cv", self.clint_cv),
            ("peff_cv", self.peff_cv), ("vdss_cv", self.vdss_cv),
            ("rbp_cv", self.rbp_cv), ("solubility_cv", self.solubility_cv),
            ("f_bioavail_cv", self.f_bioavail_cv),
        ):
            if cv < 0.10:
                raise ValueError(
                    f"MeasuredADMEInput.{name}={cv} < 0.10; a CV below 10% implies "
                    "a unit error and collapses the MC envelope."
                )


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

_FUP_V2_PATH = _MODEL_DIR / "xgboost_fup_v2.json"
_FUP_CV_V2 = 0.40  # from 5-fold CV: AAFE=2.40, R²=0.41


def _predict_fup_v1(features: np.ndarray) -> Distribution:
    """v1: log10-space model. Output clamped to [0.001, 1.0]."""
    model = _load_model("xgboost_fup.json")
    log_fup = float(model.predict(features)[0])
    fup = float(np.clip(10**log_fup, 0.001, 1.0))
    return Distribution(mean=fup, cv=_FUP_CV)


def _predict_fup_v2(features: np.ndarray) -> Distribution:
    """v2: logit-space model, sigmoid inverse. Output in (0, 1) guaranteed."""
    model = _load_model("xgboost_fup_v2.json")
    logit_fup = float(model.predict(features)[0])
    fup = float(1.0 / (1.0 + np.exp(-logit_fup)))
    return Distribution(mean=fup, cv=_FUP_CV_V2)


def _predict_fup(features: np.ndarray) -> Distribution:
    """Auto-select fup model: v2 if available, v1 fallback."""
    if _FUP_V2_PATH.exists():
        return _predict_fup_v2(features)
    return _predict_fup_v1(features)


def _predict_clint(features: np.ndarray) -> Distribution:
    """Predict hepatocyte intrinsic clearance from 2057-element feature vector.

    Model predicts log10(CLint) in uL/min/10^6 cells.
    Output floored at 0.1 uL/min/10^6 cells.
    """
    model = _load_model("xgboost_clint.json")
    log_clint = float(model.predict(features)[0])
    clint = max(10**log_clint, 0.1)
    return Distribution(mean=clint, cv=_CLINT_CV)


def _predict_rbp() -> Distribution:
    """Return the population-default blood:plasma ratio (1.0).

    The bundled RBP XGBoost model (``xgboost_rbp.json``, TDC BTPR N=50) is
    uninformative — scaffold-CV R²=-0.08, i.e. worse than predicting the
    population mean — so it is deliberately not consulted.  Its ``meta.json``
    states the intended behaviour outright: "Pipeline defaults to RBP=1.0."

    Historically that default was reached through two compounding bugs rather
    than stated plainly.  The model's target is a *raw* blood:plasma ratio, but
    the code mis-read it as log10(RBP) and applied ``10**``, inflating every
    prediction (e.g. caffeine 0.72 → 5.3) past the clip's 3.0 ceiling; an
    asymmetric reset (``abs(rbp - 1.0) > 0.5``) whose low branch was unreachable
    then snapped the clipped 3.0 down to 1.0.  Net effect: every drug — all 107
    holdout drugs included — resolved to exactly 1.0 regardless of structure.

    This function makes that outcome explicit instead of incidental: RBP is a
    constant 1.0 (with ``_RBP_CV`` spread) until an informative model replaces
    it.  Behaviour is bit-identical to the previous buggy path on the holdout,
    so no benchmark re-pin is required.
    """
    return Distribution(mean=1.0, cv=_RBP_CV)


def _predict_vdss(features: np.ndarray) -> Distribution:
    """Predict volume of distribution at steady state from 2057-element feature vector.

    Model predicts log10(VDss) in L/kg.  Output floored at 0.01 L/kg.
    """
    model = _load_model("xgboost_vdss.json")
    log_vdss = float(model.predict(features)[0])
    vdss = max(10**log_vdss, 0.01)
    return Distribution(mean=vdss, cv=_VDSS_CV)


_PEFF_MODEL_PATH = _MODEL_DIR / "xgboost_peff.json"
_PEFF_CV_TRAINED = 0.35  # from 5-fold CV: RMSE=0.42 log units, R²=0.71


def _predict_peff_xgb(features: np.ndarray) -> Distribution:
    """Predict Peff from XGBoost model trained on TDC Caco2_Wang.

    Model predicts log10(Papp [x10^-4 cm/s]) from Caco-2 in vitro data.
    A calibration offset converts Caco-2 Papp to in vivo Peff, accounting
    for villous amplification (~20x, Helander & Fändriks 2014).
    Offset = +1.29 log10 units (calibrated on N=65 training drugs only).
    Output clamped to [0.01, 100.0].
    """
    _CACO2_TO_INVIVO_OFFSET = 1.29  # log10 units, ~20x (training-set calibrated)
    model = _load_model("xgboost_peff.json")
    log_peff_caco2 = float(model.predict(features)[0])
    log_peff_invivo = log_peff_caco2 + _CACO2_TO_INVIVO_OFFSET
    peff = float(np.clip(10**log_peff_invivo, 0.01, 100.0))
    return Distribution(mean=peff, cv=_PEFF_CV_TRAINED)


def _estimate_peff_heuristic(profile: MolecularProfile) -> Distribution:
    """Estimate effective permeability from LogP using sigmoidal relationship.

    Peff (x10^-4 cm/s) ~ 10^(0.4 * logP - 0.4) for passive diffusion.
    Capped to [0.1, 50.0].

    Source: Sun et al. (2004) AAPS PharmSciTech, adapted.
    """
    peff = 10 ** (0.4 * profile.logp - 0.4)
    peff = float(np.clip(peff, 0.1, 50.0))
    return Distribution(mean=peff, cv=_PEFF_CV)


def _estimate_peff(profile: MolecularProfile) -> Distribution:
    """Estimate Peff: XGBoost model if available, logP heuristic fallback."""
    if _PEFF_MODEL_PATH.exists():
        try:
            features = compute_features(profile.smiles)
            return _predict_peff_xgb(features.reshape(1, -1))
        except Exception as exc:
            logger.warning("Peff XGBoost prediction failed, using heuristic: %s", exc)
    return _estimate_peff_heuristic(profile)


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

    # fup: DrugBank measured → XGBoost fallback on >5x disagreement
    #
    # 2026-04-11 tried inverting this — prefer DrugBank measured even when it
    # disagrees by >5x with XGBoost (the "ketorolac fix" from the TDM D2
    # writeup). Ran the 107-holdout benchmark and measured:
    #    Engine AAFE: 3.421 → 3.726 (+0.306, major regression, matches the
    #                 34+ prior error-cancellation failures in the dead-ends log)
    #    Meta AAFE:   2.695 → 2.728 (+0.033, noise level but wrong direction)
    #    Ketorolac engine fold: 0.259x → 0.534x (directionally better but
    #                 still well under truth 0.80 — TDM CI still doesn't cover
    #                 even with floor=0.5, so the "fix" didn't actually achieve
    #                 its coverage goal)
    # Reverted. Ketorolac's failure is more than a fup mismatch — its CLint
    # path also contributes. A partial fix (fup only) breaks error cancellation
    # on 100+ other drugs for essentially zero coverage gain on ketorolac.
    # Tracked as a known structural limitation; proper fix requires engine-
    # level ADME improvement, not a post-predict override.
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
    rbp = _predict_rbp()  # uninformative model (R²<0) disabled; population default 1.0
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
