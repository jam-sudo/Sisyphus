"""CL/F-based Cmax predictor (3rd track for meta-learner).

Bypasses the IVIVE chain entirely. Predicts oral clearance (CL/F)
and volume of distribution (Vd/F) directly from SMILES, then computes
Cmax via analytical 1-compartment oral PK model.

CL/F model: XGBoost, log10(CL/F in mL/min/kg), trained on MMPK AUC data.
Vd/F model: XGBoost, log10(Vd/F in L/kg), trained on MMPK t½ data.

Analytical Cmax:
    ke = CL_over_F / Vd_over_F  (same F cancels)
    tmax = ln(ka/ke) / (ka - ke)
    Cmax = (Dose * ka) / (Vd_over_F_L * (ka - ke)) * (exp(-ke*tmax) - exp(-ka*tmax))

ka determination (in priority order):
    1. Engine Tmax → numerical inversion of tmax = ln(ka/ke)/(ka-ke)
    2. Peff → ka = 2 * Peff / intestinal_radius (1.75 cm)
    3. Default ka = 1.0 /h (population mean)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import xgboost as xgb

from sisyphus.core import Distribution
from sisyphus.descriptors import compute_features

logger = logging.getLogger(__name__)

_MODEL_DIR = Path(__file__).resolve().parent.parent.parent.parent / "models" / "direct_pk"

# Body weight assumption for unit conversion
_BW_KG = 70.0

# Intestinal radius for Peff → ka conversion (cm)
_INTESTINAL_RADIUS_CM = 1.75

# Default population ka when no other method works
_DEFAULT_KA = 1.0  # /h

# Minimum separation between ka and ke to avoid numerical issues
_KA_KE_MIN_RATIO = 1.05


class CLFPredictor:
    """Predicts Cmax from SMILES via CL/F + Vd/F + analytical 1-cpt model.

    Uses pre-trained XGBoost models for CL/F and Vd/F prediction,
    then computes Cmax from the 1-compartment oral PK equation.
    """

    def __init__(self) -> None:
        self._clf_model: xgb.XGBRegressor | None = None
        self._vdf_model: xgb.XGBRegressor | None = None

    def _ensure_loaded(self) -> None:
        if self._clf_model is None:
            path = _MODEL_DIR / "xgboost_clf.json"
            self._clf_model = xgb.XGBRegressor()
            self._clf_model.load_model(str(path))
            logger.info("CL/F model loaded from %s", path)
        if self._vdf_model is None:
            path = _MODEL_DIR / "xgboost_vdf.json"
            self._vdf_model = xgb.XGBRegressor()
            self._vdf_model.load_model(str(path))
            logger.info("Vd/F model loaded from %s", path)

    def predict_clf_vdf(self, smiles: str) -> tuple[float, float]:
        """Predict CL/F (L/h) and Vd/F (L) from SMILES.

        Returns:
            (clf_L_h, vdf_L): CL/F in L/h and Vd/F in L (for BW=70 kg).
        """
        self._ensure_loaded()
        features = compute_features(smiles).reshape(1, -1)

        # CL/F: model predicts log10(CL/F in mL/min/kg)
        log10_clf = float(self._clf_model.predict(features)[0])  # type: ignore[union-attr]
        clf_mL_min_kg = 10**log10_clf

        # mL/min/kg → L/h: × BW × 60 / 1000
        clf_L_h = clf_mL_min_kg * _BW_KG * 60.0 / 1000.0

        # Vd/F: model predicts log10(Vd/F in L/kg)
        log10_vdf = float(self._vdf_model.predict(features)[0])  # type: ignore[union-attr]
        vdf_L_kg = 10**log10_vdf

        # L/kg → L: × BW
        vdf_L = vdf_L_kg * _BW_KG

        return clf_L_h, vdf_L

    def _determine_ka(
        self,
        ke: float,
        engine_tmax: float | None = None,
        peff: float | None = None,
    ) -> tuple[float, str]:
        """Determine absorption rate constant (ka) using priority methods.

        Args:
            ke: Elimination rate constant (/h).
            engine_tmax: Tmax from engine simulation (h), if available.
            peff: Effective permeability (×10⁻⁴ cm/s), if available.

        Returns:
            (ka, method): ka in /h and method name used.
        """
        # Method 1: Engine Tmax → numerical inversion
        # tmax = ln(ka/ke) / (ka - ke)
        # Solve for ka given tmax and ke using Newton's method
        if engine_tmax is not None and engine_tmax > 0:
            ka = self._ka_from_tmax(ke, engine_tmax)
            if ka is not None and ka > ke * _KA_KE_MIN_RATIO:
                return ka, "engine_tmax"

        # Method 2: Peff → ka
        # ka = 2 * Peff / R, where Peff in cm/s and R in cm
        # Peff is in ×10⁻⁴ cm/s, so multiply by 1e-4
        # ka in /s → /h: × 3600
        if peff is not None and peff > 0:
            peff_cm_s = peff * 1e-4  # ×10⁻⁴ cm/s → cm/s
            ka = 2.0 * peff_cm_s / _INTESTINAL_RADIUS_CM * 3600.0  # /h
            if ka > ke * _KA_KE_MIN_RATIO:
                return ka, "peff"

        # Method 3: Default population ka
        ka = _DEFAULT_KA
        # Ensure ka > ke
        if ka <= ke * _KA_KE_MIN_RATIO:
            ka = ke * 2.0  # force flip-flop avoidance
        return ka, "default"

    def _ka_from_tmax(self, ke: float, tmax: float, max_iter: int = 50) -> float | None:
        """Solve tmax = ln(ka/ke) / (ka - ke) for ka using Newton's method.

        Args:
            ke: Elimination rate constant (/h).
            tmax: Target Tmax (h).
            max_iter: Maximum Newton iterations.

        Returns:
            ka (/h) or None if convergence fails.
        """
        if tmax <= 0 or ke <= 0:
            return None

        # Initial guess: ka = ln(2) / tmax (rough heuristic)
        ka = max(np.log(2) / tmax, ke * 1.5)

        for _ in range(max_iter):
            if ka <= ke:
                ka = ke * 1.5  # reset
                continue

            diff = ka - ke
            if abs(diff) < 1e-12:
                return None

            # f(ka) = ln(ka/ke) / (ka - ke) - tmax
            f_val = np.log(ka / ke) / diff - tmax

            # f'(ka) = [1/ka * (ka-ke) - ln(ka/ke)] / (ka-ke)²
            f_prime = (1.0 / ka * diff - np.log(ka / ke)) / (diff**2)

            if abs(f_prime) < 1e-15:
                return None

            ka_new = ka - f_val / f_prime

            if ka_new <= ke:
                ka = (ka + ke * 1.5) / 2  # bisect toward feasible
                continue

            if abs(ka_new - ka) < 1e-8:
                return float(ka_new)
            ka = ka_new

        return None  # did not converge

    def predict_cmax(
        self,
        smiles: str,
        dose_mg: float,
        engine_tmax: float | None = None,
        peff: float | None = None,
    ) -> tuple[Distribution, str]:
        """Predict Cmax using CL/F analytical track.

        Args:
            smiles: SMILES string.
            dose_mg: Dose in mg.
            engine_tmax: Tmax from engine (h), used for ka estimation.
            peff: Effective permeability (×10⁻⁴ cm/s), used for ka estimation.

        Returns:
            (cmax_distribution, ka_method): Cmax as Distribution and method used for ka.
        """
        clf_L_h, vdf_L = self.predict_clf_vdf(smiles)

        # Clamp to physiological ranges
        clf_L_h = max(clf_L_h, 0.01)  # minimum clearance
        vdf_L = max(vdf_L, 1.0)  # minimum 1L volume

        # ke = CL/F / Vd/F (same F cancels)
        ke = clf_L_h / vdf_L  # /h

        # Determine ka
        ka, ka_method = self._determine_ka(ke, engine_tmax, peff)

        # Analytical 1-compartment oral Cmax
        # C(t) = (Dose * ka) / (Vd * (ka - ke)) * (exp(-ke*t) - exp(-ka*t))
        # tmax = ln(ka/ke) / (ka - ke)
        diff = ka - ke
        if abs(diff) < 1e-12:
            # Degenerate case
            tmax_calc = 1.0 / ke
            cmax = dose_mg / (vdf_L * np.e) * ke * tmax_calc * np.exp(-ke * tmax_calc)
        else:
            tmax_calc = np.log(ka / ke) / diff
            cmax = (dose_mg * ka) / (vdf_L * diff) * (
                np.exp(-ke * tmax_calc) - np.exp(-ka * tmax_calc)
            )

        cmax = max(float(cmax), 1e-10)  # floor

        logger.debug(
            "CLF track: CL/F=%.2f L/h, Vd/F=%.1f L, ke=%.4f, ka=%.4f (%s), "
            "tmax=%.2f h, Cmax=%.4f mg/L",
            clf_L_h, vdf_L, ke, ka, ka_method, tmax_calc, cmax,
        )

        return Distribution(mean=cmax, cv=0.5), ka_method
