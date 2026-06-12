"""Patient covariates that deterministically individualize the engine prior.

v1: renal function only — a measured creatinine clearance (CrCl). CrCl scales
the drug's renal (glomerular-filtration) clearance: the engine's reference renal
model is ``CL_renal = GFR*fup`` with GFR = 7.5 L/h (~125 mL/min), so an
individual's renal CL is scaled by ``CrCl / 125``. Weight/age covariates (via
sbi.physiology_generator) are a documented future extension — see the design spec
docs/superpowers/specs/2026-06-11-mipd-crcl-renal-individualization-design.md.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Reference glomerular filtration rate the engine's renal_clearance assumes
# (_GFR_L_PER_H = 7.5 L/h ~= 125 mL/min; src/sisyphus/predict/ivive.py:42-43).
_REFERENCE_GFR_ML_MIN = 125.0

# Cockcroft & Gault, Nephron 1976: CrCl = (140-age)*wt*(0.85 if female)/(72*SCr_mg/dL).
_CG_AGE_CONSTANT = 140.0
_CG_SCR_DENOMINATOR_DL = 72.0
_CG_FEMALE_FACTOR = 0.85
_CG_MAX_AGE_YEARS = 140.0          # validation floor: 140 - age must stay > 0
_CG_PEDIATRIC_AGE_YEARS = 18.0     # CG is not validated below this age
_CG_LOW_SCR_MG_DL = 0.6            # low SCr -> CrCl overestimation risk (low muscle mass)


def cockcroft_gault(
    age_years: float, weight_kg: float, scr_mg_dl: float, sex: str
) -> float:
    """Estimate creatinine clearance (mL/min) by the Cockcroft-Gault equation.

    ``CrCl = (140 - age) * weight_kg * (0.85 if female else 1.0) / (72 * SCr_mg/dL)``.
    ``scr_mg_dl`` is serum creatinine in mg/dL (NOT SI µmol/L; divide µmol/L by 88.42).
    ``sex`` is "male"/"female" (case-insensitive). Returns an ABSOLUTE CrCl in mL/min
    (body size enters via the weight term) — the quantity ``Covariates.crcl_ml_min``
    expects; do not pass a BSA-normalized eGFR there instead. With actual body weight,
    CG overestimates CrCl in obese patients (adjusted body weight needs height — future).

    Raises ``ValueError`` on non-finite, non-positive, or out-of-range inputs. Logs a
    (non-blocking) warning for age < 18 (CG unvalidated) and for very low SCr
    (overestimation in reduced muscle mass).
    """
    for _name, _value in (
        ("age_years", age_years), ("weight_kg", weight_kg), ("scr_mg_dl", scr_mg_dl)
    ):
        if not math.isfinite(_value):
            raise ValueError(f"{_name} must be finite, got {_value}")
    if not isinstance(sex, str):
        raise ValueError(f"sex must be 'male' or 'female', got {sex!r}")
    sex_l = sex.strip().lower()
    if sex_l not in ("male", "female"):
        raise ValueError(f"sex must be 'male' or 'female', got {sex!r}")
    if age_years <= 0 or age_years >= _CG_MAX_AGE_YEARS:
        raise ValueError(
            f"age_years must be in (0, {_CG_MAX_AGE_YEARS}), got {age_years}"
        )
    if weight_kg <= 0:
        raise ValueError(f"weight_kg must be > 0, got {weight_kg}")
    if scr_mg_dl <= 0:
        raise ValueError(f"scr_mg_dl must be > 0, got {scr_mg_dl}")

    if age_years < _CG_PEDIATRIC_AGE_YEARS:
        logger.warning(
            "Cockcroft-Gault is not validated for age < %s (got %s); consider Schwartz.",
            _CG_PEDIATRIC_AGE_YEARS, age_years,
        )
    if scr_mg_dl < _CG_LOW_SCR_MG_DL:
        logger.warning(
            "serum creatinine %s mg/dL is low; in patients with reduced muscle mass "
            "(elderly/cachectic) Cockcroft-Gault may overestimate CrCl.", scr_mg_dl,
        )

    crcl = (
        (_CG_AGE_CONSTANT - age_years) * weight_kg
        / (_CG_SCR_DENOMINATOR_DL * scr_mg_dl)
    )
    if sex_l == "female":
        crcl *= _CG_FEMALE_FACTOR
    return crcl


@dataclass(frozen=True)
class Covariates:
    """Deterministic patient covariates for the engine-as-prior individualization.

    Attributes:
        crcl_ml_min: measured creatinine clearance (mL/min). None -> no renal
            individualization (renal_factor 1.0).
    """

    crcl_ml_min: float | None = None
    body_weight_kg: float | None = None
    age_years: float | None = None

    def __post_init__(self) -> None:
        if self.crcl_ml_min is not None and self.crcl_ml_min <= 0:
            raise ValueError(f"crcl_ml_min must be > 0, got {self.crcl_ml_min}")
        if self.body_weight_kg is not None and self.body_weight_kg <= 0:
            raise ValueError(f"body_weight_kg must be > 0, got {self.body_weight_kg}")
        if self.age_years is not None and self.age_years <= 0:
            raise ValueError(f"age_years must be > 0, got {self.age_years}")

    def renal_factor(self) -> float:
        """Multiplicative scale for ``drug.renal_clearance`` (CrCl-only; 1.0 at CrCl=125).

        Weight/age never affect this — renal individualization is measured-CrCl-only
        (estimating GFR from age/weight is deferred; see the design spec).
        """
        if self.crcl_ml_min is None:
            return 1.0
        return self.crcl_ml_min / _REFERENCE_GFR_ML_MIN

    def has_physiology(self) -> bool:
        """True iff weight or age is set — triggers generate_physiology graph build."""
        return self.body_weight_kg is not None or self.age_years is not None

    def warnings(self) -> tuple[str, ...]:
        """Structured flags for physiologically extreme covariates (extrapolation risk)."""
        w: list[str] = []
        if self.crcl_ml_min is not None and not (5.0 <= self.crcl_ml_min <= 200.0):
            w.append(
                f"crcl:extreme:{self.crcl_ml_min}: the engine renal model is "
                "glomerular-filtration-only and least reliable outside [5, 200] mL/min"
            )
        if self.body_weight_kg is not None and not (2.0 <= self.body_weight_kg <= 250.0):
            w.append(
                f"weight:extreme:{self.body_weight_kg}: allometric/ontogeny scaling "
                "extrapolates poorly outside [2, 250] kg"
            )
        if self.age_years is not None and not (0.0 < self.age_years <= 100.0):
            w.append(
                f"age:extreme:{self.age_years}: ontogeny/aging scaling extrapolates "
                "poorly outside (0, 100] yr"
            )
        return tuple(w)

    @classmethod
    def from_cockcroft_gault(
        cls, age_years: float, weight_kg: float, scr_mg_dl: float, sex: str
    ) -> Covariates:
        """Build Covariates with crcl_ml_min ESTIMATED via Cockcroft-Gault (renal-only).

        Use when CrCl is not directly measured. Weight/age are inputs to the estimate
        ONLY — they are not stored, so ``has_physiology()`` stays False and no
        ``generate_physiology`` graph rebuild is triggered: this individualizes renal
        CL only. For whole-body size individualization, pass body_weight_kg/age_years to
        ``Covariates(...)`` directly. The estimate then flows through ``renal_factor()``
        exactly like a measured CrCl.
        """
        return cls(crcl_ml_min=cockcroft_gault(age_years, weight_kg, scr_mg_dl, sex))
