"""Patient covariates for renal and whole-body engine-prior individualization.

Measured creatinine clearance (CrCl) scales
the drug's renal (glomerular-filtration) clearance: the engine's reference renal
model is ``CL_renal = GFR*fup`` with GFR = 7.5 L/h (~125 mL/min), so an
individual's renal CL is scaled by ``CrCl / 125``. Weight and age individualize
the body graph via ``sbi.physiology_generator``. See the design spec
docs/_internal/specs/2026-06-11-mipd-crcl-renal-individualization-design.md.
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

# Devine 1974 IBW; FDA Appendix B: https://www.accessdata.fda.gov/cdrh_docs/pdf16/P160013c.pdf
_DEVINE_MALE_BASE_KG = 50.0
_DEVINE_FEMALE_BASE_KG = 45.5
_DEVINE_KG_PER_INCH = 2.3
_DEVINE_REFERENCE_HEIGHT_IN = 60.0
# Winter et al. 2012, PMID 22576791: adjusted body weight uses 40% excess weight.
_CG_ADJUSTED_WEIGHT_FRACTION = 0.4

# Rhodin et al., Pediatr Nephrol 24:67-76 (2009), PMID 18846389.
_RHODIN_ADULT_GFR_ML_MIN = 121.2
_RHODIN_ADULT_WEIGHT_KG = 70.0
_RHODIN_PMA50_WEEKS = 47.7
_RHODIN_WEIGHT_EXPONENT_UNITLESS = 0.75
_RHODIN_HILL_EXPONENT_UNITLESS = 3.4


def population_gfr_ml_min(weight_kg: float, pma_weeks: float) -> float:
    """Rhodin population GFR estimate (mL/min) from weight and postmenstrual age."""
    if not math.isfinite(weight_kg) or weight_kg <= 0:
        raise ValueError("weight_kg must be finite and > 0")
    if not math.isfinite(pma_weeks) or pma_weeks <= 0:
        raise ValueError("pma_weeks must be finite and > 0")
    maturity = pma_weeks ** _RHODIN_HILL_EXPONENT_UNITLESS
    return (
        _RHODIN_ADULT_GFR_ML_MIN
        * (weight_kg / _RHODIN_ADULT_WEIGHT_KG) ** _RHODIN_WEIGHT_EXPONENT_UNITLESS
        * maturity / (_RHODIN_PMA50_WEEKS ** _RHODIN_HILL_EXPONENT_UNITLESS + maturity)
    )


def cockcroft_gault(
    age_years: float, weight_kg: float, scr_mg_dl: float, sex: str,
    *, weight_type: str = "total", height_cm: float | None = None,
) -> float:
    """Estimate creatinine clearance (mL/min) by the Cockcroft-Gault equation.

    ``CrCl = (140 - age) * weight_kg * (0.85 if female else 1.0) / (72 * SCr_mg/dL)``.
    ``scr_mg_dl`` is serum creatinine in mg/dL (NOT SI µmol/L; divide µmol/L by 88.42).
    ``sex`` is "male"/"female" (case-insensitive). Returns an ABSOLUTE CrCl in mL/min
    (body size enters via the weight term) — the quantity ``Covariates.crcl_ml_min``
    expects; do not pass a BSA-normalized eGFR there instead. With actual body weight,
    ``weight_type`` selects total, ideal (Devine), or adjusted (Devine + 40% of
    excess total weight); the latter two require ``height_cm >= 152.4``.

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
    if weight_type not in ("total", "ideal", "adjusted"):
        raise ValueError("weight_type must be 'total', 'ideal', or 'adjusted'")
    if weight_type != "total":
        if height_cm is None or not math.isfinite(height_cm) or height_cm < 152.4:
            raise ValueError("height_cm must be finite and >= 152.4 for ideal/adjusted weight")
        height_in = height_cm / 2.54
        ideal_kg = (
            _DEVINE_MALE_BASE_KG if sex_l == "male" else _DEVINE_FEMALE_BASE_KG
        ) + _DEVINE_KG_PER_INCH * (height_in - _DEVINE_REFERENCE_HEIGHT_IN)
        if weight_type == "ideal":
            weight_kg = ideal_kg
        elif weight_kg > ideal_kg:
            weight_kg = ideal_kg + _CG_ADJUSTED_WEIGHT_FRACTION * (weight_kg - ideal_kg)

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
    """Patient covariates for engine-as-prior individualization.

    Attributes:
        crcl_ml_min: measured creatinine clearance (mL/min). None -> no renal
            individualization (renal_factor 1.0).
        body_weight_kg: patient body weight for whole-body physiology.
        age_years: patient age for enzyme ontogeny and organ physiology.
        pma_weeks: postmenstrual age for the Rhodin population GFR estimate.
        crcl_cv: optional measurement CV for measured CrCl; zero disables it.
    """

    crcl_ml_min: float | None = None
    body_weight_kg: float | None = None
    age_years: float | None = None
    pma_weeks: float | None = None
    crcl_cv: float = 0.0

    def __post_init__(self) -> None:
        if self.crcl_ml_min is not None and self.crcl_ml_min <= 0:
            raise ValueError(f"crcl_ml_min must be > 0, got {self.crcl_ml_min}")
        if self.body_weight_kg is not None and self.body_weight_kg <= 0:
            raise ValueError(f"body_weight_kg must be > 0, got {self.body_weight_kg}")
        if self.age_years is not None and self.age_years <= 0:
            raise ValueError(f"age_years must be > 0, got {self.age_years}")
        if self.pma_weeks is not None:
            if not math.isfinite(self.pma_weeks) or self.pma_weeks <= 0:
                raise ValueError("pma_weeks must be finite and > 0")
            if self.body_weight_kg is None:
                raise ValueError("pma_weeks requires body_weight_kg")
        if not math.isfinite(self.crcl_cv) or self.crcl_cv < 0:
            raise ValueError("crcl_cv must be finite and >= 0")
        if self.crcl_cv and self.crcl_ml_min is None:
            raise ValueError("crcl_cv requires crcl_ml_min")

    def renal_factor(self) -> float:
        """Renal clearance scale from measured CrCl or Rhodin population GFR."""
        if self.crcl_ml_min is not None:
            return self.crcl_ml_min / _REFERENCE_GFR_ML_MIN
        if self.pma_weeks is not None:
            gfr = population_gfr_ml_min(self.body_weight_kg, self.pma_weeks)
            return gfr / _REFERENCE_GFR_ML_MIN
        return 1.0

    def has_physiology(self) -> bool:
        """True iff weight or age is set — triggers generate_physiology graph build."""
        return self.body_weight_kg is not None or self.age_years is not None

    def warnings(self) -> tuple[str, ...]:
        """Structured flags for physiologically extreme covariates (extrapolation risk)."""
        w: list[str] = []
        if self.crcl_ml_min is None and self.pma_weeks is not None:
            w.append(
                "gfr:population-estimate: Rhodin weight/PMA model, not measured renal function"
            )
            if self.age_years is None:
                w.append(
                    "age:unspecified: non-renal physiology uses the adult default age; "
                    "supply age_years for pediatric body scaling"
                )
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
        cls, age_years: float, weight_kg: float, scr_mg_dl: float, sex: str,
        *, weight_type: str = "total", height_cm: float | None = None,
    ) -> Covariates:
        """Build Covariates with crcl_ml_min ESTIMATED via Cockcroft-Gault (renal-only).

        Use when CrCl is not directly measured. Weight/age are inputs to the estimate
        ONLY — they are not stored, so ``has_physiology()`` stays False and no
        ``generate_physiology`` graph rebuild is triggered: this individualizes renal
        CL only. For whole-body size individualization, pass body_weight_kg/age_years to
        ``Covariates(...)`` directly. The estimate then flows through ``renal_factor()``
        exactly like a measured CrCl.
        """
        return cls(crcl_ml_min=cockcroft_gault(
            age_years, weight_kg, scr_mg_dl, sex,
            weight_type=weight_type, height_cm=height_cm,
        ))
