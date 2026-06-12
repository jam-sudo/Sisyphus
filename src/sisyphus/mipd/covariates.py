"""Patient covariates that deterministically individualize the engine prior.

v1: renal function only — a measured creatinine clearance (CrCl). CrCl scales
the drug's renal (glomerular-filtration) clearance: the engine's reference renal
model is ``CL_renal = GFR*fup`` with GFR = 7.5 L/h (~125 mL/min), so an
individual's renal CL is scaled by ``CrCl / 125``. Weight/age covariates (via
sbi.physiology_generator) are a documented future extension — see the design spec
docs/superpowers/specs/2026-06-11-mipd-crcl-renal-individualization-design.md.
"""
from __future__ import annotations

from dataclasses import dataclass

# Reference glomerular filtration rate the engine's renal_clearance assumes
# (_GFR_L_PER_H = 7.5 L/h ~= 125 mL/min; src/sisyphus/predict/ivive.py:42-43).
_REFERENCE_GFR_ML_MIN = 125.0


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
