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

    def __post_init__(self) -> None:
        if self.crcl_ml_min is not None and self.crcl_ml_min <= 0:
            raise ValueError(f"crcl_ml_min must be > 0, got {self.crcl_ml_min}")

    def renal_factor(self) -> float:
        """Multiplicative scale for ``drug.renal_clearance`` (1.0 at CrCl=125)."""
        if self.crcl_ml_min is None:
            return 1.0
        return self.crcl_ml_min / _REFERENCE_GFR_ML_MIN
