"""Core types shared across all Sisyphus layers.

This module defines the foundational data types that flow between layers.
Every layer may import from here. No layer-specific logic belongs here.

Types defined:
    Distribution       — parameter value with uncertainty (the atomic unit)
    TissueComposition  — tissue fractions for Kp calculation
    DrugOnGraph        — drug properties mapped to graph (predict → engine)
    SimResult          — raw ODE solution (engine → pk)
    PKEndpoints        — pharmacokinetic endpoints (pk → pipeline)
    PredictionResult   — final pipeline output (pipeline → caller)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Distribution — the atomic unit of uncertainty
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Distribution:
    """A parameter value with uncertainty.

    Every physiological parameter and drug property in Sisyphus is a
    Distribution.  Use ``cv=0.0`` for deterministic values.

    Attributes:
        mean: Central value (the location parameter).
        cv: Coefficient of variation (σ / μ).  0.0 means deterministic.
        dist_type: Distribution family — ``"lognormal"`` (default),
            ``"normal"``, or ``"uniform"``.
    """

    mean: float
    cv: float = 0.0
    dist_type: str = "lognormal"

    _VALID_DIST_TYPES = frozenset({"lognormal", "normal", "uniform"})

    def __post_init__(self) -> None:
        if self.cv < 0:
            raise ValueError(f"cv must be non-negative, got {self.cv}")
        if self.dist_type not in self._VALID_DIST_TYPES:
            raise ValueError(
                f"dist_type must be one of {self._VALID_DIST_TYPES}, got {self.dist_type!r}"
            )

    def sample(self, rng: np.random.Generator) -> float:
        """Draw a single realization from this distribution.

        Returns the mean when cv == 0 (deterministic).
        """
        if self.cv == 0.0:
            return self.mean
        sigma = self.cv * abs(self.mean)
        if self.dist_type == "lognormal":
            if self.mean <= 0:
                # Lognormal undefined for non-positive mean; fall back to normal
                return float(rng.normal(self.mean, sigma))
            # Parameterize so E[X] = mean, CV = cv
            mu_ln = np.log(self.mean**2 / np.sqrt(sigma**2 + self.mean**2))
            sigma_ln = np.sqrt(np.log(1 + (sigma / self.mean) ** 2))
            return float(rng.lognormal(mu_ln, sigma_ln))
        elif self.dist_type == "normal":
            return float(rng.normal(self.mean, sigma))
        elif self.dist_type == "uniform":
            half = sigma * np.sqrt(3)  # uniform with same std
            return float(rng.uniform(self.mean - half, self.mean + half))
        raise ValueError(f"Unknown dist_type: {self.dist_type}")

    @property
    def std(self) -> float:
        """Standard deviation (mean × cv)."""
        return abs(self.mean) * self.cv


# ---------------------------------------------------------------------------
# TissueComposition — shared data type for Kp estimation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TissueComposition:
    """Fractional tissue composition used for tissue:plasma partition
    coefficient (Kp) estimation via Rodgers & Rowland or Berezhkovskiy.

    All fractions are dimensionless (volume fraction of wet tissue weight).

    Note: kept as bare floats (not Distribution) because Kp sensitivity
    to tissue composition fractions is low relative to fup/CLint
    uncertainty.  This is a conscious exception to Invariant 2.
    """

    fn: float  # neutral lipid fraction
    fp: float  # phospholipid fraction
    fw: float  # water fraction
    pH: float  # intracellular pH


# ---------------------------------------------------------------------------
# DrugOnGraph — predict → engine contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DrugOnGraph:
    """Drug properties mapped onto graph topology.

    Produced by the predict layer, consumed by the engine.
    All variable properties are Distribution — never bare floats.

    The key design: ``enzyme_affinity`` is per-enzyme intrinsic clearance,
    **not** per-organ.  The engine multiplies
    ``node.enzymes[tag] × drug.enzyme_affinity[tag]``
    at every node that carries that enzyme.  IVIVE is organ-blind.
    """

    # Identity
    name: str
    smiles: str
    dose_mg: float
    route: str  # "oral", "iv"
    administration_node: str  # "stomach_lumen" | "venous_blood"

    # Physicochemical
    mw: float
    pka: float | None
    compound_type: str  # "neutral", "acid", "base", "zwitterion"

    # Binding & partitioning
    fup: Distribution  # fraction unbound in plasma
    rbp: Distribution  # blood:plasma ratio
    kp_method: str  # "rodgers_rowland" | "berezhkovskiy" | "provided"
    kp_overrides: dict[str, Distribution]  # node_name → Kp override

    # Absorption
    peff: Distribution  # effective permeability (×10⁻⁴ cm/s)
    solubility: Distribution  # mg/mL

    # Metabolism — enzyme-level, NOT organ-level
    enzyme_affinity: dict[str, Distribution]  # enzyme_tag → CLint per unit enzyme (µL/min/pmol)

    # Renal
    renal_clearance: Distribution  # L/h, total plasma basis

    # Formulation (absorption model)
    particle_radius_um: float = 25.0  # particle radius for absorption rate calc

    # Permeability-surface area overrides for perm-limited organs
    ps_overrides: dict[str, Distribution] = field(default_factory=dict)

    def sample(self, rng: np.random.Generator) -> DrugOnGraph:
        """Sample all Distributions to produce a realized (point-value) copy.

        Returns a new ``DrugOnGraph`` where every Distribution field is
        replaced by ``Distribution(mean=sampled_value, cv=0.0)``.
        """
        return DrugOnGraph(
            name=self.name,
            smiles=self.smiles,
            dose_mg=self.dose_mg,
            route=self.route,
            administration_node=self.administration_node,
            mw=self.mw,
            pka=self.pka,
            compound_type=self.compound_type,
            particle_radius_um=self.particle_radius_um,
            fup=Distribution(mean=self.fup.sample(rng), cv=0.0),
            rbp=Distribution(mean=self.rbp.sample(rng), cv=0.0),
            kp_method=self.kp_method,
            kp_overrides={
                k: Distribution(mean=v.sample(rng), cv=0.0) for k, v in self.kp_overrides.items()
            },
            peff=Distribution(mean=self.peff.sample(rng), cv=0.0),
            solubility=Distribution(mean=self.solubility.sample(rng), cv=0.0),
            enzyme_affinity={
                k: Distribution(mean=v.sample(rng), cv=0.0) for k, v in self.enzyme_affinity.items()
            },
            renal_clearance=Distribution(mean=self.renal_clearance.sample(rng), cv=0.0),
            ps_overrides={
                k: Distribution(mean=v.sample(rng), cv=0.0) for k, v in self.ps_overrides.items()
            },
        )


# ---------------------------------------------------------------------------
# SimResult — engine → pk contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SimResult:
    """Raw simulation output from the ODE engine.

    Access is always by name (``concentrations["venous_blood"]``),
    never by index.
    """

    time_h: NDArray[np.float64]  # (T,)
    concentrations: dict[str, NDArray[np.float64]]  # node_name → mg/L  (T,)
    amounts: dict[str, NDArray[np.float64]]  # node_name → mg    (T,)
    mass_balance_error: float  # max |total − dose| / dose
    solver_success: bool


# ---------------------------------------------------------------------------
# PKEndpoints — pk → pipeline contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PKEndpoints:
    """Pharmacokinetic endpoints.

    All values are Distribution.  For a single deterministic run the cv
    is 0.  For MC results the distribution aggregates N samples.
    """

    cmax: Distribution
    tmax: Distribution
    auc_0t: Distribution
    auc_0inf: Distribution | None = None
    t_half: Distribution | None = None
    cl: Distribution | None = None
    vss: Distribution | None = None


# ---------------------------------------------------------------------------
# PredictionResult — pipeline → caller contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PredictionResult:
    """Final output of the Sisyphus pipeline.

    Combines engine-based and ML-based PK predictions via a meta-learner.
    """

    drug_name: str
    smiles: str
    dose_mg: float
    route: str
    pk: PKEndpoints
    method: str  # "engine", "ml", "hybrid"
    engine_pk: PKEndpoints | None
    ml_pk: PKEndpoints | None
    confidence: str  # "high", "medium", "low"
    in_applicability_domain: bool
    ad_flags: tuple[str, ...]
    warnings: tuple[str, ...]
    cmax_90ci: tuple[float, float] | None
