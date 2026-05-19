"""Core types shared across all Sisyphus layers.

This module defines the foundational data types that flow between layers.
Every layer may import from here. No layer-specific logic belongs here.

Types defined:
    Distribution       — parameter value with uncertainty (the atomic unit)
    TissueComposition  — tissue fractions for Kp calculation
    TransporterKinetics — Michaelis-Menten parameters for active transport
    ActiveMetabolite   — one-compartment active metabolite for prodrug routing
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
        correlation_group: Optional group tag for joint multivariate sampling.
            Distributions sharing a group label are sampled together by
            ``sisyphus.physiology.correlation_registry.sample_correlated``.
            ``None`` (default) means independent sampling.
    """

    mean: float
    cv: float = 0.0
    dist_type: str = "lognormal"
    correlation_group: str | None = None

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
# TransporterKinetics — Michaelis-Menten parameters for active transport
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TransporterKinetics:
    """Single transporter's Michaelis-Menten parameters.

    Used for saturable active transport (e.g., P-gp efflux, OATP uptake).
    Full Michaelis-Menten: rate = abundance × Jmax × C / (Km + C).

    Separate from enzyme_affinity because transporters are saturable at
    therapeutic concentrations (gut lumen C >> Km), while CYP clearance
    is approximately linear (hepatic C << Km).

    Attributes:
        jmax: Maximum transport rate (pmol/min/cm² or REF-scaled units).
        km: Michaelis constant (µM). Drug concentration at half-max rate.
    """

    jmax: Distribution
    km: Distribution


# ---------------------------------------------------------------------------
# ActiveMetabolite — prodrug activation routing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ActiveMetabolite:
    """Active species produced by in vivo conversion of a parent prodrug.

    One-compartment plasma disposition: aggregate ``CL_per_h`` and ``Vd_L``
    represent the active's apparent clearance and volume of distribution.
    No enzyme-level decomposition (out of scope; see
    docs/superpowers/specs/2026-04-24-prodrug-activation-design.md §2).

    The conversion edge connects ``conversion_site`` (a parent-graph node)
    to the active's plasma compartment. Conversion is 1st-order in parent
    amount with optional yield fraction < 1.

    Attributes:
        name: Identifier for the active species (e.g. ``"BH4"``).
        mw: Molecular weight (g/mol). Bare float (deterministic; consistent
            with ``DrugOnGraph.mw`` precedent — Invariant 2 exception).
        fup: Fraction unbound in plasma (Distribution).
        CL_per_h: Aggregate plasma clearance, L/h (Distribution).
        Vd_L: Apparent volume of distribution, L (Distribution).
        conversion_rate_per_h: First-order conversion rate constant, 1/h
            (Distribution).
        conversion_site: Name of parent-graph node where conversion occurs.
        conversion_yield_fraction: Fractional molar conversion (Distribution,
            default 1.0 = stoichiometric).
        enzyme_yields: Optional per-enzyme yield overrides (B-04). Defaults
            to empty dict.
    """

    name: str
    mw: float
    fup: Distribution
    CL_per_h: Distribution
    Vd_L: Distribution
    conversion_rate_per_h: Distribution
    conversion_site: str
    conversion_yield_fraction: Distribution = field(
        default_factory=lambda: Distribution(1.0, cv=0.0)
    )
    enzyme_yields: dict[str, Distribution] = field(default_factory=dict)
    """Per-enzyme conversion yield (B-04).

    Optional override of ``conversion_yield_fraction`` keyed by enzyme tag.
    When ``enzyme_yields[tag]`` is set, the builder uses it for edges
    catalyzed by ``tag``; when unset, edges fall back to the entry-level
    ``conversion_yield_fraction``. Empty dict (default) preserves the
    pre-B-04 single-yield behaviour.

    See docs/superpowers/specs/2026-05-17-multi-enzyme-prodrug-yield-design.md
    """


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

    # Active transport — transporter-level, NOT organ-level
    # Parallel to enzyme_affinity: tag → kinetics.  Engine multiplies
    # node.transporters[tag] × drug.transporter_kinetics[tag].(jmax,km)
    # Empty dict = no active transport (default for drugs without measured kinetics).
    transporter_kinetics: dict[str, TransporterKinetics] = field(default_factory=dict)

    # Extended Clearance Model (ECM) — closed-form hepatocyte QSSA.
    # Defaults drive ECM → well-stirred degenerate limit (<0.01% deviation).
    # Future OATP substrates override via data/transporters/hepatic_ecm.json
    # (file added in a later task).
    ps_passive: Distribution = field(default_factory=lambda: Distribution(1e6, cv=0.0))
    ps_eff: Distribution = field(default_factory=lambda: Distribution(1e6, cv=0.0))
    cl_int_bile: Distribution = field(default_factory=lambda: Distribution(0.0, cv=0.0))

    # Permeability-surface area overrides for perm-limited organs
    ps_overrides: dict[str, Distribution] = field(default_factory=dict)

    # Prodrug activation — see ActiveMetabolite + spec
    # docs/superpowers/specs/2026-04-24-prodrug-activation-design.md
    active_metabolite: ActiveMetabolite | None = None
    observation_species: str = "parent"  # "parent" | "active"

    # v2 prodrug activation — drug-side enzyme affinity for conversion
    # (separate from enzyme_affinity which is for elimination).
    # See docs/superpowers/specs/2026-04-27-prodrug-activation-v2-design.md §3.2.
    enzyme_affinity_for_conversion: dict[str, Distribution] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.observation_species not in ("parent", "active"):
            raise ValueError(
                f"observation_species must be 'parent' or 'active', "
                f"got {self.observation_species!r}"
            )
        if self.observation_species == "active" and self.active_metabolite is None:
            raise ValueError(
                "observation_species='active' requires active_metabolite config"
            )
        if self.enzyme_affinity_for_conversion and self.active_metabolite is None:
            raise ValueError(
                "enzyme_affinity_for_conversion is non-empty but active_metabolite is None; "
                "set active_metabolite or empty the dict"
            )

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
            enzyme_affinity_for_conversion={
                k: Distribution(mean=v.sample(rng), cv=0.0)
                for k, v in self.enzyme_affinity_for_conversion.items()
            },
            transporter_kinetics={
                k: TransporterKinetics(
                    jmax=Distribution(mean=v.jmax.sample(rng), cv=0.0),
                    km=Distribution(mean=v.km.sample(rng), cv=0.0),
                ) for k, v in self.transporter_kinetics.items()
            },
            renal_clearance=Distribution(mean=self.renal_clearance.sample(rng), cv=0.0),
            ps_passive=Distribution(mean=self.ps_passive.sample(rng), cv=0.0),
            ps_eff=Distribution(mean=self.ps_eff.sample(rng), cv=0.0),
            cl_int_bile=Distribution(mean=self.cl_int_bile.sample(rng), cv=0.0),
            ps_overrides={
                k: Distribution(mean=v.sample(rng), cv=0.0) for k, v in self.ps_overrides.items()
            },
            active_metabolite=ActiveMetabolite(
                name=self.active_metabolite.name,
                mw=self.active_metabolite.mw,
                fup=Distribution(mean=self.active_metabolite.fup.sample(rng), cv=0.0),
                CL_per_h=Distribution(mean=self.active_metabolite.CL_per_h.sample(rng), cv=0.0),
                Vd_L=Distribution(mean=self.active_metabolite.Vd_L.sample(rng), cv=0.0),
                conversion_rate_per_h=Distribution(
                    mean=self.active_metabolite.conversion_rate_per_h.sample(rng), cv=0.0),
                conversion_site=self.active_metabolite.conversion_site,
                conversion_yield_fraction=Distribution(
                    mean=self.active_metabolite.conversion_yield_fraction.sample(rng),
                    cv=0.0),
            ) if self.active_metabolite is not None else None,
            observation_species=self.observation_species,
        )

    def realize_means(self) -> DrugOnGraph:
        """Deterministic realization using means (RNG-independent).

        Returns a new ``DrugOnGraph`` where every Distribution field is
        replaced by ``Distribution(mean=mean, cv=0.0)`` — using existing
        means directly, NOT stochastic samples. Used by the deterministic
        prediction path (``n_mc_samples=0``) to eliminate seed-dependent
        RNG-order coupling.
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
            fup=Distribution(mean=self.fup.mean, cv=0.0),
            rbp=Distribution(mean=self.rbp.mean, cv=0.0),
            kp_method=self.kp_method,
            kp_overrides={
                k: Distribution(mean=v.mean, cv=0.0) for k, v in self.kp_overrides.items()
            },
            peff=Distribution(mean=self.peff.mean, cv=0.0),
            solubility=Distribution(mean=self.solubility.mean, cv=0.0),
            enzyme_affinity={
                k: Distribution(mean=v.mean, cv=0.0) for k, v in self.enzyme_affinity.items()
            },
            enzyme_affinity_for_conversion={
                k: Distribution(mean=v.mean, cv=0.0)
                for k, v in self.enzyme_affinity_for_conversion.items()
            },
            transporter_kinetics={
                k: TransporterKinetics(
                    jmax=Distribution(mean=v.jmax.mean, cv=0.0),
                    km=Distribution(mean=v.km.mean, cv=0.0),
                ) for k, v in self.transporter_kinetics.items()
            },
            renal_clearance=Distribution(mean=self.renal_clearance.mean, cv=0.0),
            ps_passive=Distribution(mean=self.ps_passive.mean, cv=0.0),
            ps_eff=Distribution(mean=self.ps_eff.mean, cv=0.0),
            cl_int_bile=Distribution(mean=self.cl_int_bile.mean, cv=0.0),
            ps_overrides={
                k: Distribution(mean=v.mean, cv=0.0) for k, v in self.ps_overrides.items()
            },
            active_metabolite=ActiveMetabolite(
                name=self.active_metabolite.name,
                mw=self.active_metabolite.mw,
                fup=Distribution(mean=self.active_metabolite.fup.mean, cv=0.0),
                CL_per_h=Distribution(mean=self.active_metabolite.CL_per_h.mean, cv=0.0),
                Vd_L=Distribution(mean=self.active_metabolite.Vd_L.mean, cv=0.0),
                conversion_rate_per_h=Distribution(
                    mean=self.active_metabolite.conversion_rate_per_h.mean, cv=0.0),
                conversion_site=self.active_metabolite.conversion_site,
                conversion_yield_fraction=Distribution(
                    mean=self.active_metabolite.conversion_yield_fraction.mean,
                    cv=0.0),
            ) if self.active_metabolite is not None else None,
            observation_species=self.observation_species,
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
    clf_pk: PKEndpoints | None = None
    # Ordered (gene, phenotype_code) pairs that were applied to the body
    # graph for this prediction. Empty when no phenotypes were supplied.
    # Round-trips via dict(result.phenotypes_applied).
    phenotypes_applied: tuple[tuple[str, str], ...] = ()
