"""In Vitro to In Vivo Extrapolation.

Translates ADME properties into DrugOnGraph parameters:
1. CLint decomposition into per-enzyme affinities
2. Kp calculation via Rodgers & Rowland (2005/2006)
3. Renal clearance estimation
4. DrugOnGraph assembly

All outputs are Distributions carrying prediction uncertainty.
"""

from __future__ import annotations

import logging

import numpy as np

from sisyphus.core import Distribution, DrugOnGraph
from sisyphus.graph.types import TissueComposition
from sisyphus.predict.adme import ADMEProperties
from sisyphus.predict.chemistry import MolecularProfile

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# IVIVE scaling constants
# ---------------------------------------------------------------------------

# Hepatocellularity: 120 x 10^6 cells/g liver
_HPGL = 120e6  # cells/g

# Liver weight: 1500 g (ICRP Reference Man)
_LIVER_WEIGHT_G = 1500.0  # g

# CLint scaling factor: converts uL/min/10^6 cells -> L/h (whole liver basis)
# = HPGL * liver_weight * 60 min/h / 1e6 uL/mL / 1e3 mL/L
# = 120e6 * 1500 * 60 / 1e6 / 1e3 = 10.8
_CLINT_SCALING = _HPGL * _LIVER_WEIGHT_G * 60.0 / 1e6 / 1e3  # 10.8 L/h per uL/min/10^6 cells

# IVIVE scaling factor: converts uL/min -> L/h = 60/1e6
_IVIVE_SCALING = 6e-5  # from reference_man.yaml

# GFR: glomerular filtration rate (L/h)
_GFR_L_PER_H = 7.5  # ~125 mL/min = 7.5 L/h

# ---------------------------------------------------------------------------
# Enzyme abundances (pmol, total organ) — from reference_man.yaml
# ---------------------------------------------------------------------------

_LIVER_ENZYME_ABUNDANCE: dict[str, float] = {
    "CYP3A4": 9_247_500.0,  # 137 pmol/mg * 45 MPPGL * 1500 g
    "CYP2D6": 675_000.0,  # 10 * 45 * 1500
    "CYP1A2": 3_037_500.0,  # 45 * 45 * 1500
    "CYP2C9": 6_480_000.0,  # 96 * 45 * 1500
    "CYP2E1": 3_307_500.0,  # 49 * 45 * 1500
}

# ---------------------------------------------------------------------------
# Default fraction metabolized (fm) by CYP enzyme
# Source: empirical estimates from Omega + literature defaults
# ---------------------------------------------------------------------------

_DEFAULT_FM: dict[str, float] = {
    "CYP3A4": 0.50,
    "CYP2D6": 0.10,
    "CYP1A2": 0.10,
    "CYP2C9": 0.20,
    "CYP2E1": 0.10,
}

# Adjustments for compound type
_FM_ADJUSTMENTS: dict[str, dict[str, float]] = {
    "base": {
        # Bases: increase CYP2D6 (lipophilic bases are classic 2D6 substrates)
        "CYP3A4": 0.35,
        "CYP2D6": 0.30,
        "CYP1A2": 0.05,
        "CYP2C9": 0.15,
        "CYP2E1": 0.15,
    },
    "acid": {
        # Acids: increase CYP2C9 (NSAIDs, warfarin-like)
        "CYP3A4": 0.35,
        "CYP2D6": 0.05,
        "CYP1A2": 0.10,
        "CYP2C9": 0.40,
        "CYP2E1": 0.10,
    },
    "zwitterion": {
        # Zwitterions: similar to bases but with some CYP2C9
        "CYP3A4": 0.40,
        "CYP2D6": 0.15,
        "CYP1A2": 0.05,
        "CYP2C9": 0.25,
        "CYP2E1": 0.15,
    },
}

# ---------------------------------------------------------------------------
# Tissue compositions for Kp calculation (Rodgers & Rowland)
# Source: Rodgers & Rowland (2006), Table 1
# Standard literature values — these do not change.
# ---------------------------------------------------------------------------

# Plasma composition
_PLASMA_COMP = TissueComposition(fn=0.0023, fp=0.0199, fw=0.945, pH=7.40)

# Tissue compositions (from reference_man.yaml)
_TISSUE_COMPOSITIONS: dict[str, TissueComposition] = {
    "lung": TissueComposition(fn=0.0030, fp=0.0128, fw=0.811, pH=7.0),
    "brain": TissueComposition(fn=0.0391, fp=0.0550, fw=0.620, pH=7.0),
    "heart": TissueComposition(fn=0.0117, fp=0.0166, fw=0.758, pH=7.0),
    "kidney": TissueComposition(fn=0.0121, fp=0.0242, fw=0.783, pH=7.0),
    "liver": TissueComposition(fn=0.0348, fp=0.0252, fw=0.751, pH=7.0),
    "spleen": TissueComposition(fn=0.0077, fp=0.0113, fw=0.788, pH=7.0),
    "gut_wall": TissueComposition(fn=0.0163, fp=0.0185, fw=0.718, pH=7.0),
    "pancreas": TissueComposition(fn=0.0348, fp=0.0252, fw=0.751, pH=7.0),
    "thymus": TissueComposition(fn=0.0132, fp=0.0100, fw=0.700, pH=7.0),
    "reproductive": TissueComposition(fn=0.0132, fp=0.0100, fw=0.700, pH=7.0),
    "rest": TissueComposition(fn=0.0132, fp=0.0100, fw=0.700, pH=7.0),
    "adipose_tissue": TissueComposition(fn=0.7021, fp=0.0022, fw=0.150, pH=7.0),
    "muscle_tissue": TissueComposition(fn=0.0238, fp=0.0072, fw=0.760, pH=7.0),
    "bone_tissue": TissueComposition(fn=0.0174, fp=0.0010, fw=0.439, pH=7.0),
    "skin_tissue": TissueComposition(fn=0.0284, fp=0.0111, fw=0.718, pH=7.0),
}

# CV for Kp predictions (moderate — the R&R equations are well-validated)
_KP_CV = 0.4


# ---------------------------------------------------------------------------
# CLint decomposition
# ---------------------------------------------------------------------------


def _get_fm_fractions(compound_type: str) -> dict[str, float]:
    """Get fraction metabolized by each CYP enzyme, adjusted for compound type.

    Args:
        compound_type: One of "neutral", "acid", "base", "zwitterion".

    Returns:
        Dict mapping enzyme tag -> fraction metabolized.
        Fractions sum to 1.0.
    """
    if compound_type in _FM_ADJUSTMENTS:
        fm = dict(_FM_ADJUSTMENTS[compound_type])
    else:
        fm = dict(_DEFAULT_FM)

    # Normalize to ensure sum = 1.0
    total = sum(fm.values())
    if total > 0:
        fm = {k: v / total for k, v in fm.items()}
    return fm


def _decompose_clint(
    clint: Distribution,
    compound_type: str,
    pka: float | None,
) -> dict[str, Distribution]:
    """Decompose total hepatic CLint into per-enzyme affinities.

    Converts CLint (uL/min/10^6 cells) to enzyme_affinity (uL/min/pmol)
    for each CYP enzyme using fraction metabolized and IVIVE scaling.

    The engine computes clearance as:
        CLint_enzyme = abundance * affinity * ivive_scaling
    So:
        affinity = (CLint_hepatic_L_per_h * fm) / (abundance * ivive_scaling)

    Args:
        clint: Total hepatic CLint as Distribution (uL/min/10^6 cells).
        compound_type: Compound ionization class.
        pka: pKa value (unused currently, reserved for refinement).

    Returns:
        Dict mapping enzyme tag -> CLint per pmol enzyme (uL/min/pmol)
        as Distributions.
    """
    fm = _get_fm_fractions(compound_type)

    # Scale CLint from cellular basis to whole-liver L/h
    clint_hepatic_l_per_h = clint.mean * _CLINT_SCALING

    enzyme_affinity: dict[str, Distribution] = {}
    for enzyme, fraction in fm.items():
        abundance = _LIVER_ENZYME_ABUNDANCE[enzyme]
        # affinity such that: abundance * affinity * ivive_scaling = CLint_hepatic * fm
        # affinity = (CLint_hepatic * fm) / (abundance * ivive_scaling)
        affinity = (clint_hepatic_l_per_h * fraction) / (abundance * _IVIVE_SCALING)
        # Carry CLint's CV through to affinity
        enzyme_affinity[enzyme] = Distribution(mean=max(affinity, 0.0), cv=clint.cv)

    logger.debug(
        "CLint decomposition: %.1f uL/min/10^6 cells -> %.1f L/h hepatic, compound_type=%s, fm=%s",
        clint.mean,
        clint_hepatic_l_per_h,
        compound_type,
        {k: f"{v:.2f}" for k, v in fm.items()},
    )

    return enzyme_affinity


# ---------------------------------------------------------------------------
# Kp calculation — Rodgers & Rowland (2005/2006)
# ---------------------------------------------------------------------------


def _lipid_partition(comp: TissueComposition, partition_coeff: float) -> float:
    """Compute lipid partitioning term.

    Returns: fn * P + fp * (0.3 * P + 0.7)
    where fn = neutral lipid fraction, fp = phospholipid fraction.
    """
    return comp.fn * partition_coeff + comp.fp * (0.3 * partition_coeff + 0.7)


def _compute_kp_rodgers_rowland(
    logp: float,
    pka: float | None,
    compound_type: str,
    tissue_comp: TissueComposition,
    plasma_comp: TissueComposition,
) -> float:
    """Compute tissue:plasma partition coefficient using Rodgers & Rowland.

    For neutrals:
        Kp = (fw_t + fn_t*P + fp_t*(0.3P+0.7)) / (fw_p + fn_p*P + fp_p*(0.3P+0.7))

    For acids: uses distribution coefficient D accounting for ionization.

    For bases: adds phospholipid binding correction for cationic species.

    Source: Rodgers & Rowland, Pharm Res (2005) 22:1495; (2006) 23:56.

    Args:
        logp: Octanol-water partition coefficient.
        pka: Dissociation constant. None for neutrals.
        compound_type: "neutral", "acid", "base", "zwitterion".
        tissue_comp: Tissue composition fractions.
        plasma_comp: Plasma composition fractions.

    Returns:
        Kp (dimensionless, > 0).
    """
    p = 10**logp

    if compound_type == "neutral" or pka is None:
        kp_num = tissue_comp.fw + _lipid_partition(tissue_comp, p)
        kp_den = plasma_comp.fw + _lipid_partition(plasma_comp, p)
        kp = kp_num / kp_den if kp_den > 0 else 1.0

    elif compound_type == "acid":
        # Distribution coefficient for acids
        d_tissue = p / (1.0 + 10 ** (tissue_comp.pH - pka))
        d_plasma = p / (1.0 + 10 ** (plasma_comp.pH - pka))
        # Ionization ratio for water partitioning
        ion_ratio = (1.0 + 10 ** (tissue_comp.pH - pka)) / (1.0 + 10 ** (plasma_comp.pH - pka))
        kp_num = tissue_comp.fw * ion_ratio + _lipid_partition(tissue_comp, d_tissue)
        kp_den = plasma_comp.fw + _lipid_partition(plasma_comp, d_plasma)
        kp = kp_num / kp_den if kp_den > 0 else 1.0

    elif compound_type in ("base", "zwitterion"):
        # Ionization ratio for bases
        ion_ratio = (1.0 + 10 ** (pka - tissue_comp.pH)) / (1.0 + 10 ** (pka - plasma_comp.pH))
        # Additional phospholipid binding for bases (ionized species bind phospholipids)
        phospholipid_binding = tissue_comp.fp * max(ion_ratio - 1.0, 0.0)
        kp_num = (
            tissue_comp.fw * ion_ratio + _lipid_partition(tissue_comp, p) + phospholipid_binding
        )
        kp_den = plasma_comp.fw + _lipid_partition(plasma_comp, p)
        kp = kp_num / kp_den if kp_den > 0 else 1.0

    else:
        # Fallback: treat as neutral
        kp_num = tissue_comp.fw + _lipid_partition(tissue_comp, p)
        kp_den = plasma_comp.fw + _lipid_partition(plasma_comp, p)
        kp = kp_num / kp_den if kp_den > 0 else 1.0

    # Clamp to physiological range
    kp = float(np.clip(kp, 0.01, 200.0))
    return kp


def _compute_all_kp(
    logp: float,
    pka: float | None,
    compound_type: str,
) -> dict[str, Distribution]:
    """Compute Kp for all tissues.

    Args:
        logp: Octanol-water partition coefficient.
        pka: Dissociation constant.
        compound_type: Ionization class.

    Returns:
        Dict mapping tissue name -> Kp as Distribution.
    """
    kp_overrides: dict[str, Distribution] = {}
    for tissue_name, tissue_comp in _TISSUE_COMPOSITIONS.items():
        kp = _compute_kp_rodgers_rowland(logp, pka, compound_type, tissue_comp, _PLASMA_COMP)
        kp_overrides[tissue_name] = Distribution(mean=kp, cv=_KP_CV)

    logger.debug(
        "Kp calculated for %d tissues (logP=%.2f, pKa=%s, type=%s): %s",
        len(kp_overrides),
        logp,
        pka,
        compound_type,
        {k: f"{v.mean:.2f}" for k, v in kp_overrides.items()},
    )

    return kp_overrides


# ---------------------------------------------------------------------------
# Renal clearance estimation
# ---------------------------------------------------------------------------


def _estimate_renal_clearance(fup: Distribution, profile: MolecularProfile) -> Distribution:
    """Estimate renal clearance from GFR and fraction unbound.

    For drugs undergoing glomerular filtration without active secretion:
        CL_renal = GFR * fup

    Active secretion/reabsorption would require transporter data.

    Args:
        fup: Fraction unbound in plasma.
        profile: Molecular profile (reserved for future use, e.g. charge-based
            reabsorption estimates).

    Returns:
        Renal clearance (L/h) as Distribution.
    """
    cl_renal = _GFR_L_PER_H * fup.mean
    return Distribution(mean=cl_renal, cv=fup.cv)


# ---------------------------------------------------------------------------
# Particle radius estimation
# ---------------------------------------------------------------------------


def _estimate_particle_radius(solubility: Distribution) -> float:
    """Estimate effective particle radius from solubility.

    High solubility -> small effective radius (dissolves fast).
    Low solubility -> larger effective radius.

    Default: 25 um.  Adjusted for very high/low solubility.

    Args:
        solubility: Aqueous solubility (mg/mL).

    Returns:
        Particle radius in um.
    """
    s = solubility.mean
    if s > 10.0:
        return 10.0  # Very soluble -> small particles (fast dissolution)
    elif s > 1.0:
        return 15.0
    elif s > 0.1:
        return 25.0  # Default
    elif s > 0.01:
        return 35.0
    else:
        return 50.0  # Poorly soluble -> large effective radius


# ---------------------------------------------------------------------------
# DrugOnGraph assembly
# ---------------------------------------------------------------------------


def build_drug_on_graph(
    profile: MolecularProfile,
    adme: ADMEProperties,
    dose_mg: float,
    route: str = "oral",
) -> DrugOnGraph:
    """Construct a DrugOnGraph from predicted properties.

    Converts global CLint into per-enzyme affinities using
    CYP contribution fractions.  Computes Kp for all tissues
    using Rodgers & Rowland.  Maps absorption parameters.
    Sets administration_node based on route.

    Args:
        profile: Molecular profile (physicochemical properties).
        adme: Predicted ADME properties.
        dose_mg: Dose in mg.
        route: Administration route ("oral" or "iv").

    Returns:
        A fully parameterized DrugOnGraph ready for the engine.
    """
    # Decompose CLint to per-enzyme affinities
    enzyme_affinity = _decompose_clint(adme.clint, profile.compound_type, profile.pka)

    # Compute Kp for each tissue using Rodgers & Rowland
    kp_overrides = _compute_all_kp(profile.logp, profile.pka, profile.compound_type)

    # Estimate particle radius from solubility
    particle_radius = _estimate_particle_radius(adme.solubility)

    # Estimate renal clearance
    renal_cl = _estimate_renal_clearance(adme.fup, profile)

    # Set administration node based on route
    if route == "oral":
        admin_node = "stomach_lumen"
    elif route == "iv":
        admin_node = "venous_blood"
    else:
        logger.warning("Unknown route %r, defaulting to oral", route)
        admin_node = "stomach_lumen"

    # Truncate name for display (SMILES can be long)
    name = profile.smiles[:40] if len(profile.smiles) > 40 else profile.smiles

    drug = DrugOnGraph(
        name=name,
        smiles=profile.smiles,
        dose_mg=dose_mg,
        route=route,
        administration_node=admin_node,
        mw=profile.mw,
        pka=profile.pka,
        compound_type=profile.compound_type,
        fup=adme.fup,
        rbp=adme.rbp,
        kp_method="rodgers_rowland",
        kp_overrides=kp_overrides,
        peff=adme.peff,
        solubility=adme.solubility,
        enzyme_affinity=enzyme_affinity,
        renal_clearance=renal_cl,
        particle_radius_um=particle_radius,
    )

    logger.info(
        "DrugOnGraph built: %s, dose=%.1f mg, route=%s, "
        "%d enzymes, %d Kp overrides, CL_renal=%.2f L/h",
        name,
        dose_mg,
        route,
        len(enzyme_affinity),
        len(kp_overrides),
        renal_cl.mean,
    )

    return drug
