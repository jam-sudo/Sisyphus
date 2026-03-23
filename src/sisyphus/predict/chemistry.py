"""Molecular property prediction from SMILES.

Computes physicochemical descriptors, pKa, logP, and checks
applicability domain.  Uses RDKit for descriptor calculation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from rdkit import Chem
from rdkit.Chem import Descriptors

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MolecularProfile:
    """Physicochemical properties derived from SMILES.

    Attributes:
        smiles: Canonical SMILES string.
        mw: Molecular weight (g/mol).
        logp: Octanol-water partition coefficient (log units).
        pka: Dissociation constant.  ``None`` for neutral compounds.
        compound_type: ``"neutral"``, ``"acid"``, ``"base"``, ``"zwitterion"``.
        hbd: Hydrogen bond donors.
        hba: Hydrogen bond acceptors.
        tpsa: Topological polar surface area (A^2).
        rotatable_bonds: Number of rotatable bonds.
        in_ad: Whether the compound is within the applicability domain.
        ad_flags: Reasons for being outside AD, if any.
    """

    smiles: str
    mw: float
    logp: float
    pka: float | None
    compound_type: str
    hbd: int
    hba: int
    tpsa: float
    rotatable_bonds: int
    in_ad: bool
    ad_flags: list[str]


# ---------------------------------------------------------------------------
# SMARTS patterns for ionizable-group detection
# ---------------------------------------------------------------------------
# Carboxylic acid: pKa ~4
_SMARTS_CARBOXYLIC_ACID = Chem.MolFromSmarts("[CX3](=O)[OX2H1]")
# Sulfonic acid: pKa ~1
_SMARTS_SULFONIC_ACID = Chem.MolFromSmarts("[SX4](=O)(=O)[OX2H1]")
# Phenol: pKa ~10 (aromatic OH)
_SMARTS_PHENOL = Chem.MolFromSmarts("[OX2H1]c")
# Primary/secondary/tertiary amine (not imine, amide, etc.)
_SMARTS_AMINE = Chem.MolFromSmarts("[NX3;H2,H1,H0;!$(N=*);!$(NC=O);!$(NS=O);!$(N#*)]")
# Aromatic N-H in rings of size >= 6 — protonatable heterocyclic nitrogen
# (e.g., benzodiazepine).  Excludes 5-membered ring N-H (pyrrole, indole,
# imidazole) which are not pharmacologically basic.
_SMARTS_AROMATIC_NH = Chem.MolFromSmarts("[nH1;!r5]")

# Default pKa values by compound type
_DEFAULT_PKA: dict[str, float | None] = {
    "acid": 4.5,
    "base": 9.0,
    "zwitterion": 7.0,
    "neutral": None,
}


def _estimate_pka_type(mol: Chem.Mol, logp: float) -> tuple[float | None, str]:
    """Estimate pKa and compound type from structural patterns.

    Uses SMARTS matching to detect ionizable groups.  Returns default pKa
    values per compound class — proper pKa prediction requires specialized
    models (e.g., MoKa, ChemAxon) which are out of scope for Phase 2.

    Args:
        mol: RDKit Mol object.
        logp: Crippen LogP (unused currently, reserved for refinement).

    Returns:
        (pka, compound_type) tuple.
    """
    has_acid = bool(
        mol.HasSubstructMatch(_SMARTS_CARBOXYLIC_ACID)
        or mol.HasSubstructMatch(_SMARTS_SULFONIC_ACID)
        or mol.HasSubstructMatch(_SMARTS_PHENOL)
    )

    # Check aliphatic amines (non-aromatic nitrogen with H or lone pair).
    # The SMARTS already excludes imines (N=*), amides (NC=O), sulfonamides
    # (NS=O), and nitriles (N#*).  We further filter out aromatic N since
    # pyridine/pyrimidine nitrogens are weakly basic and should not drive
    # classification alone.
    amine_matches = mol.GetSubstructMatches(_SMARTS_AMINE)
    has_base = False
    for match in amine_matches:
        atom = mol.GetAtomWithIdx(match[0])
        if not atom.GetIsAromatic():
            has_base = True
            break

    # Also detect aromatic N-H (e.g., benzodiazepine, indole-like rings).
    # These are protonatable and pharmacologically basic.
    if not has_base and mol.HasSubstructMatch(_SMARTS_AROMATIC_NH):
        has_base = True

    if has_acid and has_base:
        compound_type = "zwitterion"
    elif has_acid:
        compound_type = "acid"
    elif has_base:
        compound_type = "base"
    else:
        compound_type = "neutral"

    pka = _DEFAULT_PKA[compound_type]
    return pka, compound_type


# ---------------------------------------------------------------------------
# Applicability-domain flags (from Omega empirical findings)
# ---------------------------------------------------------------------------
_MW_CEILING = 700.0  # Da — large molecules are outside training dist.
_LOGP_CEILING = 5.5  # Extreme lipophilicity
_PGP_MW_THRESHOLD = 500.0  # P-gp efflux risk thresholds
_PGP_LOGP_THRESHOLD = 3.5
_PGP_TPSA_THRESHOLD = 100.0  # A^2


def _check_ad(mol: Chem.Mol, mw: float, logp: float, tpsa: float) -> list[str]:
    """Return applicability-domain warning flags.

    These are warnings, not hard failures — prediction continues with flags.

    Args:
        mol: RDKit Mol object.
        mw: Molecular weight.
        logp: Crippen LogP.
        tpsa: Topological polar surface area.

    Returns:
        List of flag strings (empty if within AD).
    """
    flags: list[str] = []
    if mw > _MW_CEILING:
        flags.append("HIGH_MW")
    if logp > _LOGP_CEILING:
        flags.append("EXTREME_LIPOPHILIC")
    if mw > _PGP_MW_THRESHOLD and logp > _PGP_LOGP_THRESHOLD and tpsa > _PGP_TPSA_THRESHOLD:
        flags.append("PGP_EFFLUX_RISK")
    return flags


def compute_profile(smiles: str) -> MolecularProfile:
    """Compute molecular profile from a SMILES string.

    Args:
        smiles: Input SMILES.

    Returns:
        MolecularProfile with all descriptors populated.

    Raises:
        ValueError: If the SMILES string is invalid.
    """
    if not smiles or not smiles.strip():
        raise ValueError(f"Invalid SMILES: {smiles!r}")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError(f"Invalid SMILES: {smiles!r}")

    canonical = Chem.MolToSmiles(mol)
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    rotatable_bonds = Descriptors.NumRotatableBonds(mol)

    pka, compound_type = _estimate_pka_type(mol, logp)

    ad_flags = _check_ad(mol, mw, logp, tpsa)

    return MolecularProfile(
        smiles=canonical,
        mw=mw,
        logp=logp,
        pka=pka,
        compound_type=compound_type,
        hbd=hbd,
        hba=hba,
        tpsa=tpsa,
        rotatable_bonds=rotatable_bonds,
        in_ad=len(ad_flags) == 0,
        ad_flags=ad_flags,
    )
