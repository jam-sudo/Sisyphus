"""Molecular property prediction from SMILES.

Computes physicochemical descriptors, pKa, logP, and checks
applicability domain.  Uses RDKit for descriptor calculation.
Includes SMARTS-based prodrug detection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Descriptors

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prodrug SMARTS patterns
# ---------------------------------------------------------------------------
# These detect ester, amide, and phosphate prodrug motifs.  When matched,
# the drug is flagged as PRODRUG in AD flags.  The rationale: prodrugs are
# designed to be rapidly converted to the active metabolite in vivo.
# Clinical Cmax references report the metabolite, but we simulate the
# prodrug itself.  These drugs are out-of-applicability-domain for PBPK
# prediction of parent compound Cmax.
#
# Source: Rautio et al., Nature Reviews Drug Discovery (2008) 7:255.

_PRODRUG_SMARTS_PATTERNS: list[tuple[str, str]] = [
    # Ester prodrug (sp3 carbon): R-C(=O)-O-C (e.g., valacyclovir, valganciclovir)
    ("[OX2][CX3](=O)[CX4]", "ESTER_PRODRUG_MOTIF"),
    # Ester prodrug (general, includes sp2 carbon): R-C(=O)-O-C
    # Catches vinyl/aryl esters (e.g., molnupiravir, fesoterodine)
    ("[OX2][CX3](=O)[#6]", "GENERAL_ESTER_MOTIF"),
    # Carbonate ester: R-O-C(=O)-O-R (e.g., tenofovir disoproxil)
    ("[OX2][CX3](=O)[OX2]", "CARBONATE_PRODRUG_MOTIF"),
    # Phosphonate ester prodrug: R-O-P(=O) (e.g., tenofovir disoproxil, adefovir dipivoxil)
    ("[OX2][PX4](=O)", "PHOSPHONATE_PRODRUG_MOTIF"),
]

_PRODRUG_COMPILED = [
    (Chem.MolFromSmarts(smarts), label) for smarts, label in _PRODRUG_SMARTS_PATTERNS
]

# --- Specific prodrug sub-patterns ---

# Amino acid ester: ester adjacent to a free amine (val-ester prodrugs:
# valacyclovir, valganciclovir).
_AMINO_ACID_ESTER_SMARTS = Chem.MolFromSmarts("[NX3;H2,H1][CX4][CX3](=O)[OX2]")

# Nucleoside ester: ester + heterocyclic ring with >=2 ring nitrogens
# (catches nucleoside/nucleotide prodrugs: molnupiravir, valacyclovir, valganciclovir).
_DI_N_RING_SMARTS = Chem.MolFromSmarts("[nR]~*~[nR]")

# Phenol ester without free carboxylic acid: aromatic C bonded to ester oxygen,
# where the molecule lacks a free -COOH on the same ring system.
# Catches fesoterodine (isobutyrate ester of 5-HMT phenol).
# Excludes aspirin (has -COOH, indicating the ester is part of the drug, not a prodrug linkage).
_PHENOL_ESTER_SMARTS = Chem.MolFromSmarts("[c]([c])([c])[OX2][CX3](=O)[CX4;!$(C-c)]")
_CARBOXYLIC_ACID_SMARTS = Chem.MolFromSmarts("[CX3](=O)[OX2H1]")


def _check_prodrug(mol: Chem.Mol) -> list[str]:
    """Detect prodrug motifs using SMARTS matching.

    Detection rules (each sufficient on its own):
    1. Amino acid ester — ester adjacent to free amine (val-ester prodrugs).
    2. Phosphonate + carbonate/ester — phosphonate ester prodrugs (tenofovir disoproxil).
    3. Multiple (>=3) ester/carbonate motifs — multi-ester prodrugs.
    4. Nucleoside ester — ester + di-nitrogen heterocycle (molnupiravir).
    5. Phenol ester without free carboxylic acid — phenol ester prodrugs (fesoterodine).

    Returns a list of prodrug-related AD flags (empty if no prodrug detected).
    """
    flags: list[str] = []

    # Collect unique ester-bond atom tuples to avoid double-counting
    # across the sp3-specific and general ester patterns.
    ester_atom_sets: set[frozenset[int]] = set()
    has_phosphonate = False

    for pattern, label in _PRODRUG_COMPILED:
        if pattern is None:
            continue
        matches = mol.GetSubstructMatches(pattern)
        if len(matches) > 0:
            if "ESTER" in label or "CARBONATE" in label:
                for match in matches:
                    ester_atom_sets.add(frozenset(match))
            if "PHOSPHONATE" in label:
                has_phosphonate = True

    ester_count = len(ester_atom_sets)
    has_ester = ester_count > 0

    # Check amino acid ester (val-ester prodrugs)
    has_amino_acid_ester = bool(mol.HasSubstructMatch(_AMINO_ACID_ESTER_SMARTS))

    # Check nucleoside ester (ester + heterocyclic ring with 2+ ring nitrogens)
    has_di_n_ring = bool(mol.HasSubstructMatch(_DI_N_RING_SMARTS))

    # Check phenol ester without free carboxylic acid
    has_phenol_ester = _PHENOL_ESTER_SMARTS is not None and bool(
        mol.HasSubstructMatch(_PHENOL_ESTER_SMARTS)
    )
    has_cooh = bool(mol.HasSubstructMatch(_CARBOXYLIC_ACID_SMARTS))

    # Decision logic — each rule is independently sufficient
    is_prodrug = False
    reason = ""

    if has_amino_acid_ester:
        is_prodrug = True
        reason = "amino acid ester motif (val-ester type)"
    elif has_phosphonate and ester_count >= 1:
        is_prodrug = True
        reason = "phosphonate ester motif"
    elif ester_count >= 3:
        is_prodrug = True
        reason = f"multiple ester/carbonate motifs ({ester_count})"
    elif has_ester and has_di_n_ring:
        is_prodrug = True
        reason = "nucleoside ester motif (ester + di-N heterocycle)"
    elif has_phenol_ester and not has_cooh:
        is_prodrug = True
        reason = "phenol ester without free carboxylic acid"

    if is_prodrug:
        flags.append("PRODRUG")
        logger.info("Prodrug detected: %s", reason)

    return flags


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
    ad_flags: tuple[str, ...]


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


def _classify_from_pka(pka_acidic: float, pka_basic: float) -> tuple[float | None, str]:
    """ChemAxon pKa → (pka_for_rr, compound_type).

    Henderson-Hasselbalch thresholds:
    - acidic < 7.0: >71% ionized at pH 7.4, R&R ion_ratio effect 43%+
    - basic > 8.0: >80% protonated at pH 7.4, R&R ion_ratio effect 121%+
    - pKa 7.0-8.0: uncertainty zone → neutral (safe default)
    """
    is_acidic = pka_acidic < 7.0
    is_basic = pka_basic > 8.0

    if is_acidic and is_basic:
        return pka_basic, "zwitterion"
    elif is_acidic:
        return pka_acidic, "acid"
    elif is_basic:
        return pka_basic, "base"
    else:
        return None, "neutral"


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

    canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    rotatable_bonds = Descriptors.NumRotatableBonds(mol)

    # ── DrugBank overrides ──
    from sisyphus.predict.drugbank import drugbank_lookup
    db = drugbank_lookup()

    db_logp = db.get_logp(canonical)
    if db_logp is not None:
        logp = db_logp  # experimental overrides Crippen

    # logP correction (residual learning) — only for Crippen, not experimental
    _LOGP_CORR_PATH = Path(__file__).resolve().parent.parent.parent.parent / "models" / "adme" / "logp_correction.json"
    if db_logp is None and _LOGP_CORR_PATH.exists():
        try:
            import xgboost as xgb
            if not hasattr(compute_profile, "_logp_model"):
                m = xgb.XGBRegressor()
                m.load_model(str(_LOGP_CORR_PATH))
                compute_profile._logp_model = m  # type: ignore[attr-defined]
            import numpy as np
            corr_features = np.array([[logp, mw, tpsa, float(hbd), float(hba), float(rotatable_bonds)]])
            correction = float(compute_profile._logp_model.predict(corr_features)[0])  # type: ignore[attr-defined]
            logp = logp + correction
        except Exception as e:
            logger.warning("logP correction failed: %s", e)

    # pKa: DrugBank ChemAxon → fallback SMARTS
    db_pka = db.get_pka(canonical)
    if db_pka is not None:
        pka, compound_type = _classify_from_pka(db_pka[0], db_pka[1])
    else:
        pka, compound_type = _estimate_pka_type(mol, logp)

    # _check_ad uses the local `logp` which may be overridden — correct per spec
    ad_flags = _check_ad(mol, mw, logp, tpsa)

    # Prodrug detection — adds PRODRUG flag if ester/phosphonate prodrug motifs found
    prodrug_flags = _check_prodrug(mol)
    ad_flags.extend(prodrug_flags)

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
        ad_flags=tuple(ad_flags),
    )
