"""Shared molecular descriptor computation.

Computes Morgan fingerprints and RDKit descriptors used by both
the predict and ml layers.  This is a shared utility at the package
root -- not part of any layer -- so both layers can import from here
without cross-layer dependencies.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator

# Morgan fingerprint generator (radius=2, 2048 bits). The new rdFingerprintGenerator
# API replaces the deprecated AllChem.GetMorganFingerprintAsBitVect; verified
# bit-identical across all 368 reference SMILES, so XGBoost features (and the 2.731
# headline) are unchanged. Constructed once at import — it is reusable across molecules.
_MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

# ---------------------------------------------------------------------------
# Ionization block (compute_features_v2 only)
#
# The v1 vector carries no ionization state: LogP, TPSA, MW, HBA, HBD,
# RotBonds, Rings, FractionCSP3 and MolMR are all charge-blind, and the Morgan
# bits encode the substructure without separating an acid from a base.  The
# training labels do separate them — on the holdout-excluded Omega Cmax corpus
# mean log10(Cmax/dose) runs -1.664 (acid) to -2.666 (base), a 1.0-log class
# effect — so the models had a signal they could not represent.
#
# Only groups whose ionization state at pH 7.4 is not in doubt are counted, so
# no pKa prediction enters (a predicted pKa fed to R&R was already rejected,
# DE-08).  Aryl amines are counted SEPARATELY from aliphatic amines: aniline
# pKaH 4.6 and diphenylamine pKaH 0.8 are un-protonated at pH 7.4 while an
# aliphatic amine (pKaH 9-11) is protonated.
# Source: Perrin, Dissociation Constants of Organic Bases in Aqueous Solution,
# Butterworths (1965); Williams & Lemke, Foye's Principles of Medicinal
# Chemistry, 5th ed., Ch. 2 (acid/base character of drug functional groups).
# ---------------------------------------------------------------------------
_IONIZATION_SMARTS: dict[str, str] = {
    # Strong acids — fully or largely ionized at pH 7.4.
    "cooh": "[CX3](=O)[OX2H1]",
    "sulfonic": "[SX4](=O)(=O)[OX2H1]",
    "acylsulfonamide": "[SX4](=O)(=O)[NX3H1][CX3]=O",
    "tetrazole": "c1nnn[nH]1",
    "phosphoacid": "[PX4](=O)([OX2H1])[OX2H1]",
    # Strong bases — largely protonated at pH 7.4.  Excludes amide,
    # sulfonamide, imine, nitrile, hydrazine/hydroxylamine and quaternary N.
    "aliph_amine": (
        "[NX3;H2,H1,H0;!$(N[a]);!$(NC=O);!$(NS=O);!$(N=*);!$(N#*);"
        "!$([N+]);!$(N[N,O])]"
    ),
    "guanidine": "[NX3][CX3](=[NX2])[NX3]",
    "amidine": "[NX3][CX3]=[NX2]",
    # Counted but NOT treated as basic — see the aryl-amine note above.
    "aryl_amine": "[NX3;H2,H1,H0;$(N[a]);!$(NC=O);!$(NS=O);!$(N=*)]",
    "basic_arom_n": "[nX2;!$([n+])]",
}
_IONIZATION_PATTERNS = {
    key: Chem.MolFromSmarts(smarts) for key, smarts in _IONIZATION_SMARTS.items()
}

_STRONG_ACID_KEYS = ("cooh", "sulfonic", "acylsulfonamide", "tetrazole", "phosphoacid")
_STRONG_BASE_KEYS = ("aliph_amine", "guanidine", "amidine")

IONIZATION_FEATURE_NAMES: tuple[str, ...] = (
    "n_cooh",
    "n_sulfonic",
    "n_acylsulfonamide",
    "n_tetrazole",
    "n_phosphoacid",
    "n_aliph_amine",
    "n_guanidine",
    "n_amidine",
    "n_aryl_amine",
    "n_basic_arom_n",
    "has_strong_acid",
    "has_strong_base",
    "net_anion",
    "net_cation",
    "is_zwitterion",
    "is_nonionizable",
    "formal_charge",
)
N_IONIZATION_FEATURES = len(IONIZATION_FEATURE_NAMES)


def _ionization_features(mol: Chem.Mol) -> NDArray[np.float64]:
    """Return the 17-element pH-7.4 ionization block for ``mol``."""
    counts = {
        key: len(mol.GetSubstructMatches(pattern))
        for key, pattern in _IONIZATION_PATTERNS.items()
    }
    has_acid = float(any(counts[k] for k in _STRONG_ACID_KEYS))
    has_base = float(any(counts[k] for k in _STRONG_BASE_KEYS))
    return np.array(
        [
            counts["cooh"],
            counts["sulfonic"],
            counts["acylsulfonamide"],
            counts["tetrazole"],
            counts["phosphoacid"],
            counts["aliph_amine"],
            counts["guanidine"],
            counts["amidine"],
            counts["aryl_amine"],
            counts["basic_arom_n"],
            has_acid,
            has_base,
            has_acid * (1.0 - has_base),  # net anion at pH 7.4
            has_base * (1.0 - has_acid),  # net cation at pH 7.4
            has_acid * has_base,  # zwitterion
            (1.0 - has_acid) * (1.0 - has_base),  # non-ionizable
            float(Chem.GetFormalCharge(mol)),
        ],
        dtype=np.float64,
    )


def compute_features(smiles: str) -> NDArray[np.float64]:
    """Compute 2057-element feature vector for XGBoost models.

    Features: Morgan FP (2048 bits, radius=2) + 9 normalized RDKit descriptors.

    Descriptor order and normalization:
        0. LogP (Crippen) / 1.0
        1. TPSA / 200.0
        2. MW / 600.0
        3. NumHAcceptors / 10.0
        4. NumHDonors / 5.0
        5. NumRotatableBonds / 15.0
        6. RingCount / 5.0
        7. FractionCSP3 / 1.0
        8. MolMR / 150.0

    Args:
        smiles: Input SMILES.

    Returns:
        1-D float64 array of length 2057.

    Raises:
        ValueError: If the SMILES string is invalid.
    """
    if not smiles or not smiles.strip():
        raise ValueError(f"Invalid SMILES: {smiles!r}")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError(f"Invalid SMILES: {smiles!r}")

    # Morgan fingerprint (2048 bits, radius 2)
    fp = _MORGAN_GEN.GetFingerprint(mol)
    fp_array = np.zeros(2048, dtype=np.float64)
    for bit in fp.GetOnBits():
        fp_array[bit] = 1.0

    # 9 RDKit descriptors (normalized)
    descriptors = np.array(
        [
            Descriptors.MolLogP(mol),  # LogP / 1.0
            Descriptors.TPSA(mol) / 200.0,  # TPSA
            Descriptors.MolWt(mol) / 600.0,  # MW
            Descriptors.NumHAcceptors(mol) / 10.0,  # HBA
            Descriptors.NumHDonors(mol) / 5.0,  # HBD
            Descriptors.NumRotatableBonds(mol) / 15.0,  # RotBonds
            Descriptors.RingCount(mol) / 5.0,  # Rings
            Descriptors.FractionCSP3(mol),  # FractionCSP3 / 1.0
            Descriptors.MolMR(mol) / 150.0,  # MolMR
        ],
        dtype=np.float64,
    )

    return np.concatenate([fp_array, descriptors])


def compute_features_v2(smiles: str) -> NDArray[np.float64]:
    """Compute 2074-element feature vector: the v1 vector plus ionization state.

    Layout:
        0-2047     Morgan FP (2048 bits, radius=2) — as v1
        2048-2056  the 9 normalized RDKit descriptors — as v1, same order
        2057-2073  the pH-7.4 ionization block, ``IONIZATION_FEATURE_NAMES``

    The first 2057 elements are bit-identical to :func:`compute_features`, so a
    v2 model sees the v1 inputs unchanged plus the charge state that v1 could
    not express.  Measured on each target's own holdout-excluded training CSV
    under the repo's 5-fold Murcko scaffold split (out-of-fold): Cmax R^2
    0.337->0.401, VDss 0.483->0.507, CLint 0.217->0.241, fup 0.456->0.493,
    CL/F 0.178->0.187, and the acid/base/zwitterion/neutral bias spread falls
    by about half or more on all five.

    Models record which function trained them in their manifest's
    ``feature_schema.name``; v1 artifacts stay valid and keep using
    :func:`compute_features`.

    Args:
        smiles: Input SMILES.

    Returns:
        1-D float64 array of length 2074.

    Raises:
        ValueError: If the SMILES string is invalid.
    """
    base = compute_features(smiles)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:  # pragma: no cover - compute_features raises
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    return np.concatenate([base, _ionization_features(mol)])
