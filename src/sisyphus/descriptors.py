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
