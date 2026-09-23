"""Structure identity for holdout and training-corpus exclusion."""

from __future__ import annotations

from rdkit import Chem


def _largest_organic_fragment(mol: Chem.Mol) -> Chem.Mol:
    """Remove counterions while retaining the largest carbon-containing fragment."""
    fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if len(fragments) <= 1:
        return mol
    organic = [frag for frag in fragments if any(a.GetAtomicNum() == 6 for a in frag.GetAtoms())]
    candidates = organic or list(fragments)
    return max(candidates, key=lambda frag: (frag.GetNumHeavyAtoms(), frag.GetNumAtoms()))


def ik14(smiles: str | None) -> str | None:
    """Salt-stripped, stereo-insensitive InChIKey connectivity block."""
    if not smiles or not isinstance(smiles, str):
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = _largest_organic_fragment(mol)
    try:
        key = Chem.MolToInchiKey(mol)
    except Exception:
        return None
    return key[:14] if key else None
