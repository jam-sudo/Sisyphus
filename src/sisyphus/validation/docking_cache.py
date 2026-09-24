"""Docking cache identity: retain legacy connectivity files without mixing isomers."""

import json
from hashlib import sha256
from pathlib import Path

from rdkit import Chem


def cache_path(cache_dir: Path, smiles: str, cyp: str) -> Path:
    """Key by full InChIKey and exact canonical isomeric structure."""
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError(f"Invalid docking SMILES: {smiles!r}")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError(f"Invalid docking SMILES: {smiles!r}")
    canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
    digest = sha256(canonical.encode()).hexdigest()
    return cache_dir / f"{Chem.MolToInchiKey(mol)}_{digest}_{cyp}.json"


def load_matching_cache(cache_dir: Path, smiles: str, cyp: str) -> dict | None:
    """Use a legacy connectivity cache only when its stored molecule is identical."""
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    full_key = Chem.MolToInchiKey(mol)
    canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
    paths = (
        cache_path(cache_dir, smiles, cyp),
        cache_dir / f"{full_key}_{cyp}.json",
        cache_dir / f"{full_key[:14]}_{cyp}.json",
    )
    for path in paths:
        if not path.is_file():
            continue
        data = json.loads(path.read_text())
        stored = data.get("smiles") if isinstance(data, dict) else None
        if not isinstance(stored, str) or not stored:
            continue
        cached_mol = Chem.MolFromSmiles(stored)
        if cached_mol and Chem.MolToSmiles(cached_mol, isomericSmiles=True) == canonical:
            return data
    return None
