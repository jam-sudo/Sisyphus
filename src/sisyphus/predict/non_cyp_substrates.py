"""Non-CYP enzyme substrate registries for NAT2 and UGT1A1 phenotype propagation.

Two JSON registries (data/enzymes/nat2_substrates.json,
data/enzymes/ugt1a1_substrates.json) keyed by full RDKit InChIKey hold
per-drug metabolic_fraction values. predict() calls get_non_cyp_fractions()
to obtain the dict passed downstream to _get_fm_fractions; that fraction
of XGBoost CLint is then routed through the named enzyme so phenotype
scaling on liver.enzymes[NAT2 or UGT1A1] propagates into engine rate.

Mirrors transporter_db.py (PR #29) — lru_cache JSON loaders, full
InChIKey matching only (no block-1 truncation), file-anchored paths.
"""
from __future__ import annotations

import json
import logging
import pathlib
from functools import lru_cache

logger = logging.getLogger(__name__)

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_NAT2_PATH = _REPO_ROOT / "data" / "enzymes" / "nat2_substrates.json"
_UGT1A1_PATH = _REPO_ROOT / "data" / "enzymes" / "ugt1a1_substrates.json"


def _smiles_to_inchikey(smiles: str) -> str | None:
    """Return RDKit InChIKey for a SMILES, or None on parse failure."""
    if not smiles:
        return None
    try:
        from rdkit import Chem
    except ImportError:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToInchiKey(mol)


@lru_cache(maxsize=1)
def _load_nat2_index() -> dict[str, dict]:
    """Return {inchikey: substrate_entry} for nat2_substrates.json."""
    if not _NAT2_PATH.exists():
        return {}
    data = json.loads(_NAT2_PATH.read_text())
    return {entry["inchikey"]: entry for entry in data.get("substrates", [])}


@lru_cache(maxsize=1)
def _load_ugt1a1_index() -> dict[str, dict]:
    """Return {inchikey: substrate_entry} for ugt1a1_substrates.json."""
    if not _UGT1A1_PATH.exists():
        return {}
    data = json.loads(_UGT1A1_PATH.read_text())
    return {entry["inchikey"]: entry for entry in data.get("substrates", [])}


def lookup_nat2_substrate(smiles: str) -> dict | None:
    """Return the registry entry if the SMILES matches a NAT2 substrate.

    Lookup is by full RDKit InChIKey (rejects block-1 truncation per
    issue #25 lessons). Returns None for missing / invalid SMILES.
    """
    ikey = _smiles_to_inchikey(smiles)
    if ikey is None:
        return None
    return _load_nat2_index().get(ikey)


def lookup_ugt1a1_substrate(smiles: str) -> dict | None:
    """Return the registry entry if the SMILES matches a UGT1A1 substrate."""
    ikey = _smiles_to_inchikey(smiles)
    if ikey is None:
        return None
    return _load_ugt1a1_index().get(ikey)


def get_non_cyp_fractions(smiles: str) -> dict[str, float]:
    """Aggregate NAT2 + UGT1A1 metabolic fractions for the given SMILES.

    Returns {gene: metabolic_fraction} ready to pass into _get_fm_fractions.
    Empty dict if no substrate match. If multi-gene total exceeds 1.0
    (round-off or curation overlap), values are re-normalized to sum=1.0
    and a logger.info message is emitted.
    """
    out: dict[str, float] = {}
    nat2 = lookup_nat2_substrate(smiles)
    if nat2 is not None:
        out["NAT2"] = float(nat2["metabolic_fraction"])
    ugt = lookup_ugt1a1_substrate(smiles)
    if ugt is not None:
        out["UGT1A1"] = float(ugt["metabolic_fraction"])
    total = sum(out.values())
    if total > 1.0:
        logger.info(
            "non_cyp_fractions sum %.3f > 1.0 for SMILES %r; re-normalizing",
            total, smiles,
        )
        out = {k: v / total for k, v in out.items()}
    return out
