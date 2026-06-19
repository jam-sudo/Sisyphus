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
_UGT2B7_PATH = _REPO_ROOT / "data" / "enzymes" / "ugt2b7_substrates.json"
_UGT1A9_PATH = _REPO_ROOT / "data" / "enzymes" / "ugt1a9_substrates.json"
_UGT_IVIVE_SF_PATH = _REPO_ROOT / "data" / "enzymes" / "ugt_ivive_sf.json"


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


@lru_cache(maxsize=1)
def _load_ugt2b7_index() -> dict[str, dict]:
    """Return {inchikey: substrate_entry} for ugt2b7_substrates.json."""
    if not _UGT2B7_PATH.exists():
        return {}
    data = json.loads(_UGT2B7_PATH.read_text())
    return {entry["inchikey"]: entry for entry in data.get("substrates", [])}


@lru_cache(maxsize=1)
def _load_ugt1a9_index() -> dict[str, dict]:
    """Return {inchikey: substrate_entry} for ugt1a9_substrates.json."""
    if not _UGT1A9_PATH.exists():
        return {}
    data = json.loads(_UGT1A9_PATH.read_text())
    return {entry["inchikey"]: entry for entry in data.get("substrates", [])}


@lru_cache(maxsize=1)
def _load_ugt_ivive_sf_index() -> dict[str, dict]:
    """Return {inchikey: entry} for ugt_ivive_sf.json (B-14)."""
    if not _UGT_IVIVE_SF_PATH.exists():
        return {}
    data = json.loads(_UGT_IVIVE_SF_PATH.read_text())
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


def lookup_ugt2b7_substrate(smiles: str) -> dict | None:
    """Return the registry entry if the SMILES matches a UGT2B7 substrate."""
    ikey = _smiles_to_inchikey(smiles)
    if ikey is None:
        return None
    return _load_ugt2b7_index().get(ikey)


def lookup_ugt1a9_substrate(smiles: str) -> dict | None:
    """Return the registry entry if the SMILES matches a UGT1A9 substrate."""
    ikey = _smiles_to_inchikey(smiles)
    if ikey is None:
        return None
    return _load_ugt1a9_index().get(ikey)


def get_non_cyp_fractions(smiles: str) -> dict[str, float]:
    """Aggregate NAT2 + UGT1A1 + UGT2B7 + UGT1A9 metabolic fractions for the given SMILES.

    Returns {gene: metabolic_fraction} ready to pass into _get_fm_fractions.
    Empty dict if no substrate match. If multi-gene total exceeds 1.0
    (round-off or curation overlap; the cross-registry duplicate test
    enforces no overlap, but re-normalization is a safety net), values
    are re-normalized to sum=1.0 and a logger.info message is emitted.

    B-02 Phase 2 (2026-05-26): UGT2B7 + UGT1A9 added; spec
    docs/_internal/specs/2026-05-26-B02-ugt-public-registry-design.md.
    """
    out: dict[str, float] = {}
    for gene, lookup in [
        ("NAT2",   lookup_nat2_substrate),
        ("UGT1A1", lookup_ugt1a1_substrate),
        ("UGT2B7", lookup_ugt2b7_substrate),
        ("UGT1A9", lookup_ugt1a9_substrate),
    ]:
        entry = lookup(smiles)
        if entry is not None:
            out[gene] = float(entry["metabolic_fraction"])
    total = sum(out.values())
    if total > 1.0:
        logger.info(
            "non_cyp_fractions sum %.3f > 1.0 for SMILES %r; re-normalizing",
            total, smiles,
        )
        out = {k: v / total for k, v in out.items()}
    return out


# --- IVIVE magnitude correction (B-14) ---------------------------------------
# Distinct from the fm-routing lookups above: fm decides WHICH enzyme carries the
# clearance; this SF decides HOW MUCH the in-vitro CLint under-predicts in vivo.
def get_ugt_ivive_sf(smiles: str) -> dict[str, float]:
    """Return {UGT_tag: scaling_factor} for the SMILES, or {} if unlisted/invalid.

    UNLIKE the lookup_* functions above (which return None), this returns a dict
    and NEVER raises: invalid SMILES -> {}. The {} default makes the caller's
    ``.get(enzyme, 1.0)`` a bit-identical no-op. See spec
    docs/_internal/specs/2026-05-30-hepatic-ugt-ivive-differential-design.md.
    """
    ikey = _smiles_to_inchikey(smiles)
    if ikey is None:
        return {}
    entry = _load_ugt_ivive_sf_index().get(ikey)
    if entry is None:
        return {}
    return {k: float(v) for k, v in entry.get("ivive_sf", {}).items()}
