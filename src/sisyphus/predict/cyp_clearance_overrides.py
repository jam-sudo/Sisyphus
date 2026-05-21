"""Per-drug metabolic-CLint scale factors for OATP1B1-mediated substrates.

When the engine's ECM transporter path is active for a drug, the XGBoost
hepatocyte CLint that was decomposed into per-enzyme affinities double-counts
clearance — the in-vitro hepatocyte assay already includes uptake. This
registry lets us scale the metabolic (CYP/UGT) affinities for known
transporter-dominant substrates so the engine's ECM path provides the entire
hepatic clearance without the metabolic path adding on top.

Default for any drug not in the registry is 1.0 (no scaling = current
behavior). Lookup is by canonical InChIKey (RDKit) for robustness against
SMILES variants.

Registry file: ``data/transporters/cyp_clearance_overrides.json``.
"""

from __future__ import annotations

import json
import logging
import pathlib
from functools import lru_cache

logger = logging.getLogger(__name__)

_REGISTRY_PATH = (
    pathlib.Path(__file__).resolve().parent.parent.parent.parent
    / "data"
    / "transporters"
    / "cyp_clearance_overrides.json"
)


@lru_cache(maxsize=1)
def _load() -> tuple[dict[str, float], dict[str, list[float]]]:
    """Load the registry once; index by full InChIKey and connectivity block.

    The connectivity index lets us match a non-isomeric query SMILES against
    a stereospecific override (and vice versa) — both forms share their
    first InChIKey block. Connectivity matches are only honored when the
    block is unambiguous (one override per connectivity); collisions
    fail-safe to the default 1.0.
    """
    if not _REGISTRY_PATH.exists():
        logger.debug("cyp_clearance_overrides: registry not found at %s", _REGISTRY_PATH)
        return {}, {}
    with _REGISTRY_PATH.open() as f:
        data = json.load(f)
    full_index: dict[str, float] = {}
    conn_index: dict[str, list[float]] = {}
    for entry in data.get("overrides", []):
        ikey = entry.get("inchikey")
        frac = entry.get("metabolic_fraction")
        if ikey is None or frac is None:
            continue
        full_index[ikey] = float(frac)
        conn_index.setdefault(ikey.split("-", maxsplit=1)[0], []).append(float(frac))
    return full_index, conn_index


def lookup_metabolic_fraction(smiles: str) -> float:
    """Return the metabolic_fraction for ``smiles``, or 1.0 if not registered.

    Lookup is by RDKit InChIKey to be SMILES-variant-robust. Falls back to
    InChIKey-connectivity matching when the full key misses, so a
    stereospecific override applies to a non-isomeric query SMILES (and
    vice versa). If RDKit is unavailable or the SMILES is invalid, returns
    1.0 (fail-safe default).
    """
    try:
        from rdkit import Chem
    except ImportError:
        return 1.0
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 1.0
    ikey = Chem.MolToInchiKey(mol)
    full_index, conn_index = _load()
    if ikey in full_index:
        return full_index[ikey]
    matches = conn_index.get(ikey.split("-", maxsplit=1)[0], [])
    if len(matches) == 1:
        return matches[0]
    return 1.0
