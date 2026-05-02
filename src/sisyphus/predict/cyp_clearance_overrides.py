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
def _load() -> dict[str, float]:
    """Load the registry once and index by InChIKey → metabolic_fraction."""
    if not _REGISTRY_PATH.exists():
        logger.debug("cyp_clearance_overrides: registry not found at %s", _REGISTRY_PATH)
        return {}
    with _REGISTRY_PATH.open() as f:
        data = json.load(f)
    index: dict[str, float] = {}
    for entry in data.get("overrides", []):
        ikey = entry.get("inchikey")
        frac = entry.get("metabolic_fraction")
        if ikey is None or frac is None:
            continue
        index[ikey] = float(frac)
    return index


def lookup_metabolic_fraction(smiles: str) -> float:
    """Return the metabolic_fraction for ``smiles``, or 1.0 if not registered.

    Lookup is by RDKit InChIKey to be SMILES-variant-robust. If RDKit is
    unavailable or the SMILES is invalid, returns 1.0 (default no-scaling
    behavior, fail-safe).
    """
    try:
        from rdkit import Chem
    except ImportError:
        return 1.0
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 1.0
    ikey = Chem.MolToInchiKey(mol)
    return _load().get(ikey, 1.0)
