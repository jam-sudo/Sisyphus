"""Transporter kinetics database loader.

Reads per-drug Jmax/Km from data/transporters/<transporter>.json and
returns them as TransporterKinetics instances ready to plug into
DrugOnGraph.transporter_kinetics.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path

from sisyphus.core import Distribution, TransporterKinetics

_DATA_ROOT = Path(__file__).resolve().parents[3] / "data" / "transporters"
_OATP1B1_FILE = _DATA_ROOT / "oatp1b1.json"


@functools.lru_cache(maxsize=1)
def _load_oatp1b1_table() -> dict[str, dict]:
    if not _OATP1B1_FILE.exists():
        return {}
    with _OATP1B1_FILE.open() as f:
        data = json.load(f)
    return {k.lower(): v for k, v in data.get("drugs", {}).items()}


@functools.lru_cache(maxsize=1)
def _load_oatp1b1_inchikey_index() -> dict[str, str]:
    """InChIKey-connectivity-block -> drug_name mapping for SMILES-based
    substrate detection.

    Keys are the first 14 characters of the InChIKey (the connectivity
    block, stereo-independent). This makes the lookup robust to SMILES
    sources that strip stereochemistry annotations (e.g. some reference
    datasets). False positives across the 7 currently-registered
    substrates are not a concern because all have distinct
    connectivity.
    """
    table = _load_oatp1b1_table()
    return {
        entry["inchikey"][:14]: name
        for name, entry in table.items()
        if "inchikey" in entry
    }


def find_oatp1b1_substrate_name(smiles: str) -> str | None:
    """Return the registered OATP1B1 substrate's drug name for *smiles*, or None.

    Lookup is via the InChIKey connectivity block (first 14 chars), so
    SMILES variants of the same molecule resolve regardless of whether
    stereochemistry is fully annotated in the input. Returns None if
    RDKit is unavailable, the SMILES is invalid, or the molecule is
    not a registered OATP1B1 substrate.
    """
    try:
        from rdkit import Chem
    except ImportError:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    ikey = Chem.MolToInchiKey(mol)
    return _load_oatp1b1_inchikey_index().get(ikey[:14])


def load_oatp1b1_kinetics(drug_name: str) -> dict[str, TransporterKinetics] | None:
    """Return ``{'OATP1B1': TransporterKinetics}`` for *drug_name*, or None.

    Lookup is case-insensitive on the drug name.
    """
    table = _load_oatp1b1_table()
    entry = table.get(drug_name.lower())
    if entry is None:
        return None

    jmax_spec = entry["jmax_pmol_per_min_per_mg"]
    km_spec = entry["km_uM"]
    return {
        "OATP1B1": TransporterKinetics(
            jmax=Distribution(mean=float(jmax_spec["mean"]), cv=float(jmax_spec["cv"])),
            km=Distribution(mean=float(km_spec["mean"]), cv=float(km_spec["cv"])),
        )
    }


_HEPATIC_ECM_FILE = _DATA_ROOT / "hepatic_ecm.json"


@functools.lru_cache(maxsize=1)
def _load_hepatic_ecm_table() -> dict[str, dict]:
    if not _HEPATIC_ECM_FILE.exists():
        return {}
    with _HEPATIC_ECM_FILE.open() as f:
        data = json.load(f)
    return {k.lower(): v for k, v in data.get("drugs", {}).items()}


def load_hepatic_ecm_params(drug_name: str) -> dict[str, Distribution] | None:
    """Return ``{'ps_passive': Distribution, 'ps_eff': Distribution, 'cl_int_bile': Distribution}``
    for *drug_name*, or ``None`` if the drug has no entry. Case-insensitive.
    """
    table = _load_hepatic_ecm_table()
    entry = table.get(drug_name.lower())
    if entry is None:
        return None
    return {
        "ps_passive": Distribution(
            mean=float(entry["ps_passive_L_per_h"]["mean"]),
            cv=float(entry["ps_passive_L_per_h"]["cv"]),
        ),
        "ps_eff": Distribution(
            mean=float(entry["ps_eff_L_per_h"]["mean"]),
            cv=float(entry["ps_eff_L_per_h"]["cv"]),
        ),
        "cl_int_bile": Distribution(
            mean=float(entry["cl_int_bile_L_per_h"]["mean"]),
            cv=float(entry["cl_int_bile_L_per_h"]["cv"]),
        ),
    }
