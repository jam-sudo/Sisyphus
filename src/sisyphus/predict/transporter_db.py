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


@functools.lru_cache(maxsize=1)
def _load_oatp_applicability_index() -> dict[str, bool]:
    """InChIKey → ecm_applicable flag, indexed once.

    Returns empty dict if registry missing (inherited from
    _load_oatp1b1_table fallback). Drugs whose entries omit the field
    are treated as False.
    """
    table = _load_oatp1b1_table()
    return {
        entry["inchikey"]: bool(entry.get("ecm_applicable", False))
        for entry in table.values()
        if "inchikey" in entry
    }


def is_oatp_ecm_applicable(smiles: str) -> bool:
    """Return True if SMILES's InChIKey is registered with ecm_applicable=true.

    Mirrors cyp_clearance_overrides.lookup_metabolic_fraction (PR #22) for
    the parallel ecm_applicable lookup. Uses RDKit-canonical InChIKey to
    be robust against SMILES annotation differences. Returns False on any
    error (RDKit unavailable, invalid SMILES, InChIKey not registered, or
    registered with explicit false).
    """
    if not smiles:
        return False
    try:
        from rdkit import Chem
    except ImportError:
        return False
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    ikey = Chem.MolToInchiKey(mol)
    return _load_oatp_applicability_index().get(ikey, False)


@functools.lru_cache(maxsize=1)
def _load_oatp1b1_name_index() -> dict[str, str]:
    """Full InChIKey (27 chars) → drug name reverse index from oatp1b1.json.

    Distinct from _load_oatp1b1_inchikey_index which uses the 14-char
    connectivity block for fuzzy/stereo-agnostic substrate detection.
    This index uses full InChIKey per spec §1.2 to avoid wrong-connectivity
    collisions (e.g. issue #25).  Returns empty dict if registry missing.
    """
    table = _load_oatp1b1_table()
    return {
        entry["inchikey"]: name
        for name, entry in table.items()
        if "inchikey" in entry
    }


def _smiles_to_drug_name(smiles: str) -> str | None:
    """Resolve SMILES → registered drug name via RDKit InChIKey, or None.

    Uses the full 27-character InChIKey (spec §1.2).  Mirrors the
    cyp_clearance_overrides PR #22 graceful-fallback style: empty SMILES
    → None, RDKit unavailable → None, invalid SMILES → None, InChIKey not
    in registry → None.
    """
    if not smiles:
        return None
    try:
        from rdkit import Chem
    except ImportError:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    ikey = Chem.MolToInchiKey(mol)
    return _load_oatp1b1_name_index().get(ikey)


def load_oatp1b1_kinetics_for_smiles(smiles: str) -> dict[str, TransporterKinetics] | None:
    """SMILES-keyed wrapper around load_oatp1b1_kinetics.  Returns None if unregistered.

    Mirrors the cyp_clearance_overrides.lookup_metabolic_fraction PR #22 pattern.
    Lookup is by full InChIKey (spec §1.2) so SMILES variants of the same
    molecule resolve correctly while avoiding connectivity collisions.
    """
    name = _smiles_to_drug_name(smiles)
    if name is None:
        return None
    return load_oatp1b1_kinetics(name)


def load_hepatic_ecm_params_for_smiles(smiles: str) -> dict[str, Distribution] | None:
    """SMILES-keyed wrapper around load_hepatic_ecm_params.  Returns None if unregistered.

    Same InChIKey lookup strategy as load_oatp1b1_kinetics_for_smiles.
    """
    name = _smiles_to_drug_name(smiles)
    if name is None:
        return None
    return load_hepatic_ecm_params(name)


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
