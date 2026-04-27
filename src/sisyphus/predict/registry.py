"""Prodrug activation registry — SMILES-keyed config loader.

Maps canonical SMILES → ActiveMetabolite + observation_species.
Used by predict.ivive.build_drug_on_graph to attach prodrug activation
configs to DrugOnGraph instances.

Registry file: ``data/sbi/prodrug_activation_registry.json``
Schema: see docs/superpowers/specs/2026-04-24-prodrug-activation-design.md §4.8
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path

from rdkit import Chem

from sisyphus.core import ActiveMetabolite, Distribution

logger = logging.getLogger(__name__)

_DEFAULT_REGISTRY_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "data" / "sbi" / "prodrug_activation_registry.json"
)


def _canonicalize(smiles: str) -> str | None:
    """Convert SMILES to canonical form via RDKit. Returns None on parse error."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


@lru_cache(maxsize=1)
def _load_registry_cached(path_str: str) -> dict:
    path = Path(path_str)
    if not path.exists():
        logger.warning("prodrug_activation_registry not found at %s", path)
        return {}
    with path.open() as f:
        return json.load(f)


_REQUIRED_FIELDS = frozenset({
    "name", "mw", "fup", "CL_per_h", "Vd_L",
    "conversion_rate_per_h", "conversion_site",
    "conversion_yield_fraction",
})


def _build_active_metabolite(entry: dict, smiles: str) -> ActiveMetabolite:
    """Construct ActiveMetabolite from registry entry; validate fields."""
    missing = _REQUIRED_FIELDS - set(entry.keys())
    if missing:
        raise ValueError(
            f"prodrug_activation_registry entry for SMILES {smiles!r} "
            f"missing field {sorted(missing)}"
        )

    if entry["mw"] <= 0:
        raise ValueError(f"mw must be positive, got {entry['mw']}")

    cr = entry["conversion_rate_per_h"]
    if cr["mean"] <= 0:
        raise ValueError(
            f"conversion_rate must be positive, got {cr['mean']}"
        )

    cy = entry["conversion_yield_fraction"]
    if not (0.0 <= cy["mean"] <= 1.0):
        raise ValueError(
            f"conversion_yield must be in [0, 1], got {cy['mean']}"
        )

    if entry["CL_per_h"]["mean"] <= 0 or entry["Vd_L"]["mean"] <= 0:
        raise ValueError("CL and Vd must be positive")

    return ActiveMetabolite(
        name=entry["name"],
        mw=float(entry["mw"]),
        fup=Distribution(**entry["fup"]),
        CL_per_h=Distribution(**entry["CL_per_h"]),
        Vd_L=Distribution(**entry["Vd_L"]),
        conversion_rate_per_h=Distribution(**entry["conversion_rate_per_h"]),
        conversion_site=str(entry["conversion_site"]),
        conversion_yield_fraction=Distribution(**entry["conversion_yield_fraction"]),
    )


def lookup_active_metabolite(
    smiles: str, registry_path: Path | None = None
) -> tuple[ActiveMetabolite, str] | None:
    """Look up SMILES in prodrug registry.

    Returns ``(ActiveMetabolite, observation_species)`` or ``None`` if not found.
    Raises ``ValueError`` on invalid registry entries.

    Args:
        smiles: SMILES string (any form; canonicalized internally).
        registry_path: Override registry file path (default: data/sbi/...).
    """
    canonical = _canonicalize(smiles)
    if canonical is None:
        return None

    path = registry_path if registry_path is not None else _DEFAULT_REGISTRY_PATH
    # Bypass lru_cache when a test-specific path is supplied
    if registry_path is not None:
        with path.open() as f:
            registry = json.load(f)
    else:
        registry = _load_registry_cached(str(path))

    entry = registry.get(canonical)
    if entry is None:
        return None

    obs_species = entry.get("observation_species", "active")
    if obs_species not in ("parent", "active"):
        raise ValueError(
            f"observation_species must be 'parent' or 'active', got {obs_species!r}"
        )

    am = _build_active_metabolite(entry, canonical)
    return am, obs_species
