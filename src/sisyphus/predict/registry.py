"""Prodrug activation registry — SMILES-keyed config loader (v2).

Maps canonical SMILES → (ActiveMetabolite, observation_species,
enzyme_affinity_for_conversion, enzyme_yields).
Used by predict.ivive.build_drug_on_graph to attach prodrug activation
configs to DrugOnGraph instances.

Registry file: ``data/sbi/prodrug_activation_registry.json``
Schema: see docs/superpowers/specs/2026-04-27-prodrug-activation-v2-design.md §4.7
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


_VALID_AFFINITY_SOURCES = frozenset({"literature", "literature_ivive", "class_extrapolated"})
"""v2 rejects 'infrastructure_only' (tier 3); see spec §3.3.

'literature_ivive' (added 2026-05-25 by B-03.x doctrine completion sprint):
distinguishes affinities derived from in-vitro Vmax/Km × abundance IVIVE
(literature-anchored numeric derivation) from raw 'literature' (a single
literature value taken verbatim). Both are doctrinally equivalent (no Cmax
tuning); the distinction is provenance documentation, not validation logic.
"""

_VALID_YIELD_SOURCES = frozenset({"literature", "class_extrapolated"})


def _canonicalize(smiles: str) -> str | None:
    """Convert SMILES to canonical form via RDKit. Returns None on parse error."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def _inchikey_connectivity(smiles: str) -> str | None:
    """Return the connectivity block of the RDKit InChIKey for ``smiles``.

    This fallback lets stereospecific registry entries match non-isomeric
    clinical reference SMILES for the same connectivity. The registry remains
    keyed by canonical SMILES for normal lookups.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToInchiKey(mol).split("-", maxsplit=1)[0]


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
    "conversion_yield_fraction",
    "yield_source",
    "enzyme_affinity_for_conversion",
    "affinity_source",
    "observation_species",
})


def _distribution_from_dict(d: dict) -> Distribution:
    """Construct Distribution from JSON dict, ignoring 'citation' / metadata keys."""
    return Distribution(mean=float(d["mean"]), cv=float(d.get("cv", 0.0)))


def _build_active_metabolite(entry: dict, smiles: str) -> ActiveMetabolite:
    """Construct ActiveMetabolite from v2 registry entry; validate fields.

    v2 omits conversion_rate_per_h and conversion_site at the registry level;
    however ActiveMetabolite still has those fields (legacy v1, retained
    for backward-compat). We populate conversion_rate_per_h with a sentinel
    Distribution(0.0) (unused by v2 flux) and conversion_site with empty
    string (unused by v2 augmentation). These v1 fields will be removed in
    a future cleanup task once all consumers are migrated.
    """
    missing = _REQUIRED_FIELDS - set(entry.keys())
    if missing:
        raise ValueError(
            f"prodrug_activation_registry entry for SMILES {smiles!r} "
            f"missing field {sorted(missing)}"
        )

    if entry["mw"] <= 0:
        raise ValueError(f"mw must be positive, got {entry['mw']}")

    cy = entry["conversion_yield_fraction"]
    if not (0.0 <= cy["mean"] <= 1.0):
        raise ValueError(f"conversion_yield must be in [0, 1], got {cy['mean']}")

    if entry["CL_per_h"]["mean"] <= 0 or entry["Vd_L"]["mean"] <= 0:
        raise ValueError("CL and Vd must be positive")

    return ActiveMetabolite(
        name=entry["name"],
        mw=float(entry["mw"]),
        fup=_distribution_from_dict(entry["fup"]),
        CL_per_h=_distribution_from_dict(entry["CL_per_h"]),
        Vd_L=_distribution_from_dict(entry["Vd_L"]),
        conversion_rate_per_h=Distribution(mean=0.0, cv=0.0),  # v2 unused
        conversion_site="",                                     # v2 unused
        conversion_yield_fraction=_distribution_from_dict(cy),
    )


def _build_enzyme_affinity_for_conversion(
    entry: dict, smiles: str
) -> tuple[dict[str, Distribution], dict[str, Distribution]]:
    """Parse enzyme_affinity_for_conversion dict; ignore citation keys.

    Returns ``(affinities, enzyme_yields)``. ``enzyme_yields`` is empty
    when no entry declares a per-enzyme ``yield`` field. When >=2 enzymes
    are declared, every enzyme MUST declare ``yield`` (all-or-nothing) -
    mixed declarations raise ValueError. See B-04 spec §5.4.
    """
    raw = entry["enzyme_affinity_for_conversion"]
    if not isinstance(raw, dict) or not raw:
        raise ValueError(
            f"enzyme_affinity_for_conversion must be non-empty dict for SMILES {smiles!r}"
        )
    affinities: dict[str, Distribution] = {}
    yields: dict[str, Distribution] = {}
    for tag, dist_raw in raw.items():
        if not isinstance(dist_raw, dict):
            raise ValueError(
                f"affinity entry for tag {tag!r} must be dict with 'mean'/'cv', "
                f"got {type(dist_raw).__name__}"
            )
        if "mean" not in dist_raw:
            raise ValueError(f"affinity entry for tag {tag!r} missing 'mean'")
        affinities[tag] = _distribution_from_dict(dist_raw)
        if "yield" in dist_raw:
            y = dist_raw["yield"]
            if not isinstance(y, dict) or "mean" not in y:
                raise ValueError(
                    f"per-enzyme yield for tag {tag!r} must be dict with 'mean'/'cv', "
                    f"got {y!r}"
                )
            y_mean = float(y["mean"])
            if not (0.0 <= y_mean <= 1.0):
                raise ValueError(
                    f"per-enzyme yield for tag {tag!r} must be in [0, 1], got {y_mean}"
                )
            yields[tag] = _distribution_from_dict(y)

    # All-or-nothing rule for multi-enzyme entries (spec §5.4).
    if len(affinities) >= 2 and yields and len(yields) != len(affinities):
        missing = sorted(set(affinities) - set(yields))
        raise ValueError(
            f"prodrug registry entry for SMILES {smiles!r}: multi-enzyme entries "
            f"must declare per-enzyme 'yield' for every enzyme or none. "
            f"Missing yield for: {missing}"
        )

    return affinities, yields


def lookup_active_metabolite(
    smiles: str, registry_path: Path | None = None
) -> tuple[ActiveMetabolite, str, dict[str, Distribution], dict[str, Distribution]] | None:
    """Look up SMILES in v2 prodrug registry.

    Returns ``(ActiveMetabolite, observation_species,
    enzyme_affinity_for_conversion, enzyme_yields)`` or ``None`` if not
    found. ``enzyme_yields`` is empty when the entry does not declare
    per-enzyme yields (single-enzyme entries; backward-compat path).

    Raises ``ValueError`` on invalid registry entries.
    """
    canonical = _canonicalize(smiles)
    if canonical is None:
        return None

    path = registry_path if registry_path is not None else _DEFAULT_REGISTRY_PATH
    if registry_path is not None:
        with path.open() as f:
            registry = json.load(f)
    else:
        registry = _load_registry_cached(str(path))

    entry = registry.get(canonical)
    if entry is None:
        query_connectivity = _inchikey_connectivity(canonical)
        if query_connectivity is not None:
            for registered_smiles, candidate in registry.items():
                if _inchikey_connectivity(registered_smiles) == query_connectivity:
                    entry = candidate
                    break
    if entry is None:
        return None

    obs_species = entry.get("observation_species", "active")
    if obs_species not in ("parent", "active"):
        raise ValueError(
            f"observation_species must be 'parent' or 'active', got {obs_species!r}"
        )

    affinity_source = entry.get("affinity_source")
    if affinity_source not in _VALID_AFFINITY_SOURCES:
        raise ValueError(
            f"affinity_source must be one of {sorted(_VALID_AFFINITY_SOURCES)}, "
            f"got {affinity_source!r} (v2 rejects 'infrastructure_only')"
        )

    yield_source = entry.get("yield_source")
    if yield_source not in _VALID_YIELD_SOURCES:
        raise ValueError(
            f"yield_source must be one of {sorted(_VALID_YIELD_SOURCES)}, "
            f"got {yield_source!r}"
        )

    am = _build_active_metabolite(entry, canonical)
    affinities, enzyme_yields = _build_enzyme_affinity_for_conversion(entry, canonical)
    return am, obs_species, affinities, enzyme_yields
