"""Per-drug hepatic intracellular fu correction registry (B-11).

When a drug's effective unbound fraction inside hepatocytes is higher
than its plasma fup (typically due to albumin-facilitated uptake or
intracellular protein binding), the WS / PT extraction formulas
under-predict hepatic CL. This registry holds a per-drug
``fu_correction_liver`` multiplier curated from primary literature.
At flagged hepatic nodes the engine replaces ``fup`` with
``fup × fu_correction_liver`` in the WS / PT formula.

Default for any drug not in the registry is ``Distribution(mean=1.0,
cv=0.0)`` -- no scaling. Lookup is by RDKit InChIKey with a
connectivity-block fallback so non-isomeric query SMILES still match
stereospecific registry entries.

Registry file: ``data/transporters/hepatic_fu_correction.json``
Schema: see docs/superpowers/specs/2026-05-21-B11-hepatic-fu-correction-design.md
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path

from sisyphus.core import Distribution

logger = logging.getLogger(__name__)

_DEFAULT_REGISTRY_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "data" / "transporters" / "hepatic_fu_correction.json"
)

_VALID_DISPOSITIONS = frozenset({
    "literature_applied",
    "class_extrapolated",
    "ceiling_accepted",
    "not_applicable",
})


def _default() -> Distribution:
    """Default no-scaling correction."""
    return Distribution(mean=1.0, cv=0.0)


def _parse_overrides(
    raw_data: dict,
) -> tuple[dict[str, Distribution], dict[str, list[Distribution]]]:
    """Parse and validate the registry overrides list.

    Raises ValueError on invalid disposition or sub-1.0 fu_correction_liver.
    """
    full_index: dict[str, Distribution] = {}
    conn_index: dict[str, list[Distribution]] = {}

    for entry in raw_data.get("overrides", []):
        disposition = entry.get("disposition")
        if disposition not in _VALID_DISPOSITIONS:
            raise ValueError(
                f"hepatic_fu_correction entry for {entry.get('drug')!r} has "
                f"disposition={disposition!r}; must be one of "
                f"{sorted(_VALID_DISPOSITIONS)}"
            )

        ikey = entry.get("inchikey")
        raw = entry.get("fu_correction_liver")
        if ikey is None or not isinstance(raw, dict) or "mean" not in raw:
            raise ValueError(
                f"hepatic_fu_correction entry for {entry.get('drug')!r} missing "
                f"required fields (inchikey, fu_correction_liver.mean)"
            )

        mean = float(raw["mean"])
        if mean < 1.0:
            raise ValueError(
                f"hepatic_fu_correction entry for {entry.get('drug')!r} has "
                f"fu_correction_liver.mean={mean} < 1.0; values must be >= 1.0 "
                f"per CLAUDE.md invariant #8 (no fudge to Cmax loss). To raise "
                f"hepatic CL for a highly bound drug, use a literature-derived "
                f"fu_inc/fu_plasma ratio >= 1.0."
            )

        dist = Distribution(mean=mean, cv=float(raw.get("cv", 0.0)))
        full_index[ikey] = dist
        conn_index.setdefault(ikey.split("-", maxsplit=1)[0], []).append(dist)

    return full_index, conn_index


@lru_cache(maxsize=1)
def _load(path_str: str) -> tuple[dict[str, Distribution], dict[str, list[Distribution]]]:
    """Load registry; index by full InChIKey and connectivity block.

    Validates every entry: disposition is in the allowed set and
    ``fu_correction_liver.mean >= 1.0`` (anti-fudge guard). Connectivity
    matches are only honored when unambiguous (one override per block).
    """
    path = Path(path_str)
    if not path.exists():
        logger.warning("hepatic_fu_correction registry not found at %s", path)
        return {}, {}

    with path.open() as f:
        return _parse_overrides(json.load(f))


def lookup_hepatic_fu_correction(
    smiles: str, registry_path: Path | None = None
) -> Distribution:
    """Return the hepatic fu correction Distribution for ``smiles``.

    Returns ``Distribution(mean=1.0, cv=0.0)`` (no scaling) when the
    SMILES is not in the registry, RDKit is unavailable, or the SMILES
    is invalid. Tries full InChIKey first, then falls back to
    connectivity-block matching (mirrors registry.py B-03 pattern)
    when full key misses.

    Raises ``ValueError`` only when the registry file itself is
    malformed (invalid disposition or sub-1.0 value).
    """
    try:
        from rdkit import Chem
    except ImportError:
        return _default()

    if not smiles:
        return _default()

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return _default()

    ikey = Chem.MolToInchiKey(mol)
    path = registry_path if registry_path is not None else _DEFAULT_REGISTRY_PATH

    if registry_path is not None:
        # Bypass the lru_cache for tests that write registries in tmp dirs.
        full_index, conn_index = _load_uncached(path)
    else:
        full_index, conn_index = _load(str(path))

    if ikey in full_index:
        return full_index[ikey]

    matches = conn_index.get(ikey.split("-", maxsplit=1)[0], [])
    if len(matches) == 1:
        return matches[0]

    return _default()


def _load_uncached(
    path: Path,
) -> tuple[dict[str, Distribution], dict[str, list[Distribution]]]:
    """Test-only helper that loads without lru_cache, so tmp paths work."""
    if not path.exists():
        return {}, {}
    with path.open() as f:
        return _parse_overrides(json.load(f))
