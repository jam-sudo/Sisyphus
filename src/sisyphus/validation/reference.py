"""Reference data loader -- single source of truth.

Loads ``data/reference/clinical_pk.json`` which contains observed
PK values for all benchmark drugs, with ``in_holdout`` and
``in_training`` flags.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Default data directory: resolve relative to repository root.
# src/sisyphus/validation/reference.py -> ../../../../data/reference
_DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "reference"


@dataclass(frozen=True)
class DrugReference:
    """Observed clinical PK data for a single drug.

    Attributes:
        name: Drug name.
        smiles: Canonical SMILES.
        dose_mg: Dose used in the reference study.
        route: Administration route.
        cmax_obs: Observed Cmax (mg/L).
        auc_obs: Observed AUC (mg*h/L), if available.
        in_holdout: Whether this drug is in the holdout set.
        in_training: Whether this drug is in the training set.
    """

    name: str
    smiles: str
    dose_mg: float
    route: str
    cmax_obs: float
    auc_obs: float | None
    in_holdout: bool
    in_training: bool


def _load_holdout_set(holdout_path: Path | None = None) -> set[str]:
    """Load the set of holdout drug names from holdout.json."""
    if holdout_path is None:
        holdout_path = _DATA_DIR / "holdout.json"
    if not holdout_path.exists():
        logger.warning("Holdout file not found: %s", holdout_path)
        return set()
    with open(holdout_path) as f:
        data = json.load(f)
    return set(data.get("holdout", []))


def load_reference(path: Path | None = None) -> list[DrugReference]:
    """Load clinical PK reference data.

    Skips drugs without SMILES, without dose_mg, or without Cmax.

    Args:
        path: Path to ``clinical_pk.json``.  Defaults to
            ``data/reference/clinical_pk.json``.

    Returns:
        List of DrugReference records.
    """
    if path is None:
        path = _DATA_DIR / "clinical_pk.json"

    with open(path) as f:
        data = json.load(f)

    holdout_set = _load_holdout_set()

    results: list[DrugReference] = []
    skipped = 0
    for name, info in data.get("drugs", {}).items():
        smiles = info.get("smiles", "")
        if not smiles:
            skipped += 1
            continue

        dose_mg = info.get("dose_mg")
        if dose_mg is None or dose_mg <= 0:
            skipped += 1
            continue

        pk = info.get("pk_params", {})
        cmax = pk.get("cmax_mg_L")
        if cmax is None or cmax <= 0:
            skipped += 1
            continue

        auc = pk.get("auc_mg_h_L")
        route = info.get("route", "oral")
        in_holdout = name in holdout_set
        in_training = not in_holdout

        results.append(
            DrugReference(
                name=name,
                smiles=smiles,
                dose_mg=float(dose_mg),
                route=route,
                cmax_obs=float(cmax),
                auc_obs=float(auc) if auc is not None else None,
                in_holdout=in_holdout,
                in_training=in_training,
            )
        )

    logger.info(
        "Loaded %d reference drugs (%d skipped, %d holdout, %d training)",
        len(results),
        skipped,
        sum(1 for r in results if r.in_holdout),
        sum(1 for r in results if r.in_training),
    )

    return results
