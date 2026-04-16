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
