"""Shared skip markers for proprietary-artifact-sensitive tests.

The 2026-05-09 audit established that two gitignored artifacts —
`data/drugbank/` and `models/adme/logp_correction.json` — silently shift
predict() Cmax values by 5–30% on covered drugs. The public-clone state
(no artifacts) is the SSOT for tests + CI; local-developer state with
artifacts present is for personal use only.

Tests that pin numerical Cmax values to the public-only canonical fail
locally when artifacts are present. Tests that *require* artifact-derived
enrichment (e.g., DrugBank fm fractions) fail in CI / public clones.

Use the markers in this module so both cases skip with an informative
message rather than failing opaquely.
"""
from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_DRUGBANK_FILE = ROOT / "data" / "drugbank" / "drugs.csv"
_LOGP_CORRECTION_FILE = ROOT / "models" / "adme" / "logp_correction.json"

PROPRIETARY_ARTIFACTS_PRESENT = _DRUGBANK_FILE.exists() or _LOGP_CORRECTION_FILE.exists()
DRUGBANK_PRESENT = _DRUGBANK_FILE.exists()


def _public_only_reason() -> str:
    present = []
    if _DRUGBANK_FILE.exists():
        present.append(str(_DRUGBANK_FILE.relative_to(ROOT)))
    if _LOGP_CORRECTION_FILE.exists():
        present.append(str(_LOGP_CORRECTION_FILE.relative_to(ROOT)))
    return (
        f"Local-developer artifact(s) present: {', '.join(present)}. "
        "This test is anchored to public-clone deterministic Cmax values "
        "(see AGENTS.md §\"Artifact gates\"). To run it locally, hide the "
        "artifact(s) first: `mv data/drugbank data/drugbank.local-archive` "
        "and/or `mv models/adme/logp_correction.json{,.local-archive}`. CI "
        "runs without these artifacts and evaluates the test normally."
    )


def _drugbank_required_reason() -> str:
    return (
        "DrugBank artifacts not present at `data/drugbank/` (academic license "
        "required; gitignored). This test relies on DrugBank-enriched fm fractions "
        "or other experimental properties. Engine-level architectural correctness "
        "is verified separately (see e.g. tests/unit/test_phenotype.py)."
    )


# Marker for tests anchored to public-only Cmax canonical values
# (e.g., test_predict_auto_ecm.py's pinned 0.0294 mg/L pravastatin).
skip_if_local_artifacts = pytest.mark.skipif(
    PROPRIETARY_ARTIFACTS_PRESENT, reason=_public_only_reason(),
)

# Marker for tests that require DrugBank enrichment to produce
# measurable mechanistic effect (e.g., single-CYP PM/EM ratios).
skip_if_no_drugbank = pytest.mark.skipif(
    not DRUGBANK_PRESENT, reason=_drugbank_required_reason(),
)
