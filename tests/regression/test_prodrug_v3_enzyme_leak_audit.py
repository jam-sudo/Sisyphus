"""v3 enzyme-leak audit per spec §6.2.

Verifies v3 registry/yaml changes affect only intended drugs. Drugs in
expected_unchanged set must produce byte-identical deterministic Cmax
compared to pre-v3 baseline (tests/regression/data/prodrug_v3_pre_baseline.json).

Two-dimension change tracking:
- CHANGED_ENZYME_ABUNDANCES: physiology yaml abundance changes (cross-drug)
- DRUG_SPECIFIC_CHANGES: drug-side registry changes (drug-isolated)

NB: v3's 4 prodrugs (sepiapterin/remdesivir/tebipenem_pivoxil/fostamatinib)
are NOT in the 107-holdout set, so even DRUG_SPECIFIC_CHANGES drugs only
matter if they happen to overlap with the holdout. With Item 5 ceiling
(no YAML change) and prodrugs absent from holdout, all 107 holdout drugs
should be byte-identical.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from sisyphus.pipeline.predict import predict
from sisyphus.validation.reference import load_reference

PRE_BASELINE_PATH = Path("tests/regression/data/prodrug_v3_pre_baseline.json")
REGISTRY_PATH = Path("data/sbi/prodrug_activation_registry.json")

# v3 change dimensions per Tasks 4-9 outcomes:
# Item 5 SPR: ceiling_accepted → no physiology YAML change → empty
CHANGED_ENZYME_ABUNDANCES: frozenset[str] = frozenset()
# Items 1, 4 ceiling (no value change). Items 2, 3 literature_applied.
DRUG_SPECIFIC_CHANGES: frozenset[str] = frozenset({"remdesivir", "fostamatinib"})


@pytest.fixture(scope="module")
def pre_baseline() -> dict[str, float]:
    return json.loads(PRE_BASELINE_PATH.read_text())


def _load_holdout_drugs():
    return [r for r in load_reference() if r.in_holdout]


@pytest.mark.slow
def test_enzyme_leak_audit(pre_baseline: dict[str, float]) -> None:
    """Drugs not affected by v3 changes must have byte-identical deterministic Cmax."""
    drugs = _load_holdout_drugs()

    expected_unchanged: list[str] = []
    expected_changed: list[str] = []
    for drug in drugs:
        if drug.name in DRUG_SPECIFIC_CHANGES:
            expected_changed.append(drug.name)
        else:
            # CHANGED_ENZYME_ABUNDANCES check would go here if non-empty;
            # currently empty per Item 5 ceiling, so all non-DRUG_SPECIFIC are unchanged.
            expected_unchanged.append(drug.name)

    # All-ceiling sanity: if no changes, all 107 should be unchanged
    if not CHANGED_ENZYME_ABUNDANCES and not DRUG_SPECIFIC_CHANGES:
        assert len(expected_unchanged) == len(drugs), (
            f"All-ceiling scenario expects {len(drugs)} unchanged, got {len(expected_unchanged)}"
        )

    # Verify byte-identical AND finite for each unchanged drug
    failures: list[str] = []
    skipped_nan: list[str] = []
    for drug in drugs:
        if drug.name not in expected_unchanged:
            continue
        baseline_cmax = pre_baseline.get(drug.name)
        if baseline_cmax is None or not math.isfinite(baseline_cmax):
            skipped_nan.append(drug.name)
            continue
        result = predict(drug.smiles, drug.dose_mg, drug.route)  # n_mc_samples=0 default
        cmax_v3 = float(result.pk.cmax.mean) if result.pk and result.pk.cmax else math.nan
        if not math.isfinite(cmax_v3):
            failures.append(f"{drug.name}: v3 produced non-finite Cmax {cmax_v3}")
            continue
        if cmax_v3 != baseline_cmax:
            failures.append(
                f"{drug.name}: v3 Cmax {cmax_v3:.10g} != pre-v3 baseline {baseline_cmax:.10g} "
                f"(delta {cmax_v3 - baseline_cmax:+.4g})"
            )

    assert not failures, (
        f"Leak detected ({len(failures)} expected_unchanged drug Cmax differs):\n"
        + "\n".join(failures[:20])
    )
