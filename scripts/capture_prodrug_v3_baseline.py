#!/usr/bin/env python3
"""Capture deterministic point-estimate Cmax for all 107 holdout drugs.

Run BEFORE any v3 value changes. Saves baseline used by the §6.2 enzyme
leak audit (test_prodrug_v3_enzyme_leak_audit.py): 105 of 107 drugs must
be byte-identical Cmax post-v3 under deterministic point-estimate.

Per spec §6.2: CHANGED_ENZYME_ABUNDANCES = ∅, DRUG_SPECIFIC_CHANGES =
{remdesivir, fostamatinib} — only these 2 drugs may differ post-v3.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sisyphus.pipeline.predict import predict  # noqa: E402
from sisyphus.validation.reference import load_reference  # noqa: E402


def main() -> None:
    refs = [r for r in load_reference() if r.in_holdout]
    print(f"Holdout drugs: {len(refs)}")

    baseline: dict[str, float] = {}
    nan_drugs: list[str] = []

    for i, ref in enumerate(refs):
        try:
            result = predict(ref.smiles, ref.dose_mg, ref.route)  # n_mc_samples=0 default → deterministic
            cmax = float(result.pk.cmax.mean) if result.pk and result.pk.cmax else math.nan
        except Exception as exc:
            print(f"  [{i+1:3d}/{len(refs)}] {ref.name:30s}  FAILED: {exc}")
            cmax = math.nan

        if not math.isfinite(cmax):
            nan_drugs.append(ref.name)
        baseline[ref.name] = cmax
        if (i + 1) % 20 == 0:
            print(f"  [{i+1:3d}/{len(refs)}] processed")

    out = ROOT / "tests" / "regression" / "data" / "prodrug_v3_pre_baseline.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(baseline, indent=2, sort_keys=True))
    print(f"\nSaved baseline for {len(baseline)} drugs to {out}")
    if nan_drugs:
        print(f"WARNING: {len(nan_drugs)} drug(s) produced NaN/inf: {nan_drugs}")


if __name__ == "__main__":
    main()
