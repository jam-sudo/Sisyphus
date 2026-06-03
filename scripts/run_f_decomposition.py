#!/usr/bin/env python3
"""Engine bioavailability-F decomposition via IV-vs-oral, with measured fup+CLint.

Reproduces the 2026-06-02 finding (experiment-log.md): with fup and CLint held at
their measured values (so clearance is NOT the free variable), the engine
systematically UNDER-CALLS oral bioavailability F. F is estimated as the ratio of
oral to IV exposure at matched dose: engine_F = oral AUC_0t / iv AUC_0t.

CAVEATS (the finding is directional, not a calibrated measurement):
  - Literature F values below are approximate, well-established ballparks (oral
    bioavailability) for these drugs — NOT a curated, citation-checked dataset.
  - AUC_0t (0-24 h), not AUC_inf — a systematic ratio across all drugs, but it can
    bias long-half-life drugs.
So: the DIRECTION (engine under-calls F for ~all drugs) is robust; the magnitude
(median engine-F / literature-F) is preliminary pending verified-F curation.

Usage: python scripts/run_f_decomposition.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

CLINICAL_PK = ROOT / "data" / "reference" / "clinical_pk.json"

# (name, measured fup, measured CLint, APPROXIMATE literature oral F) — the 10
# "clean" PoC drugs (montelukast/abiraterone excluded as extreme outliers).
# fup/CLint copied from scripts/measured_adme_poc.py (DrugBank fup, TDC CLint).
_DRUGS = [
    ("alprazolam", 0.20, 13.0, 0.90),
    ("carbamazepine", 0.25, 10.2, 0.80),
    ("clozapine", 0.03, 31.8, 0.55),
    ("diclofenac", 0.003, 83.5, 0.55),
    ("sildenafil", 0.04, 49.9, 0.40),
    ("etodolac", 0.01, 12.9, 1.00),
    ("quinine", 0.30, 21.1, 0.80),
    ("febuxostat", 0.008, 9.4, 0.85),
    ("dasatinib", 0.04, 28.2, 0.25),
    ("clopidogrel", 0.2175, 137.0, 0.50),
]


def main() -> int:
    from sisyphus.pipeline.predict import predict
    from sisyphus.predict.adme import MeasuredADMEInput

    drugs = json.loads(CLINICAL_PK.read_text())["drugs"]
    print(f"{'drug':<15}{'engF':>7}{'litF':>7}{'eng/lit':>9}")
    ratios = []
    for name, fup, clint, lit_f in _DRUGS:
        rec = drugs.get(name)
        if not rec:
            print(f"{name:<15} skip (not in clinical_pk.json)")
            continue
        smiles, dose = rec.get("smiles"), rec.get("dose_mg")
        if not (smiles and dose):
            print(f"{name:<15} skip (missing smiles/dose)")
            continue
        m = MeasuredADMEInput(fup=fup, clint=clint)
        oral = predict(smiles, dose, route="oral", measured_adme=m).engine_pk
        iv = predict(smiles, dose, route="iv", measured_adme=m).engine_pk
        eng_f = (oral.auc_0t.mean / iv.auc_0t.mean) if (iv and iv.auc_0t.mean > 0) else float("nan")
        r = eng_f / lit_f
        ratios.append(r)
        print(f"{name:<15}{eng_f:>7.2f}{lit_f:>7.2f}{r:>9.2f}")

    print(f"\nmedian engine-F / literature-F = {np.nanmedian(ratios):.2f}  "
          f"(<1 => engine under-calls F; N={len(ratios)})")
    print(f"under-calls (ratio<1): {sum(1 for r in ratios if r < 1)}/{len(ratios)}")
    print("\nDirectional finding (literature-F approximate, AUC_0t): the engine "
          "systematically under-predicts bioavailability F. See experiment-log.md 2026-06-02.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
