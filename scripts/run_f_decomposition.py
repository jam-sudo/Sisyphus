#!/usr/bin/env python3
"""Compare emergent engine F with cited human absolute-F references.

The production ``compute_f_engine`` path converges the matched oral/IV exposure
ratio. This is a small diagnostic, not an independently validated F benchmark.

CAVEAT: fup/CLint inputs come from the earlier measured-ADME probe; formulation,
population, and analytical conditions are not all matched to the F references.

Usage: python scripts/run_f_decomposition.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

CLINICAL_PK = ROOT / "data" / "reference" / "clinical_pk.json"

# (name, measured fup, measured CLint, human F lower, upper) — only references
# that state absolute/systemic oral F, not absorption, relative BA, or animal F.
# fup/CLint copied from scripts/measured_adme_poc.py (DrugBank fup, TDC CLint).
_DRUGS = [
    # DailyMed diclofenac sodium DR label, PK Table 1: mean 55%, N=7, CV 40%.
    # https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=a65aa738-8ce2-4276-8e54-5ecf4f461d3a
    ("diclofenac", 0.003, 83.5, 0.55, 0.55),
    # VIAGRA label §12.3: mean absolute F 41% (individual range 25–63%).
    # https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=ae2079a2-f3a9-4739-9611-0742b71e4761
    ("sildenafil", 0.04, 49.9, 0.41, 0.41),
    # Quinine sulfate label §12.3: healthy-adult oral F range 76–88%.
    # https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=f567d5c7-ea5d-49a7-a035-b47208135f73
    ("quinine", 0.30, 21.1, 0.76, 0.88),
]


def main() -> int:
    from sisyphus.pipeline.predict import predict
    from sisyphus.predict.adme import MeasuredADMEInput

    drugs = json.loads(CLINICAL_PK.read_text())["drugs"]
    print(f"{'drug':<15}{'engF':>7}{'refF':>12}{'eng/ref':>13}")
    n = 0
    for name, fup, clint, ref_low, ref_high in _DRUGS:
        rec = drugs.get(name)
        if not rec:
            print(f"{name:<15} skip (not in clinical_pk.json)")
            continue
        smiles, dose = rec.get("smiles"), rec.get("dose_mg")
        if not (smiles and dose):
            print(f"{name:<15} skip (missing smiles/dose)")
            continue
        m = MeasuredADMEInput(fup=fup, clint=clint)
        eng_f = predict(
            smiles, dose, route="oral", measured_adme=m, compute_f_engine=True
        ).engine_f
        if eng_f is None:
            print(f"{name:<15} skip (engine F did not converge)")
            continue
        if not 0 < eng_f <= 1:
            raise ValueError(f"Non-physical engine F for {name}: {eng_f}")
        ref = f"{ref_low:.2f}" if ref_low == ref_high else f"{ref_low:.2f}–{ref_high:.2f}"
        ratio = (
            f"{eng_f / ref_low:.2f}"
            if ref_low == ref_high
            else f"{eng_f / ref_high:.2f}–{eng_f / ref_low:.2f}"
        )
        n += 1
        print(f"{name:<15}{eng_f:>7.2f}{ref:>12}{ratio:>13}")

    if not n:
        raise RuntimeError("No converged engine-F estimates")
    print(f"\n{n} cited human absolute-F references; no pooled estimate "
          "from this small, unmatched diagnostic set.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
