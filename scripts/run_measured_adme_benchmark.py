#!/usr/bin/env python3
"""Engine-only measured-input benchmark: SMILES-only vs measured fup+CLint.

Reuses the 12 source-cited measured drugs from scripts/measured_adme_poc.py and
the observed Cmax / SMILES / dose from data/reference/clinical_pk.json. Calls the
PRODUCTION predict(measured_adme=...) API and reads result.engine_pk (the clean
engine-only surface). Reports SMILES-only vs measured AAFE SIDE BY SIDE — this is
SEPARATE from the 2.698 headline and is never merged into it.

Usage: python scripts/run_measured_adme_benchmark.py
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.WARNING)
ROOT = Path(__file__).resolve().parent.parent
CLINICAL_PK = ROOT / "data" / "reference" / "clinical_pk.json"

# (name, fup, clint) — copied verbatim from scripts/measured_adme_poc.py MEASURED
# (DrugBank fup + TDC Hepatocyte_AZ CLint geomean). montelukast + abiraterone are
# the documented extreme outliers (diagnosis.md §3); reported but flagged.
_MEASURED = [
    ("alprazolam", 0.20, 13.0), ("carbamazepine", 0.25, 10.2),
    ("clozapine", 0.03, 31.8), ("diclofenac", 0.003, 83.5),
    ("sildenafil", 0.04, 49.9), ("etodolac", 0.01, 12.9),
    ("montelukast", 0.01, 24.7), ("quinine", 0.30, 21.1),
    ("febuxostat", 0.008, 9.4), ("dasatinib", 0.04, 28.2),
    ("clopidogrel", 0.2175, 137.0), ("abiraterone", 0.01, 55.0),
]
_OUTLIERS = {"montelukast", "abiraterone"}

# Approximate literature oral bioavailability F — well-established ballparks, NOT
# a citation-curated dataset (same provenance/caveat as scripts/run_f_decomposition.py
# lit_F). Used to demonstrate the measured-F routing channel; the measured-F column
# is illustrative, not a calibrated benchmark.
_LIT_F = {
    "alprazolam": 0.90, "carbamazepine": 0.80, "clozapine": 0.55, "diclofenac": 0.55,
    "sildenafil": 0.40, "etodolac": 1.00, "quinine": 0.80, "febuxostat": 0.85,
    "dasatinib": 0.25, "clopidogrel": 0.50,
}


def _aafe(folds):
    return float(np.exp(np.mean(np.log(folds)))) if folds else float("nan")


def main() -> int:
    from sisyphus.pipeline.predict import predict
    from sisyphus.predict.adme import MeasuredADMEInput

    drugs = json.loads(CLINICAL_PK.read_text())["drugs"]
    rows = []
    fe_s, fe_m, fe_mf = [], [], []
    fe_s_clean, fe_m_clean, fe_mf_clean = [], [], []

    for name, fup, clint in _MEASURED:
        rec = drugs.get(name)
        if not rec:
            print(f"  skip {name}: not in clinical_pk.json")
            continue
        obs = (rec.get("pk_params") or {}).get("cmax_mg_L")
        smiles, dose, route = rec.get("smiles"), rec.get("dose_mg"), rec.get("route", "oral")
        if not (obs and smiles and dose):
            print(f"  skip {name}: missing obs/smiles/dose")
            continue

        c_s = predict(smiles, dose, route=route).engine_pk.cmax.mean
        c_m = predict(smiles, dose, route=route,
                      measured_adme=MeasuredADMEInput(fup=fup, clint=clint)).engine_pk.cmax.mean
        lit_f = _LIT_F.get(name)
        c_mf = None
        if lit_f is not None:
            c_mf = predict(
                smiles, dose, route=route,
                measured_adme=MeasuredADMEInput(fup=fup, clint=clint, f_bioavail=lit_f),
            ).engine_pk.cmax.mean

        f_s = max(c_s / obs, obs / c_s)
        f_m = max(c_m / obs, obs / c_m)
        f_mf = max(c_mf / obs, obs / c_mf) if c_mf else None
        rows.append((name, obs, c_s, c_m, c_mf, f_s, f_m, f_mf))
        fe_s.append(f_s)
        fe_m.append(f_m)
        if f_mf is not None:
            fe_mf.append(f_mf)
        if name not in _OUTLIERS:
            fe_s_clean.append(f_s)
            fe_m_clean.append(f_m)
            if f_mf is not None:
                fe_mf_clean.append(f_mf)

    hdr = (f"\n{'drug':<16}{'obs':>10}{'eng_smiles':>12}{'eng_meas':>12}"
           f"{'eng_m+F':>12}{'FE_s':>7}{'FE_m':>7}{'FE_m+F':>8}")
    print(hdr)
    for name, obs, c_s, c_m, c_mf, f_s, f_m, f_mf in rows:
        flag = " *" if name in _OUTLIERS else ""
        mf_c = f"{c_mf:>12.4f}" if c_mf is not None else f"{'--':>12}"
        mf_fe = f"{f_mf:>8.2f}" if f_mf is not None else f"{'--':>8}"
        print(f"{name:<16}{obs:>10.4f}{c_s:>12.4f}{c_m:>12.4f}{mf_c}"
              f"{f_s:>7.2f}{f_m:>7.2f}{mf_fe}{flag}")
    print(f"\nN={len(rows)} engine-only AAFE  SMILES={_aafe(fe_s):.3f}  "
          f"measured(fup+clint)={_aafe(fe_m):.3f}  "
          f"measured(fup+clint+F)={_aafe(fe_mf):.3f} [N={len(fe_mf)}]")
    print(f"N={len(fe_s_clean)} (excl montelukast/abiraterone)  "
          f"SMILES={_aafe(fe_s_clean):.3f}  measured={_aafe(fe_m_clean):.3f}  "
          f"measured+F={_aafe(fe_mf_clean):.3f} [N={len(fe_mf_clean)}]")
    print("\nNOTE: the +F column uses APPROXIMATE literature oral-F ballparks (see _LIT_F "
          "docstring) — illustrative of the measured-F channel, not a calibrated benchmark.")
    print("SEPARATE from the 2.698 headline — do not merge into 4track_holdout_predictions.json.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
