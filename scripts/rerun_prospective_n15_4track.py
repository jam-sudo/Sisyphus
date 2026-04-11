#!/usr/bin/env python3
"""Re-run the N=15 prospective 2024-2025 FDA NME validation under the
current 4-track production pipeline.

The pre-merge feat-branch numbers (overall AAFE 2.48, in-domain 1.68)
were measured on a different pipeline state. This script rebuilds the
same drug set with:
  - SMILES pulled from DrugBank via drug name
  - Dose and observed Cmax from the N=10 + N=5 curated JSONs
and runs ``sisyphus.pipeline.predict.predict`` on each, computing Meta
/ Engine / ML / fold error metrics.

Output: ``data/validation/prospective_N15_4track.json``
"""

from __future__ import annotations

import json
import logging
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))  # allow `from scripts import prospective_batch_validator`

from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)

logging.basicConfig(level=logging.WARNING)

from sisyphus.pipeline.predict import predict

N10_DOSES = {
    "resmetirom": 100.0, "aficamten": 20.0, "elafibranor": 120.0,
    "inavolisib": 9.0, "danicopan": 150.0, "berotralstat": 150.0,
    "vimseltinib": 30.0, "gepotidacin": 1500.0, "taletrectinib": 600.0,
    "suzetrigine": 100.0,
}


def fold_error(pred: float | None, obs: float) -> float | None:
    if pred is None or pred <= 0:
        return None
    return pred / obs if pred >= obs else obs / pred


def load_drugbank_map() -> dict[str, str]:
    import pandas as pd
    db = pd.read_csv(ROOT / "data" / "drugbank" / "drugs.csv")
    return {
        row["name"].lower(): row.get("canonical_smiles") or row["smiles"]
        for _, row in db.iterrows()
        if isinstance(row.get("name"), str)
    }


def load_candidates() -> list[tuple[str, str, float, float]]:
    """Return list of (name, smiles, dose_mg, obs_mg_L) for the 15 drugs."""
    db_map = load_drugbank_map()

    # N=10 from prospective_2024_2025_N10.json — obs values stored.
    with open(ROOT / "data" / "validation" / "prospective_2024_2025_N10.json") as f:
        n10 = json.load(f)
    # N=5 kinase batch — script already contains SMILES + dose + obs.
    from scripts import prospective_batch_validator as batch_mod
    n5_candidates = batch_mod._CANDIDATES

    drugs = []
    for rec in n10["drugs"]:
        name = rec["name"]
        dose = N10_DOSES.get(name.lower())
        smi = db_map.get(name.lower())
        if smi and dose:
            drugs.append((name, smi, dose, float(rec["obs"])))
        else:
            print(f"  SKIP {name}: smi={bool(smi)} dose={dose}", file=sys.stderr)
    for c in n5_candidates:
        drugs.append((c["name"], c["smiles"], float(c["dose_mg"]), float(c["obs_mg_L"])))
    return drugs


def main():
    drugs = load_candidates()
    print(f"Running 4-track predict on {len(drugs)} prospective drugs")
    print()

    rows = []
    for name, smi, dose, obs in drugs:
        try:
            result = predict(smi, dose, "oral")
            eng = float(result.engine_pk.cmax.mean) if result.engine_pk else None
            ml = float(result.ml_pk.cmax.mean) if result.ml_pk else None
            meta = float(result.pk.cmax.mean)
            fe_meta = fold_error(meta, obs)
            ad_flags = list(getattr(result, "ad_flags", []))
            in_ad = bool(getattr(result, "in_applicability_domain", True))
            rec = {
                "name": name,
                "smiles": smi,
                "dose_mg": dose,
                "obs_mg_L": obs,
                "engine": eng,
                "ml": ml,
                "meta": meta,
                "fold_meta": fe_meta,
                "in_applicability_domain": in_ad,
                "ad_flags": ad_flags,
            }
            rows.append(rec)
            marker = "✓" if fe_meta and fe_meta <= 2.0 else "✗"
            flags_str = ",".join(ad_flags) if ad_flags else ""
            print(
                f"  {marker} {name:<16s} dose={dose:>5.0f}mg  obs={obs:.4f}  "
                f"meta={meta:.4f}  fold={fe_meta:.2f}  {flags_str}"
            )
        except Exception as exc:
            print(f"  ERR  {name}: {exc}")
            rows.append({"name": name, "error": str(exc)})

    valid = [r for r in rows if "fold_meta" in r and r["fold_meta"] is not None]
    aafe_all = math.exp(
        sum(abs(math.log(r["fold_meta"])) for r in valid) / len(valid)
    ) if valid else None
    in_domain = [r for r in valid if not r["ad_flags"]]
    aafe_in = math.exp(
        sum(abs(math.log(r["fold_meta"])) for r in in_domain) / len(in_domain)
    ) if in_domain else None
    pct_2fold_all = sum(1 for r in valid if r["fold_meta"] <= 2.0) / len(valid) if valid else None

    payload = {
        "pipeline": "4-track (current production)",
        "n_candidates": len(drugs),
        "n_valid": len(valid),
        "n_in_domain": len(in_domain),
        "aafe_all": aafe_all,
        "aafe_in_domain": aafe_in,
        "pct_within_2fold_all": pct_2fold_all,
        "drugs": rows,
    }
    out = ROOT / "data" / "validation" / "prospective_N15_4track.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print()
    print(f"N_total={len(valid)}, AAFE_all={aafe_all:.3f}, AAFE_in_domain={aafe_in:.3f}, %2-fold={pct_2fold_all*100:.0f}%")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
