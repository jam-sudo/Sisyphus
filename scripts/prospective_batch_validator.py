"""Batch prospective validator — runs Sisyphus pipeline on new 2024-2025 NMEs.

For each drug: SMILES + dose + observed single-dose Cmax → Sisyphus prediction.
Compares Meta/Engine/ML predictions vs observed, computes fold error.

Adds to N=10 prospective benchmark to support statistical significance testing.
"""
from __future__ import annotations

import json
from pathlib import Path

from sisyphus.pipeline.predict import predict

# ---------------------------------------------------------------------------
# Curated prospective drugs (2024-2025 FDA NMEs, genuinely unseen by training).
#
# Single-dose Cmax obtained via:
#  (a) FDA label explicit single-dose PK, or
#  (b) SS Cmax / accumulation ratio AR where AR = 1 / (1 - exp(-ln(2) * tau / t_half))
#
# All SMILES from DrugBank (canonical). Eligibility requires the drug be ABSENT
# from ALL training/reference corpora — not just mmpk_expanded_full.csv but also
# mmpk_expanded_v2.csv, vdss_v2_training.csv, bioavailability_v1.csv,
# clinical_pk.json, and holdout.json['train'].
#
# 2026-05-31: vorasidenib REMOVED. The original "verified NOT in
# mmpk_expanded_full.csv" check was too narrow — vorasidenib is present in
# clinical_pk.json (gold-tier reference), mmpk_expanded_v2.csv,
# vdss_v2_training.csv, bioavailability_v1.csv and holdout.json['train'], so it
# was never genuinely prospective. See data/validation/prospective_2024_CORRECTED.json.
# ---------------------------------------------------------------------------

_CANDIDATES = [
    {
        "name": "pirtobrutinib",
        "smiles": "COc1ccc(F)cc1C(=O)NCc1ccc(-c2nn([C@@H](C)C(F)(F)F)c(N)c2C(N)=O)cc1",
        "dose_mg": 200.0,
        "obs_mg_L": 3.99,
        "notes": "Jaypirca 200mg QD; FDA SS Cmax 6503 ng/mL / AR 1.63 (t1/2=20h)",
        "flags": ["SS_CORRECTED_AR1.63"],
    },
    {
        "name": "pacritinib",
        "smiles": "C1=C/COCc2cc(ccc2OCCN2CCCC2)Nc2nccc(n2)-c2cccc(c2)COC/1",
        "dose_mg": 200.0,
        "obs_mg_L": 2.18,
        "notes": "Vonjo 200mg BID; FDA SS Cmax 8.4 mg/L / AR 3.86 (accumulates 386%)",
        "flags": ["SS_CORRECTED_AR3.86"],
    },
    {
        "name": "crinecerfont",
        "smiles": "C#CCN(c1nc(-c2cc(C)c(OC)cc2Cl)c(C)s1)[C@@H](CC1CC1)c1ccc(C)c(F)c1",
        "dose_mg": 100.0,
        "obs_mg_L": 1.44,
        "notes": "Crenessity 100mg BID; FDA SS Cmax 4231 ng/mL / AR_Cmax 2.94 (t1/2~20h)",
        "flags": ["SS_CORRECTED_AR2.94"],
    },
    {
        "name": "revumenib",
        "smiles": "CCN(C(=O)c1cc(F)ccc1Oc1cncnc1N1CC2(CCN(C[C@H]3CC[C@H](NS(=O)(=O)CC)CC3)CC2)C1)C(C)C",
        "dose_mg": 270.0,
        "obs_mg_L": 1.172,
        "notes": "Revuforj 270mg BID (no CYP3A4 inh); FDA SS Cmax 2344 ng/mL / AR 2.0",
        "flags": ["SS_CORRECTED_AR2.0"],
    },
]


def _fold_error(pred: float, obs: float) -> float:
    return pred / obs


def run() -> dict:
    results = []
    for c in _CANDIDATES:
        try:
            r = predict(c["smiles"], c["dose_mg"])
            eng = float(r.engine_pk.cmax.mean) if r.engine_pk else None
            ml = float(r.ml_pk.cmax.mean) if r.ml_pk else None
            meta = float(r.pk.cmax.mean)
            fe = _fold_error(meta, c["obs_mg_L"])
            ad_flags = list(r.ad_flags) if r.ad_flags else []
            results.append({
                "name": c["name"],
                "dose_mg": c["dose_mg"],
                "obs": c["obs_mg_L"],
                "eng": eng,
                "ml": ml,
                "meta": meta,
                "fold": fe,
                "conf": r.confidence,
                "ad_flags": ad_flags,
                "curation_flags": c["flags"],
                "notes": c["notes"],
            })
        except Exception as e:
            results.append({"name": c["name"], "error": str(e)})

    # Compute aggregate AAFE
    import math
    valid = [r for r in results if "fold" in r]
    aafe_all = math.exp(sum(abs(math.log(r["fold"])) for r in valid) / len(valid)) if valid else None
    in_domain = [r for r in valid if not r["ad_flags"] and not r["curation_flags"]]
    aafe_in = math.exp(sum(abs(math.log(r["fold"])) for r in in_domain) / len(in_domain)) if in_domain else None
    pct_2fold = sum(1 for r in valid if 0.5 <= r["fold"] <= 2.0) / len(valid) if valid else None

    summary = {
        "N_total": len(valid),
        "N_in_domain_clean": len(in_domain),
        "aafe_all": aafe_all,
        "aafe_in_domain_clean": aafe_in,
        "pct_within_2fold": pct_2fold,
    }
    return {"drugs": results, "summary": summary}


if __name__ == "__main__":
    out = run()
    Path("data/validation/prospective_batch_N5_kinase.json").write_text(json.dumps(out, indent=2))
    print(f"N={out['summary']['N_total']}, AAFE={out['summary']['aafe_all']:.3f}")
    n_in = out['summary']['N_in_domain_clean']
    if n_in > 0:
        print(f"In-domain clean: N={n_in}, AAFE={out['summary']['aafe_in_domain_clean']:.3f}")
    else:
        print(f"In-domain clean: N=0 (all have curation flags)")
    print(f"Within 2-fold: {out['summary']['pct_within_2fold']*100:.0f}%")
    for d in out["drugs"]:
        if "fold" in d:
            marker = "✓" if 0.5 <= d["fold"] <= 2.0 else "✗"
            print(f"  {marker} {d['name']:16s} dose={d['dose_mg']:5.0f}mg  "
                  f"obs={d['obs']:7.3f}  meta={d['meta']:7.3f}  fold={d['fold']:5.2f}  "
                  f"flags={d['curation_flags']+d['ad_flags']}")
