#!/usr/bin/env python3
"""Build the per-drug TDM method routing table from SBC results.

Policy (Track B hybrid dispatch, 2026-04-10):
    - per-drug SBC pass and coverage deviation ≤ 0.10 → "sbi"
    - all other drugs                           → "ibis" (or existing "is" override)

Drugs not in the validation set default to ``default`` (``ibis`` unless
overridden). The table is consumed by ``sisyphus.cli._resolve_auto_method``
when the user passes ``--method auto``.

The time-aware model is routed separately for multiple observations, using
the same per-drug SBC and coverage rule. A failed multi-observation gate
leaves the existing single-observation route in place.

Usage::

    python scripts/sbi_build_routing_table.py \\
        --sbc data/validation/sbi_sbc_multi_drug_current.json \\
        --out data/sbi/method_routing.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("routing")

STRICT_COVERAGE_OK = 0.10
BORDERLINE_COVERAGE = 0.15


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sbc", type=Path,
                   default=ROOT / "data" / "validation" / "sbi_sbc_multi_drug_current.json")
    p.add_argument("--out", type=Path,
                   default=ROOT / "data" / "sbi" / "method_routing.json")
    p.add_argument("--multi-observation-sbc", type=Path,
                   default=ROOT / "data" / "validation"
                   / "sbi_sbc_multi_observation_current.json")
    p.add_argument("--default", default="ibis",
                   help="default method for drugs not in the validation set")
    args = p.parse_args()

    with open(args.sbc) as f:
        sbc = json.load(f)
    existing = json.loads(args.out.read_text()) if args.out.exists() else {}

    routes: dict[str, str] = {}
    reasons: dict[str, dict] = {}
    sbi_count = ibis_count = is_count = 0
    for entry in sbc["drugs"]:
        name = entry["name"]
        cov_dev = float(entry["coverage_max_deviation"])
        ks = entry["ks_pvalues"]
        per_drug_pass = bool(entry.get("pass", False))
        existing_is = existing.get("routes", {}).get(name) == "is"
        if existing_is:
            method = "is"
            reason = "existing_is_override"
            is_count += 1
        elif per_drug_pass and cov_dev <= STRICT_COVERAGE_OK:
            method = "sbi"
            reason = "sbc_and_coverage_ok"
            sbi_count += 1
        elif not per_drug_pass:
            method = "ibis"
            reason = "sbc_failed"
            ibis_count += 1
        elif cov_dev <= BORDERLINE_COVERAGE:
            method = "ibis"
            reason = "coverage_borderline"
            ibis_count += 1
        else:
            method = "ibis"
            reason = "coverage_hard_fail"
            ibis_count += 1
        routes[name] = method
        reasons[name] = {
            "method": method,
            "reason": reason,
            "coverage_max_deviation": cov_dev,
            "ks_pvalues": ks,
            "sbc_pass": per_drug_pass,
        }
        override = existing.get("reasons", {}).get(name, {}).get("tdm_bias_override")
        if override is not None:
            reasons[name]["tdm_bias_override"] = override

    reweight = existing.get("sbi_reweight", {})
    multi_observation = existing.get("multi_observation")
    if args.multi_observation_sbc.exists():
        multi_sbc = json.loads(args.multi_observation_sbc.read_text())
        validation = json.loads((ROOT / "data/sbi/validation_drug_set.json").read_text())
        smiles = {entry["name"]: entry["smiles"] for entry in validation["drugs"]}
        decisions = {}
        for entry in multi_sbc["drugs"]:
            name = entry["name"]
            coverage = float(entry["coverage_max_deviation"])
            passed = bool(entry.get("pass", False)) and coverage <= STRICT_COVERAGE_OK
            decisions[name] = {
                "use_native": passed,
                "smiles": smiles[name],
                "sbc_pass": bool(entry.get("pass", False)),
                "coverage_max_deviation": coverage,
                "ks_pvalues": entry["ks_pvalues"],
            }
        multi_observation = {
            "source": str(args.multi_observation_sbc.resolve().relative_to(ROOT)),
            "model": "multi_observation_nsf.pt",
            "global_sbc_gate_passed": bool(multi_sbc.get("gate_passed", False)),
            "native_count": sum(row["use_native"] for row in decisions.values()),
            "drugs": decisions,
        }

    payload = {
        "source": str(args.sbc.resolve().relative_to(ROOT)),
        "policy": {
            "strict_coverage_ok": STRICT_COVERAGE_OK,
            "borderline_coverage": BORDERLINE_COVERAGE,
            "description": (
                "SBI only where per-drug SBC passes and coverage deviation "
                "is ≤10pp of nominal across all CI levels. Existing IS routes "
                "remain IS; other failures use IBIS."
            ),
        },
        "default": args.default,
        "summary": {
            "sbi": sbi_count,
            "ibis": ibis_count,
            "total": len(routes),
            "is": is_count,
            "sbi_reweight_drugs": sum(bool(value) for value in reweight.values()),
            "global_sbc_gate_passed": bool(sbc.get("gate_passed", False)),
        },
        "sbi_reweight": reweight,
        "multi_observation": multi_observation,
        "routes": routes,
        "reasons": reasons,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)

    print(
        f"[routing] {sbi_count} SBI / {is_count} IS / {ibis_count} IBIS "
        f"(default={args.default})"
    )
    print(f"[routing] wrote {args.out}")
    print()
    for name, data in reasons.items():
        mark = "✓" if data["method"] == "sbi" else "✗"
        print(
            f"  {mark} {name:<15s}  {data['method']:<5s}  "
            f"cov_dev={data['coverage_max_deviation']:.3f}  ({data['reason']})"
        )


if __name__ == "__main__":
    main()
