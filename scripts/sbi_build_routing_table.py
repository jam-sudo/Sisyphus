#!/usr/bin/env python3
"""Build the per-drug TDM method routing table from SBC results.

Policy:
    - zero exact/SMC/uncomputable SBC cases and no combined-Holm rejection → "sbi"
    - all other drugs → "ibis" (or existing "is" override)

Drugs not in the validation set default to ``default`` (``ibis`` unless
overridden). The table is consumed by ``sisyphus.cli._resolve_auto_method``
when the user passes ``--method auto``.

The time-aware model is routed separately for multiple observations, using
the same per-drug SBC, coverage, and rejection rule. A failed multi-observation gate
leaves the existing single-observation route in place.

Usage::

    python scripts/sbi_build_routing_table.py \\
        --sbc data/validation/sbi_sbc_multi_drug_corrected.json \\
        --multi-observation-sbc data/validation/sbi_sbc_multi_observation_corrected.json \\
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
                   default=ROOT / "data" / "validation" / "sbi_sbc_multi_drug_corrected.json")
    p.add_argument("--out", type=Path,
                   default=ROOT / "data" / "sbi" / "method_routing.json")
    p.add_argument("--multi-observation-sbc", type=Path,
                   default=ROOT / "data" / "validation"
                   / "sbi_sbc_multi_observation_corrected.json")
    p.add_argument("--hierarchical-sbc", type=Path,
                   default=ROOT / "data" / "validation"
                   / "sbi_sbc_hierarchical_corrected.json")
    p.add_argument("--continuous-sbc", type=Path,
                   default=ROOT / "data" / "validation"
                   / "sbi_sbc_continuous_corrected.json")
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
        raw_cov_dev = entry["coverage_max_deviation"]
        cov_dev = float(raw_cov_dev) if raw_cov_dev is not None else None
        ks = entry["ks_pvalues"]
        per_drug_pass = entry.get("pass")
        calibrated_rejections = entry.get("size_calibrated_rejections")
        rejection_cases = int(entry.get("rejection_fallback_cases", 0))
        smc_cases = int(entry.get("joint_smc_fallback_cases", 0))
        existing_is = existing.get("routes", {}).get(name) == "is"
        if existing_is:
            method = "is"
            reason = "existing_is_override"
            is_count += 1
        elif rejection_cases:
            method = "ibis"
            reason = "exact_rejection_sbc_fallback"
            ibis_count += 1
        elif smc_cases:
            method = "ibis"
            reason = "joint_smc_sbc_fallback"
            ibis_count += 1
        elif entry.get("pass") is None:
            method = "ibis"
            reason = "sampling_incomplete"
            ibis_count += 1
        elif calibrated_rejections == 0:
            method = "sbi"
            reason = "size_calibrated_sbc_ok"
            sbi_count += 1
        elif calibrated_rejections is None:
            method = "ibis"
            reason = "missing_size_calibrated_check"
            ibis_count += 1
        else:
            method = "ibis"
            reason = "size_calibrated_rejection"
            ibis_count += 1
        routes[name] = method
        reasons[name] = {
            "method": method,
            "reason": reason,
            "coverage_max_deviation": cov_dev,
            "ks_pvalues": ks,
            "sbc_pass": per_drug_pass,
            "size_calibrated_rejections": calibrated_rejections,
            "rejection_fallback_cases": rejection_cases,
            "joint_smc_fallback_cases": smc_cases,
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
            raw_coverage = entry["coverage_max_deviation"]
            coverage = float(raw_coverage) if raw_coverage is not None else None
            rejection_cases = int(entry.get("rejection_fallback_cases", 0))
            smc_cases = int(entry.get("joint_smc_fallback_cases", 0))
            passed = (entry.get("pass") is not None
                      and entry.get("size_calibrated_rejections") == 0
                      and rejection_cases == 0 and smc_cases == 0)
            decisions[name] = {
                "use_native": passed,
                "smiles": smiles[name],
                "sbc_pass": bool(entry.get("pass", False)),
                "coverage_max_deviation": coverage,
                "ks_pvalues": entry["ks_pvalues"],
                "size_calibrated_rejections": entry.get("size_calibrated_rejections"),
                "rejection_fallback_cases": rejection_cases,
                "joint_smc_fallback_cases": smc_cases,
            }
        multi_observation = {
            "source": str(args.multi_observation_sbc.resolve().relative_to(ROOT)),
            "model": "multi_observation_nsf.pt",
            "validation_status": "complete",
            "global_sbc_gate_passed": multi_sbc.get("gate_passed"),
            "size_calibrated_global_gate_status": multi_sbc.get(
                "size_calibrated_global_gate", {}
            ).get("status"),
            "production_scope_gate_status": multi_sbc.get(
                "production_scope_size_calibrated_gate", {}
            ).get("status"),
            "native_count": sum(row["use_native"] for row in decisions.values()),
            "sbc_exact_rejection_cases_total": sum(
                row["rejection_fallback_cases"] for row in decisions.values()
            ),
            "sbc_exact_rejection_cases_on_native_routes": sum(
                row["rejection_fallback_cases"] for row in decisions.values()
                if row["use_native"]
            ),
            "drugs": decisions,
        }
    else:
        validation = json.loads((ROOT / "data/sbi/validation_drug_set.json").read_text())
        multi_observation = {
            "source": str(args.multi_observation_sbc.resolve().relative_to(ROOT)),
            "model": "multi_observation_nsf.pt",
            "validation_status": "pending_corrected_sbc",
            "pending_reason": "Corrected multi-observation SBC is incomplete; use IBIS.",
            "global_sbc_gate_passed": None,
            "size_calibrated_global_gate_status": "PENDING",
            "production_scope_gate_status": "PENDING",
            "native_count": 0,
            "drugs": {
                entry["name"]: {
                    "smiles": entry["smiles"], "use_native": False,
                    "reason": "corrected_sbc_pending",
                }
                for entry in validation["drugs"]
            },
        }

    covariate_sections = {
        "hierarchical": existing.get("hierarchical"),
        "continuous": existing.get("continuous"),
    }
    for flow, path, model in (
        ("hierarchical", args.hierarchical_sbc, "hierarchical_nsf_2k.pt"),
        ("continuous", args.continuous_sbc, "continuous_hierarchical_nsf.pt"),
    ):
        if not path.exists():
            covariate_sections[flow] = {
                "source": str(path.resolve().relative_to(ROOT)),
                "model": model,
                "validation_status": "pending_corrected_sbc",
                "pending_reason": f"Corrected {flow} SBC is incomplete; use IBIS.",
                "global_sbc_gate_passed": None,
                "size_calibrated_global_gate_status": "PENDING",
                "production_scope_gate_status": "PENDING",
                "sbi_pair_count": 0,
                "pairs": [],
            }
            continue
        flow_sbc = json.loads(path.read_text())
        validation = json.loads((ROOT / "data/sbi/validation_drug_set.json").read_text())
        smiles = {entry["name"]: entry["smiles"] for entry in validation["drugs"]}
        populations = {
            entry["name"]: entry for entry in flow_sbc["populations"]
        } if flow == "continuous" else {}
        pair_decisions = []
        for entry in flow_sbc["pairs"]:
            rejection_cases = int(entry.get("rejection_fallback_cases", 0))
            smc_cases = int(entry.get("joint_smc_fallback_cases", 0))
            raw_coverage = entry["coverage_max_deviation"]
            coverage = float(raw_coverage) if raw_coverage is not None else None
            use_sbi = (entry.get("pass") is not None
                       and entry.get("size_calibrated_rejections") == 0
                       and rejection_cases == 0 and smc_cases == 0)
            decision = {
                "population": entry["population"],
                "drug": entry["drug"],
                "smiles": smiles[entry["drug"]],
                "use_sbi": use_sbi,
                "sbc_pass": bool(entry.get("pass", False)),
                "coverage_max_deviation": coverage,
                "ks_pvalues": entry["ks_pvalues"],
                "size_calibrated_rejections": entry.get("size_calibrated_rejections"),
                "rejection_fallback_cases": rejection_cases,
                "joint_smc_fallback_cases": smc_cases,
            }
            if flow == "continuous":
                population = populations[entry["population"]]
                decision["body_weight_kg"] = population["bw"]
                decision["age_years"] = population["age"]
            pair_decisions.append(decision)
        covariate_sections[flow] = {
            "source": str(path.resolve().relative_to(ROOT)),
            "model": model,
            "validation_status": "complete",
            "global_sbc_gate_passed": flow_sbc.get("gate_passed"),
            "size_calibrated_global_gate_status": flow_sbc.get(
                "size_calibrated_global_gate", {}
            ).get("status"),
            "production_scope_gate_status": flow_sbc.get(
                "production_scope_size_calibrated_gate", {}
            ).get("status"),
            "sbi_pair_count": sum(row["use_sbi"] for row in pair_decisions),
            "sbc_exact_rejection_cases_total": sum(
                row["rejection_fallback_cases"] for row in pair_decisions
            ),
            "sbc_exact_rejection_cases_on_sbi_routes": sum(
                row["rejection_fallback_cases"] for row in pair_decisions if row["use_sbi"]
            ),
            "pairs": pair_decisions,
        }

    payload = {
        "source": str(args.sbc.resolve().relative_to(ROOT)),
        "policy": {
            "original_strict_coverage_ok": STRICT_COVERAGE_OK,
            "original_borderline_coverage": BORDERLINE_COVERAGE,
            "routing_rule_decision_timing": (
                "Amended after corrected multi-drug SBC results were visible; "
                "the original per-row gate remains informational."
            ),
            "description": (
                "SBI only where per-drug or covariate-pair SBC has no exact rejection, "
                "joint SMC fallback, or "
                "uncomputable case and none of its KS or coverage tests is rejected by "
                "the flow-wide combined Holm family at FWER 0.01. Original uncorrected "
                "per-row pass is reported, not used for routing. Existing IS routes remain "
                "IS; other failures use IBIS."
            ),
        },
        "default": args.default,
        "summary": {
            "sbi": sbi_count,
            "ibis": ibis_count,
            "total": len(routes),
            "is": is_count,
            "sbi_reweight_drugs": sum(bool(value) for value in reweight.values()),
            "global_sbc_gate_passed": sbc.get("gate_passed"),
            "size_calibrated_global_gate_status": sbc.get(
                "size_calibrated_global_gate", {}
            ).get("status"),
            "production_scope_gate_status": sbc.get(
                "production_scope_size_calibrated_gate", {}
            ).get("status"),
            "sbc_exact_rejection_cases_total": sum(
                row.get("rejection_fallback_cases", 0) for row in sbc["drugs"]
            ),
            "sbc_exact_rejection_cases_on_sbi_routes": sum(
                reasons[name]["rejection_fallback_cases"]
                for name, method in routes.items() if method == "sbi"
            ),
        },
        "sbi_reweight": reweight,
        "multi_observation": multi_observation,
        "routes": routes,
        "reasons": reasons,
    }
    payload.update({key: value for key, value in covariate_sections.items() if value is not None})

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
        cov_text = (f"{data['coverage_max_deviation']:.3f}"
                    if data["coverage_max_deviation"] is not None else "incomplete")
        print(
            f"  {mark} {name:<15s}  {data['method']:<5s}  "
            f"cov_dev={cov_text}  ({data['reason']})"
        )


if __name__ == "__main__":
    main()
