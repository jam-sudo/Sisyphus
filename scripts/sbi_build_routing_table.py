#!/usr/bin/env python3
"""Build the per-drug TDM method routing table from SBC results.

Policy (Track B hybrid dispatch, 2026-04-10):
    - coverage_max_deviation ≤ 0.10          → "sbi"   (trust amortized posterior)
    - 0.10 < coverage_max_deviation ≤ 0.15   → "ibis"  (borderline; fall back)
    - coverage_max_deviation > 0.15          → "ibis"  (hard fail; never use SBI)

Drugs not in the validation set default to ``default`` (``ibis`` unless
overridden). The table is consumed by ``sisyphus.cli._resolve_auto_method``
when the user passes ``--method auto``.

Usage::

    python scripts/sbi_build_routing_table.py \\
        --sbc data/validation/sbi_sbc_multi_drug.json \\
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
                   default=ROOT / "data" / "validation" / "sbi_sbc_multi_drug.json")
    p.add_argument("--out", type=Path,
                   default=ROOT / "data" / "sbi" / "method_routing.json")
    p.add_argument("--default", default="ibis",
                   help="default method for drugs not in the validation set")
    args = p.parse_args()

    with open(args.sbc) as f:
        sbc = json.load(f)

    routes: dict[str, str] = {}
    reasons: dict[str, dict] = {}
    sbi_count = ibis_count = 0
    for entry in sbc["drugs"]:
        name = entry["name"]
        cov_dev = float(entry["coverage_max_deviation"])
        ks = entry["ks_pvalues"]
        if cov_dev <= STRICT_COVERAGE_OK:
            method = "sbi"
            reason = "coverage_strict_ok"
            sbi_count += 1
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
        }

    payload = {
        "source": str(args.sbc.relative_to(ROOT)),
        "policy": {
            "strict_coverage_ok": STRICT_COVERAGE_OK,
            "borderline_coverage": BORDERLINE_COVERAGE,
            "description": (
                "SBI for drugs with coverage deviation ≤ 10pp of nominal "
                "across all CI levels. IBIS fallback for borderline (≤15pp) "
                "and hard fails (>15pp). KS p-values recorded for reference "
                "only — not used in gate."
            ),
        },
        "default": args.default,
        "summary": {
            "sbi": sbi_count,
            "ibis": ibis_count,
            "total": len(routes),
        },
        "routes": routes,
        "reasons": reasons,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"[routing] {sbi_count} SBI / {ibis_count} IBIS  (default={args.default})")
    print(f"[routing] wrote {args.out}")
    print()
    for name, data in reasons.items():
        mark = "✓" if data["method"] == "sbi" else "✗"
        print(f"  {mark} {name:<15s}  {data['method']:<5s}  cov_dev={data['coverage_max_deviation']:.3f}  ({data['reason']})")


if __name__ == "__main__":
    main()
