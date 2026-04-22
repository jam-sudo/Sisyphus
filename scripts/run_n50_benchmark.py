#!/usr/bin/env python3
"""N50 secondary permanent holdout benchmark runner.

Per docs/superpowers/specs/2026-04-22-n50-secondary-holdout-design.md.

Two modes:

  --smoke-test   Run predictions on all VERIFIED drugs in holdout_n50.json
                 and print the result. Does NOT write any file. NOT a
                 release-cycle measurement. Use to verify script mechanics
                 while the seed is still being curated.

  --freeze-run   The one-shot release-cycle measurement. Requires
                 n_admitted == n_target (default 50), requires --confirm,
                 writes data/validation/n50_benchmark_<cycle_id>.json,
                 and REFUSES to overwrite an existing freeze file for
                 the same cycle. Per spec §7 this must only ever happen
                 once per cycle.

Route mapping:
  route='iv_infusion' + infusion_duration_min=X → predict(route='iv',
      infusion_duration_min=X)    (V3.1 Phase 1)
  route='iv_bolus'   → predict(route='iv')    (V3 bolus, windowed Cmax)
  route='oral'       → predict(route='oral')

CI: bootstrap 10,000 resamples on abs(log10(fold)), report
    10^percentile([2.5, 97.5]). Per docs/claude/cherry_picking_process_v1.md §3.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import pathlib
import subprocess
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

N50_FILE = ROOT / "data/reference/holdout_n50.json"
OUTPUT_DIR = ROOT / "data/validation"
N_BOOTSTRAP = 10_000


def _map_route(entry: dict) -> tuple[str, float | None]:
    """Translate N50 entry route to predict() arguments.

    Returns (route, infusion_duration_min).
    """
    raw = entry["route"]
    if raw == "oral":
        return "oral", None
    if raw == "iv_bolus":
        return "iv", None
    if raw == "iv_infusion":
        dur = entry.get("infusion_duration_min")
        if dur is None or dur <= 0:
            raise ValueError(
                f"{entry.get('_name', '<unknown>')}: iv_infusion requires "
                f"infusion_duration_min > 0, got {dur!r}"
            )
        return "iv", float(dur)
    raise ValueError(f"Unsupported route {raw!r} in N50 entry")


def _predict_one(entry: dict):
    """Run predict() on one N50 entry and return fold error + diagnostics."""
    from sisyphus.pipeline.predict import predict

    route, infusion_min = _map_route(entry)
    result = predict(
        smiles=entry["smiles"],
        dose_mg=float(entry["dose_mg"]),
        route=route,
        infusion_duration_min=infusion_min,
    )

    obs = float(entry["observed_cmax_mg_l"])
    meta = float(result.pk.cmax.mean)
    eng = float(result.engine_pk.cmax.mean) if result.engine_pk else None
    ml = float(result.ml_pk.cmax.mean) if result.ml_pk else None

    fold = meta / obs if meta > 0 and obs > 0 else None
    return {
        "observed_mg_l": obs,
        "meta_pred_mg_l": meta,
        "engine_pred_mg_l": eng,
        "ml_pred_mg_l": ml,
        "meta_fold": fold,
        "route": route,
        "infusion_duration_min": infusion_min,
        "in_applicability_domain": bool(result.in_applicability_domain),
        "ad_flags": list(result.ad_flags),
    }


def _aafe_with_ci(folds: list[float], n_bootstrap: int = N_BOOTSTRAP) -> dict:
    """Compute AAFE + bootstrap 95% CI on abs(log10(fold)).

    AAFE = 10 ^ mean(abs(log10(fold)))
    CI   = 10 ^ percentile(bootstrap_means, [2.5, 97.5])
    """
    arr = np.asarray([f for f in folds if f is not None and f > 0], dtype=float)
    if arr.size == 0:
        return {"aafe": None, "ci_95_low": None, "ci_95_high": None, "n": 0}

    log_abs = np.abs(np.log10(arr))
    aafe = float(10.0 ** log_abs.mean())

    rng = np.random.default_rng(20260422)  # deterministic CI per spec
    idx = rng.integers(0, arr.size, size=(n_bootstrap, arr.size))
    boot_means = log_abs[idx].mean(axis=1)
    ci_low = float(10.0 ** np.percentile(boot_means, 2.5))
    ci_high = float(10.0 ** np.percentile(boot_means, 97.5))

    return {"aafe": aafe, "ci_95_low": ci_low, "ci_95_high": ci_high, "n": int(arr.size)}


def _pct_within_n_fold(folds: list[float], n: float) -> float | None:
    arr = np.asarray([f for f in folds if f is not None and f > 0], dtype=float)
    if arr.size == 0:
        return None
    log_abs = np.abs(np.log10(arr))
    return float(100.0 * (log_abs <= np.log10(n)).mean())


def _git_head() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _load_n50(path: pathlib.Path) -> dict:
    with path.open() as f:
        return json.load(f)


def run(n50_payload: dict, *, oatp_only: bool = False) -> dict:
    """Evaluate all VERIFIED drugs in the N50 payload.

    Returns a dict with per-drug results and aggregate metrics (overall
    + OATP1B1 subset). Pure function — no file I/O.
    """
    drugs = n50_payload.get("drugs", {})
    per_drug: dict[str, dict] = {}
    skipped: list[tuple[str, str]] = []
    folds_all: list[float] = []
    folds_oatp: list[float] = []

    for name, entry in drugs.items():
        if entry.get("status") != "VERIFIED":
            skipped.append((name, f"status={entry.get('status')}"))
            continue
        entry_with_name = dict(entry)
        entry_with_name["_name"] = name
        if oatp_only and not entry.get("oatp1b1_substrate", False):
            continue
        try:
            res = _predict_one(entry_with_name)
        except Exception as e:
            skipped.append((name, f"predict failed: {e}"))
            logger.warning("Skipping %s: %s", name, e)
            continue

        per_drug[name] = res
        if res["meta_fold"] is not None:
            folds_all.append(res["meta_fold"])
            if entry.get("oatp1b1_substrate", False):
                folds_oatp.append(res["meta_fold"])

    overall = {
        **_aafe_with_ci(folds_all),
        "pct_2fold": _pct_within_n_fold(folds_all, 2.0),
        "pct_3fold": _pct_within_n_fold(folds_all, 3.0),
    }
    oatp = {
        **_aafe_with_ci(folds_oatp),
        "pct_2fold": _pct_within_n_fold(folds_oatp, 2.0),
        "pct_3fold": _pct_within_n_fold(folds_oatp, 3.0),
    }

    return {
        "per_drug": per_drug,
        "overall": overall,
        "oatp1b1_subset": oatp,
        "skipped": skipped,
    }


def _print_summary(label: str, results: dict, payload: dict) -> None:
    print("=" * 72)
    print(f"{label}")
    print("=" * 72)
    print(f"cycle_id         : {payload.get('cycle_id')}")
    print(f"n_admitted       : {payload.get('n_admitted')}")
    print(f"n_target         : {payload.get('n_target')}")
    print(f"curation_status  : {payload.get('curation_status')}")
    print(f"drugs evaluated  : {len(results['per_drug'])}")
    print(f"drugs skipped    : {len(results['skipped'])}")
    for name, reason in results["skipped"]:
        print(f"    - {name}: {reason}")
    print()

    overall = results["overall"]
    if overall["n"] > 0:
        print(
            f"OVERALL          AAFE {overall['aafe']:.3f} "
            f"[95% CI: {overall['ci_95_low']:.3f}, {overall['ci_95_high']:.3f}, "
            f"N={overall['n']}]  "
            f"%2-fold {overall['pct_2fold']:.1f}  %3-fold {overall['pct_3fold']:.1f}"
        )

    oatp = results["oatp1b1_subset"]
    if oatp["n"] > 0:
        print(
            f"OATP1B1 subset   AAFE {oatp['aafe']:.3f} "
            f"[95% CI: {oatp['ci_95_low']:.3f}, {oatp['ci_95_high']:.3f}, "
            f"N={oatp['n']}]"
        )
    print()
    print("Per-drug:")
    for name, r in results["per_drug"].items():
        fold = r["meta_fold"]
        fold_s = f"{fold:.2f}x" if fold is not None else "N/A"
        print(
            f"  {name:<15s} obs={r['observed_mg_l']:.3f}  "
            f"pred={r['meta_pred_mg_l']:.3f}  fold={fold_s}  "
            f"route={r['route']}"
            + (
                f" infusion={r['infusion_duration_min']:.0f}min"
                if r["infusion_duration_min"]
                else ""
            )
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--smoke-test",
        action="store_true",
        help="mechanism check; prints result only, writes nothing",
    )
    mode.add_argument(
        "--freeze-run",
        action="store_true",
        help="release-cycle one-shot; requires --confirm and n_admitted==n_target",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="required with --freeze-run to acknowledge single-use nature",
    )
    parser.add_argument(
        "--n50-file",
        type=pathlib.Path,
        default=N50_FILE,
        help="N50 JSON file (default: data/reference/holdout_n50.json)",
    )
    args = parser.parse_args()

    payload = _load_n50(args.n50_file)

    if args.smoke_test:
        results = run(payload)
        _print_summary("N50 SMOKE TEST (NOT A FREEZE RUN)", results, payload)
        print()
        print(
            "This is a mechanism check. No file written. Freeze run requires "
            "n_admitted==n_target, --freeze-run, and --confirm."
        )
        return 0

    assert args.freeze_run
    if not args.confirm:
        print("ERROR: --freeze-run requires --confirm (one-shot per spec §7).")
        return 2

    n_admitted = int(payload.get("n_admitted", 0))
    n_target = int(payload.get("n_target", 50))
    if n_admitted < n_target:
        print(
            f"ERROR: freeze run requires n_admitted ({n_admitted}) == "
            f"n_target ({n_target}). Complete curation first."
        )
        return 2

    cycle_id = payload.get("cycle_id", "unknown")
    output_path = OUTPUT_DIR / f"n50_benchmark_{cycle_id}.json"
    if output_path.exists():
        print(
            f"ERROR: freeze file already exists for cycle {cycle_id}: "
            f"{output_path}. Per spec §8 the benchmark may run only once "
            f"per cycle. Retire this file and curate N50' for a new cycle."
        )
        return 2

    results = run(payload)
    _print_summary(f"N50 FREEZE RUN — cycle {cycle_id}", results, payload)

    out_payload = {
        "cycle_id": cycle_id,
        "run_timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_commit_at_run": _git_head(),
        "spec_commit": payload.get("spec_commit"),
        "exclusion_inventory_sha256": payload.get("exclusion_inventory_sha256"),
        "n_admitted": n_admitted,
        **results,
    }
    with output_path.open("w") as f:
        json.dump(out_payload, f, indent=2, default=str)
    print(f"\nWrote freeze result: {output_path}")
    print("Retire holdout_n50.json per spec §8 before starting the next cycle.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
