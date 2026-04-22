#!/usr/bin/env python3
"""ECM generalization test — engine execution driver.

Pre-registered per spec docs/superpowers/specs/2026-04-21-ecm-generalization-test-design.md.

For each VERIFIED non-statin OATP1B1 substrate (valsartan + glimepiride under amendment v2):
1. Load IV dose + observed Cmax from data/validation/oatp_generalization_drugs.json
2. Load frozen Jmax/Km from data/transporters/oatp1b1.json
3. Build DrugOnGraph with route=iv, administration_node=venous_blood
4. Propagate uncertainty (N=1000 MC samples, fast mode)
5. Classify per-drug pass/fail, aggregate Mode A/B/C/D

Writes data/validation/oatp_generalization_result.json.

Usage:
    python scripts/validate_oatp_generalization.py

SMOKE mode (Task 4 plumbing check only — NOT the pre-registered run):
    SISYPHUS_OATP_GEN_SMOKE=1 python scripts/validate_oatp_generalization.py
    Writes data/validation/oatp_generalization_result.smoke.json (N=10).
"""

from __future__ import annotations

import json
import os as _os
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import sisyphus.engine.flux  # noqa: F401,E402 -- register flux specs
from sisyphus.engine.compiler import ODECompiler  # noqa: E402
from sisyphus.engine.uncertainty import UncertaintyEngine  # noqa: E402
from sisyphus.graph.builder import build_from_yaml  # noqa: E402
from sisyphus.predict.adme import predict_adme  # noqa: E402
from sisyphus.predict.chemistry import compute_profile  # noqa: E402
from sisyphus.predict.ivive import build_drug_on_graph  # noqa: E402
from sisyphus.predict.transporter_db import (  # noqa: E402
    load_hepatic_ecm_params,
    load_oatp1b1_kinetics,
)
from sisyphus.validation.oatp_generalization import (  # noqa: E402
    classify_aggregate,
    classify_drug,
)

_PHYS = ROOT / "data" / "physiology" / "reference_man.yaml"
_OBS_FILE = ROOT / "data" / "validation" / "oatp_generalization_drugs.json"
_OUT = ROOT / "data" / "validation" / "oatp_generalization_result.json"

# Frozen per spec §Execution Constraints. SMOKE mode for Task 4 Step 2 only
# (writes to .smoke.json, not the pre-registered result path).
_IS_SMOKE = _os.environ.get("SISYPHUS_OATP_GEN_SMOKE") == "1"
_MC_N_SAMPLES = 10 if _IS_SMOKE else 1000


def _run_one(name: str, entry: dict, graph, liver_enzymes: dict) -> dict:
    """Run MC for one drug. Returns per-drug result dict."""
    smiles = entry["smiles"]
    dose_mg = float(entry["dose_mg"])
    observed = float(entry["observed_cmax_mg_l"])

    profile = compute_profile(smiles)
    adme = predict_adme(profile)
    oatp_kinetics = load_oatp1b1_kinetics(name)
    ecm_params = load_hepatic_ecm_params(name)

    drug = build_drug_on_graph(
        profile,
        adme,
        dose_mg=dose_mg,
        route="iv",
        liver_enzymes=liver_enzymes,
        transporter_kinetics=oatp_kinetics,
        hepatic_ecm_params=ecm_params,
    )

    compiler = ODECompiler()
    compiled = compiler.compile(graph)
    ue = UncertaintyEngine()

    t0 = time.time()
    mc = ue.propagate_fast(
        compiled=compiled,
        graph=graph,
        drug=drug,
        n_samples=_MC_N_SAMPLES,
        seed=42,
        t_span=(0.0, 24.0),
        observation_node="venous_blood",
    )
    elapsed = time.time() - t0

    cmax_samples = mc.cmax_samples
    point_estimate = float(np.median(cmax_samples))  # spec: median
    pi_low, pi_high = mc.cmax_90ci

    outcome = classify_drug(name, observed, point_estimate, pi_low, pi_high)

    # Confound diagnostics — identify predict-layer issues (fup/rbp off vs published).
    # kp_overrides lives inside DrugOnGraph (built in ivive.py), not in ADMEProperties,
    # so we report only fup and rbp from the ADME layer here.
    try:
        fup_predicted = float(adme.fup.mean)
    except (AttributeError, TypeError):
        fup_predicted = float("nan")
    try:
        rbp_predicted = float(adme.rbp.mean)
    except (AttributeError, TypeError):
        rbp_predicted = float("nan")

    confound = {
        "fup_predicted": fup_predicted,
        "rbp_predicted": rbp_predicted,
        "notes": (
            "Flag drugs where fup/rbp >3x off published value — predict-layer confound. "
            "Manual check required. kp_liver not reported here (built inside ivive, not in ADMEProperties)."
        ),
    }

    return {
        "drug": name,
        "dose_mg": dose_mg,
        "observed_cmax_mg_l": observed,
        "point_estimate_cmax_mg_l": point_estimate,
        "pi_90_low_mg_l": float(pi_low),
        "pi_90_high_mg_l": float(pi_high),
        "log10_fe": outcome.log10_fe,
        "passed": outcome.passed,
        "wall_seconds": elapsed,
        "mc_n_samples": int(mc.n_samples),
        "mc_n_failures": int(mc.n_failures),
        "predict_layer_confound_diagnostics": confound,
    }


def main() -> None:
    with _OBS_FILE.open() as f:
        obs_data = json.load(f)

    graph = build_from_yaml(_PHYS)
    liver_enzymes = {
        tag: dist.mean for tag, dist in graph.nodes["liver"].enzymes.items()
    }

    if _IS_SMOKE:
        print(f"=== SMOKE MODE: N={_MC_N_SAMPLES} samples (NOT the pre-registered run) ===")
    else:
        print(f"=== ECM generalization test: N={_MC_N_SAMPLES} samples ===")

    drug_results = []
    outcomes = []
    for name in sorted(obs_data["drugs"].keys()):
        entry = obs_data["drugs"][name]
        if entry.get("status") != "VERIFIED":
            print(f"[{name}] skipped: status={entry.get('status')}")
            continue
        print(f"\n[{name}] dose={entry['dose_mg']}mg iv  observed={entry['observed_cmax_mg_l']} mg/L")
        result = _run_one(name, entry, graph, liver_enzymes)
        drug_results.append(result)
        outcomes.append(
            classify_drug(
                name,
                result["observed_cmax_mg_l"],
                result["point_estimate_cmax_mg_l"],
                result["pi_90_low_mg_l"],
                result["pi_90_high_mg_l"],
            )
        )
        print(
            f"  point={result['point_estimate_cmax_mg_l']:.4f} "
            f"PI=[{result['pi_90_low_mg_l']:.4f}, {result['pi_90_high_mg_l']:.4f}] "
            f"log10_FE={result['log10_fe']:+.3f}  passed={result['passed']}  "
            f"wall={result['wall_seconds']:.1f}s"
        )

    mode = classify_aggregate(outcomes)
    print(f"\n=== Aggregate Mode: {mode.value} ===")

    report = {
        "spec": "docs/superpowers/specs/2026-04-21-ecm-generalization-test-design.md",
        "spec_commit": "0d78c38",
        "spec_commit_chain": ["9115e63", "6e7ce0a", "0d78c38"],
        "amendment": "v2/v2.1 — substrate set = valsartan + glimepiride (N=2)",
        "mc_n_samples": _MC_N_SAMPLES,
        "drugs": drug_results,
        "aggregate_mode": mode.value,
        "mode_descriptions": {
            "A": "All-pass. ECM is confirmed as a general mechanism within domain.",
            "B": "Systematic bias. Same-direction failures with |median log10 FE| > 0.5.",
            "C": "Inconclusive. Fallback for patterns not matching A/B/D.",
            "D": "All-fail mixed. ECM = statin-specialized; architecture review required.",
        },
        "notes": "Single run per spec §Execution Constraints. No post-run parameter adjustment.",
    }

    out_path = _OUT.with_suffix(".smoke.json") if _IS_SMOKE else _OUT
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {out_path}")
    if _IS_SMOKE:
        print("(smoke run — not the pre-registered result; delete before Task 6)")


if __name__ == "__main__":
    main()
