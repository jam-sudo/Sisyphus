#!/usr/bin/env python3
"""Diagnostic: V3 ECM underpredict root cause — fup confound vs architecture.

Runs valsartan + glimepiride under V3 IV-Cmax with (a) predicted fup (production
path) and (b) clinical-literature fup override. Compares Cmax deltas to isolate
predict-layer fup confound from ECM architectural limitations.

Clinical fup references:
- valsartan: 0.05 (95% albumin-bound; DrugBank, Diovan label)
- glimepiride: 0.005 (99.5% protein-bound; DrugBank, Amaryl label)

Output: data/validation/v3_fup_override_diagnosis.json

NOT a pre-registered generalization run. Exploratory diagnostic only.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import sisyphus.engine.flux  # noqa: F401,E402
from sisyphus.core import Distribution  # noqa: E402
from sisyphus.engine.compiler import ODECompiler  # noqa: E402
from sisyphus.engine.solver import _IV_CMAX_DELAY_H  # noqa: E402
from sisyphus.engine.uncertainty import UncertaintyEngine  # noqa: E402
from sisyphus.graph.builder import build_from_yaml  # noqa: E402
from sisyphus.predict.adme import predict_adme  # noqa: E402
from sisyphus.predict.chemistry import compute_profile  # noqa: E402
from sisyphus.predict.ivive import build_drug_on_graph  # noqa: E402
from sisyphus.predict.transporter_db import (  # noqa: E402
    load_hepatic_ecm_params,
    load_oatp1b1_kinetics,
)

_PHYS = ROOT / "data" / "physiology" / "reference_man.yaml"
_OUT = ROOT / "data" / "validation" / "v3_fup_override_diagnosis.json"
_N_MC = 500

_DRUGS = {
    "valsartan": {
        "smiles": "CCCCC(=O)N([C@@H](C(C)C)C(=O)O)Cc1ccc(-c2ccccc2-c2nnn[nH]2)cc1",
        "dose_mg": 20.0,
        "observed_cmax_mg_l": 4.02,
        "clinical_fup": 0.05,  # DrugBank, 95% albumin-bound
    },
    "glimepiride": {
        "smiles": "CCC1=C(C)CN(C(=O)NCCc2ccc(S(=O)(=O)NC(=O)NC3CCC(C)CC3)cc2)C1=O",
        "dose_mg": 1.0,
        "observed_cmax_mg_l": 0.243,
        "clinical_fup": 0.005,  # DrugBank, 99.5% protein-bound
    },
}


def _run_one(name: str, entry: dict, graph, compiled, fup_override: float | None) -> dict:
    profile = compute_profile(entry["smiles"])
    adme = predict_adme(profile)
    fup_predicted = float(adme.fup.mean)

    if fup_override is not None:
        adme = dataclasses.replace(
            adme, fup=Distribution(mean=fup_override, cv=0.15)
        )

    oatp = load_oatp1b1_kinetics(name)
    ecm = load_hepatic_ecm_params(name)
    liver_enz = {t: d.mean for t, d in graph.nodes["liver"].enzymes.items()}
    drug = build_drug_on_graph(
        profile, adme, dose_mg=entry["dose_mg"], route="iv",
        liver_enzymes=liver_enz,
        transporter_kinetics=oatp,
        hepatic_ecm_params=ecm,
    )

    ue = UncertaintyEngine()
    mc = ue.propagate_fast(
        compiled=compiled, graph=graph, drug=drug,
        n_samples=_N_MC, seed=42,
        t_span=(0.0, 24.0),
        t_min_h=_IV_CMAX_DELAY_H,
    )
    point = float(np.median(mc.cmax_samples))
    lo, hi = mc.cmax_90ci
    log10_fe = float(np.log10(point / entry["observed_cmax_mg_l"]))

    return {
        "fup_used": fup_override if fup_override is not None else fup_predicted,
        "fup_predicted": fup_predicted,
        "point_cmax_mg_l": point,
        "pi_90_low": float(lo),
        "pi_90_high": float(hi),
        "log10_fe": log10_fe,
        "n_mc": int(mc.n_samples),
    }


def main() -> None:
    graph = build_from_yaml(_PHYS)
    compiler = ODECompiler()
    compiled = compiler.compile(graph)

    out: dict = {"drugs": {}, "n_mc": _N_MC, "note": "Exploratory fup override diagnostic — NOT a pre-registered run"}

    for name, entry in _DRUGS.items():
        print(f"\n=== {name} dose={entry['dose_mg']}mg obs={entry['observed_cmax_mg_l']} mg/L ===")
        baseline = _run_one(name, entry, graph, compiled, fup_override=None)
        override = _run_one(name, entry, graph, compiled, fup_override=entry["clinical_fup"])

        delta_cmax = override["point_cmax_mg_l"] / baseline["point_cmax_mg_l"]
        delta_log10_fe = override["log10_fe"] - baseline["log10_fe"]

        print(f"  baseline: fup={baseline['fup_used']:.4f}  Cmax={baseline['point_cmax_mg_l']:.4f}  log10_FE={baseline['log10_fe']:+.3f}")
        print(f"  override: fup={override['fup_used']:.4f}  Cmax={override['point_cmax_mg_l']:.4f}  log10_FE={override['log10_fe']:+.3f}")
        print(f"  delta: Cmax×{delta_cmax:.2f}  Δlog10_FE={delta_log10_fe:+.3f}")

        out["drugs"][name] = {
            "observed_cmax_mg_l": entry["observed_cmax_mg_l"],
            "clinical_fup_literature": entry["clinical_fup"],
            "baseline_predicted_fup": baseline,
            "override_clinical_fup": override,
            "delta_cmax_ratio": delta_cmax,
            "delta_log10_fe": delta_log10_fe,
        }

    _OUT.parent.mkdir(parents=True, exist_ok=True)
    with _OUT.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {_OUT}")


if __name__ == "__main__":
    main()
