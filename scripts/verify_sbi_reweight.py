#!/usr/bin/env python3
"""P6 Option 2 empirical verification: sbi_reweight=True on morphine.

Compares ``bayesian_update(method="sbi", sbi_reweight=False)`` vs
``sbi_reweight=True`` on the two drugs that matter:

- morphine (non-identifiable theta, known +52% bias without reweighting).
  Expected: bias drops to ~IS-level (+3%).
- clozapine (tight posterior, SBI already best).
  Expected: no regression (|bias delta| <= 5pp).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import sisyphus.engine.flux  # noqa: E402, F401
from sisyphus.engine.compiler import ODECompiler  # noqa: E402
from sisyphus.graph.builder import build_from_yaml  # noqa: E402
from sisyphus.predict.adme import predict_adme  # noqa: E402
from sisyphus.predict.chemistry import compute_profile  # noqa: E402
from sisyphus.predict.ivive import build_drug_on_graph  # noqa: E402
from sisyphus.regimen.tdm import Observation, bayesian_update  # noqa: E402
from sisyphus.regimen.types import DosingRegimen  # noqa: E402
from sisyphus.validation.reference import load_reference  # noqa: E402

DRUGS = [
    {"name": "morphine", "t_obs_h": 1.0},
    {"name": "clozapine", "t_obs_h": 2.5},
]


def build_drug(name: str, dose_mg: float, graph):
    refs = [r for r in load_reference() if r.name.lower() == name.lower()]
    if not refs:
        raise ValueError(f"{name} not in reference")
    ref = refs[0]
    profile = compute_profile(ref.smiles)
    adme = predict_adme(profile)
    liver_enzymes = None
    if "liver" in graph.nodes and graph.nodes["liver"].enzymes:
        liver_enzymes = {
            tag: dist.mean for tag, dist in graph.nodes["liver"].enzymes.items()
        }
    drug = build_drug_on_graph(
        profile, adme, dose_mg, "oral", liver_enzymes=liver_enzymes
    )
    return drug, ref


def run_sbi(compiled, graph, drug, regimen, obs, *, reweight: bool):
    t0 = time.time()
    res = bayesian_update(
        compiled,
        graph,
        drug,
        regimen,
        (obs,),
        n_prior=200,
        seed=42,
        method="sbi",
        sbi_reweight=reweight,
        sbi_fallback=False,
    )
    wall = time.time() - t0
    return res, wall


def main() -> None:
    graph = build_from_yaml(Path("data/physiology/reference_man.yaml"))
    compiled = ODECompiler().compile(graph)

    report = {"drugs": {}}

    for spec in DRUGS:
        name = spec["name"]
        print(f"\n=== {name} ===")
        drug, ref = build_drug(name, ref_dose := 0.0, graph)  # placeholder
        # rebuild with correct dose
        drug, ref = build_drug(name, float(ref.dose_mg), graph)
        regimen = DosingRegimen.single_oral(float(ref.dose_mg))
        obs = Observation(
            time_h=spec["t_obs_h"], concentration=float(ref.cmax_obs), cv=0.1
        )

        print(f"  obs: Cmax={ref.cmax_obs:.5f} mg/L at t={spec['t_obs_h']}h")

        print("  Running sbi_reweight=False...")
        res_off, wall_off = run_sbi(compiled, graph, drug, regimen, obs, reweight=False)
        bias_off = (res_off.posterior_cmax.mean - ref.cmax_obs) / ref.cmax_obs

        print("  Running sbi_reweight=True...")
        res_on, wall_on = run_sbi(compiled, graph, drug, regimen, obs, reweight=True)
        bias_on = (res_on.posterior_cmax.mean - ref.cmax_obs) / ref.cmax_obs

        drug_out = {
            "observed_cmax": float(ref.cmax_obs),
            "sbi_reweight_off": {
                "posterior_cmax_mean": float(res_off.posterior_cmax.mean),
                "posterior_cmax_cv": float(res_off.posterior_cmax.cv),
                "bias_pct": float(bias_off * 100),
                "ess": float(res_off.ess),
                "wall_s": float(wall_off),
            },
            "sbi_reweight_on": {
                "posterior_cmax_mean": float(res_on.posterior_cmax.mean),
                "posterior_cmax_cv": float(res_on.posterior_cmax.cv),
                "bias_pct": float(bias_on * 100),
                "ess": float(res_on.ess),
                "wall_s": float(wall_on),
            },
            "bias_delta_pp": float((bias_on - bias_off) * 100),
        }
        report["drugs"][name] = drug_out

        print(
            f"  OFF: Cmax={res_off.posterior_cmax.mean:.5f} "
            f"CV={res_off.posterior_cmax.cv*100:.1f}% "
            f"bias={bias_off*100:+.1f}% ESS={res_off.ess:.0f} wall={wall_off:.1f}s"
        )
        print(
            f"  ON : Cmax={res_on.posterior_cmax.mean:.5f} "
            f"CV={res_on.posterior_cmax.cv*100:.1f}% "
            f"bias={bias_on*100:+.1f}% ESS={res_on.ess:.0f} wall={wall_on:.1f}s"
        )
        print(f"  Δbias = {(bias_on - bias_off)*100:+.1f} pp")

    out_path = ROOT / "data" / "validation" / "sbi_reweight_verification.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
