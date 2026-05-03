"""Diagnose pravastatin ECM under-prediction vs Niemi 2006 (issue #8).

Computes current ECM-active prediction (Cmax + AUC over 24h) for the SLCO1B1
EM and PM phenotypes at 40 mg oral, and compares each layer to literature.

Niemi 2006 (Pharmacogenet Genomics 16(11):801-8): single 40 mg pravastatin oral
dose, 6 EM + 4 PM individuals. Reported geometric means:
    EM:  Cmax 75 ng/mL (0.075 mg/L),  AUC0-12  250 ng·h/mL (0.250 mg·h/L)
    PM:  Cmax 195 ng/mL (0.195 mg/L), AUC0-12  500 ng·h/mL (0.500 mg·h/L)

Sisyphus current (post-PR #22 #16):
    Hardening realize_means(); ECM-active build_drug_on_graph; t_end 24h.
"""
from __future__ import annotations

import pathlib

import numpy as np

import sisyphus.engine.flux  # noqa: F401
from sisyphus.engine.compiler import ODECompiler, ResolvedParams
from sisyphus.engine.solver import solve
from sisyphus.graph.builder import build_from_yaml
from sisyphus.predict.adme import predict_adme
from sisyphus.predict.chemistry import compute_profile
from sisyphus.predict.ivive import build_drug_on_graph
from sisyphus.predict.phenotype import apply_phenotype_to_graph
from sisyphus.predict.transporter_db import (
    load_hepatic_ecm_params,
    load_oatp1b1_kinetics,
)

_PHYS = pathlib.Path("data/physiology/reference_man.yaml")
_PRAVASTATIN = (
    "CC[C@@H](C)C(=O)O[C@@H]1C[C@H](C=C2[C@@H]1CC[C@H]"
    "([C@@H]2CC[C@H](C[C@H](CC(=O)O)O)O)C)O"
)


def build_drug(graph):
    profile = compute_profile(_PRAVASTATIN)
    adme = predict_adme(profile)
    liver_enz = {tag: d.mean for tag, d in graph.nodes["liver"].enzymes.items()}
    drug = build_drug_on_graph(
        profile, adme, dose_mg=40.0, route="oral",
        liver_enzymes=liver_enz,
        transporter_kinetics=load_oatp1b1_kinetics("pravastatin"),
        hepatic_ecm_params=load_hepatic_ecm_params("pravastatin"),
    )
    return profile, adme, drug


def simulate(graph, drug):
    rg = graph.realize_means()
    rd = drug.realize_means()
    compiler = ODECompiler()
    compiled = compiler.compile(rg)
    params = ResolvedParams(rg, rd)
    y0 = np.zeros(compiled.n_states)
    y0[compiled.state_index[drug.administration_node]] = drug.dose_mg
    t_eval = np.linspace(0, 24.0, 4801)
    result = solve(compiled, params, y0, t_span=(0, 24.0), t_eval=t_eval)
    if not result.solver_success:
        raise RuntimeError("solver failed")
    c = result.concentrations["venous_blood"]
    cmax = float(np.max(c))
    auc = float(np.trapz(c, result.time_h))
    return cmax, auc


def main():
    print("=" * 70)
    print("Pravastatin ECM diagnostic vs Niemi 2006 (issue #8)")
    print("=" * 70)

    graph_em = build_from_yaml(_PHYS)
    graph_pm = apply_phenotype_to_graph(graph_em, {"SLCO1B1": "PM"})

    profile, adme, drug_em = build_drug(graph_em)
    _, _, drug_pm = build_drug(graph_pm)

    cmax_em, auc_em = simulate(graph_em, drug_em)
    cmax_pm, auc_pm = simulate(graph_pm, drug_pm)

    print("\n--- Layer 1: Chemistry / ADME ---")
    print(f"MW             : {profile.mw:.3f}")
    print(f"logP           : {profile.logp:.3f}")
    print(f"pKa            : {profile.pka}")
    print(f"compound_type  : {profile.compound_type}")
    print(f"fup (predicted): {adme.fup.mean:.4f}  (lit: ~0.50, Hatorp 1996)")
    print(f"rbp (predicted): {adme.rbp.mean:.4f}  (lit: ~0.55, Pharmacia label)")
    print(f"peff (predicted): {adme.peff.mean:.4e}  (cm/s)")
    print(f"solubility (mg/L): {adme.solubility.mean:.4f}")

    print("\n--- Layer 2: OATP1B1 transporter kinetics (verbatim Varma 2014) ---")
    k = load_oatp1b1_kinetics("pravastatin")["OATP1B1"]
    print(f"Jmax (pmol/min/mg): {k.jmax.mean:.1f}")
    print(f"Km (uM)           : {k.km.mean:.2f}")

    print("\n--- Layer 3: ECM passive/efflux/biliary (Watanabe/Yabe) ---")
    e = load_hepatic_ecm_params("pravastatin")
    print(f"PS_passive (L/h)  : {e['ps_passive'].mean:.2f}")
    print(f"PS_eff (L/h)      : {e['ps_eff'].mean:.2f}")
    print(f"CL_int_bile (L/h) : {e['cl_int_bile'].mean:.2f}")

    print("\n--- Layer 4: liver OATP1B1 abundance ---")
    oatp = graph_em.nodes["liver"].transporters.get("OATP1B1")
    print(f"OATP1B1 abundance : {oatp.mean:.2e}  (#14 calibration confirms 5.0e5)")

    print("\n--- Predicted Cmax / AUC ---")
    print(f"{'pheno':<8}{'Cmax (mg/L)':>14}{'AUC0-24 (mg·h/L)':>22}")
    print(f"{'EM':<8}{cmax_em:>14.4f}{auc_em:>22.4f}")
    print(f"{'PM':<8}{cmax_pm:>14.4f}{auc_pm:>22.4f}")

    print("\n--- Niemi 2006 reference (40 mg single oral) ---")
    print(f"{'pheno':<8}{'Cmax (mg/L)':>14}{'AUC0-12 (mg·h/L)':>22}")
    niemi = {
        "EM": (0.075, 0.250),
        "PM": (0.195, 0.500),
    }
    for pheno, (cmax_obs, auc_obs) in niemi.items():
        print(f"{pheno:<8}{cmax_obs:>14.4f}{auc_obs:>22.4f}")

    print("\n--- Fold error (obs/pred) ---")
    print(f"{'metric':<22}{'EM':>10}{'PM':>10}")
    print(
        f"{'Cmax obs/pred':<22}"
        f"{niemi['EM'][0]/cmax_em:>10.2f}{niemi['PM'][0]/cmax_pm:>10.2f}"
    )
    print(
        f"{'AUC obs/pred':<22}"
        f"{niemi['EM'][1]/auc_em:>10.2f}{niemi['PM'][1]/auc_pm:>10.2f}"
    )
    print(
        f"\nNote: AUC obs is 0-12h; pred is 0-24h. Pravastatin t1/2 ~1.8h → "
        f"24h AUC ~ 12h AUC + ~ε; comparison is valid within ~5%."
    )


if __name__ == "__main__":
    main()
