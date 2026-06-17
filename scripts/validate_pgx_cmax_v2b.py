"""PGx v2.2b saturable-genotype Cmax-fold harness — feasibility SPIKE (propafenone).

Run from repo root:  /opt/miniconda3/bin/python scripts/validate_pgx_cmax_v2b.py

This is a Task-1 SPIKE. It answers four gating questions on ONE drug (propafenone,
CYP2D6) about whether the v2.2a saturable Michaelis-Menten metabolism (DrugOnGraph.
enzyme_km) actually ENGAGES at therapeutic first-pass liver concentrations — i.e.
whether a saturable gene produces a genotype Cmax/AUC fold that differs from the
linear-Km=inf null. Headline-isolated: imports nothing from the production predict
path; no predict()/reference_man.yaml/holdout change.

Recipe ported from the never-executed v2.1 scaffold
(docs/superpowers/plans/2026-06-15-pgx-cmax-fold-engine-differentiated.md, Task 3),
extended with a saturable drug builder + a saturable EM-anchor.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

import sisyphus.engine.flux  # noqa: F401  -- register flux specs
from sisyphus.core import Distribution, DrugOnGraph
from sisyphus.engine.compiler import ODECompiler, ResolvedParams
from sisyphus.engine.solver import solve
from sisyphus.graph.builder import build_from_yaml
from sisyphus.predict.phenotype import apply_phenotype_to_graph

_YAML = Path("data/physiology/reference_man.yaml")
_RESID = "RESIDUAL_HEPATIC"
_SYNTHETIC_GENE_ABUND = 1.0e6
# Dense early grid (read Cmax off the peak) + coarse tail (terminal slope for AUC_0inf).
_T_DENSE = np.linspace(0.0, 24.0, 480)
_T_TAIL = np.linspace(24.5, 600.0, 240)
_T_EVAL = np.concatenate([_T_DENSE, _T_TAIL])

# Propafenone (CYP2D6) reference constants.
PROPAF_MW = 341.4
PROPAF_FM = 0.80
PROPAF_DOSE_EM = 300.0
PROPAF_DOSE_HIGH = 400.0
PROPAF_TMAX = 2.5
PROPAF_THALF = 5.5
PROPAF_ORAL_F = 0.10  # EM, high first-pass => E_h ~= 0.90
PROPAF_EH = 1.0 - PROPAF_ORAL_F
# Km 5.3 uM (Kroemer), fu_mic 0.5 => unbound mg/L = 5.3 * MW/1000 * fu_mic.
PROPAF_KM_MGL = 5.3 * PROPAF_MW / 1000.0 * 0.5  # = 0.90471


def _well_stirred_graph(gene_tag: str):
    """Reference graph with the liver clearance edge forced to well_stirred (linear in
    CLint — the model the saturable factor multiplies). Edges are frozen => replace()."""
    g = build_from_yaml(_YAML)
    g.edges[:] = [
        replace(e, model="well_stirred")
        if getattr(e, "source", None) == "liver" and getattr(e, "model", None) == "extended"
        else e
        for e in g.edges
    ]
    liver = g.nodes["liver"]
    if gene_tag not in liver.enzymes:
        liver.enzymes[gene_tag] = Distribution(_SYNTHETIC_GENE_ABUND, 0.0)
    liver.enzymes[_RESID] = Distribution(1.0, 0.0)
    return g


def _drug(gene_tag: str, fm: float, cltot: float, abund_gene: float,
          peff: float, kp: float, fup: float = 0.3, dose_mg: float = 100.0,
          mw: float = 300.0) -> DrugOnGraph:
    """Synthetic oral drug. Hepatic CLint split fm:(1-fm) between gene and RESIDUAL.
    LINEAR (no enzyme_km) — bit-identical to the v2.2a empty-Km path."""
    tissues = ["adipose", "bone", "brain", "gut", "heart", "kidney",
               "liver", "lung", "muscle", "skin", "spleen"]
    return DrugOnGraph(
        name="syn_cmax", smiles="C", dose_mg=dose_mg, route="oral",
        administration_node="stomach_lumen", mw=mw, pka=None, compound_type="neutral",
        fup=Distribution(fup, 0.0), rbp=Distribution(1.0, 0.0), kp_method="provided",
        kp_overrides={t: Distribution(kp, 0.0) for t in tissues},
        peff=Distribution(peff, 0.0), solubility=Distribution(1000.0, 0.0),
        enzyme_affinity={
            gene_tag: Distribution(fm * cltot / abund_gene, 0.0),
            _RESID: Distribution((1.0 - fm) * cltot, 0.0),
        },
        renal_clearance=Distribution(0.0, 0.0),
    )


def _sat_drug(gene_tag: str, fm: float, cltot: float, abund_gene: float,
              peff: float, kp: float, km_mgl: float, fup: float = 0.3,
              dose_mg: float = 100.0, mw: float = 300.0) -> DrugOnGraph:
    """SATURABLE twin of _drug: gene_tag carries an enzyme_km (unbound mg/L) so its CLint
    is multiplied by 1/(1 + C_u/Km). RESIDUAL tag gets NO km (stays linear)."""
    base = _drug(gene_tag, fm, cltot, abund_gene, peff, kp, fup, dose_mg, mw)
    return replace(base, enzyme_km={gene_tag: Distribution(km_mgl, 0.0)})


def _cmax_auc_tmax(graph, drug: DrugOnGraph) -> tuple[float, float, float]:
    rg, rd = graph.realize_means(), drug.realize_means()
    compiled = ODECompiler().compile(rg)
    params = ResolvedParams(rg, rd)
    y0 = np.zeros(compiled.n_states)
    y0[compiled.state_index[drug.administration_node]] = drug.dose_mg
    res = solve(compiled, params, y0, t_span=(0.0, float(_T_EVAL[-1])), t_eval=_T_EVAL)
    conc, time = res.concentrations["venous_blood"], res.time_h
    trapz = getattr(np, "trapezoid", np.trapz)
    auc_0t = float(trapz(conc, time))
    i0 = int(len(time) * 0.7)
    tt, ct = time[i0:], conc[i0:]
    pos = ct > 0
    auc = auc_0t
    if pos.sum() >= 2:
        k = -np.polyfit(tt[pos], np.log(ct[pos]), 1)[0]
        if k > 0:
            auc = auc_0t + conc[-1] / k
    cmax = float(conc.max())
    tmax = float(time[int(np.argmax(conc))])
    return cmax, auc, tmax


def _peak_liver_cu(graph, drug: DrugOnGraph, fup: float = 0.3) -> float:
    """Peak UNBOUND liver concentration (mg/L) during the run = fup * max(c_liver).
    The quantity that must approach Km_mgL for saturation to engage."""
    rg, rd = graph.realize_means(), drug.realize_means()
    compiled = ODECompiler().compile(rg)
    params = ResolvedParams(rg, rd)
    y0 = np.zeros(compiled.n_states)
    y0[compiled.state_index[drug.administration_node]] = drug.dose_mg
    res = solve(compiled, params, y0, t_span=(0.0, float(_T_EVAL[-1])), t_eval=_T_EVAL)
    c_liver = res.concentrations["liver"]
    return float(fup * c_liver.max())


def _axial_graph(gene_tag: str, n_sub: int = 10):
    """Well_stirred synthetic skeleton with the liver clearance edge switched to
    parallel_tube and axially expanded into n_sub serial well_stirred sub-tanks
    (lookup_name='liver'). Genotype scaling reaches every sub-tank via the (A) fix."""
    import dataclasses as _dc

    from sisyphus.graph.axial import expand_axial
    from sisyphus.graph.types import ClearanceEdge
    g = _well_stirred_graph(gene_tag)
    g.nodes["liver"] = _dc.replace(g.nodes["liver"], axial_subcompartments=n_sub)
    g.edges[:] = [
        _dc.replace(e, model="parallel_tube")
        if isinstance(e, ClearanceEdge) and e.source == "liver"
        else e
        for e in g.edges
    ]
    return expand_axial(g)


def _steady_state_exposure(graph, drug, interval_h: float, n_doses: int,
                           metric: str = "css_avg") -> float:
    """Multi-dose steady-state exposure over the LAST dosing interval.
    metric: 'css_avg' (AUC_tau/tau), 'auc_tau', or 'cmax_ss'."""
    from sisyphus.regimen.solver import solve_regimen
    from sisyphus.regimen.types import DosingEvent, DosingRegimen
    rg, rd = graph.realize_means(), drug.realize_means()
    compiled = ODECompiler().compile(rg)
    params = ResolvedParams(rg, rd)
    events = tuple(
        DosingEvent(time_h=i * interval_h, dose_mg=drug.dose_mg,
                    node=drug.administration_node)
        for i in range(n_doses)
    )
    reg = DosingRegimen(events=events)
    t_total = n_doses * interval_h
    res = solve_regimen(compiled, params, reg, t_total_h=t_total, dt_output=0.05)
    conc, time = res.concentrations["venous_blood"], res.time_h
    mask = time >= (t_total - interval_h - 1e-9)
    ct, tt = conc[mask], time[mask]
    trapz = getattr(np, "trapezoid", np.trapz)
    if metric == "cmax_ss":
        return float(ct.max())
    auc_tau = float(trapz(ct, tt))
    if metric == "auc_tau":
        return auc_tau
    if metric == "css_avg":
        return auc_tau / interval_h
    raise ValueError(f"unknown metric {metric!r}")


def _single_dose_exposure(graph, drug, metric: str = "auc") -> float:
    """Single-dose exposure on the given (possibly axial) graph. metric 'auc' or 'cmax'."""
    cmax, auc, _ = _cmax_auc_tmax(graph, drug)
    if metric == "auc":
        return auc
    if metric == "cmax":
        return cmax
    raise ValueError(f"unknown metric {metric!r}")


def _iv_drug(gene_tag, fm, cltot, abund_gene, kp, fup=0.3, dose_mg=100.0, mw=300.0,
             km_mgl: float | None = None):
    """IV-bolus twin (administration at venous_blood) for an F reference. Mirrors the
    saturable/linear nature of the oral drug via km_mgl (None => linear)."""
    if km_mgl is None:
        d = _drug(gene_tag, fm, cltot, abund_gene, peff=20.0, kp=kp, fup=fup,
                  dose_mg=dose_mg, mw=mw)
    else:
        d = _sat_drug(gene_tag, fm, cltot, abund_gene, peff=20.0, kp=kp, km_mgl=km_mgl,
                      fup=fup, dose_mg=dose_mg, mw=mw)
    return replace(d, route="intravenous", administration_node="venous_blood")


def _engine_e_h(graph, gene_tag, fm, cltot, abund, peff, kp, fup=0.3, dose_mg=100.0,
                mw=300.0, km_mgl: float | None = None) -> float:
    """Engine-MEASURED hepatic extraction: E_h = 1 - AUC_oral/AUC_iv (no gut/renal loss
    on the synthetic drug, so F = 1-E_h). Captures the engine's real ivive_scaling AND
    (if km_mgl) the saturable factor — unlike the analytic fu*CLint/(Q+fu*CLint)."""
    if km_mgl is None:
        oral = _drug(gene_tag, fm, cltot, abund, peff, kp, fup, dose_mg, mw)
    else:
        oral = _sat_drug(gene_tag, fm, cltot, abund, peff, kp, km_mgl, fup, dose_mg, mw)
    iv = _iv_drug(gene_tag, fm, cltot, abund, kp, fup, dose_mg, mw, km_mgl)
    _, auc_oral, _ = _cmax_auc_tmax(graph, oral)
    _, auc_iv, _ = _cmax_auc_tmax(graph, iv)
    return 1.0 - auc_oral / auc_iv


def anchor_em(gene_tag: str, e_h_target: float, tmax_target: float, fm: float,
              kp: float = 3.0, fup: float = 0.3, dose_mg: float = 100.0,
              mw: float = 300.0, km_mgl: float | None = None) -> dict:
    """Find (cltot, peff) so the EM run's ENGINE-MEASURED E_h hits e_h_target (bisected
    against the engine) and tmax ~ tmax_target (peff, best-effort). If km_mgl is set, the
    anchor uses the SATURABLE drug, so cltot is tuned WITH saturation engaged at this dose.

    Order: peff barely moves E_h, so tune cltot for E_h first at nominal peff, then tune
    peff for tmax at the fixed cltot. Raises if E_h unreachable."""
    if not 0 < e_h_target < 0.999:
        raise ValueError(f"e_h_target out of range: {e_h_target}")
    g = _well_stirred_graph(gene_tag)
    abund = g.nodes["liver"].enzymes[gene_tag].mean

    def e_h_err(log_clt: float) -> float:
        return _engine_e_h(g, gene_tag, fm, float(np.exp(log_clt)), abund, 20.0, kp, fup,
                           dose_mg, mw, km_mgl) - e_h_target

    log_clt = brentq(e_h_err, np.log(1.0e1), np.log(1.0e9), xtol=1e-3)
    cltot = float(np.exp(log_clt))

    def tmax_err(log_peff: float) -> float:
        if km_mgl is None:
            d = _drug(gene_tag, fm, cltot, abund, float(np.exp(log_peff)), kp, fup,
                      dose_mg, mw)
        else:
            d = _sat_drug(gene_tag, fm, cltot, abund, float(np.exp(log_peff)), kp, km_mgl,
                          fup, dose_mg, mw)
        _, _, tm = _cmax_auc_tmax(g, d)
        return tm - tmax_target

    try:
        peff = float(np.exp(brentq(tmax_err, np.log(0.05), np.log(2000.0), xtol=1e-3)))
    except ValueError:
        peff = 20.0  # tmax not bracketable on this grid; default + flag

    if km_mgl is None:
        d = _drug(gene_tag, fm, cltot, abund, peff, kp, fup, dose_mg, mw)
    else:
        d = _sat_drug(gene_tag, fm, cltot, abund, peff, kp, km_mgl, fup, dose_mg, mw)
    cmax, auc, tmax_em = _cmax_auc_tmax(g, d)
    e_h_real = _engine_e_h(g, gene_tag, fm, cltot, abund, peff, kp, fup, dose_mg, mw, km_mgl)
    cu_peak = _peak_liver_cu(g, d, fup)
    return {"cltot": cltot, "peff": peff, "kp": kp, "fup": fup, "abund": abund,
            "e_h": e_h_real, "tmax_em": tmax_em, "cmax_em": cmax, "auc_em": auc,
            "cu_peak": cu_peak}


def genotype_folds(gene_tag: str, fm: float, recipe: dict, dose_mg: float, mw: float,
                   km_mgl: float | None, a_var: float = 0.0) -> dict:
    """EM vs variant (PM: a_var=0) at dose_mg on the anchored skeleton. If km_mgl, the
    SATURABLE drug is used. Returns cmax_fold, auc_fold, peak liver C_u (EM)."""
    fup, kp, abund = recipe["fup"], recipe["kp"], recipe["abund"]
    cltot, peff = recipe["cltot"], recipe["peff"]
    g = _well_stirred_graph(gene_tag)
    if km_mgl is None:
        drug = _drug(gene_tag, fm, cltot, abund, peff, kp, fup, dose_mg, mw)
    else:
        drug = _sat_drug(gene_tag, fm, cltot, abund, peff, kp, km_mgl, fup, dose_mg, mw)
    cm_em, au_em, _ = _cmax_auc_tmax(g, drug)
    cu_peak_em = _peak_liver_cu(g, drug, fup)

    gv = _well_stirred_graph(gene_tag)
    gv = apply_phenotype_to_graph(gv, {gene_tag: "PM"},
                                  phenotype_scale_overrides={gene_tag: a_var})
    cm_v, au_v, _ = _cmax_auc_tmax(gv, drug)
    return {"cmax_fold": cm_v / cm_em, "auc_fold": au_v / au_em,
            "cu_peak_em": cu_peak_em, "cmax_em": cm_em, "auc_em": au_em}


def _gate_block() -> dict:
    """Run the four-gate propafenone spike. Returns a dict of all numbers."""
    gene = "CYP2D6"
    fm = PROPAF_FM
    km = PROPAF_KM_MGL
    eh = PROPAF_EH
    out: dict = {"km_mgl": km, "fm": fm, "e_h_target": eh}

    # --- Anchor the SATURABLE EM run at the reference dose (300 mg). ---
    rec_sat = anchor_em(gene, e_h_target=eh, tmax_target=PROPAF_TMAX, fm=fm, kp=3.0,
                        fup=0.3, dose_mg=PROPAF_DOSE_EM, mw=PROPAF_MW, km_mgl=km)
    out["anchor_sat"] = rec_sat

    # --- Anchor the LINEAR (Km=inf) EM run independently at the same dose. ---
    rec_lin = anchor_em(gene, e_h_target=eh, tmax_target=PROPAF_TMAX, fm=fm, kp=3.0,
                        fup=0.3, dose_mg=PROPAF_DOSE_EM, mw=PROPAF_MW, km_mgl=None)
    out["anchor_lin"] = rec_lin

    # --- Folds. ---
    out["sat_300"] = genotype_folds(gene, fm, rec_sat, PROPAF_DOSE_EM, PROPAF_MW, km)
    out["lin_300"] = genotype_folds(gene, fm, rec_lin, PROPAF_DOSE_EM, PROPAF_MW, None)
    out["sat_400"] = genotype_folds(gene, fm, rec_sat, PROPAF_DOSE_HIGH, PROPAF_MW, km)

    # --- Low-extraction oracle control (Gate 4 fallback): E_h ~= 0.3, saturable off. ---
    rec_lowE = anchor_em(gene, e_h_target=0.30, tmax_target=PROPAF_TMAX, fm=fm, kp=3.0,
                         fup=0.3, dose_mg=PROPAF_DOSE_EM, mw=PROPAF_MW, km_mgl=None)
    out["anchor_lowE"] = rec_lowE
    out["lin_lowE_300"] = genotype_folds(gene, fm, rec_lowE, PROPAF_DOSE_EM, PROPAF_MW,
                                         None)

    # --- Gates. ---
    sat300_auc = out["sat_300"]["auc_fold"]
    lin300_auc = out["lin_300"]["auc_fold"]
    sat400_auc = out["sat_400"]["auc_fold"]
    analytic_auc = 1.0 / (1.0 - fm)  # = 5.0 for fm=0.80

    g1 = (np.isfinite(rec_sat["cltot"]) and np.isfinite(rec_sat["peff"])
          and abs(rec_sat["e_h"] - eh) / eh < 0.05)
    g2_val = abs(np.log(sat300_auc) - np.log(lin300_auc))
    g2 = g2_val > 0.10
    g3_val = abs(np.log(sat300_auc) - np.log(sat400_auc))
    g3_sign_ok = sat400_auc < sat300_auc  # saturable AUC-fold should SHRINK 300->400
    g3 = g3_val > 0.03 and g3_sign_ok
    g4_he_val = abs(out["lin_300"]["auc_fold"] - analytic_auc) / analytic_auc
    g4_he = g4_he_val < 0.02
    g4_le_val = abs(out["lin_lowE_300"]["auc_fold"] - analytic_auc) / analytic_auc
    g4_le = g4_le_val < 0.02

    out["gates"] = {
        "analytic_auc": analytic_auc,
        "g1_converge": bool(g1),
        "g2_saturation_engaged": bool(g2), "g2_val": g2_val,
        "g3_dose_dependence": bool(g3), "g3_val": g3_val, "g3_sign_ok": bool(g3_sign_ok),
        "g4_oracle_highE": bool(g4_he), "g4_highE_relerr": g4_he_val,
        "g4_oracle_lowE": bool(g4_le), "g4_lowE_relerr": g4_le_val,
    }
    return out


def main() -> None:
    o = _gate_block()
    g = o["gates"]
    km = o["km_mgl"]

    def fmt(d):
        return f"Cmax={d['cmax_fold']:.4f}  AUC={d['auc_fold']:.4f}"

    print("=" * 72)
    print("PGx v2.2b saturable harness feasibility SPIKE — propafenone (CYP2D6)")
    print("=" * 72)
    print(f"Km_unbound = {km:.5f} mg/L | fm={o['fm']} | E_h target={o['e_h_target']}")
    print()
    print("--- Anchors ---")
    print(f"SAT  EM@300: cltot={o['anchor_sat']['cltot']:.3g}  peff={o['anchor_sat']['peff']:.3g}"
          f"  E_h={o['anchor_sat']['e_h']:.4f}  tmax={o['anchor_sat']['tmax_em']:.2f}"
          f"  Cu_peak={o['anchor_sat']['cu_peak']:.4f}")
    print(f"LIN  EM@300: cltot={o['anchor_lin']['cltot']:.3g}  peff={o['anchor_lin']['peff']:.3g}"
          f"  E_h={o['anchor_lin']['e_h']:.4f}  tmax={o['anchor_lin']['tmax_em']:.2f}"
          f"  Cu_peak={o['anchor_lin']['cu_peak']:.4f}")
    print(f"LIN  EM@lowE(0.30): cltot={o['anchor_lowE']['cltot']:.3g}"
          f"  E_h={o['anchor_lowE']['e_h']:.4f}")
    print()
    print("--- Folds (PM/EM) ---")
    print(f"sat_300:      {fmt(o['sat_300'])}   Cu_peak(EM)={o['sat_300']['cu_peak_em']:.4f}")
    print(f"lin_300:      {fmt(o['lin_300'])}   Cu_peak(EM)={o['lin_300']['cu_peak_em']:.4f}")
    print(f"sat_400:      {fmt(o['sat_400'])}   Cu_peak(EM)={o['sat_400']['cu_peak_em']:.4f}")
    print(f"lin_lowE_300: {fmt(o['lin_lowE_300'])}")
    print()
    print(f"Cu_peak(EM)/Km @300 = {o['sat_300']['cu_peak_em'] / km:.3f}")
    print(f"Cu_peak(EM)/Km @400 = {o['sat_400']['cu_peak_em'] / km:.3f}")
    print()
    print("--- GATES ---")
    print(f"Gate 1 (converge):           {g['g1_converge']}  "
          f"(E_h={o['anchor_sat']['e_h']:.4f} vs {o['e_h_target']})")
    print(f"Gate 2 (saturation engaged): {g['g2_saturation_engaged']}  "
          f"|dlog AUC sat-lin|={g['g2_val']:.4f}  (>0.10)")
    print(f"Gate 3 (dose-dependence):    {g['g3_dose_dependence']}  "
          f"|dlog AUC 300-400|={g['g3_val']:.4f} (>0.03)  shrink_300to400={g['g3_sign_ok']}")
    print(f"Gate 4 (oracle highE):       {g['g4_oracle_highE']}  "
          f"relerr={g['g4_highE_relerr']:.4f} vs analytic {g['analytic_auc']:.3f}")
    print(f"Gate 4 (oracle lowE 0.30):   {g['g4_oracle_lowE']}  "
          f"relerr={g['g4_lowE_relerr']:.4f}")
    print()
    verdict = "PASS" if (g["g1_converge"] and g["g2_saturation_engaged"]) else "HALT"
    print(f"VERDICT: {verdict}  (Gate 2 is decisive)")
    print("=" * 72)
    return o


if __name__ == "__main__":
    main()
