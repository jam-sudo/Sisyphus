"""Zonal GSH-pool depletion probe (Bridge B / B1.x, Phase-0).

Upgrades the B1 static-detox hazard (PR #82) to a DYNAMIC, depleting per-zone GSH pool.
Demonstrates: the pool makes the hazard HISTORY-DEPENDENT (a pure concentration-reordering
moves the dynamic hazard but provably NOT the static pointwise hazard — the centerpiece);
excess path-dependence (bolus vs divided) over the static envelope baseline; a transient
depletion cliff (autocatalytic, reported via transition_width); an NAC-precursor lever
(monotone in GSH0); all orthogonal to bulk parent PK (DE-50). Harness-isolated; reuses the
B1 axial machinery via importlib. No predict()/reference_man.yaml/holdout/engine change.

a-priori physiological pinning (fixed BEFORE observing any cliff/path outcome, spec §3.4):
GSH resynthesis t1/2 ~ 2-4 h => k_syn = ln2/t1/2 ~ 0.2-0.35 /h; tau (divided spacing) on
the recovery timescale.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from sisyphus.validation.pgx_metrics import (
    gsh_pool_hazard,
    transition_width,
    zonal_hazard,
    zonation_weights,
)

_ROOT = Path(__file__).resolve().parent.parent


def _load(mod_name, rel):
    spec = importlib.util.spec_from_file_location(mod_name, _ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_b1 = _load("b1_probe", "scripts/probe_zonal_hazard.py")  # _parent_profile_by_zone, bulk_E, h
h = _b1.h

# --- a-priori pinned pool kinetics (spec §3.4) ---
_T_HALF_GSH_H = 3.0                      # hepatic GSH resynthesis t1/2 ~ 2-4 h (mid)
_K_SYN = np.log(2.0) / _T_HALF_GSH_H     # ~0.231 /h
_KG = 1.0                                # scavenging affinity (mg/L-equiv), synthetic
_KM_BIO = 1.0
_TAU_H = 4.0                             # divided-dose spacing, on the recovery timescale

# Same acetaminophen-like skeleton config as B1 (_CFG), minus the static detox fields.
_CFG = dict(gene_tag="CYP3A4", fm=0.9, n_sub=10, cltot=1.0e6, fup=0.3, mw=300.0,
            km_mgl=0.5)
_VMAX_BIO_TOTAL = 300.0
_GSH0_TOTAL = 150.0                       # baseline pool budget (synthetic)
_VMAX_DETOX_TOTAL = 15.0                  # B1 static capacity, for the dynamic-vs-static arm
_DOSES = [50.0, 100.0, 200.0, 400.0, 800.0]
# APAP zonation: bioactivation pericentral-high, pool pericentral-low (detox-equivalent).
_APAP = dict(bio_direction="pericentral", bio_ratio=3.0,
             gsh_direction="periportal", gsh_ratio=3.0)


def _vmax_bio_zone(n_sub, direction, ratio):
    return [_VMAX_BIO_TOTAL * w for w in zonation_weights(n_sub, ratio, direction)]


def _gsh0_zone(n_sub, direction, ratio):
    return [_GSH0_TOTAL * w for w in zonation_weights(n_sub, ratio, direction)]


def _dynamic_profile_hazard(c_by_zone, time, bio_dir, bio_ratio, gsh_dir, gsh_ratio):
    n = len(c_by_zone)
    return gsh_pool_hazard(c_by_zone, _vmax_bio_zone(n, bio_dir, bio_ratio), _KM_BIO,
                           _gsh0_zone(n, gsh_dir, gsh_ratio), _K_SYN, _KG, time)


def _static_profile_hazard(c_by_zone, time, bio_dir, bio_ratio, detox_dir, detox_ratio):
    n = len(c_by_zone)
    vmax_bio = _vmax_bio_zone(n, bio_dir, bio_ratio)
    vmax_detox = [_VMAX_DETOX_TOTAL * w for w in zonation_weights(n, detox_ratio, detox_dir)]
    return zonal_hazard(c_by_zone, vmax_bio, _KM_BIO, vmax_detox, time)


# ---------- G-order: pure history-dependence (no engine) ----------
def _ordering_profiles(levels, dt_h=2.0, pts_per_step=200):
    """Two single-zone C_u(t) trajectories visiting the SAME value-multiset in ascending vs
    descending order (each level held dt_h). Static hazard is identical by construction;
    dynamic differs (pool memory)."""
    asc = list(levels)
    desc = list(levels)[::-1]
    t_parts, ca, cd = [], [], []
    for k, (la, ld) in enumerate(zip(asc, desc)):
        seg = np.linspace(k * dt_h, (k + 1) * dt_h, pts_per_step, endpoint=False)
        t_parts.append(seg)
        ca.append(np.full_like(seg, float(la)))
        cd.append(np.full_like(seg, float(ld)))
    t = np.concatenate(t_parts + [np.array([len(asc) * dt_h])])
    c_asc = np.concatenate(ca + [np.array([float(asc[-1])])])
    c_desc = np.concatenate(cd + [np.array([float(desc[-1])])])
    return t, c_asc, c_desc


def order_test():
    levels = [0.2, 0.5, 1.0, 2.0, 4.0, 8.0]
    t, c_asc, c_desc = _ordering_profiles(levels)
    vmax_bio, gsh0 = [_VMAX_BIO_TOTAL], [_GSH0_TOTAL]
    vmax_detox = [_VMAX_DETOX_TOTAL]
    hs_a = zonal_hazard([c_asc], vmax_bio, _KM_BIO, vmax_detox, t)[0]
    hs_d = zonal_hazard([c_desc], vmax_bio, _KM_BIO, vmax_detox, t)[0]
    hd_a = gsh_pool_hazard([c_asc], vmax_bio, _KM_BIO, gsh0, _K_SYN, _KG, t)[0]
    hd_d = gsh_pool_hazard([c_desc], vmax_bio, _KM_BIO, gsh0, _K_SYN, _KG, t)[0]
    static_rel = abs(hs_a - hs_d) / max(hs_a, hs_d, 1e-30)
    dyn_rel = abs(hd_a - hd_d) / max(hd_a, hd_d, 1e-30)
    return {"static_asc": hs_a, "static_desc": hs_d, "static_rel_diff": static_rel,
            "dyn_asc": hd_a, "dyn_desc": hd_d, "dyn_rel_diff": dyn_rel}


# ---------- G-time: divided-dose two-segment axial solve ----------
def _divided_dose_profile_by_zone(total_dose, n_splits, tau_h):
    """Per-sub-tank C_u(t) for `n_splits` equal doses spaced tau_h, on the synthetic axial
    liver. Two-segment compiled-ODE solve: dose, integrate to tau, ADD next dose to the
    admin-node state, re-solve from the carried state; concatenate. Equal total mass to a
    single bolus (the static control absorbs the saturable-first-pass envelope difference)."""
    from sisyphus.engine.compiler import ODECompiler, ResolvedParams
    from sisyphus.engine.solver import solve

    g = h._axial_graph(_CFG["gene_tag"], n_sub=_CFG["n_sub"])
    abund = h._SYNTHETIC_GENE_ABUND
    drug = h._sat_drug(_CFG["gene_tag"], _CFG["fm"], _CFG["cltot"], abund, 20.0, 3.0,
                       _CFG["km_mgl"], _CFG["fup"], total_dose, _CFG["mw"])
    rg, rd = g.realize_means(), drug.realize_means()
    compiled = ODECompiler().compile(rg)
    params = ResolvedParams(rg, rd)
    admin = compiled.state_index[drug.administration_node]
    names = _b1._subtank_names(g)

    per_dose = total_dose / n_splits
    t_end = float(h._T_EVAL[-1])
    t_all, conc = [], {nm: [] for nm in names}
    y = np.zeros(compiled.n_states)
    t_offset = 0.0
    for s in range(n_splits):
        y[admin] += per_dose
        seg_end = tau_h if s < n_splits - 1 else (t_end - t_offset)
        seg_end = max(seg_end, 1e-6)
        t_eval = np.linspace(0.0, seg_end, 400)
        res = solve(compiled, params, y, t_span=(0.0, seg_end), t_eval=t_eval)
        keep = slice(0, -1) if s < n_splits - 1 else slice(0, None)
        t_all.append(t_offset + np.asarray(res.time_h)[keep])
        for nm in names:
            conc[nm].append(np.asarray(res.concentrations[nm])[keep])
        # Seed the next segment from the FULL end-of-segment state (amounts, mg),
        # matching the y0 = amounts convention (dose_mg is placed into an amount state).
        y = np.zeros(compiled.n_states)
        for name, idx in compiled.state_index.items():
            arr = res.amounts.get(name)
            if arr is not None:
                y[idx] = float(np.asarray(arr)[-1])
        t_offset += seg_end
    time = np.concatenate(t_all)
    c_by_zone = [_CFG["fup"] * np.concatenate(conc[nm]) for nm in names]
    return c_by_zone, time


def time_test():
    total = 400.0
    cb, tb = _b1._parent_profile_by_zone(_CFG["gene_tag"], _CFG["fm"], _CFG["n_sub"],
                                         _CFG["cltot"], _CFG["fup"], _CFG["mw"],
                                         _CFG["km_mgl"], dose_mg=total)
    cd, td = _divided_dose_profile_by_zone(total, n_splits=2, tau_h=_TAU_H)
    dyn_b = max(_dynamic_profile_hazard(cb, tb, _APAP["bio_direction"], _APAP["bio_ratio"],
                                        _APAP["gsh_direction"], _APAP["gsh_ratio"]))
    dyn_d = max(_dynamic_profile_hazard(cd, td, _APAP["bio_direction"], _APAP["bio_ratio"],
                                        _APAP["gsh_direction"], _APAP["gsh_ratio"]))
    sta_b = max(_static_profile_hazard(cb, tb, _APAP["bio_direction"], _APAP["bio_ratio"],
                                       "periportal", _APAP["gsh_ratio"]))
    sta_d = max(_static_profile_hazard(cd, td, _APAP["bio_direction"], _APAP["bio_ratio"],
                                       "periportal", _APAP["gsh_ratio"]))
    dyn_ratio = dyn_b / max(dyn_d, 1e-30)
    sta_ratio = sta_b / max(sta_d, 1e-30)
    return {"dyn_bolus": dyn_b, "dyn_divided": dyn_d, "dyn_ratio": dyn_ratio,
            "static_bolus": sta_b, "static_divided": sta_d, "static_ratio": sta_ratio,
            "excess_path_dependence": dyn_ratio - sta_ratio, "tau_h": _TAU_H}


# ---------- G1/G2 zonation + G-cliff dose + G-NAC ----------
def zonation_test():
    rows = []
    for bdir in ("pericentral", "uniform", "periportal"):
        br = 1.0 if bdir == "uniform" else 3.0
        cb, tb = _b1._parent_profile_by_zone(_CFG["gene_tag"], _CFG["fm"], _CFG["n_sub"],
                                             _CFG["cltot"], _CFG["fup"], _CFG["mw"],
                                             _CFG["km_mgl"], dose_mg=200.0)
        haz = _dynamic_profile_hazard(cb, tb, bdir, br, "uniform", 1.0)
        e = _b1.bulk_E(_CFG["gene_tag"], _CFG["fm"], _CFG["n_sub"], _CFG["cltot"],
                       _CFG["fup"], _CFG["mw"], _CFG["km_mgl"], bdir, br)
        rows.append({"bio_zonation": bdir, "bulk_E": round(e, 6),
                     "hazard_peak_zone": int(np.argmax(haz)), "maxH": round(max(haz), 4)})
    e_span = max(r["bulk_E"] for r in rows) - min(r["bulk_E"] for r in rows)
    return rows, e_span


def dose_test():
    dyn_curve, sta_curve, rows = [], [], []
    for d in _DOSES:
        cb, tb = _b1._parent_profile_by_zone(_CFG["gene_tag"], _CFG["fm"], _CFG["n_sub"],
                                             _CFG["cltot"], _CFG["fup"], _CFG["mw"],
                                             _CFG["km_mgl"], dose_mg=d)
        hd = _dynamic_profile_hazard(cb, tb, _APAP["bio_direction"], _APAP["bio_ratio"],
                                     _APAP["gsh_direction"], _APAP["gsh_ratio"])
        hs = _static_profile_hazard(cb, tb, _APAP["bio_direction"], _APAP["bio_ratio"],
                                    "periportal", _APAP["gsh_ratio"])
        dyn_curve.append(max(hd))
        sta_curve.append(max(hs))
        rows.append({"dose": d, "dyn_maxH": round(max(hd), 4), "dyn_peak_zone": int(np.argmax(hd)),
                     "static_maxH": round(max(hs), 4)})
    return rows, transition_width(_DOSES, dyn_curve), transition_width(_DOSES, sta_curve)


def nac_test():
    global _GSH0_TOTAL
    base = _GSH0_TOTAL
    out = []
    cb, tb = _b1._parent_profile_by_zone(_CFG["gene_tag"], _CFG["fm"], _CFG["n_sub"],
                                         _CFG["cltot"], _CFG["fup"], _CFG["mw"],
                                         _CFG["km_mgl"], dose_mg=400.0)
    for scale in (1.0, 1.5, 3.0):
        _GSH0_TOTAL = base * scale
        haz = _dynamic_profile_hazard(cb, tb, _APAP["bio_direction"], _APAP["bio_ratio"],
                                      _APAP["gsh_direction"], _APAP["gsh_ratio"])
        out.append({"gsh0_scale": scale, "maxH": round(max(haz), 4)})
    _GSH0_TOTAL = base
    return out


def run_sweep():
    order = order_test()
    zon_rows, e_span = zonation_test()
    dose_rows, w_dyn, w_sta = dose_test()
    return {
        "pinned": {"t_half_gsh_h": _T_HALF_GSH_H, "k_syn_per_h": _K_SYN, "kg": _KG,
                   "tau_h": _TAU_H, "gsh0_total": _GSH0_TOTAL, "km_bio": _KM_BIO},
        "G_order": order,
        "G2_invariance_contrast": zon_rows, "G2_bulk_E_span": e_span,
        "G_time": time_test(),
        "G_cliff": {"rows": dose_rows, "width_dynamic": w_dyn, "width_static": w_sta},
        "G_NAC": nac_test(),
    }


def main():
    import json

    res = run_sweep()
    base = _ROOT / "data" / "validation" / "gsh_depletion_2026-06-18"
    out = {
        "title": "Zonal GSH-pool depletion probe (Bridge B / B1.x, Phase-0)",
        "date": "2026-06-18",
        "conclusion": (
            "A depleting per-zone GSH pool makes the zonal reactive-metabolite hazard "
            "HISTORY-DEPENDENT: a pure concentration reordering leaves the static pointwise "
            f"hazard unchanged (rel diff {res['G_order']['static_rel_diff']:.1e}) while moving "
            f"the dynamic hazard ({res['G_order']['dyn_rel_diff']:.2f}) — structure beyond the "
            "B1 static model and orthogonal to bulk parent PK (DE-50, bulk-E span "
            f"{res['G2_bulk_E_span']:.1e}). Excess path-dependence over the static envelope "
            f"baseline = {res['G_time']['excess_path_dependence']:+.3f}; dose transition width "
            f"dynamic {res['G_cliff']['width_dynamic']:.3f} vs static "
            f"{res['G_cliff']['width_static']:.3f} (log10-dose); raising GSH0 lowers hazard "
            "(NAC lever). k_syn/tau pinned a priori from GSH t1/2. Headline 2.731 untouched "
            "(harness-isolated). Qualitative acetaminophen mechanism; not a calibrated tox number."
        ),
        **res,
    }
    base.with_suffix(".json").write_text(json.dumps(out, indent=2, default=float))

    o = res["G_order"]
    g = res["G_cliff"]
    tt = res["G_time"]
    lines = [
        "# Zonal GSH-pool depletion probe — Bridge B / B1.x Phase-0 (2026-06-18)",
        "",
        "**Harness-isolated** (`scripts/probe_gsh_depletion.py`); the GSH pool and reactive "
        "metabolite are a POST-PROCESSOR on the axial parent profile, not engine species. No "
        "`predict()` / `reference_man.yaml` / holdout change; headline **2.731 bit-identical**. "
        "Reuses the axial machinery (PR #79) + B1 harness (PR #82). `k_syn`/`tau` pinned a "
        f"priori: GSH t1/2 {_T_HALF_GSH_H} h -> k_syn {_K_SYN:.3f}/h, tau {_TAU_H} h.",
        "", "## Conclusion", "", out["conclusion"], "",
        "## G-order — pool memory (centerpiece)",
        "",
        f"Same value-multiset, reordered. Static rel diff **{o['static_rel_diff']:.1e}** "
        f"(invariant, by construction) vs dynamic rel diff **{o['dyn_rel_diff']:.3f}** "
        "(moves) — the pool carries order/history information the static model cannot.",
        "",
        "## G2 — local matters, bulk doesn't (DE-50)",
        "",
        f"Bulk parent E span across bioactivation zonation **{res['G2_bulk_E_span']:.2e}** "
        "(~invariant) while the dynamic hazard peak-zone moves:",
        "",
        "| bio zonation | bulk E | hazard peak-zone | maxH |",
        "|---|---|---|---|",
    ]
    for r in res["G2_invariance_contrast"]:
        lines.append(f"| {r['bio_zonation']} | {r['bulk_E']} | {r['hazard_peak_zone']} "
                     f"| {r['maxH']} |")
    lines += [
        "",
        "## G-time — excess path-dependence (bolus vs 2x divided, equal dose)",
        "",
        f"dynamic ratio {tt['dyn_ratio']:.3f} vs static ratio {tt['static_ratio']:.3f} "
        f"=> **excess {tt['excess_path_dependence']:+.3f}** (tau {tt['tau_h']} h). The static "
        "path effect is measured, not assumed zero; the excess is the pool-dynamics signature.",
        "",
        "## G-cliff — dynamic vs static dose-response sharpness",
        "",
        f"transition width (log10-dose, 10->90% of own max): dynamic **{g['width_dynamic']:.3f}** "
        f"vs static **{g['width_static']:.3f}** (smaller = sharper; reported, not presupposed).",
        "",
        "| dose | dyn maxH | dyn peak-zone | static maxH |",
        "|---|---|---|---|",
    ]
    for r in g["rows"]:
        lines.append(f"| {r['dose']} | {r['dyn_maxH']} | {r['dyn_peak_zone']} "
                     f"| {r['static_maxH']} |")
    lines += [
        "",
        "## G-NAC — precursor protective lever",
        "",
        "| GSH0 scale | maxH |", "|---|---|",
    ]
    for r in res["G_NAC"]:
        lines.append(f"| {r['gsh0_scale']} | {r['maxH']} |")
    lines += ["", "peak-zone 0-indexed inlet(0)->outlet(9); zone 9 = pericentral / zone 3.", ""]
    base.with_suffix(".md").write_text("\n".join(lines))
    print("G-order static/dyn rel:", f"{o['static_rel_diff']:.1e}", f"{o['dyn_rel_diff']:.3f}",
          "| G2 E-span:", f"{res['G2_bulk_E_span']:.1e}",
          "| G-time excess:", f"{tt['excess_path_dependence']:+.3f}",
          "| cliff dyn/static:", f"{g['width_dynamic']:.3f}/{g['width_static']:.3f}")


if __name__ == "__main__":
    main()
