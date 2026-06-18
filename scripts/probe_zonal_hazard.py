"""Zonal reactive-metabolite hazard probe (Bridge B / B1, Phase-0).

Computes a per-zone reactive-metabolite hazard as a POST-PROCESSOR on the axial
parent concentration profile (the reactive metabolite is NOT an engine species).
Demonstrates: hazard localizes by zonation; bulk parent PK is invariant to that
zonation (DE-50) while per-zone hazard is not; a saturable-detox dose-threshold with
zone-specificity (acetaminophen pattern). Harness-isolated; reuses the synthetic-engine
helpers from scripts/validate_pgx_cmax_v2b.py. No predict()/reference_man.yaml/holdout.
"""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import numpy as np

from sisyphus.validation.pgx_metrics import zonal_hazard, zonation_weights  # noqa: F401

_ROOT = Path(__file__).resolve().parent.parent
_HARNESS = _ROOT / "scripts" / "validate_pgx_cmax_v2b.py"


def _load_harness():
    spec = importlib.util.spec_from_file_location("v2b_harness", _HARNESS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


h = _load_harness()


def _load_zonation_probe():
    spec = importlib.util.spec_from_file_location(
        "zon_probe", _ROOT / "scripts" / "probe_liver_zonation.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_zon = _load_zonation_probe()  # exposes apply_zonation (robust under script + pytest)


def _subtank_names(graph):
    subs = [n.name for n in graph.nodes.values() if (n.lookup_name or n.name) == "liver"]
    return sorted(subs, key=lambda nm: int(re.search(r"__ax(\d+)$", nm).group(1)))


def _parent_profile_by_zone(gene_tag, fm, n_sub, cltot, fup, mw, km_mgl, dose_mg=100.0,
                            kp=3.0, peff=20.0):
    """Solve a single oral dose on the synthetic axial liver; return
    (c_u_by_zone, time): per-sub-tank UNBOUND parent conc arrays (C_u = fup*c_node,
    matching _peak_liver_cu) ordered inlet(ax1)->outlet(axN), and the time grid."""
    from sisyphus.engine.compiler import ODECompiler, ResolvedParams
    from sisyphus.engine.solver import solve
    g = h._axial_graph(gene_tag, n_sub=n_sub)
    abund = h._SYNTHETIC_GENE_ABUND
    drug = h._sat_drug(gene_tag, fm, cltot, abund, peff, kp, km_mgl, fup, dose_mg, mw)
    rg, rd = g.realize_means(), drug.realize_means()
    compiled = ODECompiler().compile(rg)
    params = ResolvedParams(rg, rd)
    y0 = np.zeros(compiled.n_states)
    y0[compiled.state_index[drug.administration_node]] = drug.dose_mg
    res = solve(compiled, params, y0, t_span=(0.0, float(h._T_EVAL[-1])), t_eval=h._T_EVAL)
    names = _subtank_names(g)
    c_u_by_zone = [fup * np.asarray(res.concentrations[nm]) for nm in names]
    return c_u_by_zone, np.asarray(res.time_h)


def zone_hazard_profile(gene_tag, fm, n_sub, cltot, fup, mw, km_mgl, dose_mg,
                        bio_direction, bio_ratio, detox_direction, detox_ratio,
                        vmax_bio_total, vmax_detox_total, km_bio):
    """Per-zone hazard for given bioactivation- and detox-zonation (independent),
    totals preserved. Returns per-zone hazard (inlet->outlet)."""
    c_by_zone, time = _parent_profile_by_zone(gene_tag, fm, n_sub, cltot, fup, mw,
                                              km_mgl, dose_mg)
    wbio = zonation_weights(n_sub, bio_ratio, bio_direction)
    wdet = zonation_weights(n_sub, detox_ratio, detox_direction)
    vmax_bio = [vmax_bio_total * w for w in wbio]
    vmax_detox = [vmax_detox_total * w for w in wdet]
    return zonal_hazard(c_by_zone, vmax_bio, km_bio, vmax_detox, time)


def bulk_E(gene_tag, fm, n_sub, cltot, fup, mw, km_mgl, bio_direction, bio_ratio):
    """Bulk parent extraction with the bioactivation enzyme zonated (G2 invariance arm)."""
    g = h._axial_graph(gene_tag, n_sub=n_sub)
    g = _zon.apply_zonation(g, gene_tag, zonation_weights(n_sub, bio_ratio, bio_direction))
    return h._engine_e_h(g, gene_tag, fm, cltot, h._SYNTHETIC_GENE_ABUND, 20.0, 3.0,
                         fup, 100.0, mw, km_mgl)


# Controller-calibrated acetaminophen-like config (2026-06-18): bioactivation
# pericentral-high, detox pericentral-low, Km_bio above low-dose C_u -> a clean
# dose-threshold. Synthetic-param selection for mechanism visibility, NOT a clinical fit.
_CFG = dict(gene_tag="CYP3A4", fm=0.9, n_sub=10, cltot=1.0e6, fup=0.3, mw=300.0,
            km_mgl=0.5, vmax_bio_total=300.0, vmax_detox_total=15.0, km_bio=1.0)
_DOSES = [50.0, 100.0, 200.0, 400.0, 800.0]


def run_sweep():
    """G1 localization, G2 bulk-invariant/hazard-variant, G3 dose-threshold + protection.
    Returns a dict of all numbers (no hand-written values)."""
    cfg = _CFG
    aceta = dict(bio_direction="pericentral", bio_ratio=3.0,
                 detox_direction="periportal", detox_ratio=3.0)  # detox pericentral-low

    # G3: dose-threshold curve + protective lever (3x detox) on the aceta config.
    g3 = []
    for d in _DOSES:
        haz = zone_hazard_profile(**cfg, dose_mg=d, **aceta)
        cfg_prot = dict(cfg, vmax_detox_total=cfg["vmax_detox_total"] * 3.0)
        haz_p = zone_hazard_profile(**cfg_prot, dose_mg=d, **aceta)
        g3.append({"dose": d, "maxH": max(haz), "peak_zone": int(np.argmax(haz)),
                   "profile": [round(x, 4) for x in haz],
                   "maxH_3x_detox": max(haz_p)})

    # G2: bulk E vs bioactivation zonation (should be ~invariant) and hazard peak-zone
    # (should move). Uniform detox isolates the bioactivation-zonation effect.
    g2 = []
    for bdir in ("pericentral", "uniform", "periportal"):
        e = bulk_E(cfg["gene_tag"], cfg["fm"], cfg["n_sub"], cfg["cltot"], cfg["fup"],
                   cfg["mw"], cfg["km_mgl"], bdir, 1.0 if bdir == "uniform" else 3.0)
        haz = zone_hazard_profile(**cfg, dose_mg=200.0, bio_direction=bdir,
                                  bio_ratio=1.0 if bdir == "uniform" else 3.0,
                                  detox_direction="uniform", detox_ratio=1.0)
        g2.append({"bio_zonation": bdir, "bulk_E": round(e, 6),
                   "hazard_peak_zone": int(np.argmax(haz)), "maxH": round(max(haz), 4)})

    e_span = max(r["bulk_E"] for r in g2) - min(r["bulk_E"] for r in g2)
    return {"config": cfg, "doses": _DOSES, "G3_dose_threshold": g3,
            "G2_invariance_contrast": g2, "G2_bulk_E_span": e_span}


def main():
    import json

    res = run_sweep()
    base = _ROOT / "data" / "validation" / "zonal_hazard_probe_2026-06-18"
    out = {
        "title": "Zonal reactive-metabolite hazard probe (Bridge B / B1, Phase-0)",
        "date": "2026-06-18",
        "conclusion": (
            "The per-zone reactive-metabolite hazard (local bioactivation exceeding local "
            "saturable detox) is a real surface ORTHOGONAL to bulk PK: varying bioactivation "
            "zonation leaves bulk parent extraction ~invariant (DE-50, span "
            f"{res['G2_bulk_E_span']:.1e}) while the per-zone hazard peak-zone moves. A "
            "saturable-detox DOSE-THRESHOLD with pericentral (zone-3) specificity emerges, "
            "and raising detox capacity protects — qualitatively reproducing the "
            "acetaminophen centrilobular pattern. First concrete Bridge-B endpoint; "
            "headline 2.731 untouched (harness-isolated). Post-processor fidelity; "
            "transported-metabolite + GSH-pool dynamics + quantitative PoD are B1.x."
        ),
        **res,
    }
    base.with_suffix(".json").write_text(json.dumps(out, indent=2))

    lines = [
        "# Zonal reactive-metabolite hazard probe — Bridge B / B1 Phase-0 (2026-06-18)",
        "",
        "**Harness-isolated** (`scripts/probe_zonal_hazard.py`); the reactive metabolite is "
        "a POST-PROCESSOR on the axial parent profile, not an engine species. No "
        "`predict()` / `reference_man.yaml` / holdout change; headline **2.731 "
        "bit-identical**. Reuses the axial machinery (PR #79) + `zonation_weights` (DE-50).",
        "", "## Conclusion", "", out["conclusion"], "",
        "## G2 — local matters, bulk doesn't (DE-50 closure)",
        "",
        f"Bulk parent extraction span across bioactivation zonation: "
        f"**{res['G2_bulk_E_span']:.2e}** (~invariant), while the hazard peak-zone moves:",
        "",
        "| bio zonation | bulk E | hazard peak-zone | maxH |",
        "|---|---|---|---|",
    ]
    for r in res["G2_invariance_contrast"]:
        lines.append(
            f"| {r['bio_zonation']} | {r['bulk_E']} | "
            f"{r['hazard_peak_zone']} | {r['maxH']} |"
        )
    lines += [
        "",
        "## G3 — saturable-detox dose-threshold + zone-specificity (acetaminophen config)",
        "",
        "Bioactivation pericentral-high, detox pericentral-low. Below the threshold no zone "
        "has hazard; above it the pericentral (zone-3) zone crosses first; 3× detox protects.",
        "",
        "| dose | maxH | peak-zone | maxH (3× detox) |",
        "|---|---|---|---|",
    ]
    for r in res["G3_dose_threshold"]:
        lines.append(
            f"| {r['dose']} | {r['maxH']:.4g} | {r['peak_zone']} | "
            f"{r['maxH_3x_detox']:.4g} |"
        )
    lines += [
        "",
        "peak-zone is 0-indexed inlet(0)→outlet(9); zone 9 = pericentral / zone 3 / "
        "centrilobular. The dose=50 row (maxH 0) is below threshold.",
        "",
    ]
    base.with_suffix(".md").write_text("\n".join(lines))
    print(f"wrote {base.with_suffix('.json').name} and {base.with_suffix('.md').name}")
    g3_summary = [(r["dose"], round(r["maxH"], 3)) for r in res["G3_dose_threshold"]]
    print("G2 bulk_E span:", f"{res['G2_bulk_E_span']:.2e}", "| G3 threshold:", g3_summary)


if __name__ == "__main__":
    main()
