"""PGx genotype-fold validation harness (engine regression + report).

Run from the repo root:  python scripts/validate_pgx_genotype_folds.py
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np

import sisyphus.engine.flux  # noqa: F401  -- register flux specs
from sisyphus.core import Distribution, DrugOnGraph
from sisyphus.engine.compiler import ODECompiler, ResolvedParams
from sisyphus.engine.solver import solve
from sisyphus.graph.builder import build_from_yaml
from sisyphus.predict.phenotype import apply_phenotype_to_graph
from sisyphus.validation.pgx_metrics import (
    a_emp,
    analytical_fold,
    fm_agreement,
    fm_invivo,
    fm_invivo_ci,
)

_YAML = Path("data/physiology/reference_man.yaml")
_RESID = "RESIDUAL_HEPATIC"

# Total synthetic hepatic CLint (summed abundance*affinity, pre-IVIVE) split
# fm/(1-fm) between the gene and RESIDUAL_HEPATIC. Kept in the LOW-EXTRACTION
# (linear) regime so the oral-AUC identity ``AUC ∝ Dose / CLint`` — and hence
# the genotype fold ``1/(1-fm+fm*a)`` — holds exactly.
#
# Why _CLTOT=2000 (not lower, not higher)?
#   HIGH end: very high CLint pushes hepatic extraction E → 1 (well-stirred
#   saturated regime) where oral AUC ∝ 1/CLint over-shoots — the closed-form
#   additivity breaks and the fold diverges from the analytic oracle.
#   LOW end: lower CLint produces a slower PM-arm terminal phase whose tail is
#   NOT fully captured within any finite integration window. The old ``_CLTOT=50``
#   path appeared to give fold ≈ 1.003 for that reason — the AUC_0t (trapezoidal)
#   under-counted PM exposure because the slow terminal phase extended beyond
#   the window, not because ivive ≈ 6e-5 made clearance negligible (that
#   narrative was INCORRECT and is retired). With the AUC_0inf extrapolation
#   (AUC_0t + C_last/k_terminal) the truncation artifact is removed, and
#   _CLTOT=2000 holds the worst-case PM fold error to <0.6% across
#   CYP2D6/CYP2C19/CYP2C9 × fm ∈ {0.7, 0.9}. See Task-2 calibration notes.
_CLTOT = 2000.0

# Integration horizon. With AUC_0inf extrapolation the fold is window-robust
# (varies <0.3% between 10000 h and 20000 h); 10000 h is ample for the slowest
# (fm=0.9 PM) terminal phase to be well-sampled for the terminal-slope fit.
_T_END_H = 10000.0

# Terminal-slope fit window: last fraction of time points used for the
# log-linear regression that yields k_terminal for the AUC_0inf tail.
_TERMINAL_FRACTION = 0.3

# Default synthetic abundance for an in-scope gene that is ABSENT from the
# liver enzymes of reference_man.yaml. CYP2C19 (an in-scope "Big 3" gene per
# the spec) is not present in the physiology YAML's liver node. The oracle is
# a correctness test, not a physiological claim: the gene abundance cancels
# analytically (a_gene = fm*CLTOT/abund_gene, multiplied back by abund_gene
# inside the clearance flux), so any positive value reproduces the same fold.
# Injecting it mirrors the RESIDUAL_HEPATIC injection. See Task-2 finding.
_SYNTHETIC_GENE_ABUND = 1.0e6


def _synthetic_drug(gene_tag: str, fm: float, abund_gene: float) -> DrugOnGraph:
    a_gene = fm * _CLTOT / abund_gene
    a_resid = (1.0 - fm) * _CLTOT
    return DrugOnGraph(
        name=f"synthetic_{gene_tag}_fm{fm}",
        smiles="C", dose_mg=100.0, route="oral",
        administration_node="stomach_lumen",
        mw=300.0, pka=None, compound_type="neutral",
        fup=Distribution(0.1, 0.0), rbp=Distribution(1.0, 0.0),
        kp_method="rodgers_rowland", kp_overrides={},
        peff=Distribution(5.0, 0.0), solubility=Distribution(1000.0, 0.0),
        enzyme_affinity={
            gene_tag: Distribution(a_gene, 0.0),
            _RESID: Distribution(a_resid, 0.0),
        },
        renal_clearance=Distribution(0.0, 0.0),
    )


def _auc_0inf(graph, drug: DrugOnGraph) -> float:
    """Engine oral AUC_0-inf at the venous-blood observation node.

    AUC_0t (trapezoidal) plus a terminal-slope extrapolation
    ``C_last / k_terminal``. The extrapolation removes the AUC truncation
    artifact that would otherwise bias the slow (low-CLint PM) arm and corrupt
    the genotype fold — see the _CLTOT note. Physics (the ODE) is untouched;
    this is post-hoc NCA on the engine's own concentration-time curve.

    Load-bearing assumption: hepatic clearance uses the ``extended`` (ECM)
    model, which is linear here only because the synthetic drug has
    ``ps_passive = ps_eff = 1e6 ≫ cl_int_h``, so the ECM uptake term
    ``fup·ps·cl_int / (ps + cl_int) → fup·cl_int`` and flux additivity
    (hence the closed-form AUC identity) is recovered.
    """
    rg, rd = graph.realize_means(), drug.realize_means()
    compiled = ODECompiler().compile(rg)
    params = ResolvedParams(rg, rd)
    y0 = np.zeros(compiled.n_states)
    y0[compiled.state_index[drug.administration_node]] = drug.dose_mg
    res = solve(compiled, params, y0, t_span=(0.0, _T_END_H))
    conc = res.concentrations["venous_blood"]
    time = res.time_h
    trapz = getattr(np, "trapezoid", np.trapz)  # numpy 2.0+ vs 1.x
    auc_0t = float(trapz(conc, time))
    # Terminal log-linear slope from the last _TERMINAL_FRACTION of samples.
    i0 = int(len(time) * (1.0 - _TERMINAL_FRACTION))
    t_tail, c_tail = time[i0:], conc[i0:]
    pos = c_tail > 0
    if pos.sum() >= 2:
        k_terminal = -np.polyfit(t_tail[pos], np.log(c_tail[pos]), 1)[0]
        if k_terminal > 0:
            return auc_0t + conc[-1] / k_terminal
    return auc_0t


def engine_auc_fold(gene_tag: str, fm: float, activity_variant: float) -> float:
    base = build_from_yaml(_YAML)
    liver = base.nodes["liver"]
    if gene_tag not in liver.enzymes:
        # In-scope gene absent from physiology YAML (CYP2C19). Inject a
        # synthetic abundance so the identity-blind clearance flux can route
        # fm through it; abundance cancels in the fold (see _SYNTHETIC_GENE_ABUND).
        liver.enzymes[gene_tag] = Distribution(_SYNTHETIC_GENE_ABUND, 0.0)
    liver.enzymes[_RESID] = Distribution(1.0, 0.0)
    drug = _synthetic_drug(gene_tag, fm, abund_gene=liver.enzymes[gene_tag].mean)
    auc_em = _auc_0inf(base, drug)
    variant = apply_phenotype_to_graph(
        base, {gene_tag: "PM"}, phenotype_scale_overrides={gene_tag: activity_variant}
    )
    auc_var = _auc_0inf(variant, drug)
    return auc_var / auc_em


_BENCH = Path("data/validation/pgx_genotype_folds.json")
_ACTIVITY = {"PM": 0.0, "IM": 0.5, "NM": 1.0, "EM": 1.0, "UM": 2.0}


def _evaluate(pairs: list[dict]) -> list[dict]:
    rows = []
    for p in pairs:
        a = _ACTIVITY[p["phenotype"]]
        fm = p["fm_invitro"]
        fold = p["obs_auc_fold_pm"]
        row = {
            **{k: p[k] for k in ("drug", "gene", "phenotype", "quantitative", "flags")},
            "fm_invitro": fm,
            "obs_fold": fold,
            "fm_invivo": fm_invivo(fold) if p["phenotype"] == "PM" else None,
            "fm_invivo_ci": (
                list(fm_invivo_ci(p["obs_auc_fold_ci"])) if p["phenotype"] == "PM" else None
            ),
            "analytical_fold": analytical_fold(fm=fm, activity=a),
            "a_emp": a_emp(fold, fm) if (p["phenotype"] != "PM" and fm >= 0.6) else None,
            "engine_fold": engine_auc_fold(p["gene"], fm, a),
        }
        row["engine_vs_analytical_rel"] = (
            abs(row["engine_fold"] - row["analytical_fold"]) / row["analytical_fold"]
        )
        rows.append(row)
    return rows


def main() -> None:
    pairs = json.loads(_BENCH.read_text())["pairs"]
    rows = _evaluate(pairs)

    pm = [r for r in rows if r["phenotype"] == "PM" and r["quantitative"]]
    agreement = fm_agreement(
        [r["fm_invitro"] for r in pm], [r["fm_invivo"] for r in pm], tol=0.15
    )
    primary_pass = agreement["frac_within_tol"] >= 0.70 and 0.7 <= agreement["slope"] <= 1.3

    engine_pass = all(r["engine_vs_analytical_rel"] < 0.02 for r in rows)

    registry = {
        r["drug"]: {
            "gene": r["gene"], "fm_invitro": r["fm_invitro"],
            "fm_invivo": r["fm_invivo"], "fm_invivo_ci": r["fm_invivo_ci"],
        }
        for r in rows if r["phenotype"] == "PM"
    }

    stamp = date.today().isoformat()
    out = {
        "created": stamp,
        "primary_pm_fm_agreement": agreement,
        "primary_pass": primary_pass,
        "engine_regression_pass": engine_pass,
        "secondary_im_um": "no IM/UM pairs in v1 benchmark (deferred)",
        "pairs": rows,
    }
    Path(f"data/validation/pgx_fold_validation_{stamp}.json").write_text(json.dumps(out, indent=2))
    Path("data/validation/pgx_fm_registry.json").write_text(json.dumps(registry, indent=2))

    md = [
        f"# PGx genotype-fold validation — {stamp}",
        "",
        f"- Primary (PM fm-agreement): frac_within_0.15 = {agreement['frac_within_tol']:.2f}, "
        f"slope = {agreement['slope']:.2f}, MAD = {agreement['mad']:.3f}  -> "
        f"**{'PASS' if primary_pass else 'REVIEW'}**",
        f"- Engine regression (<2% vs analytical): **{'PASS' if engine_pass else 'FAIL'}**",
        "",
        "| drug | gene | fm_invitro | obs_fold | fm_invivo | engine vs analytical |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        fv = f"{r['fm_invivo']:.3f}" if r["fm_invivo"] is not None else "-"
        md.append(
            f"| {r['drug']} | {r['gene']} | {r['fm_invitro']:.2f} | {r['obs_fold']:.1f} | "
            f"{fv} | {r['engine_vs_analytical_rel']:.2%} |"
        )
    Path(f"data/validation/pgx_fold_validation_{stamp}.md").write_text("\n".join(md) + "\n")
    print(f"primary_pass={primary_pass} engine_pass={engine_pass} "
          f"frac_within={agreement['frac_within_tol']:.2f} slope={agreement['slope']:.2f}")


if __name__ == "__main__":
    main()
