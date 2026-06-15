"""PGx genotype-fold validation harness (engine regression + report).

Run from the repo root:  python scripts/validate_pgx_genotype_folds.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import sisyphus.engine.flux  # noqa: F401  -- register flux specs
from sisyphus.core import Distribution, DrugOnGraph
from sisyphus.engine.compiler import ODECompiler, ResolvedParams
from sisyphus.engine.solver import solve
from sisyphus.graph.builder import build_from_yaml
from sisyphus.predict.phenotype import apply_phenotype_to_graph

_YAML = Path("data/physiology/reference_man.yaml")
_RESID = "RESIDUAL_HEPATIC"

# Total synthetic hepatic CLint (summed abundance*affinity, pre-IVIVE) split
# fm/(1-fm) between the gene and RESIDUAL_HEPATIC. Chosen in the LOW-EXTRACTION
# (linear) regime so the oral-AUC identity ``AUC = Dose / CLint`` — and hence
# the genotype fold ``1/(1-fm+fm*a)`` — holds exactly. At higher CLint the
# hepatic first-pass extraction-regime distortion (E -> 1) breaks the closed
# form (oral AUC stops being linear in 1/CLint); at lower CLint the slow PM
# terminal phase dominates and AUC truncation matters. _CLTOT=2e3 with the
# AUC_0inf extrapolation below holds the worst-case PM fold to <0.6% across
# CYP2D6/CYP2C19/CYP2C9 x fm in {0.7, 0.9}. See Task-2 calibration notes.
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
