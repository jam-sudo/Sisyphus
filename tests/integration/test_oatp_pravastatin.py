"""Integration tests for OATP1B1 hepatic uptake (post-#12 reconciliation).

The pre-#12 invariant ``cmax_on/cmax_off < 0.95`` is mathematically
incompatible with the post-reconciliation model: with the metabolic
path zeroed for pravastatin (registry entry, see issue #12), the
"off" arm (no transporter_kinetics, no ECM) has *no* hepatic
clearance for pravastatin and Cmax goes very high; the "on" arm has
ECM-only clearance. The two cannot be ordered in a way that
preserves the original test's intent.

Replacement (#13 B1): verify the phenotype path. Under SLCO1B1 PM
(OATP1B1 activity x 0.10), pravastatin systemic Cmax must be HIGHER
than under SLCO1B1 EM (x 1.00), because reduced hepatic uptake leaves
more drug in venous blood. This is the canonical clinical observation
that motivated SLCO1B1 PGx in the first place.

Depends on:
- data/physiology/reference_man.yaml having liver.transporters.OATP1B1
- data/transporters/oatp1b1.json with pravastatin entry
- data/transporters/cyp_clearance_overrides.json with pravastatin entry
- predict/ivive.py supporting transporter_kinetics + metabolic_fraction
- predict/phenotype.py for SLCO1B1 -> OATP1B1 abundance scaling

Reference choice (issue #8 closure 2026-05-03):
    Pravastatin has two commonly cited Cmax references at 40 mg single oral:
        FDA Pravachol label: 0.045 mg/L (large regulatory cohort)
        Niemi 2006 EM N=6 : 0.075 mg/L (small SLCO1B1 521TT cohort)
    These are 1.67x apart and cannot both be satisfied by a single calibration.
    Sisyphus is anchored on FDA via scripts/calibrate_oatp_abundance_ecm.py
    (issue #14 confirms 5.0e5 OATP1B1 abundance optimum). test_oatp_ecm_statins
    [pravastatin] passes at FE 1.066 vs FDA. Niemi 2006 EM/PM RATIO (~2.6x)
    is reproduced directionally by this test; absolute population magnitudes
    are not anchored against Niemi means. GenoADME Tier 1 should anchor on
    FDA Cmax 0.045 mg/L for population mean and retain Niemi as a ratio check.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

import sisyphus.engine.flux  # noqa: F401 -- register flux specs
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


def _simulate_cmax(drug, graph, t_end: float = 24.0) -> float:
    """Run a deterministic ODE solve and return venous_blood Cmax.

    Uses ``realize_means()`` (post-Hardening canonical deterministic path)
    to be consistent with ``test_oatp_ecm_statins`` and the production
    ``predict()`` default.
    """
    realized_graph = graph.realize_means()
    realized_drug = drug.realize_means()

    compiler = ODECompiler()
    compiled = compiler.compile(realized_graph)
    params = ResolvedParams(realized_graph, realized_drug)

    y0 = np.zeros(compiled.n_states)
    admin_idx = compiled.state_index[drug.administration_node]
    y0[admin_idx] = drug.dose_mg

    result = solve(compiled, params, y0, t_span=(0, t_end))

    if not result.solver_success:
        pytest.fail("solver did not converge")

    return float(np.max(result.concentrations["venous_blood"]))


def _build_pravastatin(graph):
    profile = compute_profile(_PRAVASTATIN)
    adme = predict_adme(profile)
    liver_enzymes = {
        tag: dist.mean for tag, dist in graph.nodes["liver"].enzymes.items()
    }
    return build_drug_on_graph(
        profile, adme, dose_mg=40.0, route="oral",
        liver_enzymes=liver_enzymes,
        transporter_kinetics=load_oatp1b1_kinetics("pravastatin"),
        hepatic_ecm_params=load_hepatic_ecm_params("pravastatin"),
    )


@pytest.mark.slow
def test_pravastatin_pm_higher_cmax_than_em():
    """SLCO1B1 PM reduces hepatic OATP1B1 uptake -> higher systemic Cmax.

    Replaces the pre-#12 cmax_on/cmax_off invariant. Verifies that the
    PGx integration path is wired end-to-end through the engine: the
    apply_phenotype_to_graph hook scales the OATP1B1 abundance, the
    engine's ECM machinery sees the scaled abundance via PS_active, and
    the drop in hepatic uptake lifts Cmax.
    """
    graph_em = build_from_yaml(_PHYS)
    graph_pm = apply_phenotype_to_graph(graph_em, {"SLCO1B1": "PM"})

    drug_em = _build_pravastatin(graph_em)
    drug_pm = _build_pravastatin(graph_pm)

    cmax_em = _simulate_cmax(drug_em, graph_em)
    cmax_pm = _simulate_cmax(drug_pm, graph_pm)

    print(
        f"\npravastatin SLCO1B1: cmax_em={cmax_em:.4f}, cmax_pm={cmax_pm:.4f}, "
        f"ratio={cmax_pm / cmax_em:.3f}"
    )

    assert cmax_pm > cmax_em, (
        "expected SLCO1B1 PM to raise pravastatin Cmax (less hepatic uptake), "
        f"got cmax_em={cmax_em:.4f}, cmax_pm={cmax_pm:.4f}"
    )
    # Lower bound: PM must produce a meaningful uplift, not just numerical
    # noise. CPIC PM activity = 0.10x EM, so we expect at least ~10%
    # higher Cmax. Tighter clinical observation is ~2-3x but depends on
    # absolute abundance calibration (#14).
    assert cmax_pm / cmax_em > 1.10, (
        "expected at least 10% Cmax increase under PM, "
        f"got ratio {cmax_pm / cmax_em:.3f}"
    )


@pytest.mark.slow
def test_non_oatp_drug_unaffected_by_oatp_wiring():
    """Morphine (no OATP1B1 substrate) -> transporter_kinetics empty -> MM inactive."""
    graph = build_from_yaml(_PHYS)
    morphine_smiles = "CN1CCC23C4C1CC5=C2C(=C(C=C5)O)OC3C(C=C4)O"
    profile = compute_profile(morphine_smiles)
    adme = predict_adme(profile)
    liver_enzymes = {
        tag: dist.mean for tag, dist in graph.nodes["liver"].enzymes.items()
    }
    kinetics = load_oatp1b1_kinetics("morphine")
    drug = build_drug_on_graph(
        profile, adme, dose_mg=30.0, route="oral",
        liver_enzymes=liver_enzymes,
        transporter_kinetics=kinetics,
    )
    assert drug.transporter_kinetics == {}, "morphine should not have OATP kinetics"

    realized_graph = graph.realize_means()
    realized_drug = drug.realize_means()

    compiler = ODECompiler()
    compiled = compiler.compile(realized_graph)
    params = ResolvedParams(realized_graph, realized_drug)

    y0 = np.zeros(compiled.n_states)
    admin_idx = compiled.state_index[drug.administration_node]
    y0[admin_idx] = drug.dose_mg

    result = solve(compiled, params, y0, t_span=(0, 24.0))
    assert result.solver_success
    assert np.max(result.concentrations["venous_blood"]) > 0
