"""Integration tests for predict() ECM auto-activation gating (v0.3).

Pre-existing PR #9 wiring activates ECM for any drug in both oatp1b1.json
and hepatic_ecm.json registries. v0.3 gates that activation with the
ecm_applicable=true flag (Task 1 helper, Task 2 data flag) to prevent
the empirically-documented triple-counting bug for CYP-dominant or
otherwise-not-OATP-rate-limited drugs (fluvastatin, pitavastatin,
rosuvastatin, atorvastatin).

Reference Cmax values (40 mg/2 mg oral, realize_means(), post-Hardening):
- pravastatin auto-ECM (mf=0): 0.0422 mg/L (FDA 0.045, FE 1.066)
- pitavastatin no-ECM:         0.00777 mg/L
- fluvastatin no-ECM:          0.0583 mg/L
"""
from __future__ import annotations

import numpy as np
import pytest

from sisyphus.pipeline.predict import predict


_PRAVA_SMILES = (
    "CC[C@@H](C)C(=O)O[C@@H]1C[C@H](C=C2[C@@H]1CC[C@H]"
    "([C@@H]2CC[C@H](C[C@H](CC(=O)O)O)O)C)O"
)
_FLUVA_SMILES = (
    "CC(C)N1C2=CC=CC=C2C(=C1/C=C/[C@H](O)C[C@H](O)CC(=O)O)"
    "C3=CC=C(F)C=C3"
)
_PITA_SMILES = (
    "OC(=O)C[C@H](O)C[C@H](O)/C=C/C1=C(C2CC2)N=C3C=CC=CC3=C1"
    "C4=CC=C(F)C=C4"
)


@pytest.mark.slow
def test_pravastatin_auto_ecm_activates():
    """predict(pravastatin) auto-activates ECM and produces the FDA-anchored Cmax."""
    result = predict(_PRAVA_SMILES, dose_mg=40.0, route="oral", n_mc_samples=0)
    assert result.engine_pk is not None
    cmax = result.engine_pk.cmax.mean
    assert any("oatp1b1:auto_ecm:pravastatin" in w for w in result.warnings), (
        f"expected oatp1b1:auto_ecm:pravastatin warning, got: {result.warnings}"
    )
    expected = 0.0422
    rel_err = abs(cmax - expected) / expected
    assert rel_err < 0.05, (
        f"pravastatin auto-ECM Cmax drift: actual={cmax:.4f}, expected={expected:.4f}, "
        f"rel_err={rel_err:.3f} (5% tol). Auto-activation may be misfiring."
    )


@pytest.mark.slow
def test_fluvastatin_no_auto_ecm():
    """predict(fluvastatin) does NOT auto-activate ECM (CYP2C9-dominant per Niemi 2009)."""
    result = predict(_FLUVA_SMILES, dose_mg=40.0, route="oral", n_mc_samples=0)
    assert result.engine_pk is not None
    cmax = result.engine_pk.cmax.mean
    assert not any("oatp1b1:auto_ecm" in w for w in result.warnings), (
        f"fluvastatin should NOT auto-activate ECM, but got warnings: {result.warnings}"
    )
    expected = 0.0583
    rel_err = abs(cmax - expected) / expected
    assert rel_err < 0.05, (
        f"fluvastatin Cmax shifted unexpectedly: actual={cmax:.4f}, "
        f"expected={expected:.4f} (no-ECM path). Gate may be leaking."
    )


@pytest.mark.slow
def test_pitavastatin_no_auto_ecm():
    """predict(pitavastatin) does NOT auto-activate ECM (no metabolic_fraction entry).

    The schema regression test gates this at registry level; this test gates it
    at predict() runtime. Two layers of defense.
    """
    result = predict(_PITA_SMILES, dose_mg=2.0, route="oral", n_mc_samples=0)
    assert result.engine_pk is not None
    cmax = result.engine_pk.cmax.mean
    assert not any("oatp1b1:auto_ecm" in w for w in result.warnings), (
        f"pitavastatin should NOT auto-activate ECM, but got warnings: {result.warnings}"
    )
    expected = 0.00777
    rel_err = abs(cmax - expected) / expected
    assert rel_err < 0.05, (
        f"pitavastatin Cmax drifted: actual={cmax:.5f}, expected={expected:.5f} "
        f"(no-ECM path post-gating). Gate may be leaking."
    )


@pytest.mark.slow
def test_pravastatin_auto_ecm_mass_balance():
    """Auto-ECM adds OATP1B1 saturable + ECM passive + biliary CL_int paths.
    Verify mass balance still closes (engine invariant)."""
    import pathlib

    import sisyphus.engine.flux  # noqa: F401
    from sisyphus.engine.compiler import ODECompiler, ResolvedParams
    from sisyphus.engine.solver import solve
    from sisyphus.graph.builder import build_from_yaml
    from sisyphus.predict.adme import predict_adme
    from sisyphus.predict.chemistry import compute_profile
    from sisyphus.predict.ivive import build_drug_on_graph
    from sisyphus.predict.transporter_db import (
        load_hepatic_ecm_params_for_smiles,
        load_oatp1b1_kinetics_for_smiles,
    )

    graph = build_from_yaml(pathlib.Path("data/physiology/reference_man.yaml"))
    profile = compute_profile(_PRAVA_SMILES)
    adme = predict_adme(profile)
    liver_enz = {tag: d.mean for tag, d in graph.nodes["liver"].enzymes.items()}
    drug = build_drug_on_graph(
        profile, adme, dose_mg=40.0, route="oral",
        liver_enzymes=liver_enz,
        transporter_kinetics=load_oatp1b1_kinetics_for_smiles(_PRAVA_SMILES),
        hepatic_ecm_params=load_hepatic_ecm_params_for_smiles(_PRAVA_SMILES),
    )
    rg, rd = graph.realize_means(), drug.realize_means()
    compiler = ODECompiler()
    compiled = compiler.compile(rg)
    params = ResolvedParams(rg, rd)
    y0 = np.zeros(compiled.n_states)
    y0[compiled.state_index[drug.administration_node]] = drug.dose_mg
    result = solve(compiled, params, y0, t_span=(0, 24.0))
    assert result.solver_success
    assert result.mass_balance_error < 1e-6, (
        f"auto-ECM mass balance broken: error={result.mass_balance_error:.2e}"
    )
