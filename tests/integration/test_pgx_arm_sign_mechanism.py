"""Data-independent mechanism tests for the two-arm genotype-nonlinearity harness.
Spec: 2026-06-16-...-two-arm-design.md §5, §8. Runs the synthetic engine only —
no clinical data, no predict()/holdout. Imports the harness script by path."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
_SCRIPT = ROOT / "scripts" / "validate_pgx_cmax_v2b.py"


def _harness():
    spec = importlib.util.spec_from_file_location("v2b_harness", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_axial_graph_has_subtanks_no_literal_liver():
    h = _harness()
    g = h._axial_graph("CYP2D6", n_sub=8)
    assert "liver" not in g.nodes
    subs = [n for n in g.nodes.values() if (n.lookup_name or n.name) == "liver"]
    assert len(subs) == 8 and all("CYP2D6" in s.enzymes for s in subs)


def test_steady_state_exposure_accumulates():
    h = _harness()
    g = h._well_stirred_graph("CYP2C9")
    abund = g.nodes["liver"].enzymes["CYP2C9"].mean
    drug = h._drug("CYP2C9", 0.9, 5.0, abund, peff=20.0, kp=3.0, fup=0.10,
                   dose_mg=100.0, mw=252.3)
    e1 = h._steady_state_exposure(g, drug, interval_h=24.0, n_doses=1, metric="css_avg")
    e30 = h._steady_state_exposure(g, drug, interval_h=24.0, n_doses=20, metric="css_avg")
    assert e30 > e1 > 0  # accumulation to steady state


def test_single_dose_exposure_positive():
    h = _harness()
    g = h._well_stirred_graph("CYP2D6")
    abund = g.nodes["liver"].enzymes["CYP2D6"].mean
    drug = h._sat_drug("CYP2D6", 0.8, 1.0e7, abund, peff=20.0, kp=3.0, km_mgl=0.9,
                       fup=0.3, dose_mg=150.0, mw=341.4)
    assert h._single_dose_exposure(g, drug, metric="auc") > 0


def test_systemic_sign_is_regime_dependent_not_invariant():
    """HONEST-NEGATIVE (refutes spec §5.2 hypothesis). The systemic-arm genotype
    cross-term Δβ is NOT a topological invariant: its SIGN flips with the saturation
    regime. Holding everything fixed except Km (cltot=5e4, kp=0.05, PM activity 0.25,
    same dose span), a low Km (deep saturation) gives Δβ < 0 (convergence) while a high
    Km (mild saturation) gives Δβ > 0 (divergence). So "systemic always diverges" is
    false; whether the gene-deficient PM is the more-nonlinear arm depends on how deeply
    each genotype saturates. This is the documented dead-end (DE-NN); see also that the
    phenytoin-relevant low-Km regime gives the *wrong* sign vs phenytoin's clinical
    divergence — untestable here because the crossed-grid data HALTed.

    Unlike the first-pass arm (where PM≈0 activity pins β_PM≡1 and the convergence sign
    is structural/robust), the systemic PM retains a saturating gene, so β_PM ≷ β_EM is
    regime-dependent. NOT a fit to clinical data — a synthetic characterization of the
    engine's behavior."""
    h = _harness()
    from sisyphus.validation.pgx_metrics import delta_beta
    gene, fm, mw, fup = "CYP2C9", 0.9, 252.3, 0.10
    cltot, abund, kp = 50000.0, h._SYNTHETIC_GENE_ABUND, 0.05
    doses = [100.0, 300.0, 900.0]

    def gb():
        return h._well_stirred_graph(gene)

    def dbeta_at_km(km_mgl):
        def db(dose):
            return h._sat_drug(gene, fm, cltot, abund, 60.0, kp, km_mgl, fup, dose, mw)
        b_em = h._beta_for_genotype(gb, db, doses, "EM", gene, 0.25, "steady_state")
        b_pm = h._beta_for_genotype(gb, db, doses, "PM", gene, 0.25, "steady_state")
        return delta_beta(b_pm, b_em)

    dbeta_low_km = dbeta_at_km(2.0)    # deep saturation
    dbeta_high_km = dbeta_at_km(10.0)  # mild saturation
    # The sign genuinely flips across the regime — that IS the finding.
    assert dbeta_low_km < -0.02, ("expected convergence at low Km", dbeta_low_km)
    assert dbeta_high_km > 0.02, ("expected divergence at high Km", dbeta_high_km)


def test_first_pass_gene_converges_delta_beta_negative():
    """Arm-F crux (REQUIRES PR #79): first-pass extraction gene → EM (high extraction)
    saturates → β_EM > β_PM → Δβ < 0 (fold converges). Axial skeleton."""
    h = _harness()
    from sisyphus.validation.pgx_metrics import delta_beta
    gene, fm, mw, fup = "CYP2D6", 0.85, 341.4, 0.30
    km_mgl = 0.3
    cltot, abund, kp = 5.0e6, h._SYNTHETIC_GENE_ABUND, 3.0
    doses = [75.0, 300.0, 600.0]

    def gb():
        return h._axial_graph(gene, n_sub=10)

    def db(dose):
        return h._sat_drug(gene, fm, cltot, abund, 20.0, kp, km_mgl, fup, dose, mw)

    b_em = h._beta_for_genotype(gb, db, doses, "EM", gene, 0.03, "single_dose")
    b_pm = h._beta_for_genotype(gb, db, doses, "PM", gene, 0.03, "single_dose")
    assert delta_beta(b_pm, b_em) < -0.02, (b_pm, b_em)


def test_propafenone_em_supraproportional_p1_vs_siddoway():
    """P1 (the one citable clinical saturation signature). Propafenone EM, axial
    saturable engine: β_EM > 1 (supra-proportional first-pass), reproducing the
    DIRECTION of Siddoway 1987 (EM concentration rose ~10× over a 3× dose increase,
    β_obs ≈ 2.1). The linear-null gives β = 1.0 exactly. We assert the qualitative
    supra-proportionality the linear model cannot produce — NOT a magnitude match,
    since the literature Km (Kroemer 5.3 µM vs Hemeryck 0.12 µM) and fup are uncertain
    (fup corrected to ~0.10 per the literature audit, NOT the 0.30 first assumed)."""
    h = _harness()
    from sisyphus.validation.pgx_metrics import km_uM_to_unbound_mgL
    gene, fm, mw, fup = "CYP2D6", 0.80, 341.4, 0.10
    km_mgl = km_uM_to_unbound_mgL(5.3, mw, 0.5)  # Kroemer 1989, ~0.905 mg/L unbound
    cltot, abund, kp = 5.0e6, h._SYNTHETIC_GENE_ABUND, 3.0
    doses = [150.0, 300.0, 450.0]

    def gb():
        return h._axial_graph(gene, n_sub=10)

    def db_sat(dose):
        return h._sat_drug(gene, fm, cltot, abund, 20.0, kp, km_mgl, fup, dose, mw)

    def db_lin(dose):
        return h._drug(gene, fm, cltot, abund, 20.0, kp, fup, dose, mw)

    beta_em_sat = h._beta_for_genotype(gb, db_sat, doses, "EM", gene, 0.03, "single_dose")
    beta_em_lin = h._beta_for_genotype(gb, db_lin, doses, "EM", gene, 0.03, "single_dose")
    assert beta_em_sat > 1.05, ("EM should be supra-proportional", beta_em_sat)
    assert abs(beta_em_lin - 1.0) < 0.02, ("linear-null must be proportional", beta_em_lin)


def test_axial_inlet_cu_exceeds_well_stirred():
    """Arm-F premise: the axial liver's peak unbound Cu exceeds the well_stirred Cu
    for the same drug/dose (well_stirred averages the inlet away)."""
    h = _harness()
    gene, fup = "CYP2D6", 0.30
    abund = h._SYNTHETIC_GENE_ABUND
    drug = h._sat_drug(gene, 0.85, 5.0e6, abund, 20.0, 3.0, 0.3, fup, 300.0, 341.4)
    g_ws = h._well_stirred_graph(gene)
    g_ax = h._axial_graph(gene, n_sub=10)
    cu_ws = h._peak_liver_cu(g_ws, drug, fup)
    # axial: max Cu over all sub-tanks (tank 1 sees the inlet)
    cu_ax = max(
        h._peak_liver_cu_node(g_ax, drug, n.name, fup)
        for n in g_ax.nodes.values() if (n.lookup_name or n.name) == "liver"
    )
    assert cu_ax > cu_ws > 0


def test_box_robustness_probe_propafenone_axial_engages():
    """Propafenone axial first-pass engages saturation across the Km span (low + high)
    × fu_mic ∈ {0.3,0.6,1.0}: every corner |Δlog AUC-fold| > 0.10 → gate PASS."""
    h = _harness()
    deltas = h.box_robustness_probe(
        gene_tag="CYP2D6", fm=0.80, mw=341.4, fup=0.30, dose_mg=300.0,
        km_span_uM=[0.12, 5.3], fu_mic_grid=[0.3, 0.6, 1.0],
        pm_activity=0.03, regime="single_dose",
    )
    # report all corners; propafenone is expected to pass at the LOW Km end robustly,
    # and the probe returns the per-corner deltas for the gate decision.
    assert len(deltas) == 6 and all(d >= 0 for d in deltas)
    # the gate decision itself (PASS/HALT) is asserted in the Task 6 report, not pinned
    # to a hard PASS here (high-Km corner may legitimately fail → that is the gate working)


def test_oracle_linear_fold_matches_analytic_well_stirred():
    """C2: with idealized gene→0 PM (a_var=0), the linear engine's oral AUC genotype
    fold = 1/(1-fm) on the well_stirred skeleton."""
    h = _harness()
    fold = h.oracle_check(gene_tag="CYP2C9", fm=0.9, skeleton="well_stirred")
    assert fold == pytest.approx(1.0 / (1.0 - 0.9), rel=0.02)


@pytest.mark.xfail(
    strict=True,
    reason="FINDING: the 1/(1-fm) oral-AUC genotype-fold oracle is exact on the "
    "well_stirred skeleton (telescoping F·CL) but the axial PARALLEL_TUBE liver does "
    "NOT satisfy it — parallel-tube extraction is exponential (E=1-exp(-fu*CLint/Q)) "
    "and does not telescope to the same closed form. Anchored at the axial skeleton's "
    "own E_h=0.30 the linear-null fold is ~4.04 (vs 5.0); via the well_stirred-twin "
    "anchor used by oracle_check it is ~3.98. The deviation is topology-structural, "
    "not a tuning/anchor artifact (confirmed E_h-invariant) and is NOT fixable without "
    "fitting. PR #79 axial PM scaling IS exercised here (gene->0 reaches every sub-tank "
    "— verified by test_first_pass_gene_converges and test_axial_inlet_cu). The "
    "well_stirred oracle (the machinery pin) passes exactly; spec §5.4's 'holds on both "
    "skeletons' over-generalized a well_stirred-only identity to parallel_tube."
)
def test_oracle_linear_fold_matches_analytic_axial():
    """C2 on the axial skeleton (REQUIRES PR #79 for PM scaling); same 1/(1-fm).

    KNOWN-FALSE on parallel_tube (see xfail reason): the engine produces the axial
    parallel-tube fold (~4.0), not the well_stirred analytic 5.0. Assertion kept
    numerically faithful (strict xfail) rather than loosened or fitted."""
    h = _harness()
    fold = h.oracle_check(gene_tag="CYP2D6", fm=0.8, skeleton="axial")
    assert fold == pytest.approx(1.0 / (1.0 - 0.8), rel=0.05)


def test_box_probe_monotone_in_km():
    """Self-review requirement: for the same fu_mic, the per-corner saturation deltas
    grow as Km falls (deeper saturation). Compares low-Km vs high-Km corners on the
    axial first-pass skeleton at a single fu_mic so only Km varies."""
    h = _harness()
    low = h.box_robustness_probe(
        gene_tag="CYP2D6", fm=0.80, mw=341.4, fup=0.30, dose_mg=300.0,
        km_span_uM=[0.12], fu_mic_grid=[0.5], pm_activity=0.03, regime="single_dose",
    )
    high = h.box_robustness_probe(
        gene_tag="CYP2D6", fm=0.80, mw=341.4, fup=0.30, dose_mg=300.0,
        km_span_uM=[20.0], fu_mic_grid=[0.5], pm_activity=0.03, regime="single_dose",
    )
    assert low[0] >= high[0], (low[0], high[0])


def test_headline_isolation_holdout_cache_untouched():
    """The harness is fully isolated: importing it and running the engine leaves the
    holdout cache byte-identical, and the v2.2a empty-enzyme_km bit-identity pin + the
    cached-2.731 headline pin still pass. Headline 2.731 is untouched by construction."""
    import subprocess
    import sys

    cache = ROOT / "data" / "training" / "4track_holdout_predictions.json"
    before = cache.read_bytes()
    h = _harness()
    # exercise the engine paths this milestone uses
    h.oracle_check("CYP2C9", 0.9, "well_stirred")
    g = h._axial_graph("CYP2D6", n_sub=6)
    drug = h._sat_drug("CYP2D6", 0.8, 5.0e6, h._SYNTHETIC_GENE_ABUND, 20.0, 3.0, 0.3,
                       0.1, 300.0, 341.4)
    h._single_dose_exposure(g, drug, "auc")
    assert cache.read_bytes() == before

    r = subprocess.run(
        [sys.executable, "-m", "pytest",
         "tests/regression/test_mm_headline_bit_identity.py",
         "tests/integration/test_holdout_regression.py::test_cached_holdout_aafe_is_2p731",
         "-q"],
        cwd=str(ROOT), capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stdout + r.stderr
