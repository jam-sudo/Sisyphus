import numpy as np
import pytest

from sisyphus.core import Distribution, SimResult
from sisyphus.pkpd import (
    CRNodeMCResult,
    CRSpec,
    EndpointMC,
    concentration_response,
    concentration_response_mc,
)


def mk_sim(time, conc_by_node):
    return SimResult(
        time_h=np.asarray(time, dtype=float),
        concentrations={k: np.asarray(v, dtype=float) for k, v in conc_by_node.items()},
        amounts={},
        mass_balance_error=0.0,
        solver_success=True,
    )


def test_g_reduction_nsamples0_matches_b2_exactly():
    time = np.linspace(0.0, 10.0, 11)
    sim = mk_sim(time, {"liver": np.linspace(0.0, 5.0, 11)})
    spec = CRSpec(node="liver", response="linear", params={"slope": 2.0, "intercept": 1.0})
    det = concentration_response(sim, spec)["liver"]
    mc = concentration_response_mc(sim, spec, n_samples=0)["liver"]
    assert isinstance(mc, CRNodeMCResult)
    assert mc.n_samples == 0
    for ep in ("peak", "tpeak", "nadir", "tnadir", "integral"):
        e = getattr(mc, ep)
        d = getattr(det, ep)
        assert isinstance(e, EndpointMC)
        assert e.mean == d
        assert e.std == 0.0
        assert e.p5 == d and e.p50 == d and e.p95 == d
        assert e.samples.tolist() == [d]


def test_g_reduction_cv0_single_sim_zero_std():
    time = np.linspace(0.0, 10.0, 11)
    sim = mk_sim(time, {"liver": np.linspace(0.0, 5.0, 11)})
    det = concentration_response(
        sim, CRSpec(node="liver", response="linear", params={"slope": 2.0, "intercept": 1.0})
    )["liver"]
    spec = CRSpec(
        node="liver",
        response="linear",
        params={"slope": Distribution(2.0, 0.0), "intercept": Distribution(1.0, 0.0)},
        conc_scale=Distribution(1.0, 0.0),
    )
    mc = concentration_response_mc(sim, spec, n_samples=100, seed=1)["liver"]
    assert mc.n_samples == 100
    assert mc.peak.std == 0.0
    assert mc.peak.mean == det.peak
    assert mc.integral.std == 0.0
    assert mc.integral.mean == det.integral


def test_empty_ensemble_and_negative_nsamples_raise():
    spec = CRSpec(node="liver", response="linear", params={"slope": 1.0, "intercept": 0.0})
    with pytest.raises(ValueError):
        concentration_response_mc([], spec, n_samples=4)
    sim = mk_sim([0.0, 1.0], {"liver": [0.0, 1.0]})
    with pytest.raises(ValueError):
        concentration_response_mc(sim, spec, n_samples=-1)


def test_multinode_list_params_realize_per_node():
    # node=list + params=list[dict] with Distributions: per-node realize must align by index.
    time = np.linspace(0.0, 10.0, 11)
    sim = mk_sim(
        time,
        {"liver": np.linspace(0.0, 4.0, 11), "gut": np.linspace(0.0, 8.0, 11)},
    )
    # deterministic (cv=0) so the result is exact and we can pin per-node alignment.
    spec = CRSpec(
        node=["liver", "gut"],
        response="linear",
        params=[
            {"slope": Distribution(1.0, 0.0), "intercept": Distribution(0.0, 0.0)},
            {"slope": Distribution(3.0, 0.0), "intercept": Distribution(0.0, 0.0)},
        ],
    )
    # n_samples=0 must match the deterministic two-node B2.0 call (Distributions -> .mean).
    det_spec = CRSpec(
        node=["liver", "gut"],
        response="linear",
        params=[{"slope": 1.0, "intercept": 0.0}, {"slope": 3.0, "intercept": 0.0}],
    )
    det = concentration_response(sim, det_spec)
    mc0 = concentration_response_mc(sim, spec, n_samples=0)
    assert set(mc0) == {"liver", "gut"}
    assert mc0["liver"].peak.mean == det["liver"].peak  # 1*4 = 4
    assert mc0["gut"].peak.mean == det["gut"].peak       # 3*8 = 24
    assert mc0["liver"].peak.mean != mc0["gut"].peak.mean  # slopes did not get swapped
    # cv=0 through the rng path (n_samples>0) is also exact and per-node aligned.
    mc = concentration_response_mc(sim, spec, n_samples=50, seed=2)
    assert mc["liver"].peak.mean == det["liver"].peak
    assert mc["gut"].peak.mean == det["gut"].peak
    assert mc["liver"].peak.std == 0.0 and mc["gut"].peak.std == 0.0


def test_g_param_mean_matches_analytic_expectation():
    # linear, slope=1, intercept=0: peak = conc_scale * Cmax. E[conc_scale] = mean = 2.0.
    time = np.linspace(0.0, 10.0, 11)
    sim = mk_sim(time, {"liver": np.linspace(0.0, 4.0, 11)})  # Cmax = 4.0
    spec = CRSpec(
        node="liver",
        response="linear",
        params={"slope": 1.0, "intercept": 0.0},
        conc_scale=Distribution(2.0, 0.3),
    )
    mc = concentration_response_mc(sim, spec, n_samples=5000, seed=7)["liver"]
    assert mc.peak.mean == pytest.approx(2.0 * 4.0, rel=0.05)  # 8.0
    # also pin the propagated param-source spread (a no-variance bug would give std==0):
    assert mc.peak.std == pytest.approx(0.3 * 8.0, rel=0.1)  # cv*mean = 0.3*8 = 2.4


def test_g_concentration_samples_take_per_member_values():
    # 2-member ensemble, deterministic params: each draw's peak is one of the two B2.0 peaks.
    time = np.linspace(0.0, 10.0, 11)
    simA = mk_sim(time, {"liver": np.linspace(0.0, 3.0, 11)})  # Cmax 3
    simB = mk_sim(time, {"liver": np.linspace(0.0, 9.0, 11)})  # Cmax 9
    spec = CRSpec(node="liver", response="linear", params={"slope": 1.0, "intercept": 0.0})
    peakA = concentration_response(simA, spec)["liver"].peak
    peakB = concentration_response(simB, spec)["liver"].peak
    mc = concentration_response_mc([simA, simB], spec, n_samples=2000, seed=3)["liver"]
    assert set(np.unique(mc.peak.samples).tolist()) == {peakA, peakB}
    assert min(peakA, peakB) <= mc.peak.mean <= max(peakA, peakB)
    assert mc.peak.p5 >= min(peakA, peakB) and mc.peak.p95 <= max(peakA, peakB)


def test_g_composition_both_sources_exceed_each_single_source():
    time = np.linspace(0.0, 10.0, 11)
    simA = mk_sim(time, {"liver": np.linspace(0.0, 3.0, 11)})
    simB = mk_sim(time, {"liver": np.linspace(0.0, 9.0, 11)})
    base = {"node": "liver", "response": "linear"}
    param_spec = CRSpec(
        params={"slope": 1.0, "intercept": 0.0}, conc_scale=Distribution(2.0, 0.3), **base
    )
    det_params = {"slope": 1.0, "intercept": 0.0}
    std_param = concentration_response_mc(
        simA, param_spec, n_samples=4000, seed=11
    )["liver"].peak.std
    std_conc = concentration_response_mc(
        [simA, simB], CRSpec(params=det_params, conc_scale=2.0, **base), n_samples=4000, seed=11
    )["liver"].peak.std
    std_both = concentration_response_mc(
        [simA, simB], param_spec, n_samples=4000, seed=11
    )["liver"].peak.std
    assert std_param > 0.0 and std_conc > 0.0
    assert std_both > std_param
    assert std_both > std_conc


def test_g_seed_reproducible_and_seed_sensitive():
    time = np.linspace(0.0, 10.0, 11)
    sim = mk_sim(time, {"liver": np.linspace(0.0, 4.0, 11)})
    spec = CRSpec(
        node="liver",
        response="linear",
        params={"slope": 1.0, "intercept": 0.0},
        conc_scale=Distribution(2.0, 0.3),
    )
    a = concentration_response_mc(sim, spec, n_samples=500, seed=42)["liver"]
    b = concentration_response_mc(sim, spec, n_samples=500, seed=42)["liver"]
    c = concentration_response_mc(sim, spec, n_samples=500, seed=43)["liver"]
    assert np.array_equal(a.peak.samples, b.peak.samples)
    assert not np.array_equal(a.peak.samples, c.peak.samples)


def test_g_zero_inflation_honesty():
    # mm_excess: low vmax draws keep the whole trajectory below threshold -> peak == 0.
    time = np.linspace(0.0, 10.0, 11)
    sim = mk_sim(time, {"liver": np.linspace(0.0, 10.0, 11)})  # Cmax 10, C/(km+C)=0.5 at peak
    spec = CRSpec(
        node="liver",
        response="mm_excess",
        params={"vmax": Distribution(1.0, 0.5), "km": 10.0, "threshold": 0.4},
    )
    mc = concentration_response_mc(sim, spec, n_samples=2000, seed=5)["liver"]
    assert mc.peak.p5 == 0.0  # zero-inflated tail a lognormal Distribution can't represent
    assert mc.peak.mean > 0.0
    assert (mc.peak.samples == 0.0).any() and (mc.peak.samples > 0.0).any()
