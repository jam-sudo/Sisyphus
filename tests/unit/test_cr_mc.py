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
