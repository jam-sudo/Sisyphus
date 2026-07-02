"""Unit tests for the generic concentration-response (CR) layer. All synthetic SimResults
(no engine solve)."""
import numpy as np
import pytest

from sisyphus.core import SimResult
from sisyphus.pkpd import CRNodeResult, CRSpec, concentration_response


def _sim(conc: dict, t=None):
    t = np.linspace(0.0, 10.0, 101) if t is None else t
    return SimResult(time_h=t, concentrations=conc, amounts={},
                     mass_balance_error=0.0, solver_success=True)


def test_linear_response_and_aggregations():
    t = np.linspace(0.0, 4.0, 5)            # [0,1,2,3,4]
    c = np.array([0.0, 1.0, 2.0, 1.0, 0.0])
    sim = _sim({"x": c}, t)
    spec = CRSpec(node="x", response="linear", params={"slope": 2.0, "intercept": 1.0})
    out = concentration_response(sim, spec)
    r = out["x"]
    # response = 1 + 2c = [1,3,5,3,1]
    assert np.allclose(r.trajectory, [1.0, 3.0, 5.0, 3.0, 1.0])
    assert r.peak == 5.0 and r.tpeak == 2.0
    assert r.nadir == 1.0 and r.tnadir == 0.0          # argmin -> first index
    assert np.isclose(r.integral, np.trapezoid([1, 3, 5, 3, 1], t))


def test_negative_slope_nadir_differs_from_peak():
    t = np.array([0.0, 1.0, 2.0])
    sim = _sim({"x": np.array([0.0, 1.0, 2.0])}, t)
    spec = CRSpec(node="x", response="linear", params={"slope": -1.0, "intercept": 5.0})
    r = concentration_response(sim, spec)["x"]          # [5,4,3]
    assert r.peak == 5.0 and r.tpeak == 0.0
    assert r.nadir == 3.0 and r.tnadir == 2.0           # inhibitory: nadir != peak


def test_conc_scale_equivalence():
    t = np.array([0.0, 1.0, 2.0])
    c = np.array([0.0, 2.0, 4.0])
    sim = _sim({"x": c}, t)
    scaled = concentration_response(
        sim, CRSpec(node="x", response="linear", params={"slope": 1.0, "intercept": 0.0},
                    conc_scale=0.5))["x"]
    direct = concentration_response(
        _sim({"x": 0.5 * c}, t),
        CRSpec(node="x", response="linear", params={"slope": 1.0, "intercept": 0.0}))["x"]
    assert np.allclose(scaled.trajectory, direct.trajectory)


def test_mm_excess_response():
    t = np.array([0.0, 1.0])
    sim = _sim({"x": np.array([10.0, 10.0])}, t)
    spec = CRSpec(node="x", response="mm_excess",
                  params={"vmax": 100.0, "km": 10.0, "threshold": 30.0})
    r = concentration_response(sim, spec)["x"]
    # mm = 100*10/(10+10)=50; excess = max(0,50-30)=20
    assert np.allclose(r.trajectory, [20.0, 20.0])


def test_mm_excess_floors_at_zero():
    sim = _sim({"x": np.array([1.0, 1.0])}, np.array([0.0, 1.0]))
    r = concentration_response(
        sim, CRSpec(node="x", response="mm_excess",
                    params={"vmax": 10.0, "km": 10.0, "threshold": 5.0}))["x"]
    # mm = 10*1/11 ~ 0.909 < 5 -> excess 0
    assert np.allclose(r.trajectory, [0.0, 0.0])


def test_per_node_list_and_broadcast():
    t = np.array([0.0, 1.0])
    sim = _sim({"a": np.array([1.0, 1.0]), "b": np.array([2.0, 2.0])}, t)
    # broadcast: one dict applies to both nodes
    bc = concentration_response(
        sim, CRSpec(node=["a", "b"], response="linear",
                    params={"slope": 1.0, "intercept": 0.0}))
    assert np.allclose(bc["a"].trajectory, [1.0, 1.0])
    assert np.allclose(bc["b"].trajectory, [2.0, 2.0])
    # per-node list
    pn = concentration_response(
        sim, CRSpec(node=["a", "b"], response="linear",
                    params=[{"slope": 1.0, "intercept": 0.0},
                            {"slope": 10.0, "intercept": 0.0}]))
    assert np.allclose(pn["a"].trajectory, [1.0, 1.0])
    assert np.allclose(pn["b"].trajectory, [20.0, 20.0])


def test_validation_errors():
    with pytest.raises(ValueError):                      # unknown registry key
        CRSpec(node="x", response="nope", params={})
    with pytest.raises(ValueError):                      # len(params) != len(node)
        CRSpec(node=["a", "b"], response="linear", params=[{"slope": 1.0, "intercept": 0.0}])


def test_missing_node_raises_keyerror():
    sim = _sim({"x": np.array([1.0])}, np.array([0.0]))
    with pytest.raises(KeyError):
        concentration_response(sim, CRSpec(node="absent", response="linear",
                                           params={"slope": 1.0, "intercept": 0.0}))


def test_result_is_per_node_mapping_with_crnoderesult():
    sim = _sim({"x": np.array([1.0, 2.0])}, np.array([0.0, 1.0]))
    out = concentration_response(sim, CRSpec(node="x", response="linear",
                                             params={"slope": 1.0, "intercept": 0.0}))
    assert set(out.keys()) == {"x"} and isinstance(out["x"], CRNodeResult)


def test_p1_reproduces_zonal_hazard():
    # P1: one concentration_response call reproduces validation.pgx_metrics.zonal_hazard.
    from sisyphus.validation.pgx_metrics import zonal_hazard  # ONLY here (test bridges layers)

    t = np.linspace(0.0, 8.0, 81)
    fup = 0.3
    n = 4
    rng = np.arange(1, n + 1)
    # synthetic per-zone TOTAL concentration curves (a rise-then-fall hump per zone)
    arrs = [(zi * 2.0) * np.exp(-((t - 2.0) ** 2) / 4.0) for zi in rng]
    vmax_bio = [300.0, 250.0, 200.0, 150.0]
    vmax_detox = [10.0, 12.0, 14.0, 16.0]
    km_bio = 1.0

    ref = zonal_hazard([fup * a for a in arrs], vmax_bio, km_bio, vmax_detox, t)

    sim = _sim({f"liver__ax{i+1}": arrs[i] for i in range(n)}, t)
    spec = CRSpec(
        node=[f"liver__ax{i+1}" for i in range(n)],
        response="mm_excess",
        params=[{"vmax": vmax_bio[i], "km": km_bio, "threshold": vmax_detox[i]}
                for i in range(n)],
        conc_scale=fup,
    )
    out = concentration_response(sim, spec)
    got = [out[f"liver__ax{i+1}"].integral for i in range(n)]
    assert np.allclose(got, ref, rtol=1e-12, atol=0.0)


def test_p2_reproduces_compute_effect():
    # P2: reproduces pkpd.compute_effect (and guards against future divergence).
    from sisyphus.pkpd import PDModel, compute_effect

    t = np.linspace(0.0, 12.0, 121)
    cp = 5.0 * (np.exp(-0.3 * t) - np.exp(-1.5 * t))
    sim = _sim({"venous_blood": cp}, t)
    pd = PDModel(ke0=0.5, emax=80.0, ec50=2.0, hill=1.5, baseline=5.0)
    ref = compute_effect(sim, pd)
    out = concentration_response(
        sim, CRSpec(node="venous_blood", response="sigmoid_emax",
                    params={"emax": pd.emax, "ec50": pd.ec50, "hill": pd.hill,
                            "baseline": pd.baseline}, ke0=pd.ke0))["venous_blood"]
    assert np.allclose(out.trajectory, ref.effect, rtol=1e-12, atol=0.0)
    assert np.isclose(out.peak, ref.emax_achieved, rtol=1e-12)
    assert np.isclose(out.tpeak, ref.temax, rtol=1e-12)


def test_g_layering_pkpd_does_not_import_validation():
    # pkpd.py must not depend on the validation (benchmark) layer.
    import ast
    import pathlib

    src = (pathlib.Path(__file__).resolve().parents[2]
           / "src" / "sisyphus" / "pkpd.py").read_text()
    tree = ast.parse(src)
    mods = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.append(node.module)
    assert not any("validation" in m for m in mods), \
        f"pkpd imports a validation module: {[m for m in mods if 'validation' in m]}"


def test_g_headline_isolation():
    import json
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[2]
    d = json.loads((root / "data" / "training" / "4track_holdout_predictions.json").read_text())
    assert abs(d["overall"]["meta"]["aafe"] - 2.735) < 5e-3
    # predict/pipeline must not import the CR symbols (headline path unaffected)
    for sub in ("predict", "pipeline"):
        for f in (root / "src" / "sisyphus" / sub).glob("*.py"):
            txt = f.read_text()
            assert "concentration_response" not in txt and "CRSpec" not in txt
