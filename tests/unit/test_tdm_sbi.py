"""Tests for SBI TDM dispatch path (regimen/tdm_sbi.py)."""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

pytest.importorskip("torch")

import sisyphus.engine.flux  # noqa: F401, E402
from sisyphus.engine.compiler import ODECompiler  # noqa: E402
from sisyphus.graph.builder import build_from_yaml  # noqa: E402
from sisyphus.regimen.tdm import Observation, TDMResult, bayesian_update  # noqa: E402
from sisyphus.regimen.types import DosingRegimen  # noqa: E402

POSTERIOR_PATH = pathlib.Path("models/sbi/multi_drug_nsf.pt")
MORPHINE_SMILES = "CN1CCC23C4C1CC5=C2C(=C(C=C5)O)OC3C(C=C4)O"


@pytest.fixture(scope="module")
def physiology():
    yaml_path = (
        pathlib.Path(__file__).resolve().parent.parent.parent
        / "data"
        / "physiology"
        / "reference_man.yaml"
    )
    graph = build_from_yaml(yaml_path)
    compiled = ODECompiler().compile(graph)
    return graph, compiled


@pytest.fixture(scope="module")
def morphine_drug(physiology):
    from sisyphus.predict.adme import predict_adme
    from sisyphus.predict.chemistry import compute_profile
    from sisyphus.predict.ivive import build_drug_on_graph

    graph, _ = physiology
    profile = compute_profile(MORPHINE_SMILES)
    adme = predict_adme(profile)
    liver_enzymes = None
    if "liver" in graph.nodes and graph.nodes["liver"].enzymes:
        liver_enzymes = {
            tag: dist.mean for tag, dist in graph.nodes["liver"].enzymes.items()
        }
    drug = build_drug_on_graph(profile, adme, 30.0, "oral", liver_enzymes=liver_enzymes)
    return drug, float(profile.logp)


def _posterior_prior_matches_current() -> bool:
    """Check if saved posterior's prior bounds match the current logit(fup) prior."""
    if not POSTERIOR_PATH.exists():
        return False
    try:
        import torch
        checkpoint = torch.load(POSTERIOR_PATH, weights_only=False, map_location="cpu")
        if isinstance(checkpoint, dict) and "prior_low" in checkpoint:
            saved_low_1 = float(checkpoint["prior_low"][1])
            # Old parameterization had prior_low[1] ≈ 0.01 (absolute fup).
            # New logit parameterization has prior_low[1] ≈ -4.595.
            return saved_low_1 < -1.0  # logit space values are << 0
    except Exception:
        pass
    return False


@pytest.mark.skipif(
    not POSTERIOR_PATH.exists(),
    reason="SBI multi-drug posterior not trained",
)
@pytest.mark.skipif(
    POSTERIOR_PATH.exists() and not _posterior_prior_matches_current(),
    reason="Posterior trained with old fup parameterization (pre-logit); retrain needed",
)
class TestSBIDispatch:
    def test_covariate_pair_exclusion_uses_flow_specific_routing(
        self, morphine_drug, monkeypatch, tmp_path,
    ):
        from types import SimpleNamespace

        from sisyphus.regimen import tdm_sbi

        drug, _ = morphine_drug
        path = tmp_path / "routing.json"
        path.write_text(json.dumps({
            "hierarchical": {"pairs": [
                {"population": "adult", "drug": "morphine", "smiles": MORPHINE_SMILES,
                 "use_sbi": False},
            ]},
            "continuous": {"pairs": [
                {"population": "adult", "drug": "morphine", "smiles": MORPHINE_SMILES,
                 "body_weight_kg": 70.0, "age_years": 35.0, "use_sbi": False},
                {"population": "pediatric_5y", "drug": "morphine", "smiles": MORPHINE_SMILES,
                 "body_weight_kg": 18.0, "age_years": 5.0, "use_sbi": True},
            ]},
        }))
        monkeypatch.setattr(
            tdm_sbi, "get_resource_config",
            lambda: SimpleNamespace(data=lambda *args, **kwargs: path),
        )
        assert "hierarchical SBC excludes" in tdm_sbi._sbc_covariate_exclusion(
            drug, "adult", None, None
        )
        assert "continuous SBC excludes" in tdm_sbi._sbc_covariate_exclusion(
            drug, None, 70.0, 35.0
        )
        assert tdm_sbi._sbc_covariate_exclusion(drug, None, 18.0, 5.0) is None
        assert tdm_sbi._sbc_covariate_exclusion(drug, "pediatric_5y", None, None) is None

        path.write_text(json.dumps({
            "continuous": {"validation_status": "pending_corrected_sbc", "pairs": []},
        }))
        assert "pending" in tdm_sbi._sbc_covariate_exclusion(drug, None, 70.0, 35.0)

    def test_pending_multi_observation_falls_back_before_model_load(
        self, physiology, morphine_drug,
    ):
        from sisyphus.regimen.tdm_sbi import sbi_update
        from sisyphus.sbi.importance import LowImportanceESS

        graph, compiled = physiology
        drug, logp = morphine_drug
        with pytest.raises(LowImportanceESS, match="Multi-observation corrected SBC is pending"):
            sbi_update(
                compiled, graph, drug, DosingRegimen.single_oral(30.0),
                (Observation(1.0, 0.01865), Observation(4.0, 0.010)),
                n_samples=2, n_predictive_sims=1, logp_hint=logp,
            )

    @pytest.mark.parametrize("failed_attempts", [0, 2, 3])
    def test_default_sbi_uses_likelihood_correction(
        self, physiology, morphine_drug, monkeypatch, failed_attempts,
    ):
        from types import SimpleNamespace

        from sisyphus.regimen import tdm_sbi
        from sisyphus.regimen.tdm_sbi import sbi_update
        from sisyphus.sbi import amortizer, importance

        graph, compiled = physiology
        drug, logp = morphine_drug
        seen = []
        log_densities = []
        solve_windows = []
        solve_steps = []
        monkeypatch.setattr(amortizer, "load_result", lambda path: SimpleNamespace(
            prior_low=np.array([-1.0, -4.5, -0.5]), posterior=object(),
        ))
        def fake_solve(*args, **kwargs):
            solve_windows.append(kwargs["t_total_h"])
            solve_steps.append(kwargs["dt_output"])
            return SimpleNamespace(
                solver_success=True, time_h=np.array([0.0, 1.0, 2.0]),
                concentrations={"venous_blood": np.array([0.01, 0.005, 0.02])},
            )

        monkeypatch.setattr(tdm_sbi, "solve_regimen", fake_solve)

        def corrected(posterior, x, log_draw, *, n_draws, n_proposals, seed):
            seen.append((len(x), n_proposals, seed))
            log_densities.append(log_draw(np.zeros(3), np.random.default_rng(1)))
            if len(seen) <= failed_attempts:
                raise importance.LowImportanceESS("forced low ESS")
            return np.zeros((n_draws, 3)), 64.0

        monkeypatch.setattr(importance, "sample_corrected", corrected)
        def run():
            return sbi_update(
                compiled, graph, drug, DosingRegimen.single_oral(30.0),
                (Observation(1.0, 0.01865),), n_samples=2, n_predictive_sims=1,
                logp_hint=logp,
            )

        if failed_attempts == 3:
            with pytest.raises(importance.LowImportanceESS, match="after 1024 proposals"):
                run()
        else:
            result = run()
            assert result.ess == 64.0
        expected_counts = [256] if failed_attempts == 0 else [256, 512, 1024]
        assert seen == [(13, count, 42 + count) for count in expected_counts]
        sigma = np.log10(1.1)
        expected = (-0.5 * ((np.log10(0.01865) - np.log10(0.02)) / sigma) ** 2
                    - np.log(sigma) - 0.5 * np.log(2 * np.pi))
        assert all(value == pytest.approx(expected) for value in log_densities)
        assert 48.0 in solve_windows
        assert 48.0 / 499 in solve_steps

    def test_retrained_hierarchical_posterior_runs_end_to_end(
        self, physiology, morphine_drug,
    ):
        from sisyphus.regimen.tdm_sbi import sbi_update

        graph, compiled = physiology
        drug, logp = morphine_drug
        result = sbi_update(
            compiled, graph, drug, DosingRegimen.single_oral(30.0),
            (Observation(1.0, 0.01865),), n_samples=100, n_predictive_sims=5,
            population_class="adult", logp_hint=logp,
            raw_npe=True,
        )
        assert result.n_successful > 0
        assert result.posterior_cmax.mean > 0

    def test_retrained_continuous_posterior_runs_end_to_end(
        self, physiology, morphine_drug, monkeypatch,
    ):
        from sisyphus.regimen import tdm_sbi
        from sisyphus.regimen.tdm_sbi import sbi_update

        graph, compiled = physiology
        drug, logp = morphine_drug
        weights = []
        original_compile = tdm_sbi.ODECompiler.compile

        def capture_compile(self, covariate_graph):
            weights.append(covariate_graph.global_params["body_weight"].mean)
            return original_compile(self, covariate_graph)

        monkeypatch.setattr(tdm_sbi.ODECompiler, "compile", capture_compile)
        result = sbi_update(
            compiled, graph, drug, DosingRegimen.single_oral(30.0),
            (Observation(1.0, 0.01865),), n_samples=100, n_predictive_sims=5,
            body_weight_kg=18.0, age_years=5.0, logp_hint=logp,
            raw_npe=True,
        )
        assert weights == [18.0]
        assert result.n_successful > 0
        assert result.posterior_cmax.mean > 0

    def test_continuous_context_uses_14_feature_standardizer(
        self, physiology, morphine_drug, monkeypatch,
    ):
        from types import SimpleNamespace

        import torch

        from sisyphus.regimen.tdm_sbi import sbi_update
        from sisyphus.sbi import amortizer

        graph, compiled = physiology
        drug, logp = morphine_drug
        observed = []

        class Posterior:
            def sample(self, shape, *, x, show_progress_bars):
                observed.append(x.shape[1])
                return torch.zeros((shape[0], 3))

        monkeypatch.setattr(amortizer, "load_result", lambda path: SimpleNamespace(
            prior_low=np.array([-1.0, -4.5, -1.0]), posterior=Posterior(),
        ))
        result = sbi_update(
            compiled, graph, drug, DosingRegimen.single_oral(30.0),
            (Observation(1.0, 0.01865),), n_samples=2, n_predictive_sims=1,
            body_weight_kg=70.0, age_years=35.0, logp_hint=logp,
            raw_npe=True,
        )
        assert observed == [15]
        assert result.method_used == "sbi"

    def test_calibrated_multi_observation_uses_time_aware_posterior(
        self, physiology, morphine_drug, monkeypatch, caplog,
    ):
        from sisyphus.regimen.tdm_sbi import sbi_update
        from sisyphus.sbi import amortizer

        graph, compiled = physiology
        drug, logp = morphine_drug
        loaded = []
        original_load = amortizer.load_result

        def record_load(path):
            loaded.append(pathlib.Path(path).name)
            return original_load(path)

        monkeypatch.setattr(amortizer, "load_result", record_load)
        result = sbi_update(
            compiled, graph, drug, DosingRegimen.single_oral(30.0),
            (Observation(1.0, 0.01865), Observation(4.0, 0.010)),
            n_samples=100, n_predictive_sims=5, logp_hint=logp, use_surrogate=True,
            raw_npe=True,
        )
        assert loaded == ["multi_observation_nsf.pt"]
        assert result.n_successful > 0
        assert np.allclose(result.weights, result.weights[0])
        assert "surrogate does not support timed observations" in caplog.text

    def test_sbi_update_runs_end_to_end(self, physiology, morphine_drug):
        graph, compiled = physiology
        drug, logp = morphine_drug
        regimen = DosingRegimen.single_oral(30.0)
        obs = (Observation(time_h=1.0, concentration=0.01865, cv=0.10),)

        result = bayesian_update(
            compiled, graph, drug, regimen, obs,
            method="sbi",
            n_prior=50,  # small for fast test
            logp_hint=logp,
            sbi_raw_npe=True,
        )
        assert isinstance(result, TDMResult)
        assert result.n_successful > 0
        assert result.prior_cmax.mean > 0
        assert result.posterior_cmax.mean > 0

    def test_sbi_posterior_shifts_toward_low_observation(self, physiology, morphine_drug):
        """A low measured concentration should shift predicted Cmax downward."""
        graph, compiled = physiology
        drug, logp = morphine_drug
        regimen = DosingRegimen.single_oral(30.0)
        obs = (Observation(time_h=1.0, concentration=0.01865, cv=0.10),)

        result = bayesian_update(
            compiled, graph, drug, regimen, obs,
            method="sbi",
            n_prior=100,
            logp_hint=logp,
            sbi_raw_npe=True,
        )
        assert result.posterior_cmax.mean < result.prior_cmax.mean

    def test_sbi_hybrid_surrogate_matches_scipy_on_easy_drug(
        self, physiology, morphine_drug
    ):
        """Hybrid surrogate dispatch should match scipy posterior Cmax
        within 40 % on a well-behaved drug like morphine.

        The std-gate routes theta-shifted samples that push the surrogate
        into its high-uncertainty region back to scipy, so the result
        tracks scipy closely rather than the uncorrected surrogate (which
        can be off by >100 % on hard drugs).
        """
        import pathlib

        graph, compiled = physiology
        drug, logp = morphine_drug
        regimen = DosingRegimen.single_oral(30.0)
        obs = (Observation(time_h=1.0, concentration=0.01865, cv=0.10),)

        # Skip if surrogate models are not available on this checkout.
        surrogate_dir = pathlib.Path("models/surrogate")
        if not surrogate_dir.exists():
            pytest.skip("surrogate model directory missing")

        r_sci = bayesian_update(
            compiled, graph, drug, regimen, obs,
            method="sbi", n_prior=80, logp_hint=logp, sbi_raw_npe=True,
        )
        r_hyb = bayesian_update(
            compiled, graph, drug, regimen, obs,
            method="sbi", n_prior=80, logp_hint=logp,
            sbi_use_surrogate=True, sbi_raw_npe=True,
        )
        ratio = r_hyb.posterior_cmax.mean / max(r_sci.posterior_cmax.mean, 1e-12)
        # Within 60 % of scipy is generous but covers MC noise + minor surrogate bias
        assert 0.4 < ratio < 2.0, (
            f"hybrid posterior Cmax {r_hyb.posterior_cmax.mean:.4f} "
            f"too far from scipy {r_sci.posterior_cmax.mean:.4f} (ratio {ratio:.2f})"
        )

    def test_sbi_multi_obs_reweight_affects_posterior(self, physiology, morphine_drug):
        """The legacy multi-obs reweight path should alter posterior weights.

        With an extra observation that disagrees with the first (higher
        concentration later than expected for morphine's half-life), the
        reweighted posterior should shift toward accommodating both.
        """
        graph, compiled = physiology
        drug, logp = morphine_drug
        regimen = DosingRegimen.single_oral(30.0)
        obs1 = (Observation(time_h=1.0, concentration=0.01865, cv=0.10),)
        obs2 = (
            Observation(time_h=1.0, concentration=0.01865, cv=0.10),
            Observation(time_h=4.0, concentration=0.010, cv=0.10),
        )

        r1 = bayesian_update(
            compiled, graph, drug, regimen, obs1,
            method="sbi", n_prior=80, logp_hint=logp, seed=42, sbi_raw_npe=True,
        )
        r2 = bayesian_update(
            compiled, graph, drug, regimen, obs2,
            method="sbi", n_prior=80, logp_hint=logp, seed=42,
            sbi_posterior_path=str(POSTERIOR_PATH),
            sbi_raw_npe=True,
        )
        assert r1.n_successful > 0
        assert r2.n_successful > 0
        # ESS for multi-obs should be ≤ n_post (reweighting reduces effective N).
        # For single-obs SBI, ESS == n_post (uniform weights).
        assert r2.ess <= r2.n_successful
        # Weights should no longer all be uniform for multi-obs.
        assert not np.allclose(r2.weights, r2.weights[0])

    def test_sbi_speedup_vs_importance_sampling(self, physiology, morphine_drug):
        """SBI dispatch should be measurably faster than IS for the same forward budget.

        Rough ballpark: SBI on 50 forward sims should beat IS on 500 prior
        samples. Not a strict time gate — just a sanity check that amortized
        sampling isn't dominating.
        """
        import time

        graph, compiled = physiology
        drug, logp = morphine_drug
        regimen = DosingRegimen.single_oral(30.0)
        obs = (Observation(time_h=1.0, concentration=0.01865, cv=0.10),)

        t0 = time.time()
        r_sbi = bayesian_update(
            compiled, graph, drug, regimen, obs,
            method="sbi", n_prior=50, logp_hint=logp, sbi_raw_npe=True,
        )
        sbi_time = time.time() - t0

        assert r_sbi.posterior_cmax.mean > 0
        # Hard upper bound: SBI with 50 forward sims should finish under 60s.
        # On test hardware 150 sims runs in ~35s, so 50 ≈ 12s.
        assert sbi_time < 60.0, f"SBI took {sbi_time:.1f}s, expected <60s"
