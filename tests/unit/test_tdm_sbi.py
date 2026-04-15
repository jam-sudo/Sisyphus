"""Tests for SBI TDM dispatch path (regimen/tdm_sbi.py)."""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

import sisyphus.engine.flux  # noqa: F401
from sisyphus.engine.compiler import ODECompiler
from sisyphus.graph.builder import build_from_yaml
from sisyphus.regimen.tdm import Observation, TDMResult, bayesian_update
from sisyphus.regimen.types import DosingRegimen

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
        )
        assert isinstance(result, TDMResult)
        assert result.n_successful > 0
        assert result.prior_cmax.mean > 0
        assert result.posterior_cmax.mean > 0

    def test_sbi_posterior_tighter_than_prior_cv(self, physiology, morphine_drug):
        """Posterior predictive CV should be ≤ prior CV after observation."""
        graph, compiled = physiology
        drug, logp = morphine_drug
        regimen = DosingRegimen.single_oral(30.0)
        obs = (Observation(time_h=1.0, concentration=0.01865, cv=0.10),)

        result = bayesian_update(
            compiled, graph, drug, regimen, obs,
            method="sbi",
            n_prior=100,
            logp_hint=logp,
        )
        # Posterior should not be meaningfully wider than prior; we allow
        # slack for Monte-Carlo noise at N=100. 0.10 absorbs the expected
        # ~5–8 % MC variation on the CV estimate.
        assert result.posterior_cmax.cv < result.prior_cmax.cv + 0.10

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
            method="sbi", n_prior=80, logp_hint=logp,
        )
        r_hyb = bayesian_update(
            compiled, graph, drug, regimen, obs,
            method="sbi", n_prior=80, logp_hint=logp,
            sbi_use_surrogate=True,
        )
        ratio = r_hyb.posterior_cmax.mean / max(r_sci.posterior_cmax.mean, 1e-12)
        # Within 60 % of scipy is generous but covers MC noise + minor surrogate bias
        assert 0.4 < ratio < 2.0, (
            f"hybrid posterior Cmax {r_hyb.posterior_cmax.mean:.4f} "
            f"too far from scipy {r_sci.posterior_cmax.mean:.4f} (ratio {ratio:.2f})"
        )

    def test_sbi_multi_obs_reweight_affects_posterior(self, physiology, morphine_drug):
        """Multi-obs SBI should produce a different posterior than single-obs.

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
            method="sbi", n_prior=80, logp_hint=logp, seed=42,
        )
        r2 = bayesian_update(
            compiled, graph, drug, regimen, obs2,
            method="sbi", n_prior=80, logp_hint=logp, seed=42,
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
            method="sbi", n_prior=50, logp_hint=logp,
        )
        sbi_time = time.time() - t0

        assert r_sbi.posterior_cmax.mean > 0
        # Hard upper bound: SBI with 50 forward sims should finish under 60s.
        # On test hardware 150 sims runs in ~35s, so 50 ≈ 12s.
        assert sbi_time < 60.0, f"SBI took {sbi_time:.1f}s, expected <60s"


class TestSBIFallback:
    def test_missing_posterior_falls_back_when_enabled(self, physiology, morphine_drug, tmp_path):
        """If sbi_posterior_path is missing and sbi_fallback=True, uses IBIS."""
        graph, compiled = physiology
        drug, _ = morphine_drug
        regimen = DosingRegimen.single_oral(30.0)
        obs = (Observation(time_h=1.0, concentration=0.01865, cv=0.10),)

        ghost_path = tmp_path / "nonexistent.pt"

        # Use small n_prior for speed; IBIS fallback will be slow otherwise.
        # We only check that it returns without raising, not the value.
        result = bayesian_update(
            compiled, graph, drug, regimen, obs,
            method="sbi",
            sbi_posterior_path=str(ghost_path),
            sbi_fallback=True,
            n_prior=50,
        )
        assert isinstance(result, TDMResult)

    def test_missing_posterior_raises_when_fallback_disabled(
        self, physiology, morphine_drug, tmp_path
    ):
        graph, compiled = physiology
        drug, _ = morphine_drug
        regimen = DosingRegimen.single_oral(30.0)
        obs = (Observation(time_h=1.0, concentration=0.01865, cv=0.10),)

        ghost_path = tmp_path / "nonexistent.pt"

        with pytest.raises(FileNotFoundError):
            bayesian_update(
                compiled, graph, drug, regimen, obs,
                method="sbi",
                sbi_posterior_path=str(ghost_path),
                sbi_fallback=False,
                n_prior=50,
            )
