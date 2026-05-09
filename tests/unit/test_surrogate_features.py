"""Tests for the surrogate feature extractor OOD fix (Track D1)."""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

import sisyphus.engine.flux  # noqa: F401
from sisyphus.engine.compiler import ODECompiler, ResolvedParams
from sisyphus.ml.surrogate import (
    FEATURE_NAMES,
    features_in_distribution,
    params_to_features_single,
    recover_drug_level_clint,
)
from sisyphus.graph.builder import build_from_yaml


DRUGS = (
    # (name, smiles, dose_mg, expected_clint_approx_within_20pct)
    ("morphine", "CN1CCC23C4C1CC5=C2C(=C(C=C5)O)OC3C(C=C4)O", 30.0, 23.0),
    ("amantadine", "C1C2CC3CC1CC(C2)(C3)N", 100.0, 29.0),
    ("ketorolac", "O=C(c1ccccc1)c1ccc2n1CCC2C(=O)O", 10.0, 24.0),
)


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


def _build_resolved(graph, smiles: str, dose_mg: float) -> ResolvedParams:
    from sisyphus.predict.adme import predict_adme
    from sisyphus.predict.chemistry import compute_profile
    from sisyphus.predict.ivive import build_drug_on_graph

    profile = compute_profile(smiles)
    adme = predict_adme(profile)
    liver_enzymes = None
    if "liver" in graph.nodes and graph.nodes["liver"].enzymes:
        liver_enzymes = {
            tag: dist.mean for tag, dist in graph.nodes["liver"].enzymes.items()
        }
    drug = build_drug_on_graph(
        profile, adme, dose_mg, "oral", liver_enzymes=liver_enzymes
    )
    return ResolvedParams(graph, drug)


class TestRecoverDrugLevelClint:
    def test_returns_positive_for_known_drugs(self, physiology):
        graph, _ = physiology
        for name, smi, dose, _expected in DRUGS:
            params = _build_resolved(graph, smi, dose)
            clint = recover_drug_level_clint(params)
            assert clint > 0, f"{name} recovered clint was {clint}"

    def test_matches_adme_predictor_output(self, physiology):
        """Recovered CLint should closely match the ADME predictor's CLint.

        ``build_drug_on_graph`` calls ``_decompose_clint`` which converts
        the ADME CLint into per-enzyme affinities. Inverting that operation
        here should recover the original scalar within a few percent
        (the loss comes from rounding in ``_LIVER_ENZYME_ABUNDANCE``).
        """
        from sisyphus.predict.adme import predict_adme
        from sisyphus.predict.chemistry import compute_profile

        graph, _ = physiology
        for name, smi, dose, _expected in DRUGS:
            profile = compute_profile(smi)
            adme = predict_adme(profile)
            params = _build_resolved(graph, smi, dose)
            recovered = recover_drug_level_clint(params)
            predicted = float(adme.clint.mean)
            rel_err = abs(recovered - predicted) / max(predicted, 1e-9)
            assert rel_err < 0.05, (
                f"{name}: recovered {recovered:.2f} vs predicted {predicted:.2f} "
                f"(rel_err={rel_err:.3f})"
            )


class TestParamsToFeaturesSingle:
    def test_feature_shape_and_layout(self, physiology):
        graph, _ = physiology
        params = _build_resolved(graph, DRUGS[0][1], DRUGS[0][2])
        feats = params_to_features_single(params)
        assert feats.shape == (12,)
        assert feats.dtype == np.float64
        assert len(FEATURE_NAMES) == 12

    def test_clint_dimension_in_training_range(self, physiology):
        """The log10_clint feature must land in the training range [-0.5, 3.0]
        for every real drug — this is the OOD bug this fix targets."""
        graph, _ = physiology
        for name, smi, dose, _expected in DRUGS:
            params = _build_resolved(graph, smi, dose)
            feats = params_to_features_single(params)
            assert -0.5 <= feats[0] <= 3.0, (
                f"{name} log10_clint={feats[0]:.3f} is out of training range"
            )

    def test_features_in_distribution_passes_for_known_drugs(self, physiology):
        graph, _ = physiology
        for name, smi, dose, _expected in DRUGS:
            params = _build_resolved(graph, smi, dose)
            feats = params_to_features_single(params)
            ok, offenders = features_in_distribution(feats)
            assert ok, f"{name} OOD: {offenders}"


class TestFeaturesInDistribution:
    def test_detects_out_of_range_log10_clint(self):
        feats = np.zeros(12)
        feats[0] = 10.0  # absurdly high log10_clint
        ok, offenders = features_in_distribution(feats)
        assert not ok
        assert any("log10_clint" in s for s in offenders)

    def test_slack_allows_small_edge_deviation(self):
        feats = np.zeros(12)
        feats[0] = 3.1  # just above 3.0 upper bound
        feats[8] = 1.0  # log10_dose in range
        feats[9] = 1.0  # log10_peff in range
        feats[10] = 1.0  # log10_sol in range
        feats[11] = 1.0  # log10_renal_cl in range
        feats[2] = 2.0  # logp in range
        feats[3] = 7.0  # pka_or_7
        feats[7] = 0.5  # mw_scaled
        feats[1] = 0.5  # fup
        ok, offenders = features_in_distribution(feats, slack=0.1)
        # 3.1 is within 3.0 + 0.1*(3.0-(-0.5)) = 3.35 → should pass
        assert ok, f"offenders: {offenders}"
