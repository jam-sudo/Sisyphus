"""Unit tests for the kp_method= parameter on pipeline.predict.predict.

These verify that the parameter is threaded correctly through the pipeline
to ``build_drug_on_graph``. Magnitude validation (does Berezhkovskiy
produce different Cmax than Rodgers-Rowland?) belongs to integration
tests; this file's job is the wiring contract only.

Closes hardening backlog B3 (the per-drug DrugOnGraph.kp_method field
was unreachable through pipeline.predict.predict() before 2026-05-02).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from sisyphus.core import PredictionResult
from sisyphus.pipeline.predict import predict


_CAFFEINE = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"


class TestPredictKpMethodWiring:
    def test_default_is_rodgers_rowland(self):
        """Existing callers (no kp_method kwarg) must continue to use
        Rodgers-Rowland — the historical default and what every existing
        benchmark / test expects.
        """
        baseline = predict(_CAFFEINE, dose_mg=100.0)
        assert isinstance(baseline, PredictionResult)
        assert baseline.pk.cmax.mean > 0

    def test_default_call_unchanged(self):
        """predict(smiles, dose) without kp_method must be bit-identical
        to predict(smiles, dose, kp_method="rodgers_rowland").
        """
        a = predict(_CAFFEINE, dose_mg=100.0)
        b = predict(_CAFFEINE, dose_mg=100.0, kp_method="rodgers_rowland")
        assert a.pk.cmax.mean == pytest.approx(b.pk.cmax.mean)
        assert a.pk.auc_0t.mean == pytest.approx(b.pk.auc_0t.mean)

    def test_kp_method_is_keyword_only(self):
        """kp_method must be keyword-only to keep the positional
        signature stable for callers that pass route/n_mc_samples
        positionally.
        """
        with pytest.raises(TypeError):
            predict(_CAFFEINE, 100.0, "oral", 0, None, "rodgers_rowland")  # type: ignore[misc]

    def test_kp_method_forwarded_to_build_drug_on_graph(self):
        """The string passed to predict() must reach build_drug_on_graph.
        Spying via patch — pipeline calls build_drug_on_graph twice
        (initial + liver_enzymes rebuild); both calls must forward the
        kp_method.
        """
        from sisyphus.predict import ivive

        real_build = ivive.build_drug_on_graph
        seen_kp_methods: list[str] = []

        def spy(*args, **kwargs):
            seen_kp_methods.append(kwargs.get("kp_method", "<missing>"))
            return real_build(*args, **kwargs)

        with patch("sisyphus.predict.ivive.build_drug_on_graph", side_effect=spy) as p:
            predict(_CAFFEINE, dose_mg=100.0, kp_method="berezhkovskiy")
            assert p.call_count >= 1, "expected build_drug_on_graph to be called at least once"
            assert all(m == "berezhkovskiy" for m in seen_kp_methods), (
                f"expected every build_drug_on_graph call to receive kp_method='berezhkovskiy', "
                f"got {seen_kp_methods}"
            )

    def test_default_forwarded_to_build_drug_on_graph(self):
        """Default kp_method='rodgers_rowland' must be forwarded explicitly
        (not omitted), so the pipeline contract does not silently rely on
        build_drug_on_graph's own default — that decoupling lets the
        function default change without surprising the API.
        """
        from sisyphus.predict import ivive

        real_build = ivive.build_drug_on_graph
        seen_kp_methods: list[str] = []

        def spy(*args, **kwargs):
            seen_kp_methods.append(kwargs.get("kp_method", "<missing>"))
            return real_build(*args, **kwargs)

        with patch("sisyphus.predict.ivive.build_drug_on_graph", side_effect=spy):
            predict(_CAFFEINE, dose_mg=100.0)
            assert all(m == "rodgers_rowland" for m in seen_kp_methods), (
                f"expected every call to receive kp_method='rodgers_rowland', "
                f"got {seen_kp_methods}"
            )
