"""Unit tests for the phenotypes= parameter on pipeline.predict.predict.

These verify the parameter is threaded correctly through the pipeline.
Magnitude validation (does PM actually shift AUC by the expected
clinical amount?) belongs to integration tests — for SLCO1B1 see
``tests/integration/test_slco1b1_phenotype.py``. This file's job is
the wiring contract only.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from sisyphus.core import PredictionResult
from sisyphus.pipeline.predict import predict


class TestPredictPhenotypesWiring:
    def test_phenotypes_none_is_default(self):
        """phenotypes=None must produce a normal PredictionResult."""
        result = predict("Cn1c(=O)c2c(ncn2C)n(C)c1=O", dose_mg=100.0, phenotypes=None)
        assert isinstance(result, PredictionResult)
        assert result.pk.cmax.mean > 0

    def test_default_call_unchanged(self):
        """Existing callers (no phenotypes kwarg) must work bit-for-bit."""
        baseline = predict("Cn1c(=O)c2c(ncn2C)n(C)c1=O", dose_mg=100.0)
        explicit = predict(
            "Cn1c(=O)c2c(ncn2C)n(C)c1=O", dose_mg=100.0, phenotypes=None
        )
        assert explicit.pk.cmax.mean == pytest.approx(baseline.pk.cmax.mean, rel=1e-12)
        assert explicit.pk.auc_0t.mean == pytest.approx(
            baseline.pk.auc_0t.mean, rel=1e-12
        )

    def test_empty_phenotypes_is_no_op(self):
        """An empty dict must behave identically to phenotypes=None."""
        baseline = predict("Cn1c(=O)c2c(ncn2C)n(C)c1=O", dose_mg=100.0)
        with_empty = predict(
            "Cn1c(=O)c2c(ncn2C)n(C)c1=O", dose_mg=100.0, phenotypes={}
        )
        assert with_empty.pk.cmax.mean == pytest.approx(baseline.pk.cmax.mean, rel=1e-12)

    def test_phenotypes_is_keyword_only(self):
        """Positional call must reject phenotypes — keyword-only by design."""
        with pytest.raises(TypeError):
            predict(
                "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
                100.0,
                "oral",
                0,
                None,
                {"CYP1A2": "PM"},  # type: ignore[misc]
            )

    def test_apply_phenotype_to_graph_is_called(self):
        """A non-empty phenotypes dict must invoke apply_phenotype_to_graph."""
        with patch(
            "sisyphus.predict.phenotype.apply_phenotype_to_graph",
            wraps=__import__(
                "sisyphus.predict.phenotype", fromlist=["apply_phenotype_to_graph"]
            ).apply_phenotype_to_graph,
        ) as spy:
            predict(
                "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
                dose_mg=100.0,
                phenotypes={"CYP1A2": "PM"},
            )
            assert spy.called, "apply_phenotype_to_graph was not invoked"
            args, kwargs = spy.call_args
            # Second positional or 'phenotypes' kwarg must equal what we passed.
            passed_dict = args[1] if len(args) > 1 else kwargs.get("phenotypes")
            assert passed_dict == {"CYP1A2": "PM"}

    def test_apply_phenotype_to_graph_not_called_for_none(self):
        """phenotypes=None must NOT invoke apply_phenotype_to_graph."""
        with patch(
            "sisyphus.predict.phenotype.apply_phenotype_to_graph"
        ) as spy:
            predict("Cn1c(=O)c2c(ncn2C)n(C)c1=O", dose_mg=100.0, phenotypes=None)
            assert not spy.called

    def test_apply_phenotype_to_graph_not_called_for_empty(self):
        """phenotypes={} must NOT invoke apply_phenotype_to_graph (no-op)."""
        with patch(
            "sisyphus.predict.phenotype.apply_phenotype_to_graph"
        ) as spy:
            predict("Cn1c(=O)c2c(ncn2C)n(C)c1=O", dose_mg=100.0, phenotypes={})
            assert not spy.called

    # -- PredictionResult.phenotypes_applied audit metadata (B-06) --

    def test_phenotypes_applied_empty_when_none(self):
        """phenotypes=None must produce PredictionResult.phenotypes_applied = ()."""
        result = predict("Cn1c(=O)c2c(ncn2C)n(C)c1=O", dose_mg=100.0, phenotypes=None)
        assert result.phenotypes_applied == ()

    def test_phenotypes_applied_empty_when_empty_dict(self):
        """phenotypes={} must produce PredictionResult.phenotypes_applied = ()."""
        result = predict("Cn1c(=O)c2c(ncn2C)n(C)c1=O", dose_mg=100.0, phenotypes={})
        assert result.phenotypes_applied == ()

    def test_phenotypes_applied_round_trips_single(self):
        """Single-gene phenotypes round-trips as ordered (gene, code) pairs."""
        result = predict(
            "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
            dose_mg=100.0,
            phenotypes={"CYP1A2": "PM"},
        )
        assert result.phenotypes_applied == (("CYP1A2", "PM"),)
        assert dict(result.phenotypes_applied) == {"CYP1A2": "PM"}

    def test_phenotypes_applied_round_trips_multi(self):
        """Multi-gene phenotypes round-trip; dict reconstruction is order-free."""
        passed = {"CYP1A2": "PM", "CYP3A5": "EM"}
        result = predict(
            "Cn1c(=O)c2c(ncn2C)n(C)c1=O", dose_mg=100.0, phenotypes=passed,
        )
        assert dict(result.phenotypes_applied) == passed
