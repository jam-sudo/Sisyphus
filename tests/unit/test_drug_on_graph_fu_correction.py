"""Unit tests for DrugOnGraph.fu_correction_liver field (B-11)."""
from __future__ import annotations

import numpy as np
import pytest

from sisyphus.core import Distribution, DrugOnGraph


def _minimal_dog(**overrides) -> DrugOnGraph:
    """Construct a DrugOnGraph with required fields only; B-11-related override allowed."""
    defaults = dict(
        name="test_drug",
        smiles="CCO",
        dose_mg=100.0,
        route="oral",
        administration_node="stomach_lumen",
        mw=46.07,
        pka=None,
        compound_type="neutral",
        fup=Distribution(mean=0.1, cv=0.0),
        rbp=Distribution(mean=1.0, cv=0.0),
        kp_method="rodgers_rowland",
        kp_overrides={},
        peff=Distribution(mean=1.0e-4, cv=0.0),
        solubility=Distribution(mean=1.0, cv=0.0),
        enzyme_affinity={},
        renal_clearance=Distribution(mean=0.0, cv=0.0),
        particle_radius_um=25.0,
        transporter_kinetics={},
        ps_passive=Distribution(mean=1e6, cv=0.0),
        ps_eff=Distribution(mean=1e6, cv=0.0),
        cl_int_bile=Distribution(mean=0.0, cv=0.0),
        ps_overrides={},
        active_metabolite=None,
        observation_species="parent",
        enzyme_affinity_for_conversion={},
    )
    defaults.update(overrides)
    return DrugOnGraph(**defaults)


class TestFuCorrectionLiverField:
    def test_default_is_one_with_zero_cv(self):
        dog = _minimal_dog()
        assert dog.fu_correction_liver.mean == pytest.approx(1.0)
        assert dog.fu_correction_liver.cv == pytest.approx(0.0)

    def test_explicit_value_preserved(self):
        dog = _minimal_dog(fu_correction_liver=Distribution(mean=8.5, cv=0.5))
        assert dog.fu_correction_liver.mean == pytest.approx(8.5)
        assert dog.fu_correction_liver.cv == pytest.approx(0.5)

    def test_realize_means_carries_value(self):
        """realize_means() returns the central tendency as cv=0 Distribution."""
        dog = _minimal_dog(fu_correction_liver=Distribution(mean=8.5, cv=0.5))
        realized = dog.realize_means()
        assert realized.fu_correction_liver.mean == pytest.approx(8.5)
        assert realized.fu_correction_liver.cv == pytest.approx(0.0)

    def test_sample_carries_value(self):
        """sample(rng) draws a stochastic point; result is cv=0 Distribution at the draw."""
        dog = _minimal_dog(fu_correction_liver=Distribution(mean=8.5, cv=0.5))
        rng = np.random.default_rng(seed=42)
        sampled = dog.sample(rng)
        assert sampled.fu_correction_liver.mean > 0.0
        assert np.isfinite(sampled.fu_correction_liver.mean)
        assert sampled.fu_correction_liver.cv == pytest.approx(0.0)

    def test_default_realize_means_is_one(self):
        dog = _minimal_dog()
        assert dog.realize_means().fu_correction_liver.mean == pytest.approx(1.0)

    def test_default_sample_is_one_when_cv_zero(self):
        dog = _minimal_dog()
        rng = np.random.default_rng(seed=42)
        assert dog.sample(rng).fu_correction_liver.mean == pytest.approx(1.0)


class TestBuildDrugOnGraphAttachesFuCorrection:
    def test_default_one_when_smiles_unregistered(self):
        """Drugs not in hepatic_fu_correction.json get the default 1.0."""
        from sisyphus.predict.adme import predict_adme
        from sisyphus.predict.chemistry import compute_profile
        from sisyphus.predict.ivive import build_drug_on_graph

        # ethanol — never in any registry
        profile = compute_profile("CCO")
        adme = predict_adme(profile)
        dog = build_drug_on_graph(profile, adme, dose_mg=100.0, route="oral")
        assert dog.fu_correction_liver.mean == pytest.approx(1.0)
        assert dog.fu_correction_liver.cv == pytest.approx(0.0)
