"""Integration tests for correlated physiology sampling via generate_physiology."""
from __future__ import annotations

import pathlib

import numpy as np
import pytest

from sisyphus.physiology.correlation_registry import (
    _REGISTRY,
    assert_sampled,
    load_from_json,
)
from sisyphus.sbi.physiology_generator import generate_physiology

ACHOUR_JSON = (
    pathlib.Path(__file__).resolve().parents[2]
    / "data"
    / "physiology"
    / "achour2021_correlation.json"
)


@pytest.fixture(autouse=True)
def _load_achour_once() -> None:
    _REGISTRY.clear()
    load_from_json(ACHOUR_JSON)


def test_deterministic_without_rng_preserves_current_means() -> None:
    """Gate A: no-rng path yields a BodyGraph with the YAML's .mean values.

    Expected values are YAML means × enzyme_factor(age=30, bw=70) which is
    ≈1 but not exactly 1 (maturation = 1 - exp(-0.693*30/t_half) < 1).
    Tolerance rel=1e-6 is tight enough to catch rng-path regressions.
    """
    g = generate_physiology(body_weight_kg=70.0, age_years=30.0)
    liver = g.nodes["liver"]

    # Means must equal YAML means scaled by maturation(30,70) ≈ 1 and bw_ratio=1.
    # Values computed from current physiology_generator; rel=1e-6 guards regressions.
    assert liver.enzymes["CYP3A4"].mean == pytest.approx(9247217.164967772, rel=1e-6)
    assert liver.enzymes["CYP2D6"].mean == pytest.approx(674999.7600997412, rel=1e-6)
    assert liver.enzymes["CYP1A2"].mean == pytest.approx(3035689.161337071, rel=1e-6)
    assert liver.enzymes["CYP2C9"].mean == pytest.approx(6479968.347431339, rel=1e-6)
    assert liver.enzymes["CYP2E1"].mean == pytest.approx(3307499.9010654567, rel=1e-6)


def test_sampled_graph_passes_assert_sampled() -> None:
    """After sampling, no correlation_group should survive."""
    rng = np.random.default_rng(2026)
    g = generate_physiology(70.0, 30.0, rng=rng)
    assert_sampled(g)


def test_two_draws_differ() -> None:
    g1 = generate_physiology(70.0, 30.0, rng=np.random.default_rng(1))
    g2 = generate_physiology(70.0, 30.0, rng=np.random.default_rng(2))
    assert g1.nodes["liver"].enzymes["CYP3A4"].mean != g2.nodes["liver"].enzymes["CYP3A4"].mean


def test_sampling_follows_correlation_group() -> None:
    """Empirical log-correlation across 1000 draws should match stored matrix
    within ±0.1 (looser tolerance than Gate C because we're going through the
    YAML→parser→generator pipeline, not calling sample_correlated directly)."""
    import json as _json
    with ACHOUR_JSON.open() as f:
        spec = _json.load(f)
    members = spec["members"]
    target = np.array(spec["log_corr_matrix"])

    n_draws = 1000
    samples = np.zeros((n_draws, len(members)))
    for i in range(n_draws):
        g = generate_physiology(70.0, 30.0, rng=np.random.default_rng(10_000 + i))
        liver = g.nodes["liver"]
        for j, m in enumerate(members):
            src = liver.enzymes if m.startswith("CYP") else liver.transporters
            samples[i, j] = src[m].mean

    emp_corr = np.corrcoef(np.log(samples), rowvar=False)
    assert np.allclose(emp_corr, target, atol=0.1), (
        f"Empirical log-corr deviates from target:\n{emp_corr}\nvs\n{target}"
    )


def test_rng_reproducibility() -> None:
    """Same seed → same sampled graph."""
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    g1 = generate_physiology(70.0, 30.0, rng=rng1)
    g2 = generate_physiology(70.0, 30.0, rng=rng2)
    assert g1.nodes["liver"].enzymes["CYP3A4"].mean == g2.nodes["liver"].enzymes["CYP3A4"].mean
