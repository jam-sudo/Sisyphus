"""Registry of log-space correlation matrices for correlated abundance priors.

Each entry (keyed by ``correlation_group`` name, e.g. ``liver_achour2021``) holds
  - members: tuple of enzyme/transporter tags in a fixed order
  - cvs: per-member coefficient of variation on the raw scale
  - log_corr_matrix: NxN Pearson correlation on log-transformed per-donor data

This module provides a thread-unsafe global registry (single-process, batch use
only) and the ``sample_correlated`` function that draws correlated lognormal
variates consistent with the stored matrices.

Spec: docs/_internal/specs/2026-04-22-achour-abundance-correlation-design.md
"""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CorrelationSpec:
    """One group's correlation specification.

    All three arrays must agree in size (len(members) == len(cvs) == shape[0]
    of log_corr_matrix). log_corr_matrix must be symmetric PSD with unit diagonal.
    """

    members: tuple[str, ...]
    log_corr_matrix: np.ndarray
    cvs: np.ndarray

    def __post_init__(self) -> None:
        n = len(self.members)
        if self.log_corr_matrix.shape != (n, n):
            raise ValueError(
                f"log_corr_matrix shape {self.log_corr_matrix.shape} "
                f"does not match members count {n}"
            )
        if len(self.cvs) != n:
            raise ValueError(f"cvs length {len(self.cvs)} does not match members count {n}")


_REGISTRY: dict[str, CorrelationSpec] = {}


def register(name: str, spec: CorrelationSpec) -> None:
    """Register a correlation group. Overwrites existing entry with the same name."""
    _REGISTRY[name] = spec


def get(name: str) -> CorrelationSpec | None:
    """Return the correlation spec for ``name`` or None if not registered."""
    return _REGISTRY.get(name)


def load_from_json(path: pathlib.Path) -> None:
    """Load a JSON file and register the correlation group it defines.

    Expected JSON schema (see data/physiology/achour2021_correlation.json):
        {
          "name": "liver_achour2021",
          "members": ["CYP3A4", ...],
          "cv": [0.763, ...],
          "log_corr_matrix": [[1.0, ...], ...]
        }
    """
    with pathlib.Path(path).open() as f:
        data = json.load(f)
    name = data["name"]
    members = tuple(data["members"])
    cvs = np.asarray(data["cv"], dtype=float)
    matrix = np.asarray(data["log_corr_matrix"], dtype=float)
    register(name, CorrelationSpec(members=members, log_corr_matrix=matrix, cvs=cvs))


def assert_sampled(graph) -> None:  # type: BodyGraph; avoid circular import at module scope
    """Fail loudly if any Distribution in the graph still carries a
    non-None correlation_group (i.e., sampling was intended but forgotten).

    The contract: ``_resample_correlated_abundances`` replaces every grouped
    Distribution with a collapsed ``Distribution(mean=sampled, cv=0,
    correlation_group=None)``. If this check fails after a call path that
    should have sampled, an ``rng=`` argument was omitted.

    Raises:
        AssertionError: if any node has a Distribution with
            ``correlation_group is not None``.
    """
    for node_name, node in graph.nodes.items():
        for tag, dist in node.enzymes.items():
            if dist.correlation_group is not None:
                raise AssertionError(
                    f"Node {node_name!r} enzyme {tag!r} still has "
                    f"correlation_group={dist.correlation_group!r}. "
                    f"Caller forgot to pass rng= to generate_physiology?"
                )
        for tag, dist in node.transporters.items():
            if dist.correlation_group is not None:
                raise AssertionError(
                    f"Node {node_name!r} transporter {tag!r} still has "
                    f"correlation_group={dist.correlation_group!r}. "
                    f"Caller forgot to pass rng= to generate_physiology?"
                )


def sample_correlated(
    means: np.ndarray,
    cvs: np.ndarray,
    log_corr: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw one sample of N correlated lognormal variates.

    For each i:  log(X_i) ~ Normal(mu_i, sigma_i)  where
        sigma_i^2 = log(1 + cv_i^2)
        mu_i     = log(mean_i) - sigma_i^2 / 2

    Joint structure: corr(log X_i, log X_j) = log_corr[i,j].

    The mean and CV of X_i on the raw scale reproduce ``means[i]``/``cvs[i]``
    exactly in expectation (subject to finite-sample noise in any empirical
    check).

    Args:
        means: shape (N,), strictly positive.
        cvs: shape (N,), non-negative.
        log_corr: shape (N, N), symmetric PSD with unit diagonal.
        rng: numpy random Generator.

    Returns:
        A numpy array of shape (N,) with one draw for each variable.
    """
    means = np.asarray(means, dtype=float)
    cvs = np.asarray(cvs, dtype=float)
    log_corr = np.asarray(log_corr, dtype=float)

    if (means <= 0).any():
        raise ValueError("sample_correlated requires strictly positive means")
    if (cvs < 0).any():
        raise ValueError("sample_correlated requires non-negative cvs")

    sigmas = np.sqrt(np.log1p(cvs ** 2))
    mus = np.log(means) - 0.5 * sigmas ** 2

    cov = log_corr * np.outer(sigmas, sigmas)
    z = rng.multivariate_normal(mean=np.zeros(len(means)), cov=cov)
    return np.exp(mus + z)
