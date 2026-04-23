"""Registry of log-space correlation matrices for correlated abundance priors.

Each entry (keyed by ``correlation_group`` name, e.g. ``liver_achour2021``) holds
  - members: tuple of enzyme/transporter tags in a fixed order
  - cvs: per-member coefficient of variation on the raw scale
  - log_corr_matrix: NxN Pearson correlation on log-transformed per-donor data

This module provides a thread-unsafe global registry (single-process, batch use
only) and the ``sample_correlated`` function that draws correlated lognormal
variates consistent with the stored matrices.

Spec: docs/superpowers/specs/2026-04-22-achour-abundance-correlation-design.md
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
