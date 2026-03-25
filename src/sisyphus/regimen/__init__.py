"""Regimen layer — multi-dose simulation support.

Wraps the single-dose engine solver to support repeated dosing,
IV infusions, and steady-state detection without modifying the
engine RHS.

Types:
    DosingEvent    — a single dose event (time, amount, target node)
    DosingRegimen  — ordered collection of DosingEvents with factory methods
    ConcentrationProfile — time-course with uncertainty bounds
    SteadyStateMetrics   — Css, accumulation ratio, SS detection
"""

from sisyphus.regimen.profile import ConcentrationProfile, SteadyStateMetrics
from sisyphus.regimen.tdm import Observation, TDMResult, bayesian_update
from sisyphus.regimen.types import DosingEvent, DosingRegimen

__all__ = [
    "ConcentrationProfile",
    "DosingEvent",
    "DosingRegimen",
    "Observation",
    "SteadyStateMetrics",
    "TDMResult",
    "bayesian_update",
]
