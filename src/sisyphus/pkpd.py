"""PK/PD link -- effect compartment and Emax response.

Computes pharmacodynamic effects from plasma concentration time course.
Uses analytical convolution for effect-site equilibration, not ODE.
This keeps the engine identity-blind and avoids adding PD-specific
flux types.

The effect compartment equilibrates with plasma via ke0::

    dCe/dt = ke0 * (Cp - Ce)

Solved exactly per time step::

    Ce(t+dt) = Ce(t) * exp(-ke0*dt) + Cp(t) * (1 - exp(-ke0*dt))

The pharmacodynamic effect uses a sigmoid Emax model::

    E = baseline + Emax * Ce^n / (EC50^n + Ce^n)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from sisyphus.core import Distribution, SimResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PD model specification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PDModel:
    """Pharmacodynamic model specification.

    Attributes:
        ke0: Effect-site equilibration rate constant (h^-1).
            Controls delay between plasma and effect-site concentrations.
            Typical range: 0.1-10 h^-1.
        emax: Maximum effect (arbitrary units, 0-100%).
        ec50: Concentration producing 50% of Emax (mg/L).
        hill: Hill coefficient (sigmoidicity). 1.0 = standard Emax.
        baseline: Baseline effect (at zero concentration).
    """

    ke0: float  # h^-1
    emax: float = 100.0  # max effect (%)
    ec50: float = 1.0  # mg/L
    hill: float = 1.0  # sigmoidicity
    baseline: float = 0.0  # baseline effect

    def __post_init__(self) -> None:
        if self.ke0 <= 0:
            raise ValueError(f"ke0 must be positive, got {self.ke0}")
        if self.ec50 <= 0:
            raise ValueError(f"ec50 must be positive, got {self.ec50}")
        if self.hill <= 0:
            raise ValueError(f"hill must be positive, got {self.hill}")


# ---------------------------------------------------------------------------
# PD result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PDResult:
    """PK/PD simulation result.

    Attributes:
        time_h: Time points (h).
        cp: Plasma concentration (mg/L).
        ce: Effect-site concentration (mg/L).
        effect: Pharmacodynamic effect (0-Emax + baseline).
        emax_achieved: Peak effect.
        temax: Time of peak effect (h).
    """

    time_h: NDArray[np.float64]
    cp: NDArray[np.float64]
    ce: NDArray[np.float64]
    effect: NDArray[np.float64]
    emax_achieved: float
    temax: float


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def _compute_effect_site(
    time: NDArray[np.float64],
    cp: NDArray[np.float64],
    ke0: float,
) -> NDArray[np.float64]:
    """Compute effect-site concentration via exact discrete convolution.

    Solves dCe/dt = ke0 * (Cp - Ce) stepwise using the exact solution
    for each interval (piecewise-constant Cp assumption)::

        Ce(t+dt) = Ce(t) * exp(-ke0*dt) + Cp(t) * (1 - exp(-ke0*dt))

    Args:
        time: Time points (h), monotonically increasing.
        cp: Plasma concentration at each time point (mg/L).
        ke0: Effect-site equilibration rate constant (h^-1).

    Returns:
        Effect-site concentration array, same shape as cp.
    """
    ce = np.zeros_like(cp)
    for i in range(1, len(time)):
        dt = time[i] - time[i - 1]
        decay = np.exp(-ke0 * dt)
        ce[i] = ce[i - 1] * decay + cp[i - 1] * (1.0 - decay)
    return ce


def _sigmoid_emax(
    ce: NDArray[np.float64],
    emax: float,
    ec50: float,
    hill: float,
    baseline: float,
) -> NDArray[np.float64]:
    """Apply sigmoid Emax model to effect-site concentrations.

    E = baseline + Emax * Ce^n / (EC50^n + Ce^n)

    Args:
        ce: Effect-site concentration (mg/L).
        emax: Maximum effect.
        ec50: Concentration at 50% Emax (mg/L).
        hill: Hill coefficient.
        baseline: Baseline effect.

    Returns:
        Effect array, same shape as ce.
    """
    ce_pos = np.maximum(ce, 0.0)
    ce_n = ce_pos**hill
    ec50_n = ec50**hill
    return baseline + emax * ce_n / (ec50_n + ce_n)


def compute_effect(
    sim_result: SimResult,
    pd_model: PDModel,
    observation_node: str = "venous_blood",
) -> PDResult:
    """Compute PK/PD effect from simulation result.

    1. Extract plasma C(t) from SimResult
    2. Compute effect-site Ce(t) via discrete convolution with ke0
    3. Apply Emax model: E = baseline + Emax * Ce^n / (EC50^n + Ce^n)

    Args:
        sim_result: ODE simulation result.
        pd_model: PD model parameters.
        observation_node: Node for plasma concentration.

    Returns:
        PDResult with time courses and peak effect.

    Raises:
        KeyError: If observation_node not found in sim_result.
    """
    if observation_node not in sim_result.concentrations:
        raise KeyError(
            f"Observation node {observation_node!r} not found in SimResult. "
            f"Available: {sorted(sim_result.concentrations.keys())}"
        )

    time = sim_result.time_h
    cp = sim_result.concentrations[observation_node]

    # Effect-site equilibration
    ce = _compute_effect_site(time, cp, pd_model.ke0)

    # Sigmoid Emax model
    effect = _sigmoid_emax(ce, pd_model.emax, pd_model.ec50, pd_model.hill, pd_model.baseline)

    emax_achieved = float(np.max(effect))
    temax = float(time[np.argmax(effect)])

    logger.debug(
        "PK/PD: peak effect=%.2f at t=%.2f h (ke0=%.3f, EC50=%.4f, hill=%.1f)",
        emax_achieved,
        temax,
        pd_model.ke0,
        pd_model.ec50,
        pd_model.hill,
    )

    return PDResult(
        time_h=time,
        cp=cp,
        ce=ce,
        effect=effect,
        emax_achieved=emax_achieved,
        temax=temax,
    )


# ---------------------------------------------------------------------------
# Preset PD models for common drug-effect pairs
# ---------------------------------------------------------------------------

# Sources:
#   Mandema et al. (1996) J Pharmacokinet Biopharm 24:25-46 (midazolam sedation)
#   Holford (1986) Clin Pharmacokinet 11:483-504 (warfarin PD)
#   Sheiner et al. (1979) Clin Pharmacol Ther 25:358-371 (warfarin INR)

MIDAZOLAM_SEDATION = PDModel(
    ke0=0.5,  # 0.5 h^-1 (moderate equilibration delay)
    emax=100.0,  # % sedation
    ec50=0.05,  # mg/L (typical sedation EC50)
    hill=2.0,  # steep dose-response
    baseline=0.0,
)

WARFARIN_INR = PDModel(
    ke0=0.02,  # 0.02 h^-1 (very slow -- INR response takes days)
    emax=5.0,  # INR units above baseline
    ec50=1.0,  # mg/L
    hill=1.0,
    baseline=1.0,  # baseline INR = 1.0
)


# ---------------------------------------------------------------------------
# B2.0 — generic direct/pointwise concentration-response (CR) layer
# ---------------------------------------------------------------------------

from collections.abc import Callable  # noqa: E402  (grouped with the CR layer)


def _resp_linear(c, slope, intercept):
    return intercept + slope * c


def _resp_mm_excess(c, vmax, km, threshold):
    return np.maximum(0.0, vmax * c / (km + c) - threshold)


def _resp_sigmoid_emax(c, emax, ec50, hill, baseline):
    return _sigmoid_emax(c, emax, ec50, hill, baseline)


# registry key -> pure C_array->rate_array response (self-contained; no validation import)
_CR_RESPONSES: dict[str, Callable] = {
    "linear": _resp_linear,
    "mm_excess": _resp_mm_excess,
    "sigmoid_emax": _resp_sigmoid_emax,
}


@dataclass(frozen=True)
class CRSpec:
    """Config for one concentration-response evaluation (spec §2).

    node: node name(s) to read C(t) from. response: a registry key (see _CR_RESPONSES) or a
    raw ``(C_array, **params) -> rate_array`` callable. params: a dict broadcasts to all
    nodes; a list aligns per-node (len == len(node)). conc_scale: multiplies C BEFORE the
    response (set to fup for unbound-driven effect). ke0: optional effect-site first-order
    delay (None => direct C).
    """

    node: str | list[str]
    response: str | Callable
    params: dict | list[dict]
    conc_scale: float = 1.0
    ke0: float | None = None

    def __post_init__(self) -> None:
        if isinstance(self.response, str) and self.response not in _CR_RESPONSES:
            raise ValueError(
                f"unknown response {self.response!r}; known: {sorted(_CR_RESPONSES)}"
            )
        if isinstance(self.node, list) and isinstance(self.params, list):
            if len(self.params) != len(self.node):
                raise ValueError(
                    f"params list len {len(self.params)} != node list len {len(self.node)}"
                )


@dataclass(frozen=True)
class CRNodeResult:
    """Per-node CR output (all aggregations always computed)."""

    trajectory: NDArray[np.float64]
    peak: float
    tpeak: float
    nadir: float
    tnadir: float
    integral: float


def concentration_response(
    sim_result: SimResult, spec: CRSpec
) -> dict[str, CRNodeResult]:
    """Apply a configured direct/pointwise concentration-response to node concentration(s).

    Per node: C_drive = conc_scale * C; optional effect-site delay (ke0); response_fn;
    then trajectory/peak/tpeak/nadir/tnadir/integral. Pure post-processor — no engine.
    KeyError if a node is absent (mirrors compute_effect). Spec §2.
    """
    nodes = [spec.node] if isinstance(spec.node, str) else list(spec.node)
    if isinstance(spec.params, list):
        params_per_node = spec.params
    else:
        params_per_node = [spec.params] * len(nodes)
    fn = spec.response if callable(spec.response) else _CR_RESPONSES[spec.response]
    trapz = getattr(np, "trapezoid", np.trapz)

    time = np.asarray(sim_result.time_h, dtype=float)
    out: dict[str, CRNodeResult] = {}
    for name, params in zip(nodes, params_per_node):
        if name not in sim_result.concentrations:
            raise KeyError(
                f"node {name!r} not in SimResult; available: "
                f"{sorted(sim_result.concentrations)}"
            )
        c_drive = spec.conc_scale * np.asarray(sim_result.concentrations[name], dtype=float)
        driver = c_drive if spec.ke0 is None else _compute_effect_site(time, c_drive, spec.ke0)
        r = np.asarray(fn(driver, **params), dtype=float)
        out[name] = CRNodeResult(
            trajectory=r,
            peak=float(np.max(r)),
            tpeak=float(time[int(np.argmax(r))]),
            nadir=float(np.min(r)),
            tnadir=float(time[int(np.argmin(r))]),
            integral=float(trapz(r, time)),
        )
    return out


@dataclass(frozen=True)
class EndpointMC:
    """MC summary of one scalar CR endpoint (spec §2): moments + percentiles + raw samples."""

    mean: float
    std: float
    p5: float
    p50: float
    p95: float
    samples: NDArray[np.float64]  # (n_draws,) ground truth — derive any quantile


@dataclass(frozen=True)
class CRNodeMCResult:
    """Per-node CR endpoints with propagated uncertainty (spec §2)."""

    peak: EndpointMC
    tpeak: EndpointMC
    nadir: EndpointMC
    tnadir: EndpointMC
    integral: EndpointMC
    n_samples: int


_CR_ENDPOINTS = ("peak", "tpeak", "nadir", "tnadir", "integral")


def _realize_param_value(v, rng: np.random.Generator | None):
    """One param value -> float. Distribution -> .mean (rng is None) or a draw; else passthrough."""
    if isinstance(v, Distribution):
        return v.mean if rng is None else v.sample(rng)
    return v


def _realize_spec(spec: CRSpec, rng: np.random.Generator | None) -> CRSpec:
    """Realize any Distribution in params/conc_scale to floats; structure/node/response/ke0 kept.

    rng is None => realize to means (deterministic). Param dict order then conc_scale fixes the
    RNG draw order, so a given seed is reproducible (spec §3 step 1).
    """
    if isinstance(spec.params, list):
        params = [
            {k: _realize_param_value(val, rng) for k, val in d.items()} for d in spec.params
        ]
    else:
        params = {k: _realize_param_value(val, rng) for k, val in spec.params.items()}
    conc_scale = _realize_param_value(spec.conc_scale, rng)
    return CRSpec(
        node=spec.node,
        response=spec.response,
        params=params,
        conc_scale=conc_scale,
        ke0=spec.ke0,
    )


def _summarize_endpoint(arr: NDArray[np.float64]) -> EndpointMC:
    p5, p50, p95 = (float(x) for x in np.percentile(arr, [5, 50, 95]))
    return EndpointMC(
        mean=float(np.mean(arr)),
        std=float(np.std(arr)),  # population std (ddof=0), MC sample summary
        p5=p5,
        p50=p50,
        p95=p95,
        samples=arr,
    )


def _wrap_scalar_endpoint(v: float) -> EndpointMC:
    return EndpointMC(mean=v, std=0.0, p5=v, p50=v, p95=v, samples=np.array([v], dtype=float))


def concentration_response_mc(
    sim_or_ensemble: SimResult | list[SimResult],
    spec: CRSpec,
    n_samples: int = 0,
    seed: int = 0,
) -> dict[str, CRNodeMCResult]:
    """MC-propagate CR-param and concentration uncertainty to the CR endpoints (spec §2-§4).

    Two sources, one joint loop: per draw realize Distribution params/conc_scale AND sample one
    ensemble member, then reuse the deterministic ``concentration_response``. ``n_samples=0`` is
    an exact deterministic reduction to B2.0 (Distributions realized to ``.mean``). Pure
    post-processor — no engine. Headline 2.731 untouched.
    """
    if n_samples < 0:
        raise ValueError(f"n_samples must be >= 0, got {n_samples}")
    ensemble = (
        [sim_or_ensemble] if isinstance(sim_or_ensemble, SimResult) else list(sim_or_ensemble)
    )
    if not ensemble:
        raise ValueError("ensemble is empty; pass a SimResult or a non-empty list[SimResult]")

    nodes = [spec.node] if isinstance(spec.node, str) else list(spec.node)

    if n_samples == 0:
        det = concentration_response(ensemble[0], _realize_spec(spec, None))
        return {
            name: CRNodeMCResult(
                **{ep: _wrap_scalar_endpoint(getattr(det[name], ep)) for ep in _CR_ENDPOINTS},
                n_samples=0,
            )
            for name in nodes
        }

    rng = np.random.default_rng(seed)
    raw: dict[str, dict[str, list[float]]] = {
        name: {ep: [] for ep in _CR_ENDPOINTS} for name in nodes
    }
    for _ in range(n_samples):
        realized = _realize_spec(spec, rng)  # step 1: params first (fixes RNG order)
        member = ensemble[int(rng.integers(len(ensemble)))]  # step 2: one C(t)
        res = concentration_response(member, realized)
        for name in nodes:
            r = res[name]
            for ep in _CR_ENDPOINTS:
                raw[name][ep].append(getattr(r, ep))

    return {
        name: CRNodeMCResult(
            **{
                ep: _summarize_endpoint(np.asarray(raw[name][ep], dtype=float))
                for ep in _CR_ENDPOINTS
            },
            n_samples=n_samples,
        )
        for name in nodes
    }
