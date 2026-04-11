"""Amortized SBI TDM update (Phase 2.0 Track B production integration).

Wraps the multi-drug conditional NPE posterior trained in Track A as a
drop-in TDM method on the same interface as the existing importance
sampling / IBIS / EnKF code paths. One ``sbi_update`` call returns a
``TDMResult`` comparable to the other methods.

The amortized speedup is in the posterior sampling step (milliseconds
instead of minutes). Posterior predictive Cmax still requires forward
simulations — but only ``n_predictive_sims`` of them (≈ 100) instead of
the 2000 particles IBIS propagates. Typical wall time: 5–30 seconds per
query on CPU, compared with IBIS's 1200–1600 seconds on the same drug.

Prior Cmax is computed the same way the other TDM paths compute it:
draw from the drug's population ``Distribution`` prior, simulate, and
collect Cmax. This keeps ``TDMResult.prior_cmax`` comparable across
methods.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from sisyphus.core import Distribution, DrugOnGraph
from sisyphus.engine.compiler import CompiledODE, ResolvedParams
from sisyphus.graph.body import BodyGraph
from sisyphus.regimen.solver import solve_regimen
from sisyphus.regimen.types import DosingRegimen

if TYPE_CHECKING:
    from sisyphus.regimen.tdm import Observation, TDMResult

logger = logging.getLogger(__name__)

_DEFAULT_POSTERIOR_PATH = Path("models/sbi/multi_drug_nsf.pt")
_DEFAULT_AUX_SUFFIX = ".aux.pt"


def _extract_params(drug: DrugOnGraph) -> dict[str, float]:
    """Mirror of tdm._extract_params (avoided circular import)."""
    params = {"fup": float(drug.fup.mean), "peff": float(drug.peff.mean)}
    if drug.enzyme_affinity:
        params["total_enzyme_affinity"] = float(
            sum(d.mean for d in drug.enzyme_affinity.values())
        )
    return params


def sbi_update(
    compiled: CompiledODE,
    graph: BodyGraph,
    drug: DrugOnGraph,
    regimen: DosingRegimen,
    observations: tuple["Observation", ...] | list["Observation"],
    posterior_path: Path | str | None = None,
    n_samples: int = 5000,
    n_predictive_sims: int = 100,
    seed: int = 42,
    observation_node: str = "venous_blood",
    logp_hint: float | None = None,
) -> "TDMResult":
    """Amortized SBI TDM update via the multi-drug conditional posterior.

    Args:
        compiled: Pre-compiled ODE skeleton (same as other TDM paths).
        graph: BodyGraph with distributional parameters.
        drug: DrugOnGraph with distributional parameters (population prior).
        regimen: Dosing regimen the patient received.
        observations: Observed concentrations. Only the first observation is
            currently conditioned on — multi-observation amortization is a
            Phase 2.1 follow-up.
        posterior_path: Path to the trained posterior .pt file. Defaults to
            ``models/sbi/multi_drug_nsf.pt``. Expects a sibling ``.aux.pt``
            holding the feature standardizer (mean / std) produced by
            ``scripts/sbi_train_multi_drug.py``.
        n_samples: Number of theta samples drawn from the amortized posterior.
        n_predictive_sims: Number of posterior theta samples forwarded
            through the engine for the posterior-predictive Cmax distribution.
        seed: RNG seed base for the prior-Cmax and posterior-predictive loops.
        observation_node: Node name used for Cmax extraction.
        logp_hint: Optional logP value to inject into the feature vector.
            Without this the feature extractor defaults to 2.0 (sentinel).
            When called from ``bayesian_update`` with a fully built drug,
            pass ``compute_profile(drug.smiles).logp`` if available.

    Returns:
        TDMResult with prior/posterior Cmax distributions, parameter means,
        and uniform weights on the posterior samples (NPE gives exact
        samples, so the "ESS" field is set to the number of successful
        posterior-predictive simulations for API consistency).

    Raises:
        FileNotFoundError: if the posterior file is missing. The dispatcher
        in :func:`sisyphus.regimen.tdm.bayesian_update` can catch this and
        fall back to IBIS.
    """
    # Late imports to avoid cycles with tdm.py.
    from sisyphus.regimen.tdm import TDMResult
    from sisyphus.sbi.amortizer import load_result
    from sisyphus.sbi.multi_drug import extract_drug_features
    from sisyphus.sbi.simulator import apply_theta_to_drug

    observations = tuple(observations)
    if not observations:
        raise ValueError("At least one observation is required")

    ppath = Path(posterior_path) if posterior_path is not None else _DEFAULT_POSTERIOR_PATH
    if not ppath.exists():
        raise FileNotFoundError(
            f"SBI posterior file not found: {ppath}. "
            "Run scripts/sbi_train_multi_drug.py or fall back to method='ibis'."
        )

    aux_path = ppath.with_suffix(_DEFAULT_AUX_SUFFIX)
    import torch  # local import to keep optional-dep policy clean

    t0_total = time.time()

    logger.info("SBI TDM: loading posterior %s", ppath)
    result = load_result(ppath)
    feat_mean = feat_std = None
    if aux_path.exists():
        aux = torch.load(aux_path, weights_only=False)
        feat_mean = np.asarray(aux["feat_mean"], dtype=np.float64)
        feat_std = np.asarray(aux["feat_std"], dtype=np.float64)
    else:
        logger.warning(
            "SBI TDM: aux standardizer missing at %s — using raw features",
            aux_path,
        )

    feats = extract_drug_features(drug=drug, graph=graph, logp=logp_hint)
    feats_used = feats.copy()
    if feat_mean is not None:
        feats_used = (feats - feat_mean) / feat_std

    # Packed observation: [log10(first_obs_concentration), drug_features]
    # The POC and Track A amortizer both condition on a single observation.
    obs0 = observations[0]
    log10_cmax_obs = float(np.log10(max(obs0.concentration, 1e-12)))
    x_vec = np.concatenate([[log10_cmax_obs], feats_used]).astype(np.float32)
    x_t = torch.as_tensor(x_vec[None, :], dtype=torch.float32)

    t0_sample = time.time()
    samples_t = result.posterior.sample(
        (n_samples,), x=x_t, show_progress_bars=False
    )
    samples_np = samples_t.detach().cpu().numpy()  # (n_samples, 3)
    posterior_sample_time = time.time() - t0_sample
    logger.info(
        "SBI TDM: posterior sampling (%d) took %.3fs",
        n_samples,
        posterior_sample_time,
    )

    # Determine simulation window (match importance_sampling path)
    max_obs_t = max(obs.time_h for obs in observations)
    t_total = max(max_obs_t + 24.0, regimen.last_dose_time_h + 24.0)

    # ── Prior Cmax (from drug Distribution prior, same as IS/IBIS) ──
    prior_cmax_list: list[float] = []
    prior_param_list: list[dict[str, float]] = []
    n_prior_fail = 0
    for i in range(n_predictive_sims):
        rng = np.random.default_rng(seed + i)
        realized_graph = graph.sample(rng)
        realized_drug = drug.sample(rng)
        params = ResolvedParams(realized_graph, realized_drug)
        try:
            sim = solve_regimen(
                compiled, params, regimen, t_total_h=t_total, dt_output=0.25
            )
            if not sim.solver_success:
                n_prior_fail += 1
                continue
        except Exception:
            n_prior_fail += 1
            continue
        conc = sim.concentrations.get(observation_node)
        if conc is None or len(conc) == 0:
            n_prior_fail += 1
            continue
        cmax = float(np.max(conc))
        if cmax <= 0:
            n_prior_fail += 1
            continue
        prior_cmax_list.append(cmax)
        prior_param_list.append(_extract_params(realized_drug))

    # ── Posterior predictive Cmax (apply each sampled theta, simulate) ──
    # For each theta draw we *fix* the inferred ADME at the posterior mean
    # by collapsing the CV on overridden fields to zero. Theta-driven spread
    # then dominates posterior Cmax variance; graph physiology still samples.
    # This avoids double-counting variance (theta uncertainty + CV noise)
    # that was inflating posterior CV above prior CV in early tests.
    from dataclasses import replace as dc_replace

    from sisyphus.core import Distribution as _Distribution

    nominal_peff = float(drug.peff.mean)
    idx_rng = np.random.default_rng(seed + 9000)
    n_pick = min(n_predictive_sims, n_samples)
    idx = idx_rng.choice(n_samples, size=n_pick, replace=False)
    post_cmax_list: list[float] = []
    post_param_list: list[dict[str, float]] = []
    n_post_fail = 0
    for k, j in enumerate(idx):
        theta_j = samples_np[j].astype(np.float64)
        overridden = apply_theta_to_drug(drug, theta_j, nominal_peff)
        overridden = dc_replace(
            overridden,
            fup=_Distribution(mean=overridden.fup.mean, cv=0.0,
                              dist_type=overridden.fup.dist_type),
            peff=_Distribution(mean=overridden.peff.mean, cv=0.0,
                               dist_type=overridden.peff.dist_type),
            enzyme_affinity={
                tag: _Distribution(mean=d.mean, cv=0.0, dist_type=d.dist_type)
                for tag, d in overridden.enzyme_affinity.items()
            },
        )
        rng = np.random.default_rng(seed + 10000 + k)
        realized_graph = graph.sample(rng)
        realized_drug = overridden.sample(rng)
        params = ResolvedParams(realized_graph, realized_drug)
        try:
            sim = solve_regimen(
                compiled, params, regimen, t_total_h=t_total, dt_output=0.25
            )
            if not sim.solver_success:
                n_post_fail += 1
                continue
        except Exception:
            n_post_fail += 1
            continue
        conc = sim.concentrations.get(observation_node)
        if conc is None or len(conc) == 0:
            n_post_fail += 1
            continue
        cmax = float(np.max(conc))
        if cmax <= 0:
            n_post_fail += 1
            continue
        post_cmax_list.append(cmax)
        post_param_list.append(_extract_params(realized_drug))

    # ── Summarize ──
    if not prior_cmax_list or not post_cmax_list:
        logger.warning(
            "SBI TDM: insufficient successful simulations "
            "(prior=%d, post=%d of %d)",
            len(prior_cmax_list), len(post_cmax_list), n_predictive_sims,
        )
        return TDMResult(
            prior_cmax=Distribution(0.0),
            posterior_cmax=Distribution(0.0),
            n_prior=n_predictive_sims,
            n_successful=len(post_cmax_list),
            ess=float(len(post_cmax_list)),
            observations=observations,
        )

    prior_arr = np.array(prior_cmax_list)
    post_arr = np.array(post_cmax_list)
    prior_mean = float(np.mean(prior_arr))
    prior_cv = float(np.std(prior_arr) / prior_mean) if prior_mean > 0 else 0.0
    post_mean = float(np.mean(post_arr))
    post_cv = float(np.std(post_arr) / post_mean) if post_mean > 0 else 0.0

    prior_params: dict[str, float] = {}
    posterior_params: dict[str, float] = {}
    if prior_param_list and post_param_list:
        keys = set(prior_param_list[0].keys())
        for key in keys:
            prior_params[key] = float(np.mean([p[key] for p in prior_param_list]))
            posterior_params[key] = float(np.mean([p[key] for p in post_param_list]))

    # Uniform weights on posterior predictive samples (NPE gives exact samples).
    n_post = len(post_cmax_list)
    uniform_weights = np.full(n_post, 1.0 / n_post)

    total_time = time.time() - t0_total
    logger.info(
        "SBI TDM done: posterior=%.3fs, forward=%.1fs, total=%.1fs. "
        "Prior Cmax=%.4f (CV=%.0f%%) → posterior=%.4f (CV=%.0f%%), "
        "successful post=%d/%d",
        posterior_sample_time,
        total_time - posterior_sample_time,
        total_time,
        prior_mean, 100 * prior_cv,
        post_mean, 100 * post_cv,
        n_post, n_predictive_sims,
    )

    return TDMResult(
        prior_cmax=Distribution(prior_mean, cv=prior_cv),
        posterior_cmax=Distribution(post_mean, cv=post_cv),
        prior_params=prior_params,
        posterior_params=posterior_params,
        n_prior=len(prior_cmax_list),
        n_successful=n_post,
        ess=float(n_post),
        observations=observations,
        weights=uniform_weights,
        log_likelihoods=np.zeros(n_post),
    )
