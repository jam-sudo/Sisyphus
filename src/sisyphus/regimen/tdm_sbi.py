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


_DEFAULT_HIERARCHICAL_POSTERIOR_PATH = Path("models/sbi/hierarchical_nsf_2k.pt")
_DEFAULT_CONTINUOUS_POSTERIOR_PATH = Path("models/sbi/continuous_hierarchical_nsf.pt")


def sbi_update(
    compiled: CompiledODE,
    graph: BodyGraph,
    drug: DrugOnGraph,
    regimen: DosingRegimen,
    observations: tuple[Observation, ...] | list[Observation],
    posterior_path: Path | str | None = None,
    n_samples: int = 5000,
    n_predictive_sims: int = 100,
    seed: int = 42,
    observation_node: str = "venous_blood",
    logp_hint: float | None = None,
    use_surrogate: bool = False,
    surrogate_model_dir: Path | str | None = None,
    surrogate_ensemble_std_threshold: float = 0.02,
    population_class: str | None = None,
    body_weight_kg: float | None = None,
    age_years: float | None = None,
    sbi_reweight: bool = False,
) -> TDMResult:
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
        use_surrogate: If ``True`` and the surrogate ensemble loads
            successfully, replace the scipy forward simulations in the
            prior and posterior-predictive loops with a vectorised
            neural-network call. Drops wall time from ~50 s to sub-second
            per drug at the cost of ~20 % mean absolute bias in Cmax
            (see ``data/validation/surrogate_production_accuracy.json``).
            Default ``False`` — conservative scipy engine fallback.
        surrogate_model_dir: Override the default ``models/surrogate``
            directory for the ensemble weights + scaler.
        population_class: If set (e.g. ``"adult"``, ``"pediatric_5y"``),
            use the hierarchical posterior that conditions on population.
            Loads ``models/sbi/hierarchical_nsf_2k.pt`` by default.
            Requires the ``.aux.pt`` to contain ``population_order``.

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

    # Track A amortizer conditions on the first observation only. Additional
    # observations are used as a post-hoc importance-reweighting likelihood:
    # each theta sample from the amortizer is re-weighted by the probability
    # of producing the non-first observations under the engine forward sim.
    # See _multi_obs_reweight below for implementation. Single-obs path is
    # unchanged.
    _extra_obs = observations[1:]

    if posterior_path is not None:
        ppath = Path(posterior_path)
    elif body_weight_kg is not None and age_years is not None:
        ppath = _DEFAULT_CONTINUOUS_POSTERIOR_PATH
    elif population_class is not None:
        ppath = _DEFAULT_HIERARCHICAL_POSTERIOR_PATH
    else:
        ppath = _DEFAULT_POSTERIOR_PATH
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
    from sisyphus.sbi.priors import PRIOR_LOW
    if len(result.prior_low) == len(PRIOR_LOW) and result.prior_low[1] > -1.0:
        raise ValueError(
            f"Posterior at {ppath} uses pre-Phase-2.0.5 absolute fup prior "
            f"(prior_low[1]={result.prior_low[1]:.3f}, expected ~{PRIOR_LOW[1]:.3f}). "
            f"Retrain the model before using SBI TDM inference."
        )
    feat_mean = feat_std = None
    aux: dict = {}
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

    # Packed observation: [log10(first_obs_concentration), drug_features, (pop_onehot or bw/age)]
    obs0 = observations[0]
    log10_cmax_obs = float(np.log10(max(obs0.concentration, 1e-12)))

    if body_weight_kg is not None and age_years is not None:
        log_bw_norm = float(np.log10(body_weight_kg / 70.0))
        log_age_norm = float(np.log10(max(age_years, 0.5)))
        if feat_mean is not None and len(feat_mean) == 14:
            feats_ext = np.concatenate([feats, [log_bw_norm, log_age_norm]])
            feats_ext_std = (feats_ext - feat_mean) / feat_std
            parts = [np.array([log10_cmax_obs]), feats_ext_std]
        else:
            parts = [np.array([log10_cmax_obs]), feats_used,
                     np.array([log_bw_norm, log_age_norm])]
    elif population_class is not None:
        pop_order = aux.get("population_order") if aux_path.exists() else None
        if pop_order is None:
            raise ValueError(
                f"population_class='{population_class}' requested but "
                f"posterior aux file has no population_order"
            )
        pop_order = list(pop_order)
        if population_class not in pop_order:
            raise ValueError(
                f"population_class='{population_class}' not in "
                f"posterior population_order={pop_order}"
            )
        pop_oh = np.zeros(len(pop_order), dtype=np.float64)
        pop_oh[pop_order.index(population_class)] = 1.0
        parts = [np.array([log10_cmax_obs]), feats_used, pop_oh]
    else:
        parts = [np.array([log10_cmax_obs]), feats_used]
    x_vec = np.concatenate(parts).astype(np.float32)
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

    # ── Optional: load surrogate ensemble for fast forward sims ──
    # D1 follow-up (2026-04-10): the nominal-feature OOD guard was not
    # enough — when the amortizer posterior shifts a parameter (e.g.
    # clozapine fup 0.03 → 0.6) individual per-sample features can land
    # outside the training box even though the *nominal* drug is safe.
    # The surrogate bundle is now always loaded when ``use_surrogate`` is
    # requested, and the hybrid batch wrapper ``_hybrid_cmax_batch``
    # routes each sample independently: in-distribution rows go through
    # the vectorised surrogate call, out-of-distribution rows fall back
    # to a scipy forward sim. This preserves most of the speedup on
    # well-behaved drugs while restoring accuracy on the hard ones.
    if _extra_obs and use_surrogate:
        logger.warning(
            "SBI TDM: multi-obs reweighting is not yet supported under the "
            "surrogate path. Falling back to scipy forward sims for accurate "
            "multi-observation conditioning."
        )
        use_surrogate = False

    surrogate_bundle: tuple | None = None
    if use_surrogate:
        try:
            from pathlib import Path as _Path

            from sisyphus.ml.surrogate import (
                features_in_distribution,
                load_surrogate_ensemble,
                params_to_features_single,
                predict_surrogate_ensemble,
            )

            sdir = _Path(surrogate_model_dir) if surrogate_model_dir is not None \
                else _Path("models/surrogate")
            if not sdir.exists():
                raise FileNotFoundError(f"surrogate dir {sdir} missing")
            s_models = load_surrogate_ensemble(sdir, n_ensemble=5)
            scaler = np.load(sdir / "scaler.npz")
            s_mean = scaler["mean"]
            s_std = scaler["std"]

            # Cheap nominal-feature sanity check — if the *drug itself*
            # is outside training bounds there is no point enabling the
            # surrogate at all (the OOD rate will be 100 %).
            sample_params = ResolvedParams(
                graph.sample(np.random.default_rng(seed)),
                drug.sample(np.random.default_rng(seed)),
            )
            nominal_feats = params_to_features_single(sample_params)
            ok, offenders = features_in_distribution(nominal_feats)
            if not ok:
                logger.warning(
                    "SBI TDM: surrogate OOD for this drug at nominal (%s), "
                    "falling back to pure scipy forward sims",
                    ", ".join(offenders[:3]),
                )
            else:
                surrogate_bundle = (
                    s_models, s_mean, s_std,
                    params_to_features_single,
                    predict_surrogate_ensemble,
                    features_in_distribution,
                )
                logger.info("SBI TDM: surrogate forward-sim enabled (hybrid)")
        except Exception as exc:
            logger.warning(
                "SBI TDM: surrogate load failed (%s), falling back to scipy",
                exc,
            )

    def _scipy_cmax(params: ResolvedParams) -> float:
        """Scipy engine single-sample forward sim (fallback path)."""
        try:
            sim = solve_regimen(
                compiled, params, regimen, t_total_h=t_total, dt_output=0.25
            )
            if not sim.solver_success:
                return float("nan")
            conc = sim.concentrations.get(observation_node)
            if conc is None or len(conc) == 0:
                return float("nan")
            cmax = float(np.max(conc))
            return cmax if cmax > 0 else float("nan")
        except Exception:
            return float("nan")

    def _scipy_cmax_and_obs_conc(params: ResolvedParams) -> tuple[float, np.ndarray]:
        """Return (cmax, conc_at_obs_times) for multi-obs reweighting.

        conc_at_obs_times has one entry per observation, including the first.
        Used to compute the importance weight for observations beyond the
        first when multi-obs reweighting is enabled.
        """
        try:
            sim = solve_regimen(
                compiled, params, regimen, t_total_h=t_total, dt_output=0.25
            )
            if not sim.solver_success:
                return float("nan"), np.full(len(observations), np.nan)
            conc = sim.concentrations.get(observation_node)
            if conc is None or len(conc) == 0:
                return float("nan"), np.full(len(observations), np.nan)
            cmax = float(np.max(conc))
            if cmax <= 0:
                return float("nan"), np.full(len(observations), np.nan)
            t_axis = sim.time_h
            obs_conc = np.array(
                [float(np.interp(o.time_h, t_axis, conc)) for o in observations]
            )
            return cmax, obs_conc
        except Exception:
            return float("nan"), np.full(len(observations), np.nan)

    def _hybrid_cmax_batch(params_list: list, label: str = "") -> tuple[np.ndarray, int, int]:
        """Per-sample hybrid routing for surrogate vs scipy.

        Two-stage acceptance:

        1. **Training-box guard** (``features_in_distribution``) — rejects
           samples whose features fall outside the training parameter
           ranges (log10_clint, fup, logp, etc.).
        2. **Ensemble-std guard** — runs the surrogate ensemble on the
           candidate rows and rejects any whose ensemble std exceeds
           ``surrogate_ensemble_std_threshold`` (default 0.02). This
           catches the subtle failure mode where features are technically
           in the training box but the surrogate's local response surface
           is poorly calibrated (see docs/surrogate_ood_fix.md — clozapine
           under theta shift was the motivating case).

        Rejected samples fall back to scipy forward sims for accuracy.
        Accepted samples use the vectorised surrogate prediction.

        Returns ``(cmax, n_surrogate, n_scipy)``.
        """
        if surrogate_bundle is None:
            raise RuntimeError("surrogate_bundle unavailable")
        (s_models, s_mean, s_std, _p2f, _predict, _in_dist) = surrogate_bundle

        n = len(params_list)
        cmax = np.full(n, np.nan, dtype=np.float64)
        feats = np.stack([_p2f(p) for p in params_list])

        # Stage 1: training-box OOD check.
        box_mask = np.zeros(n, dtype=bool)
        for i in range(n):
            ok_i, _ = _in_dist(feats[i])
            box_mask[i] = ok_i

        # Stage 2: run the surrogate on box-accepted rows and filter
        # by ensemble std.
        candidate_mask = box_mask.copy()
        if candidate_mask.any():
            feats_cand = feats[candidate_mask]
            doses_cand = np.array(
                [params_list[i].drug_param("dose_mg")
                 for i in np.where(candidate_mask)[0]]
            )
            mean_pred, std_pred = _predict(s_models, feats_cand, s_mean, s_std)
            std_ok = std_pred <= surrogate_ensemble_std_threshold

            # Build the final surrogate acceptance mask over all n rows.
            cand_indices = np.where(candidate_mask)[0]
            accept_indices = cand_indices[std_ok]
            reject_std_indices = cand_indices[~std_ok]

            # Write surrogate predictions for accepted rows.
            if accept_indices.size > 0:
                accept_pred = mean_pred[std_ok]
                accept_doses = doses_cand[std_ok]
                cmax_surrogate = 10 ** accept_pred * accept_doses
                cmax_surrogate = np.where(
                    np.isfinite(cmax_surrogate) & (cmax_surrogate > 0),
                    cmax_surrogate,
                    np.nan,
                )
                cmax[accept_indices] = cmax_surrogate

            # Mark rejected rows (high std) for scipy fallback.
            candidate_mask[reject_std_indices] = False

        n_surrogate = int(candidate_mask.sum())
        scipy_mask = ~candidate_mask
        n_scipy = int(scipy_mask.sum())

        if n_scipy > 0:
            scipy_indices = np.where(scipy_mask)[0]
            for idx in scipy_indices:
                cmax[idx] = _scipy_cmax(params_list[idx])

        if label:
            n_box = int((~box_mask).sum())
            n_std = n_scipy - n_box
            logger.info(
                "SBI TDM %s: %d surrogate + %d scipy (%d box-OOD, %d std-OOD)",
                label, n_surrogate, n_scipy, n_box, n_std,
            )
        return cmax, n_surrogate, n_scipy

    # ── Prior Cmax (from drug Distribution prior, same as IS/IBIS) ──
    prior_cmax_list: list[float] = []
    prior_param_list: list[dict[str, float]] = []
    n_prior_fail = 0
    prior_n_surrogate = prior_n_scipy = 0
    if surrogate_bundle is not None:
        # Pre-realize all prior draws, then hybrid-route per sample.
        all_prior_params = []
        all_prior_drugs = []
        for i in range(n_predictive_sims):
            rng = np.random.default_rng(seed + i)
            realized_graph = graph.sample(rng)
            realized_drug = drug.sample(rng)
            all_prior_params.append(ResolvedParams(realized_graph, realized_drug))
            all_prior_drugs.append(realized_drug)
        cmax_arr, prior_n_surrogate, prior_n_scipy = _hybrid_cmax_batch(
            all_prior_params, label="prior"
        )
        for cmax, rd in zip(cmax_arr, all_prior_drugs):
            if not np.isfinite(cmax) or cmax <= 0:
                n_prior_fail += 1
                continue
            prior_cmax_list.append(float(cmax))
            prior_param_list.append(_extract_params(rd))
    else:
        for i in range(n_predictive_sims):
            rng = np.random.default_rng(seed + i)
            realized_graph = graph.sample(rng)
            realized_drug = drug.sample(rng)
            params = ResolvedParams(realized_graph, realized_drug)
            cmax = _scipy_cmax(params)
            if not np.isfinite(cmax) or cmax <= 0:
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
    post_n_surrogate = post_n_scipy = 0
    if surrogate_bundle is not None:
        all_post_params = []
        all_post_drugs = []
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
            all_post_params.append(ResolvedParams(realized_graph, realized_drug))
            all_post_drugs.append(realized_drug)
        cmax_arr, post_n_surrogate, post_n_scipy = _hybrid_cmax_batch(
            all_post_params, label="posterior"
        )
        for cmax, rd in zip(cmax_arr, all_post_drugs):
            if not np.isfinite(cmax) or cmax <= 0:
                n_post_fail += 1
                continue
            post_cmax_list.append(float(cmax))
            post_param_list.append(_extract_params(rd))
    else:
        post_log_weights: list[float] = []
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
            cmax, obs_conc = _scipy_cmax_and_obs_conc(params)
            if not np.isfinite(cmax) or cmax <= 0:
                n_post_fail += 1
                continue
            post_cmax_list.append(float(cmax))
            post_param_list.append(_extract_params(realized_drug))

            # Reweighting: importance weights under log-normal assay noise
            # (matches IS observation model). ``sbi_reweight`` extends this
            # to the first observation too — turning SBI into an IS with
            # NPE as the proposal distribution. Used to mitigate wide
            # posteriors from non-identifiable theta-space (e.g. morphine).
            if _extra_obs or sbi_reweight:
                log_w = 0.0
                start_idx = 0 if sbi_reweight else 1
                for oi, obs in enumerate(observations[start_idx:], start=start_idx):
                    pred = obs_conc[oi]
                    if not np.isfinite(pred) or pred <= 0:
                        log_w = -np.inf
                        break
                    sigma = max(obs.cv, 1e-3)
                    log_w += -0.5 * (
                        (np.log(obs.concentration) - np.log(pred)) / sigma
                    ) ** 2
                post_log_weights.append(float(log_w))
            else:
                post_log_weights.append(0.0)

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

    # Compute posterior weights. Without multi-obs reweighting, weights are
    # uniform (NPE gives exact samples conditioned on first obs). With extra
    # observations, we log-normalize the likelihood weights via log-sum-exp.
    n_post = len(post_cmax_list)
    multi_obs_used = False
    if (_extra_obs or sbi_reweight) and surrogate_bundle is None and 'post_log_weights' in dir() and post_log_weights:  # noqa: E501
        lw = np.array(post_log_weights[:n_post])
        finite_mask = np.isfinite(lw)
        if finite_mask.sum() > 0:
            lw_max = lw[finite_mask].max()
            w = np.where(finite_mask, np.exp(lw - lw_max), 0.0)
            w_sum = w.sum()
            if w_sum > 0:
                weights = w / w_sum
                multi_obs_used = True
            else:
                weights = np.full(n_post, 1.0 / n_post)
        else:
            weights = np.full(n_post, 1.0 / n_post)
    else:
        weights = np.full(n_post, 1.0 / n_post)

    if multi_obs_used:
        post_mean = float(np.sum(weights * post_arr))
        post_var = float(np.sum(weights * (post_arr - post_mean) ** 2))
        post_cv = float(np.sqrt(max(post_var, 0)) / post_mean) if post_mean > 0 else 0.0
        # Weighted quantile for CI
        order = np.argsort(post_arr)
        sorted_vals = post_arr[order]
        sorted_w = weights[order]
        cum = np.cumsum(sorted_w)
        ci_lo = float(sorted_vals[np.searchsorted(cum, 0.05)])
        ci_hi_idx = int(np.searchsorted(cum, 0.95))
        ci_hi_idx = min(ci_hi_idx, len(sorted_vals) - 1)
        ci_hi = float(sorted_vals[ci_hi_idx])
        ci90 = (ci_lo, ci_hi)
        ess = float(1.0 / np.sum(weights ** 2))
        logger.info(
            "SBI TDM: multi-obs reweighting applied (%d extra obs), ESS=%.1f/%d",
            len(_extra_obs), ess, n_post,
        )
    else:
        post_mean = float(np.mean(post_arr))
        post_cv = float(np.std(post_arr) / post_mean) if post_mean > 0 else 0.0
        ci90 = (
            float(np.quantile(post_arr, 0.05)),
            float(np.quantile(post_arr, 0.95)),
        )
        ess = float(n_post)

    prior_mean = float(np.mean(prior_arr))
    prior_cv = float(np.std(prior_arr) / prior_mean) if prior_mean > 0 else 0.0

    prior_params: dict[str, float] = {}
    posterior_params: dict[str, float] = {}
    if prior_param_list and post_param_list:
        keys = set(prior_param_list[0].keys())
        for key in keys:
            prior_params[key] = float(np.mean([p[key] for p in prior_param_list]))
            # Use weighted mean when multi-obs reweighting is active
            if multi_obs_used:
                posterior_params[key] = float(
                    np.sum(weights * np.array([p[key] for p in post_param_list]))
                )
            else:
                posterior_params[key] = float(
                    np.mean([p[key] for p in post_param_list])
                )

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
        ess=ess,
        observations=observations,
        weights=weights,
        log_likelihoods=np.zeros(n_post),
        cmax_ci_90=ci90,
    )
