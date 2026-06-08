"""Pipeline orchestrator -- SMILES -> PredictionResult.

Thin coordination layer that wires predict, engine, ml, and pk
together.  All logic lives in the sub-layers; this module only
calls them in the right order and combines results.
"""

from __future__ import annotations

import functools
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from sisyphus.core import Distribution, DrugOnGraph, PKEndpoints, PredictionResult
from sisyphus.engine.contracts import FuCorrectionContractError

if TYPE_CHECKING:
    from sisyphus.predict.adme import MeasuredADMEInput

logger = logging.getLogger(__name__)

_CONFORMAL_CALIBRATION = Path("data/validation/conformal_calibration.json")


@functools.lru_cache(maxsize=1)
def _conformal_q90_meta() -> float | None:
    """Train-calibrated split-conformal 90% half-width (log10) for the meta track.

    The user-facing 90% Cmax PI is the conformal interval (holdout-validated ~0.95
    coverage at nominal 0.90), superseding the parameter-only MC interval (~0.30).
    Calibrated on the TRAIN set (Invariant #5: never the holdout). Returns None if
    the calibration artifact is absent/unreadable (the pipeline then falls back to
    the MC interval, or None). See validation/conformal.py + scripts/calibrate_conformal.py.
    """
    try:
        art = json.loads(_CONFORMAL_CALIBRATION.read_text())
        q = float(art["tracks"]["meta"]["0.1"])
        return q if np.isfinite(q) else None
    except Exception:
        return None


def _resolve_observation_node(drug: DrugOnGraph, base_node: str = "venous_blood") -> str:
    """Resolve which graph node to read PK from, accounting for active species.

    Returns ``base_node + ACTIVE_SUFFIX`` if the drug has an active metabolite
    AND ``observation_species == "active"``; otherwise returns ``base_node``.
    """
    from sisyphus.graph.builder import ACTIVE_SUFFIX
    if drug.active_metabolite is not None and drug.observation_species == "active":
        return base_node + ACTIVE_SUFFIX
    return base_node


def _adjust_ad_for_prodrug(
    drug: DrugOnGraph, ad_flags: list[str]
) -> tuple[bool, list[str]]:
    """Adjust applicability-domain interpretation for prodrugs.

    - PRODRUG flag + active_metabolite present -> in_domain=True (PRODRUG removed),
      warn "routed via activation".
    - PRODRUG flag + no active_metabolite -> in_domain=False (existing behavior).
    - No PRODRUG flag + active_metabolite present -> in_domain=True (other flags
      may still flip it), warn "non-structural activation".
    - Otherwise -> flags drive in_domain (any flag -> False).

    Returns ``(in_applicability_domain, warnings_list)``.
    """
    warnings: list[str] = []
    flags_for_domain = list(ad_flags)
    has_prodrug = "PRODRUG" in ad_flags
    has_active = drug.active_metabolite is not None

    if has_prodrug and has_active:
        site = drug.active_metabolite.conversion_site
        warnings.append(
            f"Prodrug {drug.name!r} routed via activation to "
            f"{drug.active_metabolite.name!r} at {site}."
        )
        flags_for_domain = [f for f in ad_flags if f != "PRODRUG"]
    elif has_active and not has_prodrug:
        warnings.append(
            f"Active metabolite declared for {drug.name!r} without "
            "structural prodrug motif; registry override applied."
        )

    in_domain = len(flags_for_domain) == 0
    return in_domain, warnings


# ── Measured-F routing (exposure-scaling) ────────────────────────────────
# F (oral bioavailability) is emergent in the engine (F = fa*Fg*Fh); there is no
# F input to set. A caller-supplied measured F sets the systemic exposure SCALE:
# compute the engine's own oral F via an IV-reference solve and scale engine
# Cmax/AUC by F_measured/F_engine. Pipeline-layer only (engine stays identity-
# blind). See docs/superpowers/specs/2026-06-03-measured-f-routing-design.md.
_F_K_MIN = 0.05  # clamp bounds on the F-correction factor k, to bound numerical
_F_K_MAX = 50.0  # absurdity when the engine catastrophically mis-calls F.


def _engine_oral_bioavailability(
    compiled, params, drug: DrugOnGraph, oral_auc: float, observation_node: str
) -> float | None:
    """Engine's emergent oral F = oral AUC / IV-reference AUC (matched dose).

    Both AUCs are the 0-24h truncated AUC, so clearance cancels only
    approximately (exactly at infinite time) — F_engine carries a mild truncation
    bias for drugs whose t1/2 approaches the 24h window, so the reported F_engine
    is a 24h-truncated F, not a pure fa*Fg*Fh structural fraction. The IV
    reference uses the SAME compiled graph and params as the oral solve, so the
    exposure-scaling stays self-consistent and target-hitting (corrected oral
    AUC / IV AUC == F_measured) is exact regardless. Returns None if the reference
    solve fails or an AUC is non-positive (caller then skips correction).
    """
    import numpy as np

    from sisyphus.engine.solver import _IV_CMAX_DELAY_H, solve
    from sisyphus.pk.endpoints import compute_endpoints

    iv_idx = compiled.state_index.get("venous_blood")
    if iv_idx is None or oral_auc <= 0:
        return None
    y0 = np.zeros(compiled.n_states)
    y0[iv_idx] = drug.dose_mg
    iv_sim = solve(compiled, params, y0, t_span=(0, 24), t_min_h=_IV_CMAX_DELAY_H)
    if not iv_sim.solver_success:
        return None
    iv_pk = compute_endpoints(
        iv_sim, observation_node=observation_node, t_min_h=_IV_CMAX_DELAY_H
    )
    iv_auc = iv_pk.auc_0t.mean
    if iv_auc <= 0:
        return None
    return oral_auc / iv_auc


def _apply_measured_f(
    engine_pk: PKEndpoints, f_engine: float, f_measured: float, f_cv: float
) -> tuple[PKEndpoints, float, bool]:
    """Scale engine Cmax/AUC by k = F_measured/F_engine (clamped).

    f_cv (measurement CV of F) is folded into the Cmax/AUC CV in quadrature.
    Tmax / t_half / cl / vss are unchanged (F sets exposure scale, not shape).
    Returns ``(scaled_pk, k, was_clamped)``.
    """
    import dataclasses
    import math

    k_raw = f_measured / f_engine
    k = min(max(k_raw, _F_K_MIN), _F_K_MAX)
    clamped = not math.isclose(k, k_raw, rel_tol=1e-9)

    def _scale(dist: Distribution | None) -> Distribution | None:
        if dist is None:
            return None
        return Distribution(mean=dist.mean * k, cv=math.sqrt(dist.cv ** 2 + f_cv ** 2))

    scaled = dataclasses.replace(
        engine_pk,
        cmax=_scale(engine_pk.cmax),
        auc_0t=_scale(engine_pk.auc_0t),
        auc_0inf=_scale(engine_pk.auc_0inf),
    )
    return scaled, k, clamped


# Resolve physiology YAML relative to repository root.
# src/sisyphus/pipeline/predict.py -> ../../../../data/physiology
_PHYSIOLOGY_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "physiology"

def predict(
    smiles: str,
    dose_mg: float,
    route: str = "oral",
    n_mc_samples: int = 0,
    infusion_duration_min: float | None = None,
    *,
    phenotypes: dict[str, str] | None = None,
    phenotype_scale_overrides: dict[str, float] | None = None,
    kp_method: str = "rodgers_rowland",
    measured_adme: MeasuredADMEInput | None = None,
) -> PredictionResult:
    """End-to-end prediction: SMILES -> PredictionResult.

    Pipeline:
    1. predict layer: SMILES -> MolecularProfile -> ADMEProperties -> DrugOnGraph
    2. engine layer: DrugOnGraph + BodyGraph -> SimResult (with MC)
    3. pk layer: SimResult -> PKEndpoints
    4. ml layer: SMILES -> direct PKEndpoints
    5. meta-learner: combine engine + ML -> final PKEndpoints

    Args:
        smiles: Input SMILES string.
        dose_mg: Dose in mg.
        route: Administration route (``"oral"`` or ``"iv"``).
        n_mc_samples: Number of Monte Carlo samples for uncertainty
            propagation.  When > 0, runs MC and computes 90% PI
            for Cmax.  0 (default) skips MC for speed.
        infusion_duration_min: IV infusion duration in minutes. Only valid
            for route='iv'. When > 0.5, routes the deterministic simulation
            through ``regimen.solver`` (zero-order input) instead of the
            bolus y0 shortcut. ``None`` or <= 0.5 falls back to V3 bolus.
            MC for infusion is V3.1 Phase 2 scope and is skipped with a
            warning in Phase 1. See
            docs/superpowers/specs/2026-04-22-v3.1-iv-infusion-design.md.
        phenotypes: Optional CPIC phenotype assignments mapping enzyme or
            transporter gene tag (e.g., ``"CYP2D6"``, ``"SLCO1B1"``) to
            phenotype label (``"PM"``/``"IM"``/``"EM"``/``"NM"``/``"UM"``/
            ``"RM"``). When provided, the body graph's enzyme and
            transporter abundances at the liver node are scaled by the
            corresponding CPIC activity multipliers before simulation.
            Liver-sourced enzyme abundances passed to IVIVE inherit the
            scaling, so CLint and downstream PK endpoints are
            phenotype-conditional. ``None`` (default) is the
            average-patient prediction. See
            ``sisyphus.predict.phenotype.PHENOTYPE_SCALES``.
        phenotype_scale_overrides: Optional ``{gene: effective_scale}`` dict
            forwarded to ``apply_phenotype_to_graph``. When provided and a
            gene matches a key in ``phenotypes``, the override replaces the
            CPIC default scale for that gene. Values are caller-supplied
            and caller-justified — Sisyphus does not endorse specific values.
        kp_method: Tissue partition coefficient (Kp) calculation method.
            One of ``"rodgers_rowland"`` (default, suitable for most
            small-molecule drugs), ``"berezhkovskiy"`` (alternative
            ionization handling, differs from Rodgers-Rowland mainly for
            bases and zwitterions), or ``"provided"`` (use values
            supplied via ``DrugOnGraph.kp_overrides`` instead of
            computing). Forwarded to ``build_drug_on_graph``; the prior
            implementation hard-coded the default and made the per-drug
            field unreachable through the public API (hardening backlog
            B3, 2026-05-02).

    Returns:
        PredictionResult with combined PK endpoints and uncertainty.

    Raises:
        ValueError: If the SMILES string is invalid, or if
            ``infusion_duration_min`` is set for a non-IV route, or if
            ``infusion_duration_min`` is negative.
    """
    if infusion_duration_min is not None:
        if route != "iv":
            raise ValueError(
                f"infusion_duration_min={infusion_duration_min!r} requires "
                f"route='iv', got route={route!r}"
            )
        if infusion_duration_min < 0:
            raise ValueError(
                f"infusion_duration_min must be non-negative, "
                f"got {infusion_duration_min}"
            )
    # Import sub-layers here to avoid circular imports and to register flux specs.
    import sisyphus.engine.flux  # noqa: F401 -- register flux specs
    from sisyphus.engine.compiler import ODECompiler, ResolvedParams
    from sisyphus.engine.solver import _IV_CMAX_DELAY_H, solve
    from sisyphus.engine.uncertainty import UncertaintyEngine
    from sisyphus.graph.builder import build_from_yaml
    from sisyphus.ml.ensemble import MetaLearner
    from sisyphus.ml.models import PKPredictor
    from sisyphus.pk.endpoints import compute_endpoints
    from sisyphus.predict.adme import predict_adme
    from sisyphus.predict.chemistry import compute_profile
    from sisyphus.predict.ivive import build_drug_on_graph
    from sisyphus.predict.non_cyp_substrates import get_non_cyp_fractions
    from sisyphus.predict.transporter_db import (
        find_oatp1b1_substrate_name,
        is_oatp_ecm_applicable,
        load_hepatic_ecm_params_for_smiles,
        load_oatp1b1_kinetics_for_smiles,
    )

    warnings_list: list[str] = []
    cmax_90ci: tuple[float, float] | None = None
    graph = None
    compiled = None
    f_correction_k: float | None = None  # measured-F exposure-scaling factor

    # ── Step 1: Chemistry + ADME ─────────────────────────────────────────
    profile = compute_profile(smiles)
    adme = predict_adme(profile)

    # ── Step 1b: measured-ADME override (additive; no-op when None) ──────
    # predict_adme() always runs first, so the ML/VDss tracks never see None.
    # When measured_adme is supplied, REBIND the `adme` name via
    # dataclasses.replace so BOTH build_drug_on_graph calls (this step's and the
    # phenotype back-solve at the liver_enzymes_pre path) consume the substituted
    # values. When None, this block is skipped and `adme` is the unmodified
    # predicted object — the SMILES-only path is bit-identical. Reported
    # measured-input results read result.engine_pk (peff/vdss overrides also move
    # the CLF/VDss meta-tracks; only the engine track is clean).
    if measured_adme is not None:
        from dataclasses import replace as _dc_replace
        _ov: dict[str, Distribution] = {}
        if measured_adme.fup is not None:
            _ov["fup"] = Distribution(mean=measured_adme.fup, cv=measured_adme.fup_cv)
        if measured_adme.clint is not None:
            _ov["clint"] = Distribution(mean=measured_adme.clint, cv=measured_adme.clint_cv)
        if measured_adme.peff is not None:
            _ov["peff"] = Distribution(mean=measured_adme.peff, cv=measured_adme.peff_cv)
        if measured_adme.vdss is not None:
            _ov["vdss"] = Distribution(mean=measured_adme.vdss, cv=measured_adme.vdss_cv)
        if measured_adme.rbp is not None:
            _ov["rbp"] = Distribution(mean=measured_adme.rbp, cv=measured_adme.rbp_cv)
        if measured_adme.solubility is not None:
            _ov["solubility"] = Distribution(
                mean=measured_adme.solubility, cv=measured_adme.solubility_cv
            )
        if _ov:
            adme = _dc_replace(adme, **_ov)
            warnings_list.append(f"measured_adme:overrides={sorted(_ov)}")
        # measured-F is oral-only (F=1 by definition for IV); warn + ignore else.
        if measured_adme.f_bioavail is not None and route != "oral":
            warnings_list.append(
                "measured_adme:f_bioavail ignored for non-oral route (F=1 for IV)"
            )

    # Auto-activate ECM (OATP1B1 saturable + ECM passive + biliary CL_int)
    # ONLY for drugs flagged ecm_applicable=true in oatp1b1.json. The flag
    # gates against the empirically-documented triple-counting bug: drugs
    # like fluvastatin (CYP2C9-dominant) and pitavastatin (no
    # metabolic_fraction entry) shipped to predict() WITHOUT this gate
    # would have XGBoost-CYP enzyme affinities running at full strength
    # plus OATP1B1 saturable plus ECM passive — triple-counting hepatic
    # clearance.
    #
    # The gate uses full InChIKey matching (spec §1.2). The cyp_clearance
    # _overrides registry is checked downstream by ivive.build_drug_on_graph
    # for the metabolic_fraction scaling (PR #22 mechanism).
    auto_oatp_kinetics = None
    auto_ecm_params = None
    if is_oatp_ecm_applicable(profile.smiles):
        auto_oatp_kinetics = load_oatp1b1_kinetics_for_smiles(profile.smiles)
        auto_ecm_params = load_hepatic_ecm_params_for_smiles(profile.smiles)
        if auto_oatp_kinetics is not None and auto_ecm_params is not None:
            substrate_name = find_oatp1b1_substrate_name(profile.smiles) or "unknown"
            warnings_list.append(f"oatp1b1:auto_ecm:{substrate_name}")
        else:
            # Flag set but registry data missing — log and disable ECM. This
            # should be caught by the schema regression test, but defend at
            # runtime too.
            auto_oatp_kinetics = None
            auto_ecm_params = None

    # Per-gene non-CYP fm registry lookup (NAT2 / UGT1A1).
    # Empty dict for non-substrates is a no-op downstream.
    non_cyp_fractions = get_non_cyp_fractions(profile.smiles)

    drug = build_drug_on_graph(
        profile, adme, dose_mg, route,
        kp_method=kp_method,
        transporter_kinetics=auto_oatp_kinetics,
        hepatic_ecm_params=auto_ecm_params,
        non_cyp_fractions=non_cyp_fractions,
    )

    # ── DrugBank enrichment tags ──────────────────────────────────────
    # NOTE: fup tag checks data availability + sanity range but does NOT
    # replicate the 5x cross-validation guard from adme.py.  This means
    # a drug whose DrugBank fup was rejected by the 5x guard will still
    # be tagged as "drugbank:fup" (~5-10% of cases).  This is an accepted
    # imprecision per spec §5.5 — gold group has minor silver contamination.
    try:
        from sisyphus.predict.drugbank import drugbank_lookup
        db = drugbank_lookup()
        canonical = profile.smiles
        if db.get_substrate_enzymes(canonical) is not None:
            warnings_list.append("drugbank:enzyme_fm")
        db_fup = db.get_fup(canonical)
        if db_fup is not None and 0.001 <= db_fup <= 1.0:
            warnings_list.append("drugbank:fup")
        if db.get_pka(canonical) is not None:
            warnings_list.append("drugbank:pka")
        if db.get_logp(canonical) is not None:
            warnings_list.append("drugbank:logp")
    except Exception:
        pass  # DrugBank tagging is advisory, never blocks pipeline

    # ── Step 2: Engine (PBPK simulation) ─────────────────────────────────
    # Route-conditional Cmax window and simulation path.
    # IV bolus skips t=0 spike (V3); IV infusion routes through
    # regimen.solver for a physically correct zero-order input (V3.1).
    # For non-IV routes t_min_h=0.0 is a no-op (V2-compatible).
    is_infusion = (
        route == "iv"
        and infusion_duration_min is not None
        and infusion_duration_min > 0.5
    )
    if is_infusion:
        infusion_duration_h = infusion_duration_min / 60.0
        t_min_h = infusion_duration_h + _IV_CMAX_DELAY_H
    else:
        t_min_h = _IV_CMAX_DELAY_H if route == "iv" else 0.0

    engine_pk: PKEndpoints | None = None
    try:
        graph = build_from_yaml(_PHYSIOLOGY_DIR / "reference_man.yaml")

        # CRITICAL (v0.3.2): snapshot pre-phenotype liver enzyme abundances.
        # `_decompose_clint` back-solves enzyme affinity from abundance, so
        # passing scaled abundances would cause phenotype scaling to cancel
        # out at engine multiplication time (the bug that silently nulled
        # all CYP/UGT/NAT phenotype effects pre-v0.3.2; SLCO1B1 escaped
        # only because OATP1B1 uses saturable MM kinetics, not back-solve).
        # We snapshot BEFORE phenotype application so affinity is computed
        # from the unscaled baseline; phenotype then propagates through the
        # engine as scaled_abundance × pre_affinity = scale × original_rate.
        liver_enzymes_pre: dict[str, float] | None = None
        if "liver" in graph.nodes and graph.nodes["liver"].enzymes:
            liver_enzymes_pre = {
                tag: dist.mean for tag, dist in graph.nodes["liver"].enzymes.items()
            }

        # Apply CPIC phenotype scaling AFTER snapshot. Engine reads scaled
        # abundances from the graph; affinity (computed below from
        # liver_enzymes_pre) carries the unscaled baseline.
        if phenotypes:
            from sisyphus.predict.phenotype import apply_phenotype_to_graph
            graph = apply_phenotype_to_graph(
                graph, phenotypes,
                phenotype_scale_overrides=phenotype_scale_overrides,
            )

        # Rebuild drug with PRE-phenotype abundances. Phenotype's effect on
        # the graph remains (scaled abundances flow into engine multiplication);
        # affinity is back-solved from unscaled abundances so the multiplication
        # propagates the scaling rather than cancelling it.
        if liver_enzymes_pre is not None:
            drug = build_drug_on_graph(
                profile, adme, dose_mg, route,
                liver_enzymes=liver_enzymes_pre,
                kp_method=kp_method,
                transporter_kinetics=auto_oatp_kinetics,
                hepatic_ecm_params=auto_ecm_params,
                non_cyp_fractions=non_cyp_fractions,
            )

        from sisyphus.graph.builder import augment_for_active_species
        graph = augment_for_active_species(graph, drug)

        from sisyphus.graph.axial import expand_axial
        graph = expand_axial(graph)  # no-op unless a parallel_tube edge is present

        # WS-2 contract guard: a curated (non-1.0) fu_correction_liver that would
        # be ENTIRELY dropped (flagged node, drop-model clearance, no honoring
        # flux) is a contract violation — fail loud rather than silently no-op.
        # No-op today (every registry value is 1.0) → headline bit-identical.
        from sisyphus.engine.contracts import assert_fu_correction_honored
        assert_fu_correction_honored(graph, drug.fu_correction_liver.mean)

        compiler = ODECompiler()
        compiled = compiler.compile(graph)

        # Deterministic mean-only realization (RNG-independent).
        # Hardening: replaced graph.sample(rng=42) → realize_means() to
        # eliminate seed-dependent RNG-order coupling. Adding a new
        # Distribution to physiology YAML no longer shifts realized values
        # for unrelated drugs. Restores 2026-04-14 baseline (Engine 3.421,
        # Meta 2.695) by removing the lognormal-stochastic artifact.
        realized_graph = graph.realize_means()
        realized_drug = drug.realize_means()
        params = ResolvedParams(realized_graph, realized_drug)

        if is_infusion:
            from sisyphus.regimen.solver import solve_regimen
            from sisyphus.regimen.types import DosingRegimen
            regimen = DosingRegimen.single_iv(
                dose_mg=drug.dose_mg, duration_h=infusion_duration_h
            )
            sim_result = solve_regimen(compiled, params, regimen, t_total_h=24.0)
        else:
            y0 = np.zeros(compiled.n_states)
            admin_idx = compiled.state_index[drug.administration_node]
            y0[admin_idx] = drug.dose_mg
            sim_result = solve(compiled, params, y0, t_span=(0, 24), t_min_h=t_min_h)

        if sim_result.solver_success:
            _obs_node = _resolve_observation_node(drug)
            engine_pk = compute_endpoints(sim_result, observation_node=_obs_node, t_min_h=t_min_h)
            logger.info(
                "Engine PK: Cmax=%.4f mg/L, Tmax=%.2f h, AUC=%.4f mg*h/L",
                engine_pk.cmax.mean,
                engine_pk.tmax.mean,
                engine_pk.auc_0t.mean,
            )
            # ── Measured-F routing (oral only; no-op when f_bioavail is None) ──
            # Scale the engine track to honor a caller-supplied oral F. Runs the
            # IV-reference solve ONLY when f_bioavail is set, so the SMILES-only
            # path stays bit-identical.
            if (
                measured_adme is not None
                and measured_adme.f_bioavail is not None
                and route == "oral"
            ):
                _f_eng = _engine_oral_bioavailability(
                    compiled, params, drug, engine_pk.auc_0t.mean, _obs_node
                )
                if _f_eng is not None and _f_eng > 0:
                    engine_pk, f_correction_k, _clamped = _apply_measured_f(
                        engine_pk, _f_eng, measured_adme.f_bioavail,
                        measured_adme.f_bioavail_cv,
                    )
                    warnings_list.append(
                        f"measured_adme:f_bioavail={measured_adme.f_bioavail} "
                        f"f_engine={_f_eng:.3f} k={f_correction_k:.2f}"
                        + (" (clamped)" if _clamped else "")
                    )
                    logger.info(
                        "Measured-F: F_engine=%.3f -> F_measured=%.3f (k=%.2f)",
                        _f_eng, measured_adme.f_bioavail, f_correction_k,
                    )
                else:
                    warnings_list.append(
                        "measured_adme:f_bioavail skipped "
                        "(engine F-reference solve unavailable)"
                    )
        else:
            warnings_list.append("ODE solver did not converge")
            logger.warning("ODE solver did not converge")
    except FuCorrectionContractError:
        # WS-2 contract violation (entirely-dropped fu_correction) is a hard
        # contract error — propagate. Genuine engine ValueErrors (compile/solve)
        # fall through to the soft ML-only fallback below.
        raise
    except Exception as e:
        warnings_list.append(f"Engine failed: {e}")
        logger.warning("Engine simulation failed: %s", e)

    # ── Step 2b: MC uncertainty propagation ────────────────────────────
    # MC via regimen.solver is V3.1 Phase 2 scope. In Phase 1 we skip MC
    # for infusion rather than emit a misleading bolus-centered 90% PI.
    if n_mc_samples > 0 and compiled is not None and graph is not None:
        if is_infusion:
            warnings_list.append("MC skipped for IV infusion (V3.1 Phase 2 scope)")
            logger.info("Skipping MC propagation for IV infusion (Phase 2 feature)")
        else:
            try:
                ue = UncertaintyEngine()
                mc = ue.propagate_fast(
                    compiled, graph, drug, n_samples=n_mc_samples, t_min_h=t_min_h
                )
                if mc.n_samples > 0:
                    cmax_90ci = mc.cmax_90ci
                    logger.info(
                        "MC propagation: %d samples, Cmax 90%% PI = (%.4f, %.4f)",
                        mc.n_samples,
                        cmax_90ci[0],
                        cmax_90ci[1],
                    )
            except Exception as e:
                warnings_list.append(f"MC propagation failed: {e}")
                logger.warning("MC propagation failed: %s", e)

    # MC recomputes Cmax independently of engine_pk, so apply the same measured-F
    # exposure-scaling to the PI to keep it consistent with the corrected point.
    if cmax_90ci is not None and f_correction_k is not None:
        cmax_90ci = (cmax_90ci[0] * f_correction_k, cmax_90ci[1] * f_correction_k)

    # ── Step 3: ML direct Cmax ───────────────────────────────────────────
    ml_pk: PKEndpoints | None = None
    try:
        predictor = PKPredictor()
        ml_cmax = predictor.predict_cmax(smiles, dose_mg)
        ml_pk = PKEndpoints(
            cmax=ml_cmax,
            tmax=Distribution(1.0),  # ML does not predict Tmax
            auc_0t=Distribution(0.0),  # ML does not predict AUC
        )
        logger.info("ML PK: Cmax=%.4f mg/L", ml_cmax.mean)
    except Exception as e:
        warnings_list.append(f"ML prediction failed: {e}")
        logger.warning("ML prediction failed: %s", e)

    # ── Step 3b: CL/F analytical Cmax (3rd track) ─────────────────────
    clf_pk: PKEndpoints | None = None
    try:
        from sisyphus.ml.clf_predictor import CLFPredictor
        clf_pred = CLFPredictor()
        # Pass engine Tmax for ka estimation (method 1)
        engine_tmax = engine_pk.tmax.mean if engine_pk is not None else None
        # Pass Peff for ka estimation (method 2)
        peff_val = adme.peff.mean if adme is not None else None
        clf_cmax, ka_method = clf_pred.predict_cmax(
            smiles, dose_mg, engine_tmax=engine_tmax, peff=peff_val,
        )
        clf_pk = PKEndpoints(
            cmax=clf_cmax,
            tmax=Distribution(1.0),
            auc_0t=Distribution(0.0),
        )
        logger.info("CL/F PK: Cmax=%.4f mg/L (ka=%s)", clf_cmax.mean, ka_method)
    except Exception as e:
        warnings_list.append(f"CL/F prediction failed: {e}")
        logger.warning("CL/F prediction failed: %s", e)

    # ── Step 3b: VDss analytical Cmax = dose / (VDss_L_per_kg * 70 kg) ──
    # LOOCV-validated: adds 4th track to meta-learner, Δ=-0.113 AAFE on holdout.
    _BW_KG_VDSS = 70.0
    try:
        vdss_cmax_val: float | None = dose_mg / (adme.vdss.mean * _BW_KG_VDSS)
        logger.info("VDss analytical: Cmax=%.4f mg/L (VDss=%.2f L/kg)",
                    vdss_cmax_val, adme.vdss.mean)
    except Exception as e:
        vdss_cmax_val = None
        logger.warning("VDss analytical failed: %s", e)

    # ── Step 4: Meta-learner ─────────────────────────────────────────────
    meta = MetaLearner()
    final_pk = meta.combine(
        engine_pk,
        ml_pk,
        dose_mg=dose_mg,
        logp=profile.logp,
        tpsa=profile.tpsa,
        mw=profile.mw,
        fup=adme.fup.mean,
        clint=adme.clint.mean,
        compound_type=profile.compound_type,
        pgp_flag="PGP_EFFLUX_RISK" in profile.ad_flags,
        clf_pk=clf_pk,
        vdss_cmax=vdss_cmax_val,
    )

    # ── Determine method ─────────────────────────────────────────────────
    if engine_pk and ml_pk:
        method = "hybrid"
    elif engine_pk:
        method = "engine"
    elif ml_pk:
        method = "ml"
    else:
        method = "none"

    # ── P7: acid-low-fup flag (ketorolac-class) ──────────────────────────
    # Highly protein-bound acids (pKa<5, DrugBank measured fup<0.02) are a
    # documented engine structural limitation — 2026-04-11 fup-override
    # attempt regressed engine AAFE by 0.306 across 107-holdout. Flag
    # informationally only. Uses DrugBank measured fup (not XGBoost-predicted)
    # because the XGBoost model systematically overpredicts fup on this class.
    extra_flags = list(profile.ad_flags)
    pka_val = getattr(profile, "pka", None)
    if pka_val is not None and pka_val < 5.0:
        try:
            from sisyphus.predict.drugbank import drugbank_lookup
            _db_fup = drugbank_lookup().get_fup(profile.smiles)
        except Exception:
            _db_fup = None
        if (_db_fup is not None and _db_fup < 0.02
                and "HIGH_ACID_LOW_FUP" not in extra_flags):
            extra_flags.append("HIGH_ACID_LOW_FUP")
    in_ad, prodrug_warnings = _adjust_ad_for_prodrug(drug, extra_flags)
    warnings_list = list(warnings_list) + prodrug_warnings

    # ── Conformal calibrated 90% PI (user-facing interval) ────────────────
    # Supersede the parameter-only MC interval (~30% coverage at nominal 90%)
    # with the train-calibrated split-conformal interval (holdout-validated
    # ~0.95 at nominal 0.90). Multiplicative: meta /÷ 10**q90, from the FINAL
    # meta point (f-correction already applied). Skipped for infusion (the
    # calibration regime is single-dose oral / IV-bolus) and when the artifact
    # is unavailable (then cmax_90ci keeps the MC value, or None).
    _q90 = _conformal_q90_meta()
    if not is_infusion and _q90 is not None and final_pk.cmax.mean > 0:
        _factor = 10.0 ** _q90
        cmax_90ci = (final_pk.cmax.mean / _factor, final_pk.cmax.mean * _factor)

    # ── Confidence ────────────────────────────────────────────────────────
    if not in_ad:
        confidence = "low"
    elif engine_pk and engine_pk.cmax.mean > 0:
        confidence = "high"
    else:
        confidence = "medium"

    return PredictionResult(
        drug_name=profile.smiles[:30],
        smiles=profile.smiles,
        dose_mg=dose_mg,
        route=route,
        pk=final_pk,
        method=method,
        engine_pk=engine_pk,
        ml_pk=ml_pk,
        clf_pk=clf_pk,
        confidence=confidence,
        in_applicability_domain=in_ad,
        ad_flags=tuple(extra_flags),
        warnings=tuple(warnings_list),
        cmax_90ci=cmax_90ci,
        phenotypes_applied=tuple(phenotypes.items()) if phenotypes else (),
    )
