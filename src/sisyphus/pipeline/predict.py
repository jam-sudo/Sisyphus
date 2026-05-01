"""Pipeline orchestrator -- SMILES -> PredictionResult.

Thin coordination layer that wires predict, engine, ml, and pk
together.  All logic lives in the sub-layers; this module only
calls them in the right order and combines results.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from sisyphus.core import Distribution, DrugOnGraph, PKEndpoints, PredictionResult

logger = logging.getLogger(__name__)


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


# Resolve physiology YAML relative to repository root.
# src/sisyphus/pipeline/predict.py -> ../../../../data/physiology
_PHYSIOLOGY_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "physiology"


def predict(
    smiles: str,
    dose_mg: float,
    route: str = "oral",
    n_mc_samples: int = 0,
    infusion_duration_min: float | None = None,
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

    warnings_list: list[str] = []
    cmax_90ci: tuple[float, float] | None = None
    graph = None
    compiled = None

    # ── Step 1: Chemistry + ADME ─────────────────────────────────────────
    profile = compute_profile(smiles)
    adme = predict_adme(profile)
    drug = build_drug_on_graph(profile, adme, dose_mg, route)

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

        # Pass enzyme abundances from the graph to IVIVE (fix DRY violation).
        # Rebuild DrugOnGraph with graph-sourced enzyme abundances.
        if "liver" in graph.nodes and graph.nodes["liver"].enzymes:
            liver_enzymes: dict[str, float] = {
                tag: dist.mean for tag, dist in graph.nodes["liver"].enzymes.items()
            }
            drug = build_drug_on_graph(profile, adme, dose_mg, route, liver_enzymes=liver_enzymes)

        from sisyphus.graph.builder import augment_for_active_species
        graph = augment_for_active_species(graph, drug)

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
        else:
            warnings_list.append("ODE solver did not converge")
            logger.warning("ODE solver did not converge")
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
    )
