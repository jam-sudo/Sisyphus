"""Build the engine CL grid: solve the engine over a clint-scale grid.

Reproduces the ``pipeline.predict`` engine path (build_drug_on_graph -> graph ->
compile), then re-solves at ``enzyme_affinity`` scaled by each clint-scale
(compile-once / parameterize-many — Invariant 3). For each scale it records the
venous concentration-time curve (re-gridded onto a common time axis), Cmax, AUC
(from compute_endpoints — exact), and the emergent F_engine (oral AUC / IV-AUC).

Scope (v1): the standard oral/IV path with no phenotypes, no measured-ADME
overrides, and no auto OATP1B1 ECM. Faithfulness at clint-scale 1.0 is pinned to
predict()'s engine PK by ``test_cl_grid_at_unit_scale_reproduces_predict_engine_pk``.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from sisyphus.core import Distribution
from sisyphus.mipd.clgrid import CLGrid


def _default_t_grid() -> np.ndarray:
    # dense early (peak/absorption), coarser late (elimination tail), to 24 h.
    return np.unique(
        np.concatenate([np.linspace(0.0, 6.0, 121), np.linspace(6.0, 24.0, 73)])
    )


def _fill_nan_log_s(values: np.ndarray, s_grid: np.ndarray) -> np.ndarray:
    """Replace NaNs by log-s interpolation over the valid points (edge-clamped)."""
    values = np.asarray(values, dtype=float)
    ok = np.isfinite(values) & (values > 0)
    if ok.all() or not ok.any():
        return values
    ls = np.log(s_grid)
    values[~ok] = np.exp(np.interp(ls[~ok], ls[ok], np.log(values[ok])))
    return values


def _nearest_finite_backfill(conc: np.ndarray) -> np.ndarray:
    """Replace any NaN-containing rows by the nearest fully-finite row.

    Nearest is by absolute row-index distance; ties go to the lower index. Unlike
    an adjacent-copy, this never copies a still-NaN neighbor. Assumes at least one
    fully-finite row exists (the caller guards total engine failure separately).
    """
    conc = np.asarray(conc, dtype=float)
    finite = np.array([g for g in range(conc.shape[0]) if np.all(np.isfinite(conc[g]))])
    if finite.size == 0:
        raise ValueError("no finite concentration row to backfill from")
    for g in range(conc.shape[0]):
        if not np.all(np.isfinite(conc[g])):
            conc[g] = conc[finite[int(np.argmin(np.abs(finite - g)))]]
    return conc


def build_cl_grid(
    smiles: str,
    dose_mg: float,
    route: str = "oral",
    *,
    n_grid: int = 13,
    s_range: tuple[float, float] = (0.1, 10.0),
    kp_method: str = "rodgers_rowland",
    t_grid: np.ndarray | None = None,
) -> CLGrid:
    """Solve the engine over a clint-scale grid and return a ``CLGrid``."""
    from sisyphus.engine.compiler import ODECompiler, ResolvedParams
    from sisyphus.engine.solver import _IV_CMAX_DELAY_H, solve
    from sisyphus.graph.axial import expand_axial
    from sisyphus.graph.builder import augment_for_active_species, build_from_yaml
    from sisyphus.pipeline.predict import (
        _PHYSIOLOGY_DIR,
        _engine_oral_bioavailability,
        _resolve_observation_node,
    )
    from sisyphus.pk.endpoints import compute_endpoints
    from sisyphus.predict.adme import predict_adme
    from sisyphus.predict.chemistry import compute_profile
    from sisyphus.predict.ivive import build_drug_on_graph, detect_disposition

    if t_grid is None:
        t_grid = _default_t_grid()
    s_grid = np.geomspace(s_range[0], s_range[1], n_grid)

    profile = compute_profile(smiles)
    adme = predict_adme(profile)
    graph = build_from_yaml(_PHYSIOLOGY_DIR / "reference_man.yaml")

    # Use predict()'s shared disposition detector (single source of truth) so the
    # grid's engine path is faithful for OATP/ECM and non-CYP (UGT/NAT) substrates,
    # not just CYP drugs. Without these args the s=1 grid diverges from predict()
    # (e.g. ~18% for codeine, ~50% for morphine).
    auto_oatp_kinetics, auto_ecm_params, non_cyp_fractions = detect_disposition(profile)

    # Snapshot pre-phenotype liver abundances and rebuild the drug from them, so
    # back-solved affinities match the engine multiplication (mirrors predict()).
    liver_pre: dict[str, float] | None = None
    if "liver" in graph.nodes and graph.nodes["liver"].enzymes:
        liver_pre = {tag: d.mean for tag, d in graph.nodes["liver"].enzymes.items()}
    drug = build_drug_on_graph(
        profile, adme, dose_mg, route,
        liver_enzymes=liver_pre,
        kp_method=kp_method,
        transporter_kinetics=auto_oatp_kinetics,
        hepatic_ecm_params=auto_ecm_params,
        non_cyp_fractions=non_cyp_fractions,
    )
    graph = augment_for_active_species(graph, drug)
    graph = expand_axial(graph)

    compiler = ODECompiler()
    compiled = compiler.compile(graph)
    realized_graph = graph.realize_means()
    obs_node = _resolve_observation_node(drug)
    t_min_h = _IV_CMAX_DELAY_H if route == "iv" else 0.0
    admin_idx = compiled.state_index[drug.administration_node]

    conc_rows: list[np.ndarray] = []
    cmaxs: list[float] = []
    aucs: list[float] = []
    fengs: list[float] = []
    for s in s_grid:
        # Scale METABOLIC intrinsic clearance only (enzyme_affinity: CYP/UGT/NAT).
        # renal_clearance and any transporter/biliary terms are left untouched, so
        # the clint-scale latent is a metabolic-CL scale, not a total-CL scale.
        drug_s = dataclasses.replace(
            drug,
            enzyme_affinity={
                k: Distribution(mean=v.mean * float(s), cv=v.cv)
                for k, v in drug.enzyme_affinity.items()
            },
        )
        realized_drug_s = drug_s.realize_means()
        params_s = ResolvedParams(realized_graph, realized_drug_s)

        y0 = np.zeros(compiled.n_states)
        y0[admin_idx] = dose_mg
        sim = solve(compiled, params_s, y0, t_span=(0, 24), t_min_h=t_min_h)
        if not sim.solver_success:
            conc_rows.append(np.full(t_grid.size, np.nan))
            cmaxs.append(np.nan)
            aucs.append(np.nan)
            fengs.append(np.nan)
            continue
        pk = compute_endpoints(sim, observation_node=obs_node, t_min_h=t_min_h)
        conc_rows.append(np.interp(t_grid, sim.time_h, sim.concentrations[obs_node]))
        cmaxs.append(pk.cmax.mean)
        aucs.append(pk.auc_0t.mean)
        feng = _engine_oral_bioavailability(
            compiled, params_s, realized_drug_s, pk.auc_0t.mean, obs_node
        )
        fengs.append(feng if (feng is not None and feng > 0) else np.nan)

    cmaxs_arr = np.array(cmaxs)
    fengs_arr = np.array(fengs)
    if not np.isfinite(cmaxs_arr).any():
        raise ValueError(
            f"engine failed at all {n_grid} clint-scale grid points; cannot build CL grid"
        )
    if not np.isfinite(fengs_arr).any():
        raise ValueError(
            f"engine produced no valid oral bioavailability at any of {n_grid} grid points"
        )
    cmax = _fill_nan_log_s(cmaxs_arr, s_grid)
    auc = _fill_nan_log_s(np.array(aucs), s_grid)
    f_engine = np.clip(_fill_nan_log_s(fengs_arr, s_grid), 1e-4, 1.0)
    # backfill any failed curve rows from the nearest finite row so interp stays valid.
    conc = _nearest_finite_backfill(np.array(conc_rows))

    return CLGrid(
        s_grid=s_grid, t_grid=t_grid, conc=conc, cmax=cmax, auc=auc, f_engine=f_engine
    )
