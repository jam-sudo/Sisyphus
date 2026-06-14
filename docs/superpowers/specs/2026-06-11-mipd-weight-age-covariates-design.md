# MIPD Weight/Age Covariates — Graph Individualization (v1)

**Date:** 2026-06-11
**Author:** Hypatia (with Jae Min Yoon)
**Status:** Design — approved to write the implementation plan. No implementation until the plan is reviewed.
**One-line:** Individualize the engine prior for body weight and age by swapping the reference graph for `sbi.physiology_generator.generate_physiology(weight, age)` (volumes, flows, and metabolic enzyme ontogeny/aging) at the shared `_build_grid_engine` point — benefiting both the oral CL-grid and the IV steady-state TDM paths. Renal individualization stays **CrCl-only**; estimating renal CL from age/weight is deferred (a `go over` pass found `_gfr_aging_factor` is elderly-only and double-channels with perfusion).

This completes the covariate set begun by the CrCl feature (`docs/superpowers/specs/2026-06-11-mipd-crcl-renal-individualization-design.md` §8). It reuses the verified-drop-in `generate_physiology` (CrCl spec verification C1).

---

## 1. Problem

The MIPD prior is built on the 70 kg / ~30 yr ICRP reference adult (`reference_man.yaml`). Body weight and age are major individualizing covariates — pediatric, elderly, and obese patients have different organ volumes, blood flows, and enzyme abundances (ontogeny in children, decline in the elderly). The CrCl feature individualizes renal clearance; weight/age individualize the rest of the physiology. Neither path (oral `predict_posterior`, IV `predict_tdm`) currently accepts weight/age.

`sbi.physiology_generator.generate_physiology(body_weight_kg, age_years)` already produces a fully-scaled `BodyGraph` (volumes ×BW/70; cardiac output & flows ×(BW/70)^0.75 × flow-aging; diffusion ps ×BW/70; enzyme abundances × ontogeny/aging factor; transporters ×BW/70). CrCl-spec verification C1 confirmed it is a structural drop-in for `build_from_yaml` in the grid path. It is **not** currently wired into the predict/grid layer (only the SBI training pipeline uses it).

---

## 2. Decisions (from brainstorming + a `go over` pass)

- **Both entry points.** Weight/age enter at the shared `_build_grid_engine` (used by both `build_cl_grid` and `build_renal_cl_grid`), exposed via `Covariates(body_weight_kg, age_years)` on `predict_posterior` (oral) and `predict_tdm` (IV).
- **Graph individualization only; renal stays CrCl-only.** The `go over` pass found three problems with estimating renal CL from age/weight, so all of it is deferred:
  - **`_gfr_aging_factor` is elderly-only.** It returns `1.0` for every age ≤ 40 (verified: `if age <= 40: return 1.0`), so it models elderly GFR decline but **not** pediatric GFR immaturity — applying it would give neonates an adult GFR (wrong for pediatric renal drugs).
  - **Double-channel.** `generate_physiology`'s `_flow_aging_factor` already reduces renal **perfusion** with age; a separate `_gfr_aging_factor` on `renal_cl` would partially double-count the (correlated) age→renal effect.
  - **Weight inconsistency.** Scaling renal CL by weight separately from `generate_physiology` (which scales the kidney *volume/flow* but not `renal_cl`) is inconsistent.
  - → `Covariates.renal_factor()` stays **CrCl-only and unchanged**. Age/weight drive only the graph. *(Elderly metabolic and perfusion decline still flow through the engine via `generate_physiology`'s enzyme-aging and flow-aging; only explicit GFR-from-age/weight estimation is deferred.)*

---

## 3. Scope

**In v1:**
- `Covariates` gains `body_weight_kg` and `age_years` (optional); a `has_physiology()` predicate.
- `_build_grid_engine` builds the graph via `generate_physiology(weight or 70, age or 30, base_yaml=<absolute reference_man>)` when either is supplied, else `build_from_yaml(reference_man)` (unchanged).
- Both entry points thread weight/age to their grid builders; `predict_posterior`'s "individualized" dispatch condition broadens to `renal_factor != 1.0 OR has_physiology()`.
- An extreme-weight/age `warnings` entry (consistent with the extreme-CrCl rule).

**Deferred (§9):** estimated renal CL from age (elderly GFR decline) or weight (allometric/BSA-normalized GFR); pediatric renal ontogeny; `generate_physiology`'s `rng` inter-individual sampling; a pediatric/elderly PK benchmark.

---

## 4. Design

### 4.1 `Covariates` extension

```python
@dataclass(frozen=True)
class Covariates:
    crcl_ml_min: float | None = None
    body_weight_kg: float | None = None     # NEW
    age_years: float | None = None          # NEW

    def __post_init__(self) -> None:
        if self.crcl_ml_min is not None and self.crcl_ml_min <= 0: raise ValueError(...)
        if self.body_weight_kg is not None and self.body_weight_kg <= 0: raise ValueError(...)
        if self.age_years is not None and self.age_years <= 0: raise ValueError(...)

    def renal_factor(self) -> float:        # UNCHANGED — CrCl only
        return self.crcl_ml_min / _REFERENCE_GFR_ML_MIN if self.crcl_ml_min is not None else 1.0

    def has_physiology(self) -> bool:       # NEW — triggers generate_physiology
        return self.body_weight_kg is not None or self.age_years is not None
```

### 4.2 `_build_grid_engine` graph swap

`_build_grid_engine(smiles, dose_mg, route, renal_factor, kp_method)` gains `body_weight_kg=None, age_years=None`. The only change is how the base graph is built:

```python
if body_weight_kg is not None or age_years is not None:
    from sisyphus.sbi.physiology_generator import generate_physiology
    graph = generate_physiology(
        body_weight_kg if body_weight_kg is not None else 70.0,
        age_years if age_years is not None else 30.0,
        base_yaml=_PHYSIOLOGY_DIR / "reference_man.yaml",   # absolute — avoid CWD dependence
    )
else:
    graph = build_from_yaml(_PHYSIOLOGY_DIR / "reference_man.yaml")
```

Everything after (renal_factor scaling, augment → expand → compile → realize → obs_node) is unchanged. So **both** grids (oral `build_cl_grid`, IV `build_renal_cl_grid`) gain weight/age individualization through this one point. `base_yaml` is passed as the **absolute** path because `generate_physiology`'s default `_DEFAULT_YAML` is a CWD-relative `Path("data/physiology/...")`.

### 4.3 Entry-point wiring

- `build_cl_grid` and `build_renal_cl_grid` gain `body_weight_kg=None, age_years=None`, threaded straight to `_build_grid_engine`.
- `predict_tdm` (IV): pass `covariates.body_weight_kg` / `covariates.age_years` to `build_renal_cl_grid`.
- `predict_posterior` (oral): broaden the dispatch — `individualized = renal_factor != 1.0 or has_physiology`; pass weight/age to `build_cl_grid` on both the grid path (`needs_grid`) and the single-individualized-solve path (`elif individualized`). The reference F-only path (no covariates, no obs) is unchanged.

### 4.4 Gating & faithfulness

- `generate_physiology` is called **only** when weight or age is supplied (`has_physiology()`). With neither, `build_from_yaml` runs exactly as today → the grids and the SMILES-only headline are **bit-identical** (the existing faithfulness pin `test_cl_grid_at_unit_scale_reproduces_predict_engine_pk` and the holdout pin hold).
- `generate_physiology(70, 30)` is **not** bit-identical to `build_from_yaml` (CrCl-spec C1: enzyme maturation never reaches exactly 1.0; ~0.1% on a slow-maturing UGT). This is hit only when the caller explicitly supplies weight/age (opting into the covariate model) — acceptable and documented.

### 4.5 Warnings

Append a `PosteriorPK.warnings` entry for physiologically extreme inputs (the ontogeny/allometry extrapolates poorly): body weight outside ~[2, 250] kg or age outside ~[0, 100] yr. The prediction still proceeds (consistent with the extreme-CrCl rule).

---

## 5. Invariants

1. **Engine identity-blind (Inv 1):** `generate_physiology` produces a `BodyGraph`; no `engine/` change.
2. **Distributions (Inv 2):** `generate_physiology` preserves `Distribution`s (scaled means).
3. **Reuse, don't duplicate:** the weight/age scaling is `sbi.physiology_generator.generate_physiology`, reused — no reimplementation in `mipd/`.
4. **Existing contracts untouched:** `predict()`, the SMILES-only headline, the CrCl path, and the steady-state path are unchanged. `Covariates.renal_factor()` is unchanged (CrCl-only). `covariates=None` (or a `Covariates()` with all-None) is **bit-identical** to today.
5. **Headline untouched:** no change to `predict()` or any holdout artifact.

---

## 6. Error handling

- `body_weight_kg <= 0` or `age_years <= 0` → `ValueError` (`Covariates.__post_init__`).
- Extreme weight/age → `PosteriorPK.warnings` entry; prediction proceeds.
- `generate_physiology` failure (e.g. a missing correlation group) → propagates as the underlying error; the grid's all-points-failed guard (`ValueError`) still applies if the solve fails everywhere.

---

## 7. Testing & validation

Honest scope: **no pediatric/elderly PK benchmark in the repo** → mechanism + directional validation only (§8); a population benchmark is a data-acquisition effort (§9).

- `Covariates`: `has_physiology()` true iff weight or age set; `renal_factor()` unchanged (CrCl-only, weight/age don't affect it); `weight<=0`/`age<=0` → `ValueError`.
- **Integration — BOTH grids (the C1 gap):** `build_cl_grid(... body_weight_kg=, age_years=)` (oral single-bolus) AND `build_renal_cl_grid(... body_weight_kg=, age_years=)` (IV `solve_regimen`) both run `generate_physiology → augment → expand → compile → solve(_regimen)` with `solver_success` (this path has no existing caller — it must be proven here).
- **Bit-identity guard:** `build_cl_grid(... body_weight_kg=None, age_years=None)` is unchanged from no-covariate; `predict_posterior(covariates=None)` bit-identical (pins Invariant 4/5).
- **Directional — weight:** for a fixed dose, a low body weight → lower absolute clearance → higher AUC (`build_cl_grid`/`predict_posterior` AUC rises as weight falls).
- **Directional — age:** higher age → reduced enzyme abundance (aging) + reduced flow → higher exposure for a metabolized drug.
- **Dispatch:** `predict_posterior(covariates=Covariates(body_weight_kg=10), observations=[])` routes through the individualized solve (not the reference F-only path) — the weight reaches the engine.
- **predict_tdm:** `predict_tdm(... covariates=Covariates(age_years=75))` individualizes the IV grid.
- **Warnings:** extreme weight (e.g. 1 kg) or age (e.g. 120 yr) → `warnings` entry; normal values → none.

---

## 8. Honest framing — what v1 is and is NOT

- **Solid:** metabolic / volume / flow individualization for weight and age, for **both** pediatric (enzyme ontogeny via `enzyme_factor` maturation) and elderly (enzyme + flow aging). This is the well-modeled, literature-grounded part.
- **Weight-only is "a size-scaled adult."** With weight set but age unset, age defaults to 30 → `generate_physiology(weight, 30)` applies adult enzyme maturity at the scaled size. Correct for unusually small/large **adults**; **wrong for children** (who also have immature enzymes/kidneys). For pediatric, supply **both** weight and age.
- **Renal is CrCl-only (§2).** Age/weight do not estimate GFR; supply a measured CrCl for renal-drug renal individualization. (Elderly metabolic + perfusion decline still flow through the graph; only GFR-from-age/weight is deferred.)
- **Models are literature-based, not benchmark-validated here** (enzyme ontogeny Tanaka 1998; flow aging Lindeman/Lindsay). v1 ships a correct, directionally-validated mechanism — not drug- or population-specific clinical accuracy. `generate_physiology` is used in the predict/grid path for the first time. Follows the correctness-over-benchmark discipline.

---

## 9. Deferred (not in v1)

- **Estimated renal CL from age** (elderly GFR decline, with the pediatric-immaturity model that `_gfr_aging_factor` lacks) and **from weight** (allometric / BSA-normalized GFR).
- **Pediatric renal ontogeny** (neonatal GFR maturation).
- **`generate_physiology` `rng` inter-individual sampling** (correlated abundance draws) as a posterior-widening source.
- **Pediatric / elderly / obese PK benchmark** (data acquisition).

---

## 10. File-level change list

- **MODIFY** `src/sisyphus/mipd/covariates.py` — add `body_weight_kg`, `age_years` fields, validation, `has_physiology()`; `renal_factor()` unchanged.
- **MODIFY** `src/sisyphus/mipd/grid.py` — `_build_grid_engine` gains `body_weight_kg`/`age_years` (the `generate_physiology` swap); `build_cl_grid` threads them.
- **MODIFY** `src/sisyphus/mipd/renal_grid.py` — `build_renal_cl_grid` threads `body_weight_kg`/`age_years`.
- **MODIFY** `src/sisyphus/mipd/api.py` — `predict_posterior` dispatch: broaden `individualized`; thread weight/age; extreme-weight/age warnings.
- **MODIFY** `src/sisyphus/mipd/tdm.py` — `predict_tdm` threads weight/age; extreme-weight/age warnings.
- **Tests:** additions to `tests/unit/test_mipd_covariates.py`, `tests/unit/test_mipd_grid.py`, `tests/unit/test_mipd_renal_grid.py`, `tests/unit/test_mipd_api.py`, `tests/unit/test_mipd_tdm.py`.

No changes to `engine/`, `predict/predict()`, `sbi/physiology_generator.py` (reused as-is), the holdout, or any headline artifact.
