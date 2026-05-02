# Hardening Backlog (2026-04-23, amended)

Derived from external audit (v2, 2026-04-23) after cross-verification against actual repo state and existing dead-ends/invariants. 5 items adopted from 16 proposed; 6 rejected as already-done / superseded / DE-conflict; 5 deferred as YAGNI or requires architectural spec cycle.

**Amendment log (2026-04-23)**: applied 8 targeted edits from review pass — H3 risk reclassified Low + MC-gating made explicit + 3-state acceptance; H2 `feature_schema_sha256` added; H4 biological/calibrated/placeholder classification criterion; H1 entrypoint specified (`sisyphus.validation.benchmark.run_benchmark` direct, not `scripts/run_engine_benchmark.py`); H5 RDKit fallback strategy; B1 deferral gets trigger conditions; "SMILES-first strawman" rephrased as "already satisfied by project scope"; Non-goals section added. Amendments preserve invariants and do not reopen rejected items.

**Scope**: all items are pure infrastructure / documentation. None change headline metrics, engine topology, or invariants. Each still requires a spec-plan cycle before execution per `CLAUDE.md` invariants §8.

**Not shipped**: this file is a backlog, not a commitment. Prioritization decisions belong to the user.

---

## Adopted Items

### H1 — CI workflow

**Gap**: `/home/jam/Sisyphus/.github/workflows/` does not exist. No automated check on push/PR.

**Proposed scope**:
- `.github/workflows/ci.yml` running on push + PR
- Jobs: `ruff check`, `pytest tests/unit`, `pytest tests/integration/test_engine_validation.py`, benchmark smoke
- Benchmark smoke entrypoint: call `sisyphus.validation.benchmark.run_benchmark(holdout_only=True, max_drugs=5)` directly from a pytest test (NOT `scripts/run_engine_benchmark.py`, which currently has no `--max-drugs` flag plumbed). `cli.py:570` already supports `--max-drugs`; reuse that pattern.
- Cache pip dependencies
- Status badge in README

**Risk**: 0 (read-only CI infra).

**Acceptance**:
- CI runs green on `main` HEAD
- Red on deliberate test regression (verify by breaking a unit test in a throwaway PR)
- Full holdout benchmark stays out-of-CI (too slow); smoke only (N≤5 drugs)
- If model artifact load exceeds CI memory/time budget, split into `ci-fast` (unit+ruff) and `ci-full` (integration+smoke)

**Dependencies**: none (H5 lockfile benefits CI but doesn't block it).

**Estimated effort**: ~1 hour including first-pass debugging.

---

### H2 — ADME model metadata unification

**Gap**: `/home/jam/Sisyphus/models/direct_pk/meta.json` exists with rich provenance (training dataset, holdout_aafe, hyperparameters, retrain reason). `/home/jam/Sisyphus/models/adme/*.json` — 13 XGBoost artifacts (fup_v2, clint, peff, rbp, pka_acidic, pka_basic, vdss_v2, clearance_v1, thalf_v1, bioavailability, logp_correction, fup, vdss_v2_meta) — have no accompanying `.meta.json`. Asymmetric provenance.

**Proposed scope**:
- Define `docs/science/model_manifest_schema.md` matching existing `direct_pk/meta.json` shape
- Create `{model_name}.meta.json` for each ADME artifact
  - Required keys: `version`, `trained_on` (dataset path + `sha256`), `trained_at`, `n_drugs_original`, `n_drugs_excluded`, `holdout_version`, `feature_schema` (`name`, `n_features`, `sha256`), `holdout_metric` (name + value), `target`, `hyperparameters`, `retrained_reason`
  - **`feature_schema.sha256` is required, not just `n_features`**. Feature count alone does not catch order/identity changes (e.g. descriptor swap, Morgan radius change). Hash the feature-vector construction pipeline output on a canonical input set.
- Backfill retrospectively from git log / training scripts where possible; mark unrecoverable fields as `"unknown_legacy"` with explanation
- Extend `ml/registry.py` to validate manifest presence at load time. First-pass policy: missing-legacy-manifest → warning; present-but-invalid manifest → warning (promote to error later if needed). Do not block existing workflows.

**Risk**: 0 (additive metadata, no code path changes unless registry enforcement enabled).

**Acceptance**:
- All 13 ADME artifacts have `.meta.json` next to them
- Every manifest includes `feature_schema.sha256`
- `ml/registry.py` `register()` and `get()` stop raising `NotImplementedError`; instead validate schema and return `ModelRecord`
- Loading a model without manifest emits warning but succeeds
- Loading with feature-schema-hash mismatch emits warning (hard error deferred to future decision, not this item)

**Dependencies**: none.

**Estimated effort**: ~3-4 hours (mostly archaeology for legacy fields).

---

### H3 — `pi_coverage_90` wiring

**Gap**: `src/sisyphus/validation/benchmark.py:286` hardcodes `pi_coverage_90=None`. The `pi_coverage()` function already exists at `src/sisyphus/validation/metrics.py:54` and is unused.

**MC gating (important — not pure reporting)**:
- `pipeline.predict()` has `n_mc_samples: int = 0` default (`predict.py:28`).
- Bound computation is gated at `predict.py:187`: `if n_mc_samples > 0: ...`.
- `scripts/run_engine_benchmark.py` currently calls `predict(ref.smiles, ref.dose_mg, ref.route)` with no MC — so `cmax_90ci` is never computed.
- Also: when MC disabled, `uncertainty.py:340` returns `cmax_90ci=(0.0, 0.0)` (zero-tuple, not `None`) — benchmark logic must distinguish this sentinel from real bounds.

To make `pi_coverage_90` non-None, benchmark must **enable MC sampling per drug**. Default is ~1000 samples/drug → significant runtime cost on N=107 holdout.

**Proposed scope**:
- Add explicit PI benchmark mode: `run_benchmark(..., compute_pi=False, n_mc_samples=0)` flag, or dedicated `scripts/run_benchmark_with_pi.py`. Do NOT silently flip default benchmark into MC mode.
- When `compute_pi=True`, thread `n_mc_samples` through `predict()`, collect `cmax_90ci` tuples into lower/upper arrays, distinguish `(0.0, 0.0)` sentinel from computed bounds.
- Replace `pi_coverage_90=None` with `pi_coverage_90=pi_coverage(observed, lower, upper)` on the filtered-for-valid-bounds subset.
- Keep variable name `pi_coverage_90` (project-wide vocabulary) but print a caveat in benchmark output: *"these intervals reflect propagated parameter uncertainty only; not empirically-calibrated predictive intervals until residual structural error is added and validated"*.

**Risk**: Low. Reporting-only if per-drug intervals are already computed; runtime-sensitive if MC must be turned on (full-holdout PI benchmark can be ~50-200× slower than default).

**Acceptance**:
- Three distinct states in benchmark output:
  1. *interval not computed* (default fast benchmark, `compute_pi=False`)
  2. *parameter-uncertainty interval computed* (MC enabled, caveat shown)
  3. *calibrated prediction interval computed* (future work, not this item)
- Default benchmark runtime unchanged (no silent MC enable)
- Explicit PI mode emits `pi_coverage_90: <float>` and the caveat line
- Value stored in benchmark result artifact with mode label
- Test verifies: `pi_coverage_90 is None` when `compute_pi=False`; `pi_coverage_90 is not None` when `compute_pi=True` with MC > 0

**Dependencies**: none.

**Estimated effort**: ~2-3 hours (verifying bound propagation + sentinel handling + MC-mode plumbing).

---

### H4 — ECM unit chain consolidated documentation

**Gap**: The ECM extended clearance formula (`src/sisyphus/engine/flux.py:291-334`) depends on:
- `abundance` constant `1.0e11` at `liver.transporters.OATP1B1` (calibrated on pravastatin)
- `ivive_scaling=0.00006` (`60/1e6`, µL/min → L/h) at `data/physiology/reference_man.yaml:66`
- Per-drug `jmax_pmol_per_min_per_mg` + `km_uM` in `data/transporters/oatp1b1.json`
- Amendment v2.1 flat-CLuptake scaling: `Jmax_val = (Jmax_prava/Km_prava) × Km_val`

The dimensional chain (abundance × Jmax/Km × ivive → L/h) is scattered across 5 files with partial explanations. No single document traces it end-to-end with units labeled.

This is the root dimensional scaffold for DE-33 (OATP1B1 non-statin 2.5× underpredict per V3 ECM test). Consolidation does not resolve DE-33 but makes any future architectural fix (one of 4 paths in `project_ecm_generalization_test.md`) easier to evaluate.

**Proposed scope**:
- `docs/science/ecm_unit_audit.md`
- Sections:
  1. Dimensional chain from `abundance × jmax/Km × ivive` to `L/h`, with sample pravastatin numerics
  2. What `abundance=1.0e11` really is (pravastatin Phase-1 calibration, not MPPGL × organ weight; dimensional identity lost)
  3. Flat-CLuptake scaling derivation and its Km-invariance (confirms dead-end #12)
  4. V3 ECM test result interpretation (valsartan/glimepiride 2.5× underpredict as expected consequence of non-statin transfer)
  5. Cross-references to spec `2026-04-20-oatp-ecm-hepatic-clearance-design.md`, amendment 6e7ce0a, `project_ecm_generalization_test.md`
- No code changes, no YAML changes

**Risk**: 0 (pure documentation).

**Acceptance**:
- Document traces every symbol in `flux.py:291-334` to its origin and units
- Every parameter is classified as one of: (a) *biological abundance* (MPPGL, organ weight, literature kinetics), (b) *calibrated effective value* (e.g. `abundance=1.0e11` from pravastatin Phase-1 fit), (c) *dimensional placeholder/scaffold* (values whose numeric magnitude is forced by unit-conversion convention, not measurement)
- Any future contributor can reproduce the pravastatin calibration numerically from this file alone
- DE-33 architectural options (real per-drug Jmax / different scaling / drop non-statins / OATP1B3+NTCP) are enumerated with pre/con from the unit chain perspective

**Dependencies**: none.

**Estimated effort**: ~4 hours (tracing + writing + numerical verification).

---

### H5 — Dependency lockfile

**Gap**: `pyproject.toml` specifies dependencies but no `uv.lock` / `requirements-lock.txt` / `conda-lock.yml` committed. Benchmark reproducibility relies on loose version constraints; NumPy/SciPy/XGBoost/RDKit version drift can shift numerical outputs.

**Proposed scope**:
- Try `uv lock` first (preferred — project uses modern Python tooling) → commit `uv.lock`
- **RDKit fallback strategy**: `pyproject.toml` declares unpinned `"rdkit"`; RDKit has historically had platform-specific resolution issues on pip. If `uv lock` fails to resolve RDKit cleanly, fall back to `pip freeze > requirements-lock.txt` OR document a conda-based install path (`conda-lock.yml`). Document the final chosen path in `docs/reproducibility.md`.
- CI (H1) must install from whichever lockfile format is committed
- Add `docs/reproducibility.md` documenting the lock workflow + RDKit install path

**Risk**: 0 (lockfile can always be regenerated).

**Acceptance**:
- Fresh checkout + `uv sync` (or `pip install -r requirements-lock.txt`, or conda install path) reproduces the current environment
- CI runner successfully installs the dependency set from the committed lockfile — including RDKit
- Benchmark AAFE reproduces on a fresh environment within floating-point tolerance

**Dependencies**: ordering — H1 benefits from lockfile but doesn't block on it.

**Estimated effort**: ~1-2 hours (extra hour budget for RDKit resolution fallback).

---

## Suggested Ordering

If executed sequentially as separate spec-plan cycles:

```
H5 (lockfile)  →  H1 (CI)  →  H3 (pi_coverage)  →  H2 (ADME meta.json)  →  H4 (ECM unit doc)
  1-2h              1h            2-3h                  3-4h                    4h
```

Total ~11-14 hours if no deviations. Each can ship independently; no item blocks another's core value.

**Alternative**: bundle H1 + H5 as single "infra reproducibility" spec (strongest coupling). H2-H4 independent.

---

## Rejected / Deferred (for reference)

**Rejected** (already done or DE-conflict):
- `C2` new blind holdout — N50 cycle 2026Q2 frozen 2026-04-23 (`b366035`)
- `S1` better CLint — DE-1, DE-14 (14+ attempts failed)
- `S3` re-enable UGT — DE-11-adjacent, `ivive.py:589-590` documents rationale
- `S8` automated leakage checks — partially covered by N50 pre-freeze audit
- Issue #0 "preserve SMILES-first" — already satisfied by project scope; SMILES + dose + route is the enforced inference contract per `CLAUDE.md`. No separate implementation item needed.
- `HIGH_ACID_LOW_FUP` AD flag — already shipped P7

**Deferred** (requires architectural spec-plan cycle, in `project_engine_improvements_from_n50.md`):
- `S2` enzyme fraction training (overlaps DE-7)
- `S4` renal transporter module (overlaps engine_improvements direction #2)
- `S5` active metabolite support (overlaps direction #1)
- `S6` formulation modeling (overlaps direction #3)

**Deferred** (YAGNI, no current consumer):
- `B1` `pipeline.predict` refactor (currently 326 lines). Reconsider when any trigger fires:
  - `predict.py` exceeds ~500 LOC
  - A new prediction track is added
  - Endpoint provenance (`C5`) is promoted from deferred
  - PI calibration logic (H3 follow-up) is added directly to `predict.py`
- `B2` `merge_overlay()` implementation (no caller found in repo)
- `C5` endpoint provenance (real inconsistency, but no downstream TDM/MIPD bug reported)
- ~~`B3` pipeline-level `kp_method` selection.~~ **Resolved 2026-05-02** — `pipeline.predict.predict()` now accepts a `kp_method=` keyword-only kwarg (default `"rodgers_rowland"`, preserves prior behavior bit-for-bit) and forwards it to both `build_drug_on_graph` call sites. Berezhkovskiy / Provided options are now reachable through the public API. 5 unit tests in `tests/unit/test_pipeline_kp_method.py` lock the wiring contract. Originally discovered by `feature/3d-cyp-multidist` Experiment C (archive tag `archive/3d-cyp-multidist-2026-04-01`).

---

## Non-goals for this hardening batch

Explicit scope guard for Claude/Codex execution. This batch must **not**:

- Change engine topology or the ODE compilation path
- Retune meta-learner weights or VDss activation
- Change holdout membership (107 dev-validation set, N50 frozen secondary holdout)
- Add new biological mechanisms (prodrug activation, non-CYP CL, OATP1B3/NTCP, formulation — all separately tracked in `project_engine_improvements_from_n50.md`)
- Modify Sisyphus's SMILES + dose + route inference contract
- Change headline AAFE metrics except by adding additional reporting metadata
- Require formulation, food state, salt form, or patient covariates as mandatory inputs
- Modify invariants listed in `CLAUDE.md` §8 (engine identity-blind, Distribution-native, compile-once, flow conservation, holdout inviolable, no drug-specific branches)

If an execution agent encounters a task that appears to require any of the above, it must halt and escalate — not silently extend scope.

## Invariants Preserved

None of the adopted items:
- Modify engine identity-blind invariant
- Change `DrugOnGraph` fields
- Touch the holdout drug list
- Introduce drug-specific branches
- Alter headline metric pipelines

All items operate on infrastructure layer (CI, lockfile), metadata layer (manifests, docs), or reporting layer (pi_coverage wire-up).

## Status

**Not started**. This file is a backlog, not an execution plan. Each item needs its own `docs/superpowers/specs/` + `docs/superpowers/plans/` cycle before implementation per CLAUDE.md §8.
