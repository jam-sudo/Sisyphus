# Codebase Concerns

**Analysis Date:** 2026-03-24

## Critical Issues

### __pycache__ Files Committed to Git
- **Location**: Every `__pycache__/` directory under `src/sisyphus/` and `tests/`
- **Impact**: ~30 `.pyc` binary files are tracked in git despite `__pycache__/` appearing in `.gitignore`. These were committed before the gitignore rule took effect. They bloat the repo, cause merge noise, and show up as modified in `git status` after every code change.
- **Suggested fix**: Run `git rm -r --cached **/__pycache__` to untrack all `.pyc` files. The existing `.gitignore` rule will prevent re-addition.

### Model Artifacts Not in `.gitignore` (Selective Tracking)
- **Location**: `models/adme/*.json`, `models/direct_pk/*.json`, `models/meta_learner/*.json`
- **Impact**: XGBoost JSON model files (ranging 96KB to 1.7MB) are committed to git. The `.gitignore` has `models/**/*.json` but also `!models/**/.gitkeep`, and the files were committed before the ignore rule. Binary model artifacts in git cause repo bloat and make diffs unreadable. The `models/adme/xgboost_fup_v2.json` (1.7MB) and `models/adme/logp_correction.json` (316KB) are untracked (local only), creating a divergence between git state and working state.
- **Suggested fix**: Migrate model artifacts to Git LFS or a dedicated artifact store. Ensure `.gitignore` actually excludes them, then `git rm --cached` the existing ones.

### `build_drug_on_graph` Called Twice in Pipeline
- **Location**: `src/sisyphus/pipeline/predict.py` lines 71, 106
- **Impact**: `build_drug_on_graph` is called once at line 71 (without graph enzyme abundances), then again at line 106 (with liver enzyme abundances from the graph). This means the entire DrugBank lookup, CLint decomposition, Kp calculation, and DrugOnGraph assembly runs twice per prediction. Wastes ~50% of the predict layer compute time.
- **Suggested fix**: Defer `build_drug_on_graph` until after graph loading. Call it once with `liver_enzymes` when available, falling back to the hardcoded default only when graph loading fails.

## Technical Debt

### Duplicated CLint Computation in `ClearanceFluxSpec.apply`
- **Location**: `src/sisyphus/engine/flux.py` lines 215-272
- **Type**: duplication
- **Effort**: S
- **Risk if ignored**: The `well_stirred` and `parallel_tube` branches share identical CLint computation (10 lines each: enzyme loop, IVIVE scaling, fup lookup, concentration extraction). Any bug fix or performance improvement to one must be manually replicated in the other.
- **Fix**: Extract a `_compute_clint_organ` helper and a `_compute_c_out` helper to share between the two models.

### Duplicated Kp Computation Between R&R and Poulin-Theil
- **Location**: `src/sisyphus/predict/ivive.py` lines 296-413
- **Type**: duplication
- **Effort**: S
- **Risk if ignored**: `_compute_kp_rodgers_rowland` and `_compute_kp_poulin_theil` share identical logic for neutrals and acids (~25 lines each). Only bases/zwitterions differ (phospholipid binding term). The neutral and acid branches are copy-pasted.
- **Fix**: Extract shared neutral/acid logic into a common helper, dispatch only the base/zwitterion path per method.

### Hardcoded Enzyme Abundances Duplicated Between YAML and Code
- **Location**: `src/sisyphus/predict/ivive.py` lines 49-60 (`_LIVER_ENZYME_ABUNDANCE`), `data/physiology/reference_man.yaml`
- **Type**: coupling | duplication
- **Effort**: M
- **Risk if ignored**: Enzyme abundances are hardcoded in `ivive.py` as a fallback and also defined in `reference_man.yaml`. If YAML values change, the hardcoded fallback silently uses stale values. The pipeline already passes graph-sourced abundances at line 106, but the fallback path (graph load failure) uses outdated constants.
- **Fix**: Load abundances from YAML at module import or make the graph-supplied path mandatory.

### Tissue Compositions Hardcoded in `ivive.py`
- **Location**: `src/sisyphus/predict/ivive.py` lines 113-129 (`_TISSUE_COMPOSITIONS`), `data/physiology/reference_man.yaml` (node compositions)
- **Type**: duplication
- **Effort**: M
- **Risk if ignored**: Tissue compositions for Kp calculation are hardcoded as a separate dict in `ivive.py` with different naming conventions than the graph nodes (e.g., `"gut_wall"` in ivive vs the graph node `"gut_wall"` with `lookup_name`). If physiology data changes in YAML, the Kp calculations will use stale tissue compositions.
- **Fix**: Load tissue compositions from the graph (already available via node compositions) rather than maintaining a parallel dict.

### Path Resolution via Fragile `.parent` Chains
- **Location**: 7 occurrences across `src/sisyphus/predict/adme.py:28`, `src/sisyphus/predict/chemistry.py:336`, `src/sisyphus/predict/drugbank.py:44`, `src/sisyphus/pipeline/predict.py:21`, `src/sisyphus/graph/presets.py:14`, `src/sisyphus/validation/reference.py:19`, `src/sisyphus/ml/models.py:20`
- **Type**: complexity | coupling
- **Effort**: M
- **Risk if ignored**: Each module resolves the project root via `Path(__file__).resolve().parent.parent.parent.parent`. This breaks if the package structure changes (e.g., moving a file one level deeper). There is no single source of truth for the project root.
- **Fix**: Define a single `ROOT_DIR` in `sisyphus/__init__.py` or `sisyphus/config.py` and import it everywhere.

### `PipelineConfig` Defined but Never Used
- **Location**: `src/sisyphus/pipeline/config.py`
- **Type**: obsolete
- **Effort**: S
- **Risk if ignored**: `PipelineConfig` is imported nowhere. The pipeline function `predict()` in `src/sisyphus/pipeline/predict.py` uses hardcoded defaults (seed=42, t_span=(0,24), observation_node="venous_blood") instead of reading from config.
- **Fix**: Either wire `PipelineConfig` into the pipeline or remove it to reduce confusion.

### Three `NotImplementedError` Stubs
- **Location**: `src/sisyphus/graph/builder.py:183` (`merge_overlay`), `src/sisyphus/validation/split.py:29` (`scaffold_split`), `src/sisyphus/ml/registry.py:37,41` (`ModelRegistry.register`, `.get`)
- **Type**: missing-test
- **Effort**: S-M each
- **Risk if ignored**: `merge_overlay` is needed for Phase 3 extensibility (overlay YAML for SC injection, pediatric, tumor). `scaffold_split` is needed to reproduce the holdout split. `ModelRegistry` was designed for Phase 4. All are dead code that occupies mental space.
- **Fix**: Implement `merge_overlay` and `scaffold_split` as they are roadmap items. Remove `ModelRegistry` if not needed in the near term, or implement it.

### `reference_woman.yaml` Referenced but Missing
- **Location**: `src/sisyphus/graph/presets.py:26-32` calls `build_from_yaml("reference_woman.yaml")`, file does not exist
- **Type**: missing-test
- **Effort**: M
- **Risk if ignored**: `reference_woman()` will raise `FileNotFoundError` at runtime. No test covers this.

### DrugBank Singleton Pattern Prevents Configuration After Init
- **Location**: `src/sisyphus/predict/drugbank.py` lines 256-277
- **Type**: complexity
- **Effort**: S
- **Risk if ignored**: `drugbank_lookup()` creates a singleton on first call. If a different `DrugBankConfig` is passed on subsequent calls, it is silently ignored. This makes ablation studies and testing with different configurations fragile. The `_reset_singleton()` function exists but is undocumented and only for tests.

### Function-Level Model Cache via `hasattr` Hack
- **Location**: `src/sisyphus/predict/chemistry.py` lines 340-347
- **Type**: complexity
- **Effort**: S
- **Risk if ignored**: The logP correction model is cached by monkey-patching `compute_profile._logp_model`. This is fragile, type-checker unfriendly (`type: ignore[attr-defined]`), and invisible to code analysis tools. Also, `xgboost` is imported inside the function body on every call if the model exists but the attribute hasn't been set yet.
- **Fix**: Use a module-level cache like `_model_cache` in `adme.py`.

## Architecture Risks

- **`ivive.py` is the largest file (654 lines) and contains four distinct responsibilities**: CLint decomposition, Kp calculation (two methods + BZ correction), renal clearance estimation, and DrugOnGraph assembly. This violates single-responsibility and is approaching the 20-file-per-directory limit justification for splitting. Extracting `kp.py` (Kp computation) from `ivive.py` would improve modularity.

- **Engine weight for non-base drugs is 0.00**: Per `src/sisyphus/ml/ensemble.py` line 34, `_W_ENGINE_OTHER = 0.00`. This means the entire PBPK engine (graph, ODE compilation, solver) contributes nothing to final predictions for ~70% of drugs (neutrals, acids, zwitterions). The engine is used only as a secondary signal for basic drugs. This raises a strategic question: is the engine's value proposition limited to bases, or is the engine undertrained/miscalibrated for other compound types?

- **Meta-learner model (`models/meta_learner/xgboost_meta.json`) is not used in production**: The `MetaLearner` class in `src/sisyphus/ml/ensemble.py` uses a hand-tuned geometric weighting (line 110: `log_cmax = w_eng * log_eng + w_ml * log_ml`), not a trained model. The `xgboost_meta.json` file exists in the models directory but is not loaded anywhere in the codebase.

- **No parallelization for MC sampling**: `UncertaintyEngine.propagate` and `propagate_fast` in `src/sisyphus/engine/uncertainty.py` run MC iterations sequentially (lines 130-152, 205-230). Each iteration is independent (separate RNG seed). For 1000 samples this is a significant bottleneck. Python's multiprocessing or `joblib` could parallelize this easily since each sample is stateless.

## Performance Concerns

- **Sequential MC loop**: `src/sisyphus/engine/uncertainty.py` lines 130-152 and 205-230. Each MC iteration creates a fresh `BodyGraph.sample()` and `DrugOnGraph.sample()`, allocating new dicts and Distribution objects. For N=1000, this creates ~34,000 nodes and ~100+ edges worth of temporary objects per run.

- **RHS function creates a new `np.zeros(n)` array on every call**: `src/sisyphus/engine/compiler.py` line 241. The ODE solver calls `rhs()` hundreds of times per solve; each call allocates a new zeros array. Pre-allocating and zeroing in-place would reduce GC pressure.

- **`compute_features()` recomputes Morgan FP from scratch every call**: `src/sisyphus/descriptors.py` lines 49-53. In the pipeline, `compute_features` is called at least twice per drug (once in `predict_adme()` line 231, once in `PKPredictor.predict_cmax()` line 59). The features are identical. No caching exists.

- **`np.trapz` compatibility shim is evaluated on every call**: `src/sisyphus/engine/solver.py` line 142 and `src/sisyphus/pk/nca.py` line 22. `getattr(np, "trapezoid", np.trapz)` runs attribute lookup every time. Should be resolved once at module level.

## Security Considerations

- **DrugBank XML in working directory**: `full database.xml` (1.9GB) exists at repo root. It is in `.gitignore` and not committed, but its presence is a data license risk if accidentally committed. The `.gitignore` rule uses exact filename matching which is fragile.

- **Broad `except Exception` clauses swallow errors**: 14 instances across the codebase (see grep results above). In `src/sisyphus/pipeline/predict.py` lines 92, 136, 153, 168, exceptions are caught and downgraded to warnings. While this keeps the pipeline running, it masks potentially important failures (e.g., wrong SMILES canonicalization, corrupted model files, YAML parse errors).

- **`yaml.safe_load` is used correctly**: `src/sisyphus/graph/builder.py` line 55 uses `safe_load`, not `load`. No YAML deserialization vulnerability.

## Missing Capabilities

- **No solubility-limited absorption**: The absorption model in `src/sisyphus/engine/flux.py` lines 319-357 computes `ka = 2.88 * Peff * ka_fraction / radius` but does not limit absorption rate by available dissolved drug (Noyes-Whitney dissolution). For poorly soluble drugs (BCS II/IV), this overpredicts absorption rate. -- Effort: M

- **No `reference_woman.yaml`**: Referenced in `src/sisyphus/graph/presets.py:26` but missing from `data/physiology/`. Needed for sex-based physiology modeling (different organ volumes, cardiac output, enzyme abundances). -- Effort: M

- **Overlay merge not implemented**: `src/sisyphus/graph/builder.py:183` (`merge_overlay`) raises `NotImplementedError`. Required for Phase 3 extensibility (SC injection, pediatric, tumor compartment). Overlay YAMLs exist (`data/physiology/sc_overlay.yaml`, `tumor_overlay.yaml`) but cannot be applied. -- Effort: M

- **No multi-dose simulation**: The engine only supports single-dose simulation (`y0[admin_idx] = drug.dose_mg` at `src/sisyphus/engine/uncertainty.py` line 141). Steady-state PK (multiple doses, accumulation) is not supported. The MMPK training data includes multi-dose entries. -- Effort: L

- **RBP prediction is functionally disabled**: Per `src/sisyphus/predict/adme.py` lines 126-140, the RBP XGBoost model predicts but any value >0.5 deviation from 1.0 is reset to 1.0. The model has R^2 = -0.08 (worse than random). RBP effectively defaults to ~1.0 for most drugs. -- Effort: M (need better training data)

- **No AUC prediction from ML**: `src/sisyphus/pipeline/predict.py` line 165 sets `auc_0t=Distribution(0.0)` for ML predictions. The meta-learner only combines Cmax, not AUC. AUC predictions come solely from the engine. -- Effort: M

- **UGT metabolism disabled**: `src/sisyphus/predict/ivive.py` line 589 hardcodes `ugt_enzymes = None` despite full UGT infrastructure being implemented. Comment says sensitivity test showed AAFE degradation (2.861 to 3.090). The code path exists but is dead. -- Effort: S (re-enable and investigate)

## Code Smells

| Location | Smell | Severity |
|----------|-------|----------|
| `src/sisyphus/predict/ivive.py` (654 lines) | God file: CLint decomposition + Kp calculation (2 methods) + renal CL + DrugOnGraph assembly | Medium |
| `src/sisyphus/engine/flux.py:215-272` | Copy-paste between `well_stirred` and `parallel_tube` branches (~20 identical lines) | Medium |
| `src/sisyphus/predict/chemistry.py:336-349` | Inline model loading with `hasattr` monkey-patching inside `compute_profile()` | Medium |
| `src/sisyphus/pipeline/predict.py:50-61` | 12 deferred imports inside function body (avoids circular imports but hides dependencies) | Low |
| `src/sisyphus/predict/ivive.py:634` | `kp_method="rodgers_rowland"` hardcoded, ignoring the `kp_method` parameter passed to `build_drug_on_graph` | Medium |
| `src/sisyphus/engine/flux.py:353` | Magic number `2.88` for absorption calibration constant (documented in comment but not a named constant) | Low |
| `src/sisyphus/predict/drugbank.py:269-270` | Config argument silently ignored on singleton after first init | Low |
| `src/sisyphus/cli.py:55-70` | Uses `print()` for output (acceptable in CLI but breaks structured output) | Low |
| 7 files | `.resolve().parent.parent.parent.parent` chain for project root resolution | Medium |
| `src/sisyphus/ml/features.py` | 13-line file that only re-exports `compute_features` from `descriptors.py` | Low |
| `src/sisyphus/engine/result.py` | 9-line file that only re-exports `SimResult` from `core.py` | Low |

## Test Coverage Gaps

**Untested source modules** (no test file imports from them):
- `src/sisyphus/cli.py` -- CLI entry point, no integration test
- `src/sisyphus/descriptors.py` -- Feature computation (tested indirectly via `test_features_chemistry.py` but no dedicated test)
- `src/sisyphus/graph/presets.py` -- `reference_man()` and `reference_woman()` convenience functions
- `src/sisyphus/pk/analytical.py` -- 1-compartment and 2-compartment analytical solutions
- `src/sisyphus/pipeline/config.py` -- `PipelineConfig` (unused, so not testable)
- `src/sisyphus/ml/registry.py` -- `ModelRegistry` (not implemented)
- `src/sisyphus/validation/split.py` -- `scaffold_split` (not implemented)
- `src/sisyphus/engine/result.py` -- Re-export module (trivial)

**Specific untested paths:**
- `src/sisyphus/predict/ivive.py` -- `_compute_kp_poulin_theil` and `_apply_bz_correction` (Kp method variants)
- `src/sisyphus/engine/flux.py` -- `parallel_tube` clearance model branch, `ActiveTransportFluxSpec`
- `src/sisyphus/engine/uncertainty.py` -- `propagate_fast` error path (all samples fail)
- `src/sisyphus/pipeline/predict.py` -- MC propagation path (`n_mc_samples > 0`)
- `src/sisyphus/predict/adme.py` -- DrugBank fup 5x cross-validation guard (line 240)

---

*Concerns audit: 2026-03-24*
