# Changelog

All notable changes to Sisyphus are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Version numbers
track `pyproject.toml`.

> **Note on versioning:** The git tag `v1.0.0` (commit `67b7064`, 2026-03-xx)
> predates the current `pyproject.toml` version `0.1.0`. The tag records
> an earlier milestone on a feature branch whose work was absorbed into
> the current `main` line after merges and rewrites. The repository will
> align on a single scheme in an upcoming release decision.

## [Unreleased]

### Added
- **Prodrug activation routing infrastructure** (branch `feat/prodrug-activation`,
  commits `0db67a8`..`a2ad03e`, 17 commits, 2026-04-25/26):
  - `ActiveMetabolite` dataclass (`core.py`) + `DrugOnGraph.active_metabolite`
    + `observation_species` fields with `__post_init__` validation.
  - 2 new edge types (`graph/types.py`): `ProdrugActivationEdge` (asymmetric
    parent→active mass transfer with MW × yield scaling) and
    `OneCompartmentEliminationEdge` (aggregate 1st-order CL/Vd elimination).
  - 2 new FluxSpecs (`engine/flux.py`): registered via `@register_flux`,
    asymmetric mass transfer + 1-compartment elimination.
  - `ResolvedParams._build_edge_params` additive branches for the new types
    (existing logic untouched).
  - `BodyGraph.sample()` resampling branches for new edge types.
  - `graph/builder.py::augment_for_active_species` — adds 1 plasma node +
    2 edges per prodrug (no-op when `active_metabolite=None`).
  - `predict/registry.py` — SMILES-keyed registry loader with RDKit
    canonicalization + JSON validation.
  - `data/sbi/prodrug_activation_registry.json` — 4 N50 evidence drugs:
    sepiapterin→BH4, remdesivir→GS-441524, tebipenem_pivoxil→tebipenem,
    fostamatinib→R406.
  - `pipeline/predict.py` integration — augmentation hook before compile,
    `_resolve_observation_node` + `_adjust_ad_for_prodrug` helpers.
  - 50+ new tests across unit/integration/regression categories.
- **Validation gate failure (KNOWN LIMITATION, 2026-04-26)**: 4-drug 3-fold
  validation gate (`tests/integration/test_prodrug_pipeline_smoke.py`) FAILS
  with current 1st-order conversion architecture. Fold-errors:
  sepiapterin 5356×, tebipenem_pivoxil 8.63×, fostamatinib 4.78×,
  remdesivir 4.45×. Diagnosis: gut_wall flow-through residence time is
  ~64s; first-order `conversion_rate_per_h=12` converts only ~19% per pass,
  insufficient for fast first-pass conversion. Architecture works correctly
  for synthetic mass-balance test (commit `f8e8d16`) but the 1st-order
  rate-constant model cannot match clinical reality without literature-
  inconsistent values. **Successor spec planned**: enzyme-abundance MM
  kinetics (reuse CYP3A4 well-stirred pattern) replacing
  `conversion_rate_per_h` with `enzyme_affinity_for_conversion: dict[str,
  Distribution]`. Infrastructure preserved (12/15 tasks reusable).
- **H1-H5 hardening infrastructure** (PRs #3-#6, 2026-04-23):
  - H5 GitHub Actions CI (`.github/workflows/ci.yml`): Python 3.10 ubuntu,
    unit + integration + benchmark smoke, ruff advisory.
  - H1 `requirements-lock.txt` pinned from fresh venv, RDKit-coupled hashes.
  - H3 `--compute-pi` CLI flag wires `pi_coverage_90` via Monte Carlo MC
    propagation. Diagnostic only (parameter uncertainty; not calibrated).
  - H2 `ModelRegistry` + `<model>.meta.json` sidecar manifests with
    `feature_schema.sha256` on all 12 actively-loaded XGBoost artifacts.
  - H4 `docs/science/ecm_unit_audit.md` — dimensional chain analysis of
    ECM extended clearance.
- **First empirical 90% PI coverage measurement** (2026-04-24, commit
  `bbedd9f`): 29.9% at N=107 holdout × 1000 MC samples. AAFE 2.719
  reproduces the 2.695 headline within MC noise. PI captures ~1/3 of
  observed residual spread — interval is ~3× too narrow relative to
  structural error. Recorded in CLAUDE.md as diagnostic, not calibrated.
  Artifact: `data/validation/holdout_pi_coverage_2026-04-24.json`.
- **P4.5 Achour correlated abundance prior** (merge `2275932`, 2026-04-23):
  `Distribution.correlation_group` field + `physiology.correlation_registry`
  + Achour 2021 5-way CYP log-correlation matrix at the liver node.
  AAFE 2.6946 invariant. SBC retraining deferred to P4.5a.
- **N50 secondary holdout — FROZEN cycle 2026Q2** (commit `b366035`,
  2026-04-23): AAFE **5.249 [95% CI 3.79-7.77, N=50]**. Cannot re-run
  per spec §7 single-use-per-cycle. Retire `holdout_n50.json` before
  2026Q3+.

### Changed
- `README.md` structure diagram:
  - Replaced non-existent `ml/vdss_predictor.py` with `registry.py`; VDss
    is a track weight in `ensemble.py` (`_W_VDSS=0.20`), not a module.
  - Replaced non-existent `sbi/hierarchical.py` with `multi_drug.py`,
    `priors.py`, `physiology_generator.py` (the Achour sampler).
  - Removed `validation/split.py` (deleted; see below).
- `graph/presets.py::reference_woman()` now raises `NotImplementedError`
  with migration guidance instead of failing opaquely on a missing YAML
  file. Use `reference_man()` + body-weight override via the
  continuous-hierarchical path.
- `CLAUDE.md` — added Prediction-interval coverage section under
  Current Performance (diagnostic caveat emphasized).
- `docs/current_directory_completion_audit_2026-04-24.md` — counter-audit
  patch Option α (commit `c52b62f`): 3-way `NotImplementedError`
  reclassification, P0-1 option (c) flagged inadequate, §7 ceiling split
  into scientific vs release axes, §4.6 CHANGELOG/tag gap noted, §6 P2-5
  CI fail-closed gates proposed.

### Removed
- `src/sisyphus/validation/split.py` (commit `dd4282f`): `scaffold_split`
  was a `NotImplementedError` stub with no callers (grep ∅). The holdout
  split is already frozen in `data/reference/holdout.json`.

### Fixed
- `.gitignore` — added `.claude/` (agent/session-local config) as part
  of the counter-audit patch.

### Security
- No security-relevant changes in this cycle.

## [v1.0.0] — legacy tag, 2026-03 (historical reference only)

This tag records an earlier milestone on a feature branch (now merged
into `main` after rewrites). Retained for historical traceability.

### Reported achievements
- In-domain holdout AAFE: 1.697 (target ≤ 1.7)
- PK/PD link (effect compartment + Emax) implemented
- DDI module (inhibition + induction via enzyme abundance) implemented
- MC optimization (fast solve + propagate_fast for N=1000)
- Prodrug detection, Kp capping, in-domain AAFE ≤ 1.7

Note: the current `main` line (301 commits ahead of `v1.0.0`) supersedes
these numbers with a more conservative measurement regime (N=107 full
holdout AAFE 2.695, secondary holdout AAFE 5.249). The v1.0.0
acceptance criteria are therefore historical, not current.
