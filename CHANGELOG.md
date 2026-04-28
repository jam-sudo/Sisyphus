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
- **Prodrug activation v2 — enzyme-abundance mechanistic** (branch
  `feat/prodrug-activation-v2`, 2026-04-27/28): replaces v1's kinetic
  1st-order conversion (`rate = k × A_parent`) with well-stirred extraction
  at flow-through nodes (mirrors existing CYP3A4 elimination pattern).
  Drug declares `enzyme_affinity_for_conversion: dict[str, Distribution]`;
  augmentation discovers conversion sites by enzyme intersection with
  physiology. Affinity values sourced from in-vitro literature or
  substrate-class kinetics (no clinical fit; tier 3 / "infrastructure_only"
  rejected by registry loader per spec §3.3 mechanistic-A promise).

  **Architectural changes** (additive to engine; identity-blind preserved):
  - `ProdrugActivationEdge.conversion_rate: Distribution` removed; replaced
    by `enzyme_tags: frozenset[str]` (compile-time set).
  - `ProdrugActivationFluxSpec.apply()` rewritten as well-stirred extraction
    parallel to `ClearanceFluxSpec(model="well_stirred")`, destination
    redirected to active species pool with MW × yield scaling.
  - `DrugOnGraph.enzyme_affinity_for_conversion: dict[str, Distribution]`
    new additive field (`__post_init__` enforces non-empty + active_metabolite
    pairing).
  - `ResolvedParams.drug_enzyme_affinity_for_conversion(tag) -> float` new
    method (parallel to `drug_enzyme_affinity` for elimination).
  - `augment_for_active_species` rewritten: multi-site discovery via
    `enzyme_tags ∩ node.enzymes`; one ProdrugActivationEdge per site.
  - `lookup_active_metabolite` returns 3-tuple `(am, obs, affinities)`;
    schema requires `affinity_source` + `yield_source` enums.

  **Physiology YAML additions**: SPR (1e5 liver, 3e3 gut, 3e4 kidney, all
  CV~1.0–1.2 class-estimated), CES1 (8e7 liver, CV 0.47, Boberg 2017),
  CES2 (8.4e6 liver, 3e6 gut, CV 0.6), ALPI (2.3e4 gut, CV 0.9,
  Al-Majdoub 2020). All independent lognormal (no Achour matrix entry).

  **v1-vs-v2 fold-error comparison** (deterministic Cmax at registered
  doses; per-drug parametrized 3-fold gate per spec §6.1):

  | Drug              | v1 fold-error | v2 fold-error | Δ          |
  |-------------------|---------------|---------------|------------|
  | sepiapterin       | 5356×         | 4692×         | -12%       |
  | remdesivir        | 4.45×         | 4.43×         | ~unchanged |
  | tebipenem_pivoxil | 8.63×         | 9.02×         | +5%        |
  | fostamatinib      | 4.78×         | 4.51×         | -6%        |

  All four drugs fail the 3-fold validation gate (xfail with documented
  reasons). This is **expected per spec §3.3 mechanistic-A promise**:
  affinity values are NOT refit to clinical data. Failure is informative,
  not project-failing.

  **Disposition decisions** (from brainstorming review, 2026-04-28):
  - **D1 (deferred to v3)**: v1 active species CL/Vd values retained
    unchanged. T1 literature (`docs/superpowers/specs/2026-04-27-prodrug-v2-task1-literature.md`
    §4) found inconsistencies for BH4, GS-441524, R406 (1.5–50× literature
    deviation). Deferred to preserve clean architectural attribution
    (v2 = conversion math + affinity sourcing only). v3 task to update
    active CL/Vd from popPK literature with 1C-vs-2C reduction analysis.
  - **D2 (applied)**: fostamatinib `conversion_yield_fraction` 0.7 → 0.9.
    v1 conflated hydrolysis stoichiometry with bioavailability; v2 yield
    is pure hydrolysis (absorption losses captured by parent peff
    upstream).
  - **D3 (kept)**: sepiapterin retained in v2 registry. Tier 2
    (SPR abundance class-estimated, T1 caution flag). Removing would
    weaken architecture stress test (Eg ≈ 99.99% case).

  **Validation tests added**:
  - Mass balance well-stirred (flow-loop synthetic) — `tests/integration/test_prodrug_v2_mass_balance.py`.
  - Per-prodrug pipeline smoke (4 drugs end-to-end) — `tests/integration/test_prodrug_v2_pipeline_smoke.py`.
  - DDI smoke (halving CES1 → 0.53× remdesivir active Cmax) — `tests/integration/test_prodrug_v2_ddi_smoke.py`.
  - Identity-blind random tag rename invariance — `tests/regression/test_prodrug_v2_identity_blind.py`.
  - 107-holdout numerical invariance verified (no leak from new enzymes).
  - Per-prodrug Cmax ±5% snapshots — `tests/regression/test_prodrug_v2_snapshot.py`.
  - Per-drug parametrized 3-fold validation gate — `tests/regression/test_prodrug_v2_validation_gate.py`.

  **Reuse from v1**: 60% (9/15 components unchanged: ActiveMetabolite,
  OneCompartmentEliminationEdge, observation_routing, AD adjustment,
  pipeline integration shape, holdout regression, ACTIVE_SUFFIX/sink
  constants, DrugOnGraph 19 existing fields, edge_id sampling pattern).
  Changed: 6 (edge struct, flux body, registry schema, registry loader,
  augmentation, mass balance test). New: 3 (physiology entries, drug
  field, ResolvedParams method).

  v1 known-limitation note (now superseded) covered the kinetic
  1st-order architecture's inability to capture fast first-pass
  extraction — diagnosed root cause was gut_wall residence ~64s.
  v2 well-stirred resolves this architectural limitation; remaining
  prediction errors trace to literature-input quality (per T1 caution
  flags), addressable in v3 via CL/Vd updates and tighter SPR/CES2/ALPI
  abundance measurements.
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
