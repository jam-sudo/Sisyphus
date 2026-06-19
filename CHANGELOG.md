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

### MIPD — engine-as-prior posterior PK + individualized TDM (2026-06-09 → 2026-06-12)

New module `src/sisyphus/mipd/`. Repositions the mechanistic engine from a one-shot
SMILES→C<sub>max</sub> oracle (walled at the holdout ceiling) into a **structural prior that
any sparse measured observation sharply updates**. With zero measured data it reduces to the
a-priori prediction, so the **107-holdout headline (2.731) is untouched** — this is new product
surface in the regimes where the ML/meta stack has no signal by construction (out-of-domain
chemistry, dose/regimen/population extrapolation, individualized MIPD).

- **Engine-as-prior posterior PK core (PR #69, `4cb76ed`, 2026-06-09).** Bioavailability F as a
  latent with a wide prior centered on the engine's emergent `F_engine`, updated from measured
  observations (F, C<sub>max</sub>, AUC, plasma conc) by SIR. `mipd/core.py`, `mipd/clgrid.py`
  (CL latent + conc grid surrogate), `mipd/meta.py` (route the posterior through the meta blend),
  `mipd/grid.py`. `predict` surfaces `engine_f` via a shared `detect_disposition`. Gate 0b/0c
  passed (one measured anchor materially improves C<sub>max</sub>, ~3× more out-of-domain). Charter:
  `docs/_internal/specs/2026-06-09-engine-as-prior-mipd-charter.md`.
- **CrCl renal individualization + steady-state IV TDM + weight/age covariates (2026-06-11).**
  `predict_posterior(covariates=Covariates(crcl_ml_min=...))` scales renal CL by `CrCl/125` and
  surfaces the individualized posterior `post.cmax`; `predict_tdm(...)` conditions the prior on a
  steady-state IV trough to individualize a renal-CL latent over a multi-dose `solve_regimen`
  (vancomycin/aminoglycoside scope); weight/age swap the reference graph for
  `sbi.physiology_generator.generate_physiology`. Added `PosteriorPK.{warnings,renal_scale}` and an
  opt-in `ci_floor`. Fixed an IV-infusion float-overshoot in `regimen.solver.solve_regimen`.
- **Dose recommendation + Cockcroft-Gault CrCl (2026-06-12).** `recommend_dose(...)` inverts the IV
  TDM posterior into a target-attainment dose (Css,max/Css,min/AUC) under constraints; Cockcroft-Gault
  estimates CrCl from age/weight/sex/SCr when no measured CrCl is supplied.

### Holdout headline 2.784 → 2.731 — canonical CI-stack batch regen (2026-06-10)

Same-stack regen (`.github/workflows/flux1-regen.yml`) of origin/main + paracellular absorption:
Meta AAFE **2.7310** (engine 4.244, ML 2.998, in-domain 2.777, N=107). The pinned 2.784 was stale
(predated the merged oxybutynin reference fix). Attribution: oxybutynin −0.026 + paracellular −0.031,
both within the bootstrap CI — correctness-driven, not a distinguishable accuracy gain. Cache, baseline,
bootstrap CIs (`data/validation/4track_ci_2026-06-10_flux1.json`), and the cache-pin (renamed
`test_cached_holdout_aafe_is_2p731`) all regenerated.

### June correctness/contract cycles (2026-06-03 → 2026-06-09)

All correctness-first; the net holdout effect is captured by the 2.731 batch regen above.

- **FLUX-1 — flow-limitation double-count fix (PR #65, 2026-06-04).** The clearance flux applied the
  whole-organ `CL_h` (Q-embedded) to the compartment outlet alongside the convective `Q·c_out` edge,
  double-counting flow and capping hepatic/gut extraction at E→0.5 instead of →1.0. Fixed to apply the
  intrinsic clearance across all 4 clearance models; gut CYP3A4 abundance re-anchored to hold midazolam
  E_gut invariant. Correct physics; **regresses** the benchmark (2.698 → 2.784) — the wrong formula was
  load-bearing as calibration. This is the DE-41/42/43 first-pass-F root cause.
- **RBP-2 — blood:plasma concentration-basis cleanup (2026-06-04).** Holdout-bit-identical; corrects
  drugs with R<sub>B:P</sub> ≠ 1 on non-holdout paths.
- **Conformal prediction intervals (2026-06-04).** User-facing 90% C<sub>max</sub> PI is now a
  train-calibrated split-conformal interval (q90=1.111, /÷12.92), holdout-validated to 0.953 coverage at
  nominal 0.90. Replaces the MC interval (29.9% @ nominal 90%). Point estimates bit-identical.
- **OATP1B1 re-anchor to pitavastatin (2026-06-04).** Re-anchored the liver OATP1B1 uptake abundance
  5.0e5 → 1.3e5 against a non-holdout substrate (was pravastatin = holdout → Invariant #5 erosion);
  un-xfailed pravastatin + pitavastatin ECM tests. Headline-neutral.
- **Measured-F routing (PR #64, 2026-06-03).** `MeasuredADMEInput.f_bioavail` exposure-scales engine
  C<sub>max</sub>/AUC by F_measured/F_engine; bit-identical when unused (headline untouched).
- **Engine contract hardening (WS-2..6, `f8edfbc`, 2026-06-07).** Fail-loud `engine/contracts.py`
  fu-correction guard, real parallel-tube via `graph/axial.py` axial sub-compartment expansion,
  active-transport direction, JAX↔SciPy RHS parity. Headline bit-identical (no production
  `parallel_tube`/`active_transport` edges).
- **Oxybutynin holdout reference correction (PR #68, 2026-06-09).** `cmax_mg_L` 0.001 → 0.008 (FDA
  Ditropan IR single-dose ~8 ng/mL; a decimal/unit slip); AUC + ct_curve rescaled ×8. Primary-source,
  model-blind. Regen-folded into the 2.731 headline above.

### B-13 gut UGT expansion + B-14 hepatic UGT IVIVE + audit hardening (2026-05-29 → 2026-05-31)

- **B-13 — gut UGT2B7 correction (PR #49, `242b100`, 2026-05-29).** Gut-wall `UGT2B7 = 3.6e3 pmol` (0.60 pmol/mg total-mucosal × 6000; Al-Majdoub 2021 / Couto 2020); gut `UGT1A9` dropped (not expressed in human small intestine; Oda 2012). Drug-level UGT1A9 affinity still acts at liver. Metric-neutral: Meta AAFE 2.69828 → 2.69825 (103/107 bit-identical; only the 4 UGT2B7 gut seeds shift, all down). A citation-confabulation audit (11-agent adversarial verification) re-derived both gut abundances from primary sources after two fabricated citations were caught.
- **B-14 — hepatic UGT IVIVE differential = DE-40 no-op (PR #50, `3b0b72b`, 2026-05-30).** Predict-side per-enzyme UGT scaling-factor hook ships as audited no-op infrastructure: `data/enzymes/ugt_ivive_sf.json` (all-1.0 registry) + `get_ugt_ivive_sf()` loader in `predict/non_cyp_substrates.py` + a one-line `scaled_affinity *= sf` in `_decompose_clint`. Engine untouched (identity-blind preserved); 107/107 bit-identical. No verified per-substrate hepatocyte-basis hepatic-fraction SF exists (DE-40); the clean hook remains available for any future curated value.
- **gitignore housekeeping (PR #52, `7749f41`, 2026-05-30).** Added `.ruff_cache/`, `.mypy_cache/`, `.hypothesis/`, `*.orig`, `*.rej`, and `Sisyphus_Preprint_*.{docx,pdf}` (local-only manuscript drafts). No tracked files affected.
- **Pravastatin holdout→MMPK leak closed + JAX RHS guard (PR #53, `d424688`, 2026-05-31).** Forward-looking audit follow-up: pravastatin was the only holdout drug surviving both filters in `ml_cmax_improvement.load_mmpk_data` (in_holdout=False rows + an InChIKey-14 connectivity mismatch the `ho_ik` filter missed). Corrected the `in_holdout` flag in `mmpk_expanded_{full,v2}.csv`, added a name-based exclusion (`load_holdout_names()`), and pinned it with `tests/regression/test_mmpk_holdout_leak.py`. The shipped `xgboost_cmax.json` was trained via Omega's own 3-key exclusion, so the headline cache is **unaffected** (stays Meta 2.698). Also hardened `make_jax_rhs` with a pure-Python `_unsupported_flux_specs()` guard that raises `NotImplementedError` instead of silently dropping prodrug-activation / 1-compartment-elimination fluxes (dead path; no production caller uses `backend="jax"`).

### Docs reorganization — internal scratchpad split to `docs/_internal/` (PR #51, gitignored, 2026-05-30)

Agent-operational docs that external readers do not need are now gitignored under
`docs/_internal/`, mirroring the existing local-only project-context decision (2026-05-02).
Moved out of the tracked tree (via `git mv`, history preserved): `backlog.md`,
`landmarks.md`, `phase-completion.md`, `hardening_backlog.md`, `propranolol_cmax_drift.md`
(from `docs/research/`), `next_steps_plan.md` + `current_directory_completion_audit_2026-04-24.md`
(from `docs/`), and the entire `docs/_internal/plans/` directory (implementation
scaffolding — specs stay public as audit-trail per `cherry_picking_process_v1.md` §2).
The full internal development superset lives locally under `docs/_internal/`.
The committed `docs/research/` set is now the
externally-meaningful five: `dead-ends`, `experiment-log`, `diagnosis`,
`cherry_picking_process_v1`, `cherry_picking_audit_2026-04-22`. **Note:** historical
CHANGELOG / spec / experiment-log entries that cite the moved paths
(`docs/_internal/plans/…`, `docs/research/backlog.md`, etc.) are left as immutable dated
records; those files now live under `docs/_internal/` and resolve only in a working tree
that retains the internal docs.

### v3 — Prodrug Activation Input-Data Quality Refresh (2026-05-01, branch `feat/prodrug-activation-v3`)

Resolution of T1 caution flags deferred from v2. Architecture unchanged
(pure data-quality refresh per spec §3.3 mechanistic-A doctrine).

**Items resolved**:
- Item 1 (BH4 CL/Vd, sepiapterin): **ceiling_accepted** (F_sapropterin not located in primary literature; FDA Kuvan + EMA EPAR explicit "absolute bioavailability not known")
- Item 2 (GS-441524 CL/Vd, remdesivir): **literature_applied** (Tamura 2023 + Leegwater 2022 popPK geomean: CL=17.4 L/h, V=535 L)
- Item 3 (R406 CL/Vd, fostamatinib): **literature_applied** (Matsukane 2022 IV microdose review: CL=15.7 L/h, Vss=256 L)
- Item 4 (tebipenem CL/Vd): **ceiling_accepted** (F_tebipenem absolute not located; Eckburg V/F=46.2 surrogate rejected per §4.1 Gap 5)
- Item 5 (SPR primary proteomic abundance): **ceiling_accepted** (no quantitative MS-based human SPR pmol/mg located; HPA + Wu 2020 review animal-only)
- Item 6 (CES2/tebipenem direct CLint): **ceiling_accepted** (no in vitro tebipenem-pivoxil/CES2 Vmax/Km located; Gupta 2023 generic intestinal esterases)

**v1→v2→v3 fold-error progression** (per `docs/_internal/specs/2026-04-29-prodrug-v3-literature.md`):

| Drug | v1 | v2 | v3 | gate |
|---|---|---|---|---|
| sepiapterin | 5356× over | 4692× over | 4748× over | xfail (Item 1 ceiling) |
| remdesivir | 4.45× under | 4.43× under | 4.44× under | xfail (Item 2 lit_applied; parent obs not active) |
| fostamatinib | 4.78× under | 4.51× under | 4.50× under | xfail (Item 3 lit_applied; extraction rate-limited) |
| tebipenem_pivoxil | 8.63× under | 9.02× under | 9.05× under | xfail (Items 4+6 ceiling) |

**Headline AAFE delta**: zero shift (4 prodrugs not in 107-holdout). Meta 2.702, Engine 3.572, ML 3.057 unchanged from v2 baseline. Verified by §6.2 enzyme-leak audit (107/107 byte-identical).

Per spec §3.3 mechanistic-A promise, gate-fail with mechanistic-A-compliant values is acceptable outcome (informative not failing). v4 candidates require new mechanistic terms beyond data refresh: extra-hepatic esterase distribution, BH4 first-pass depletion, in vitro CES2/tebipenem kinetics.

References: `docs/_internal/specs/2026-04-29-prodrug-activation-v3-design.md` + `docs/_internal/specs/2026-04-29-prodrug-v3-literature.md` + plan `docs/_internal/plans/2026-04-29-prodrug-activation-v3.md`.

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
    unchanged. T1 literature (`docs/_internal/specs/2026-04-27-prodrug-v2-task1-literature.md`
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
  structural error. Recorded in the project README as diagnostic, not calibrated.
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
- `README.md` — added Prediction-interval coverage section under
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
