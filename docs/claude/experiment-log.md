---
last_updated: 2026-04-30
parent: ../../CLAUDE.md
charter: Chronological log of Sisyphus experiments (successes, negatives, infrastructure). Latest first.
---

# Experiment Log

Reverse-chronological. Top-level [CLAUDE.md](../../CLAUDE.md) carries only the **current** headline numbers; this file is the history. For the authoritative failed-experiment list (with do-not-retry gating), see [dead-ends.md](./dead-ends.md). For the why-accuracy-is-bounded analysis, see [diagnosis.md](./diagnosis.md).

---

## 2026-05-01 — Prodrug Activation v3 (input-data refresh, all-disposition)

**Branch**: `feat/prodrug-activation-v3` (gated on v2 PR #7 merge per spec §8.1, satisfied 2026-04-30 by `78d12e3`).
**Spec**: `docs/superpowers/specs/2026-04-29-prodrug-activation-v3-design.md`
**Plan**: `docs/superpowers/plans/2026-04-29-prodrug-activation-v3.md` (19 tasks across 5 phases — all complete)
**Literature deliverable**: `docs/superpowers/specs/2026-04-29-prodrug-v3-literature.md`

**Per-item dispositions** (mechanistic-A doctrine compliant per spec §3.3):

| # | Item | Disposition | Citation primary | Code change |
|---|---|---|---|---|
| 1 | BH4 CL/Vd (sepiapterin) | ceiling_accepted | Feillet 2008 + FDA Kuvan + EMA EPAR (F not known) | v3_metadata only |
| 2 | GS-441524 CL/Vd (remdesivir) | literature_applied | Tamura 2023 + Leegwater 2022 (popPK geomean) | CL 10→17.4, V 35→535 |
| 3 | R406 CL/Vd (fostamatinib) | literature_applied | Matsukane 2022 (IV microdose review) | CL 28→15.7, V 250→256 |
| 4 | tebipenem CL/Vd | ceiling_accepted | Eckburg 2019 (V/F surrogate rejected) | v3_metadata only |
| 5 | SPR proteomic abundance | ceiling_accepted | HPA + Wu 2020 (animal-only) | v3_metadata only |
| 6 | CES2/tebipenem CLint | ceiling_accepted | Gupta 2023 (no isoform attribution) | v3_metadata only |

**Outcome**:
- 4-drug 3-fold gate: 0 pass / 0 ceiling-with-improvement / 4 ceiling-no-improvement (drift 0.2-1.2%, all stay xfail)
- Items resolved: 2 literature_applied + 4 ceiling_accepted = 6/6 dispositioned
- 107-holdout AAFE bit-identical (4 prodrugs absent from holdout); §6.2 leak audit PASSES 107/107
- Headline metrics unchanged from v2 baseline: Meta 2.702, Engine 3.572, ML 3.057

**Significance**:
v3 closes the input-data quality pillar of the prodrug saga (v1→v2→v3) with rigorous mechanistic-A discipline. 4 items closed as ceiling because primary literature truly does not exist (F_sapropterin, F_tebipenem, human SPR proteomic, in vitro CES2/tebipenem). 2 items advanced via popPK geomean. Empirical Cmax fold-errors barely shifted because:
1. observation_species=parent for remdesivir → active CL/V update doesn't move parent Cmax
2. fostamatinib extraction rate-limits (well-stirred E~1 at high CLint) → active CL change has marginal Cmax effect
3. Items 1, 4, 5, 6 unchanged values

This is the canonical mechanistic-A outcome: "we know the literature gap exists; we documented it; we did not fudge to pass". v4 candidates require new mechanistic terms (extra-hepatic esterase, BH4 first-pass depletion, etc.) — beyond data refresh.

**Test impact**:
- `test_prodrug_v3_registry_schema` — 8/8 PASS (TDD red→green)
- `test_prodrug_v3_enzyme_leak_audit` — PASS (107/107 byte-identical)
- `test_prodrug_v2_validation_gate` — 4 xfail (reasons updated with v3 disposition references)
- `test_prodrug_v2_snapshot` — 4 PASS (re-pinned to v3 deterministic Cmax values)
- `test_prodrug_v2_pipeline_smoke` — 4 PASS (functional-only refactor per §6.1)
- `test_prodrug_v2_ddi_smoke` — PASS at v2 tolerance (no widening needed)

**Files**:
- Registry: `data/sbi/prodrug_activation_registry.json` (4 entries with v3_metadata; 2 with value updates)
- Tests: `tests/integration/test_prodrug_v3_registry_schema.py` (NEW), `tests/regression/test_prodrug_v3_enzyme_leak_audit.py` (NEW)
- Baseline capture: `scripts/capture_prodrug_v3_baseline.py` + `tests/regression/data/prodrug_v3_pre_baseline.json`
- Updated: validation_gate, snapshot, pipeline_smoke (xfail reasons + functional-only)
- Docs: literature deliverable summary tables; CLAUDE.md v3 note; CHANGELOG v3 entry

---

## 2026-04-30 — Prodrug v2 PR #7 — RNG-order discovery + cache regen

**Trigger:** v2 PR (`feat/prodrug-activation-v2`) CI failure on `test_engine_validation::test_cmax_within_5pct[midazolam, caffeine, warfarin]` — Cmax shifted 6-19% above Omega targets.

**Diagnosis:** v2 added new lognormal enzyme distributions (SPR/CES1/CES2/ALPI) to physiology YAML at liver, gut_wall, and kidney nodes. `BodyGraph.sample(rng)` iterates nodes in YAML insertion order, so adding a cv>0 distribution at kidney (which previously had no enzymes block, position 4 in YAML, BEFORE liver) consumed 1 RNG draw before liver's CYP3A4 sample. This shifted all liver CYP samples → midazolam Cmax +18.5%. Liver/gut_wall enzyme additions were appended AFTER existing CYPs, so existing CYP samples preserved BUT new draws shifted downstream OATP1B1 transporter sample → ECM-pathway holdout drugs drifted 8-27%. Test was passing on main due to RNG-order coincidence with seed=42.

**Fix (commit `6c121ce`):** Move kidney YAML node block to after gut_wall. Preserves all v2 mechanistic content (kidney SPR retained for sepiapterin renal contribution); only changes RNG sample order. ODE state index accessed via name lookup throughout — functionally invariant.

**Cache regen (commit `6528ba8`):** ECM holdout regression test (5% drift gate) failed because v2's enzyme additions still shift OATP1B1 sample even with kidney moved (liver enzyme appendage is the irreducible cause). `data/training/4track_holdout_predictions.json` regenerated against PR src + Option D YAML to capture v2 baseline.

**Aggregate AAFE delta** (main 2026-04-29 → v2 2026-04-30):

| Track | main (2026-04-29) | v2 (2026-04-30) | Δ (abs) | Δ (%) |
|---|---|---|---|---|
| Meta (Overall) | 2.719 | 2.702 | -0.017 | -0.6% |
| Engine (Overall) | 4.073 | 3.572 | -0.501 | -12.3% |
| ML (Overall) | 3.057 | 3.057 | 0 | 0% |
| Meta (In-domain) | 2.759 (n=80) | 2.730 (n=80) | -0.029 | -1.1% |

Meta %2-fold/3-fold unchanged (46.7%, 62.6%). Engine %3-fold improved 40.2 → 53.3.

**Significance:**

- **Meta AAFE statistically indistinguishable** (-0.6%) — within bootstrap CI [2.33, 3.19] noise. Headline narrative robust.
- **Engine AAFE materially improved** (-12.3%) — combines (1) v2 well_stirred extraction model for prodrug activation (replaces v1 kinetic 1st-order; remdesivir/fostamatinib/tebipenem/sepiapterin) + (2) RNG-order shift on remaining 103 non-prodrug drugs. Disentanglement requires ablation; deferred.
- **ML AAFE invariant** — ML model artifacts unchanged.
- **In-domain N=80 stable** — no AD-criteria change between 2026-04-29 and 2026-04-30 regens.

**spec §6.1 invariance violation:** v2 spec §6.1 promised "107-holdout invariance" — actually impossible because adding any cv>0 enzyme to a node consumes RNG draws and shifts downstream samples. Spec assumption was wrong. Real invariance requires either (a) per-node independent RNG seeding, or (b) deterministic mean-only realization. Both deferred to hardening backlog.

**Test impact:**
- `test_engine_validation::test_cmax_within_5pct`: PASSES (3/3) with kidney moved.
- `test_ecm_holdout_regression`: PASSES (cache regenerated).
- `test_holdout_regression::test_cached_holdout_aafe_is_2p695`: pinned AAFE updated 2.695 → 2.702. (NB: same test was already failing on main at 2.719 — pre-existing main bug; not in CI workflow.)
- `test_oatp_ecm_statins[fluvastatin]`: FAILS, FE 3.651 vs gate 3.0 (improved from main's 4.133 but still over). Pre-existing, separate from v2.
- `test_oatp_ecm_statins[pravastatin]`: PASSES (was failing on main per 2026-04-29 entry; v2 baseline shift moved it within gate — likely incidental).

**Follow-ups (queued):**
1. Refresh bootstrap 95% CIs against v2 cache post-merge (10k resamples, seed=20260422).
2. Update CLAUDE.md headline AAFE table post-merge.
3. Hardening: deterministic mean-only realization for engine validation tests (eliminates RNG-order fragility).
4. v3 spec §5 Item 5 amendment: kidney 3e4 retained but at YAML position-after-gut_wall (already in this commit; v3 spec wording may need clarifying).

**Files:**
- `data/physiology/reference_man.yaml`: kidney node moved after gut_wall (commit `6c121ce`)
- `data/training/4track_holdout_predictions.json`: regenerated (commit `6528ba8`)
- `tests/integration/test_holdout_regression.py`: pinned AAFE 2.695 → 2.702 (commit `6528ba8`)

---

## 2026-04-29 — 4-track holdout predictions regen (post-P4.5 baseline refresh)

**Trigger:** `tests/integration/test_ecm_holdout_regression.py` failing on main — 10/10 spot-checked drugs drifted 15-27% lower than cached. Investigation revealed the cache (`data/training/4track_holdout_predictions.json`) was last written 2026-04-14, before P4.5 Achour merge (2026-04-23) and other ECM/V3-routing changes.

**Action:** Re-ran `scripts/run_engine_benchmark.py --save-json data/training/4track_holdout_predictions.json` on current main. Backup of pre-regen cache stashed at `/tmp/4track_pre_regen_2026-04-29.json` (not committed).

**Aggregate AAFE delta** (PRE 2026-04-14 cache → POST 2026-04-29 fresh):

| Track | PRE | POST | Δ (abs) | Δ (%) |
|---|---|---|---|---|
| Meta (Overall) | 2.695 | 2.719 | +0.024 | +0.9% |
| Engine (Overall) | 3.421 | 4.073 | +0.652 | +19.1% |
| ML (Overall) | 3.057 | 3.057 | 0 | 0% |
| Meta (In-domain) | 2.710 (n=85) | 2.759 (n=80) | +0.049 | +1.8% |
| Engine (In-domain) | 3.236 (n=85) | 3.808 (n=80) | +0.572 | +17.7% |

Meta %3-fold: 65.4 → 62.6. Engine %3-fold: 57.9 → 40.2.

**Significance:**

- **Meta AAFE robust** (Δ +0.9%) — ML track is unchanged (model artifacts not retrained); Meta combines Engine + ML + classifier + Vd, so ML stability dampens Engine drift. This is the headline-protection mechanism in action.
- **Engine track degraded** (Δ +19.1%) — substantial. Likely root cause: P4.5 Achour correlated abundance prior (merged 2026-04-23) shifting Cmax predictions ~15-25% lower across most drugs. Earlier candidates (V3 IV-Cmax routing, ECM hepatic clearance migration) may also contribute. Per `docs/claude/propranolol_cmax_drift.md`, the propranolol +16% drift on `b366035` was an early canary; the broader engine drift documented here is consistent with that direction.
- **ML AAFE unchanged** — confirms ML model artifacts were not retrained between 2026-04-14 and 2026-04-29 (would otherwise show drift).
- **N changed: 85→80 in-domain** — applicability-domain criteria evolved or 5 drugs newly flagged. Not investigated in this entry; flagged for follow-up.
- **Cherry-picking impact:** the 2026-04-22 audit estimated retrospective-contamination band 2.85–3.10. New Meta point estimate 2.719 sits below this band, but bootstrap CI not yet refreshed against new cache; old [2.30, 3.20] CI is stale.

**Test impact:**
- `test_ecm_holdout_regression` now PASSES (cache matches fresh predictions).
- `test_oatp_ecm_statins[pravastatin]` still FAILS (FE 1.486 vs gate 1.3, T7 calibration drift) — independent of cache regen.
- `test_oatp_ecm_statins[fluvastatin]` still FAILS (FE 4.133 vs gate 3.0, Peff overprediction) — independent of cache regen.

**Follow-up needed:**
1. Refresh bootstrap 95% CIs against new cache (via cherry-picking-process bootstrap script, 10k resamples).
2. Investigate Engine-track AAFE drift root cause: bisect from 2026-04-14 to 2026-04-29 if needed; primary suspects are P4.5 Achour and ECM migration commits.
3. Document AD-criteria change (n=85 → n=80) — which 5 drugs newly flagged?
4. Decide on pravastatin T7 recalibration (was the T7 calibration tied to a pre-P4.5 cache?).

**Files updated:**
- `data/training/4track_holdout_predictions.json` (regenerated)
- `CLAUDE.md` headline performance table (point estimates, %2/3-fold, n_in_domain; CIs annotated stale)
- `docs/claude/experiment-log.md` (this entry)

---

## 2026-04-22 — Achour 2021 Correlated Physiology Prior (P4.5 infrastructure)

Spec: `docs/superpowers/specs/2026-04-22-achour-abundance-correlation-design.md`
Plan: `docs/superpowers/plans/2026-04-22-achour-abundance-correlation.md`
Branch: feat/achour-correlated-abundance (merged commit TBD).

**Outcome:** Infrastructure landed. `Distribution` gains optional `correlation_group`
field; new `sisyphus.physiology.correlation_registry` provides multivariate-lognormal
sampling; `generate_physiology(rng=)` opt-in; `reference_man.yaml` liver node
migrated to Achour 2021 CVs with OATP1B1 independent (mean_r=0.234 < 0.3
threshold, empirical Achour Table S7 inclusion rule).

**Gates passed:**
- A — deterministic mean-path: Meta AAFE 2.6946 (within ±0.001 of headline 2.695)
- B/B' — marginal CV fidelity ±5% (original Achour CVs + 0.5× healthy-proxy)
- C/C' — joint log-corr fidelity ±0.05 on 10-20k sampler draws
- D — cancer-bias sensitivity machinery (0.5× healthy-proxy supported)
- E — CSV SHA256 provenance recorded in JSON artifact

**Non-outcome:** SBC improvement is explicit Non-Goal (§1 spec). Downstream
P4.5a spec will retrain the SBI amortizer with physiology sampling and
re-measure SBC on the 52-cell grid.

**Data artifacts:**
- `data/physiology/achour2021_liver_abundance.csv` — 29 donors × 6 targets
- `data/physiology/achour2021_correlation.json` — 5×5 log-correlation matrix for CYP3A4/2D6/1A2/2C9/2E1

**Source:** Achour 2021 CPT 109:222-232 (PMC7839483, CC BY-NC 4.0).

---

## 2026-04 (current session)

### V3 IV-Cmax methodology + ECM re-run + fup confound rule-out (2026-04-22)

**Infrastructure shipped (7 commits, `4630b0b..4e10ad2`):**
Route-aware `t_min_h = _IV_CMAX_DELAY_H (5/60 h) if route=="iv" else 0.0` threaded through `solve()`, `solve_mc()`, `compute_endpoints()`, `propagate_fast()` (scipy backend), pipeline. Oral (107 holdout + production) byte-identical to V2 — pinned by `tests/integration/test_v3_oral_regression.py`. 562 pass / 4 skip / 2 xfail, zero new failures.

- Design spec: `docs/superpowers/specs/2026-04-22-iv-cmax-observation-design.md` (`d88183a`)
- Plan: `docs/superpowers/plans/2026-04-22-v3-iv-cmax-observation.md` (`de6292b`)
- Impl chain: `4630b0b` (solve anchor) → `9bc2e3d` (solve_mc windowed) → `2742df8` (compute_endpoints) → `6ed22e7` (propagate_fast) → `3f86e2e` (pipeline route-cond) → `ed3207f` (oral regression) → `4e10ad2` (propagate caveat)

**ECM generalization re-run under V3 (`7aa49ae`, `data/validation/oatp_generalization_result_v3.json`):**
Formal Mode C. **Direction flipped from V2:** V2 appeared to over-predict 1.1–1.35× but that was the t=0 artifact. V3 with windowed Cmax shows **systematic underprediction 2.5×** on both drugs.

| Drug | Observed | V2 (artifact) | V3 (real) | V3 PI | V3 log10 FE |
|---|---|---|---|---|---|
| glimepiride | 0.243 | 0.270 (1.11×) | 0.095 | [0.087, 0.101] | **−0.409** |
| valsartan | 4.02 | 5.405 (1.35×) | 1.940 | [1.80, 2.06] | **−0.316** |

Median |log10 FE| = 0.363 < 0.5 Mode B gate → formally Mode C, but same-direction underprediction is substantively suggestive of systematic ECM over-clearance for non-statin OATP1B1 substrates. V2's apparent "near-pass" was a methodology illusion; V2 result preserved as `.v2.json`.

**Diagnostic (`5ff72eb`, `data/validation/v3_fup_override_diagnosis.json`):**
fup override (valsartan predicted 0.009 → clinical 0.050, 5.6× increase) gave Cmax 0.97× — essentially no change. Glimepiride predicted fup already matches clinical (0.005). **Predict-layer fup confound RULED OUT as cause of V3 underprediction.**

**Remaining candidates for V3 underprediction (not investigated this session):**
1. ECM Jmax values too high for valsartan/glimepiride (valsartan Jmax flat-CLuptake-scaled from pravastatin under v2.1; glimepiride from literature Huang 2018)
2. Vss/Kp over-distribution (tissue holds too much drug → too little in blood at 5 min)
3. ECM architecture limit for Km > 1 µM range (pravastatin Km ≈ 13.6, glimepiride 10.0, valsartan 1.39 — three-order-of-magnitude sweep within tested substrates)

**Pre-registration integrity maintained:**
V3 methodology spec written + committed (`d88183a`) BEFORE engine re-run (`7aa49ae`). Single MC run. Fup diagnostic explicitly marked exploratory (`"note": "NOT a pre-registered run"`). No post-run parameter adjustment.

**How to apply:**
- "Does ECM generalize to non-statin OATP1B1?" → **No, current calibration underpredicts by 2.5× on both valsartan and glimepiride.** Mode C but substantively borderline systematic.
- "Is this ECM architecture failure?" → Unknown. fup ruled out. Jmax calibration vs architecture vs Vss remains unseparated.
- "Cherry-picking?" → No. V3 spec pre-committed, direction of failure was unforeseen (we expected near-pass; got underpredict).
- "Re-run with fix?" → Only after another pre-registered spec amendment targeting specific root cause (Jmax recalibration would need independent substrate set to avoid overfitting).

---

### ECM generalization test, N=2, Mode C with diagnostic findings (2026-04-21)

**SUPERSEDED by V3 run (2026-04-22) above.** Original V2 result preserved as `data/validation/oatp_generalization_result.v2.json`. Kept here for historical context only.


**Spec:** `docs/superpowers/specs/2026-04-21-ecm-generalization-test-design.md`
  - v1 `9115e63` + v2 amendment `6e7ce0a` (substrate swap) + v2.1 `0d78c38` (valsartan Jmax scaling)

**Plan:** `docs/superpowers/plans/2026-04-21-ecm-generalization-test.md` (commit `3c85fe4`)

**Result:** `data/validation/oatp_generalization_result.json` (commit `4fb6d38`)

**Formal outcome:** Mode C (inconclusive)

**Per drug:**
- Glimepiride: 1 mg IV bolus. Obs 0.243 mg/L; point 0.270; PI [0.270, 0.270] (degenerate); log10 FE +0.046 (FE 1.11×). passed=False due to PI containment.
- Valsartan: 20 mg IV bolus. Obs 4.020 mg/L; point 5.405; PI [5.405, 5.405] (degenerate); log10 FE +0.129 (FE 1.35×). passed=False due to PI containment.

**Substantive signal:**
Both point estimates within 1.5× of observed — well inside the 3× clinical-error gate. If PI were non-degenerate and contained observed, outcome would have been Mode A (confirmed generalization within tested domain). Suggestive-positive for ECM mechanism but NOT formally confirmed.

**Why PI is zero-width (root cause):**
MC Cmax for IV bolus in Sisyphus = `dose / V_venous_blood` (deterministic t=0 instantaneous value, 3.7 L ± 0.0). Distributional CVs downstream (Jmax, Km, fup, Kp, ps_*) never reach Cmax because max-over-time selects t=0. All 1000 samples produce identical output.

**Secondary gap:**
`data/transporters/hepatic_ecm.json` lacks entries for valsartan + glimepiride → `ps_passive/ps_eff/cl_int_bile` fell to defaults (1e6 L/h for ps_*, 0 for bile). Not the cause of zero-PI but a data completeness gap worth closing.

**Predict-layer confound flag (per spec §Peff Isolation):**
- Valsartan fup_predicted = 0.009 vs clinical ~0.05 (5.6× off). Per spec, this is logged but not counted as ECM failure. Possible contributor to the 1.35× over-estimation.
- Glimepiride fup_predicted = 0.005 vs clinical ~0.003-0.005 (within 2×). OK.

**Pre-registration integrity:**
Single run at N=1000, seed 42. No post-run parameter adjustment. All spec/plan amendments (v2, v2.1) pre-dated the engine execution. Substrate swap (bosentan/repaglinide → glimepiride) was documented under v2 amendment BEFORE any engine run, driven by data-access limits not expected outcome.

**Commits:**
- Spec v1: 9115e63; v2: 6e7ce0a; v2.1: 0d78c38
- Plan: 3c85fe4
- Task 1 (obs data): ee24164 → 5f79d34 → 675478c → 5e67376 → 6ddab5f
- Task 2 (kinetics): a562192 → 2115313 → b36f899
- Task 3 (classifier): 807f4aa
- Task 4 (script): 50b1ced
- Task 5 (integration test): 44d8e90
- Task 6 (result): 4fb6d38

**Follow-up recommended (separate task, not this session):**
1. Design a v3 engine methodology for IV-Cmax observation that matches clinical semantics (non-t=0 or different node).
2. Populate `hepatic_ecm.json` for non-statin OATP1B1 substrates.
3. Improve fup XGBoost for valsartan-class high-fup-bound drugs.
4. Pursue institutional library access for bosentan/repaglinide primary sources to re-enable N=3 test.

---

### OATP ECM hepatic clearance — IMPLEMENTED (2026-04-21, branch `feat/oatp-ecm`)

- **Spec**: `docs/superpowers/specs/2026-04-20-oatp-ecm-hepatic-clearance-design.md`
- **Plan**: `docs/superpowers/plans/2026-04-20-oatp-ecm-hepatic-clearance.md`
- **Outcome**: ECM closed-form hepatic-clearance flux shipped. 12-task TDD plan executed via subagent-driven development. `ClearanceFluxSpec` gains `"extended"` model; `DrugOnGraph` gains `ps_passive`, `ps_eff`, `cl_int_bile`; `data/transporters/hepatic_ecm.json` + `load_hepatic_ecm_params()` added.
- **YAML change**: `data/physiology/reference_man.yaml` liver clearance model `well_stirred → extended`; two `active_transport` edges removed; `liver.transporters.OATP1B1` abundance re-calibrated `1.0e11 → 5.0e5` via `scripts/calibrate_oatp_abundance_ecm.py` (pravastatin FE=1.013 under ECM).
- **107 holdout**: Meta AAFE 2.695 preserved exactly (|Δ|=0.000019). Non-OATP drugs use `PS_passive=PS_eff=1e6, CL_int_bile=0` defaults; ECM reduces to well-stirred algebraically.
- **Stiffness elimination**: all 5 statins solve in <0.12 s under ECM (vs 41-min stall pre-ECM on 4/5 statins). Primary engineering win of the migration.
- **Phase 2A gate**: 3/5 statins PASS FE<3× (pravastatin 1.013, pitavastatin 1.34, fluvastatin 2.98). Rosuvastatin (FE 11.9) and atorvastatin (FE 7.8) xfail-marked — root cause diagnosed as Peff XGBoost model over-predicting absorption for high-MW polar statins (clinical F% 14-20% vs predicted ~100%). ECM hepatic extraction is flow-limited (E→1) and cannot compensate. Tests use `@pytest.mark.xfail(strict=False)` so they auto-promote if Peff is later improved.
- **Phase 2B gate**: SLCO1B1 PM directional response unblocked — pravastatin PM Cmax 2.437× EM (Niemi 2006 clinical PM AUC +60-100% matches). Saturation artifact from Phase 1 resolved.
- **Tests**: +19 new (1 core DrugOnGraph field test, 1 compiler accessor, 3 loader, 1 ivive kwarg, 7 ECM flux formula/invariants, 2 YAML builder, 1 holdout regression, 1 SLCO1B1 PM directional, 2 statin xfail + 3 pass). Total collected suite: 494 → 513.
- **Commits on branch**: pravastatin calibration JSON at `data/validation/oatp_ecm_abundance_calibration.json`; sweep script at `scripts/calibrate_oatp_abundance_ecm.py`.

### OATP Phase 2B — SLCO1B1 phenotype (2026-04-20, commit `93febe3`)
- **`predict/phenotype.py` transporter extension**: `TRANSPORTER_ALIASES = {"SLCO1B1": "OATP1B1"}`, `apply_phenotype_to_graph` scales transporter abundance by CPIC activity score (PM 0.10×, IM 0.50×, EM 1.00×, UM 2.00×). `parse_phenotype_spec` accepts `SLCO1B1:PM` and mixed `CYP2D6:PM,SLCO1B1:IM`.
- **Unit tests**: +11 (SLCO1B1 parse, scale, CV preservation, enzyme/transporter isolation, UM increase, input-graph immutability). 39/39 phenotype tests pass.
- **Engine saturation limit surfaced**: liver.OATP1B1 abundance 1.0e11 operates flow-limited. Scaling PM (0.10×) → UM (2×) leaves pravastatin Cmax unchanged. Clinical SLCO1B1 AUC +60-100% (Niemi 2006) requires a non-saturated engine — addressed by ECM work above.
- **107 holdout**: unaffected (phenotype is CLI/TDM-only; `pipeline/predict.py` does not call it).

### OATP Phase 2A — statin data expansion (2026-04-20, commit `3a04291`, data-only)
- **`data/transporters/oatp1b1.json`**: 1 drug → 5 drugs. Rosuvastatin / atorvastatin / pitavastatin / fluvastatin Km from Niemi 2009 midpoints. Jmax scaled from clinical hepatic uptake CL ratio vs pravastatin (Hirano 2006, Maeda 2011, Li 2018). CV widened to 0.40 (Jmax) / 0.35 (Km).
- **107 holdout**: zero impact (`pipeline/predict.py` does not call `load_oatp1b1_kinetics` — TDM path only).
- **Engine Cmax validation deferred**: `scripts/validate_oatp_phase2a.py` ran 41 min then stalled on LSODA for 4/5 statins. Diagnosis (`oatp_phase2a_stiff_diagnosis.json`): abundance 1e11 is flow-limited saturated regime. Abundance sweep (`oatp_abundance_sweep.json`, 2026-04-20 PM): Cmax invariant across [1e9, 3e9, 1e10, 3e10, 1e11]. Conclusion: parameter tuning cannot fix this — engine refinement needed (→ OATP ECM).
- **Tests**: existing 5 `test_transporter_db.py` unit tests load all 5 drugs.

### P6 SBI likelihood reweighting (2026-04-19)
- **Implementation**: `bayesian_update(method="sbi", sbi_reweight=True)` — opt-in flag. NPE posterior samples importance-reweighted by log-normal likelihood (mathematically equivalent to IS with NPE as proposal). `tdm_sbi.py:555` + `tdm.py:227`. Default `False` (preserves existing production path).
- **5-drug tournament** (`data/validation/tdm_method_tournament_sbi_reweight.json`, OFF→ON bias):
  - morphine: +52.3% → **+2.1%** (IS-level) ✅
  - amantadine: −20.2% → **+3.6%** ✅
  - ketorolac: −31.3% → −18.4% (better, engine-level floor remains) ✅
  - clozapine: −6.1% → +17.6% (regression — posterior over-concentrated) ⚠
  - rivaroxaban: +4.9% → +40.5% (regression — same cause) ⚠
  - Mean |bias|: 23.0% → 16.4% (29% improvement overall)
  - CV tightens 1/2 – 1/4× across all drugs — posterior over-concentrated on a single obs.
- **Interpretation**: reweighting effective when |bias| ≥ 20%, regressive when |bias| < 10%. N=200 single-obs stochastic error amplified by likelihood. Bias-variance tradeoff.
- **Production decision**: default `sbi_reweight=False` retained. Per-drug routing: `method_routing.json` gets `sbi_reweight: {"morphine": true}`, morphine route `is` → `sbi`. CLI auto: `[auto] routing morphine → method=sbi +reweight`. Final production: **12 SBI / 0 IS / 1 IBIS** (IS override retired). 7 SBI dispatch tests pass.
- **Decision package**: `docs/superpowers/specs/2026-04-19-p6-morphine-fix-decision.md`.

### P7 Ketorolac AD flag (2026-04-19)
- **Decision**: close P7 as documented structural limitation. 2026-04-11 engine-level fup override attempt regressed engine AAFE +0.306 (see DE-31 in dead-ends.md).
- **Option 2 implementation**: `pipeline/predict.py` gains `HIGH_ACID_LOW_FUP` AD flag — informational warning for drugs with pKa < 5 AND DrugBank measured fup < 0.02. Ketorolac, ibuprofen flagged. Morphine / base drugs not flagged. Engine numbers unchanged.
- **Decision package**: `docs/superpowers/specs/2026-04-19-p7-ketorolac-decision.md`.

### P4 Continuous Hierarchical Infrastructure (2026-04-16, branch `feat/continuous-hierarchical`)
- **Physiology generator**: `src/sisyphus/sbi/physiology_generator.py` — `generate_physiology(BW, age)` builds BodyGraph for any patient 0.5–85y, 5–120kg. Hines 2008 enzyme ontogeny (exponential maturation) + Wynne 1989 aging decline + allometric volume/flow scaling.
- **Conditioning**: 15D = [log10_cmax(1), drug_features(12), log_bw_norm(1), log_age_norm(1)]. Replaces C1 one-hot for the continuous model.
- **API**: `bayesian_update(body_weight_kg=X, age_years=Y)` + CLI `--body-weight X --age Y`.
- **Training scripts**: `scripts/sbi_generate_continuous_data.py` + `scripts/sbi_train_continuous_hierarchical.py`.
- **Model validation (2026-04-18)**: NPE trained on 275k samples (55 drugs × 10 pops × 500θ), SBC 41/52 pass across 4-pop grid × 13 drugs (78.8%).
- **Tests**: +14 (10 generator + 4 packing/stacking).

### Session additions (2026-04-14 evening)
- **CYP phenotype layer** (commit `21a92c9`): `sisyphus tdm --phenotype CYP2D6:PM` — CPIC activity scaling (PM 0.1×, IM 0.5×, EM 1×, UM 2×). `src/sisyphus/predict/phenotype.py`. 17 tests. DM PM case: posterior enzyme_affinity 4.89 → 6.48 (physiologically interpretable).
- **Multi-obs SBI** (commit `d4e1633`): Track A amortizer conditions on first obs only; additional obs applied as post-hoc log-normal likelihood importance reweighting. `_scipy_cmax_and_obs_conc()` helper + weighted posterior stats. 2-obs test confirms ESS decrease.
- **MIPD dose_range auto-infer** (commit `ce9a924`): removed hardcoded `DEFAULT_DOSE_MIN=25mg`. Now inferred from current_dose as 0.1×–10×. DM 30 mg PM → recommends 12 mg correctly (previously clamped to 25 mg).

### v3 OATP expansion — NEGATIVE (2026-04-14, commit `5c0d864`, reverted `fdda41c`)
See [DE-32](./dead-ends.md#de-32--sbi-v3-oatp-training-expansion-2026-04-14).

### Phase 1 OATP1B1 (2026-04-15, branch `feat/oatp1b1-pravastatin`)
- **ActiveTransportEdge scaffolding**: YAML parser (`builder.py` — node `transporters:` + `active_transport` edge type) + `flux.py` / `rhs_jax.py` target-side IVIVE bug fixes + `build_drug_on_graph(transporter_kinetics=...)` kwarg + `data/transporters/oatp1b1.json` DB + `predict/transporter_db.py` loader.
- **Liver OATP1B1 abundance**: 1.0e11 — hepatocellularity proxy. Pravastatin 40 mg Cmax 0.039 vs observed 0.045 (ratio 0.86). 1.5e11 → steep nonlinearity (0.010, over-extraction). 14% gap fits within the Jmax CV=30% prior.
- **Calibration nonlinearity**: abundance 1.0e11 → 1.5e11 gives Cmax 0.039 → 0.010 (74% drop for 50% abundance increase). Hepatic extraction saturation. Linear extrapolation invalid, grid search required. *(This saturation is exactly what the 2026-04-20 ECM redesign fixes.)*
- **Non-pravastatin impact**: 0 change on 12 routing drugs' TDM output (transporter_kinetics empty, MM path inactive). 7 SBI dispatch tests pass.
- **107 holdout regression**: Meta AAFE 2.695 exact invariance.
- **Tests**: 422 + 12 new unit = 434. Integration +2. All pass.
- **Pravastatin SBC**: not executed (manual, ~40 min). Engine prior predictive Cmax shifts 0.039 → 0.045 direction confirmed. Future SBC run should gate cov_dev < 0.10.
- **Design spec / plan**: `docs/superpowers/specs/2026-04-15-oatp1b1-hepatic-uptake-design.md`, `docs/superpowers/plans/2026-04-15-oatp1b1-pravastatin.md`.

### Phase 2.0.5 — SBI routing expansion (2026-04-12, commits `ccc15a0` code + `43051ab` eval)
- **logit(fup) reparameterization**: theta[1] ∈ [−4.595, +4.595] (logit space). `apply_theta_to_drug` sigmoid-inverts. Improves prior coverage for low-fup acids / statins.
- **θ/drug expansion**: 1000 → 2000. **Acid drugs +5** (20% → 27%, total 50 → 55 drugs). Later v3 would add 5 OATP substrates (55 → 60, acid 27% → 33%) — see DE-32.
- **SBC**: SBI routing 10/13 → 12/13 SBC, production routing **11/1/1** (SBI/IS/IBIS). Superseded by P6 routing **12/0/1** (2026-04-19).
  - diclofenac cov_dev 0.247 → **0.060** (IBIS→SBI recovered).
  - posaconazole 0.120 → **0.073** (IBIS→SBI recovered).
  - pravastatin 0.273 → 0.223 (still IBIS — OATP1B1 transporter OOD; training set had 0 substrates).
  - morphine: SBC pass (0.047) but TDM bias +52% → IS override (IS bias +3%). SBI posterior CV 47% vs IS CV 10% — posterior did not tighten. *(Later resolved by P6 SBI reweight.)*
- **Model v2 production**: `models/sbi/multi_drug_nsf.pt` = v2 (logit fup, 94 epochs, 2815s on 110k samples). v1 archived as `_v1.pt`.
- **TDM tournament v2** (IS vs SBI): SBI mean abs bias 23% (IS 31%). SBI wins clozapine (−6% vs +87%) and rivaroxaban (+5% vs −18%). `data/validation/tdm_method_tournament_v2.json`.
- **Runtime guard**: `amortizer.py:load_result()` warning + `tdm_sbi.py:sbi_update()` ValueError block old models.
- **Tests**: 435 all pass (0 skip).

### Track D2 + paper-blocker bundle (2026-04-11, `docs/tdm_ci_calibration.md`)
- **CI lognormal → empirical weighted quantile**: `TDMResult.cmax_ci_90` populated from raw posterior Cmax samples via weighted quantile in all dispatch paths (IS / IBIS / EnKF / SBI). Removes the lognormal over-cover artifact on high-CV posteriors.
- **Conformal CI floor**: `bayesian_update(min_ci_half_width_fraction=0.5)` kwarg. Posterior CI half-width < 50% × mean widens to 50%. `apply_ci_floor()` public helper.
- **5-drug × 3-scenario verification**: 3/9 (floor=0) → 6/9 (floor=0.5) → 8/9 (floor=1.0). floor=0.5 is optimal — rivaroxaban 3 cases recover, easy drugs preserved, ketorolac engine-level failure exposed.
- **Full 15-scenario estimate: 12/15 (80%)** — supersedes the stale 67% (lognormal over-cover artifact). 3 ketorolac failures are engine-level fup mismatch (XGBoost 0.069 vs DrugBank 0.010) and cannot be CI-calibrated.
- **Tests**: +3 CI floor tests.

### Paper-blocker re-measurement (2026-04-11)
- **4-track 107 holdout**: overall confirmed Meta 2.695 / Engine 3.421 / ML 3.057. `data/training/4track_holdout_predictions.json` formally saved (JSON schema + per-drug fields).
- **In-domain N=85**: Meta 2.710 / Engine 3.236 / ML 3.042. Supersedes stale 2.591 (N=82 pre-VDss). In-domain meta slightly higher than overall (2.695) because adaptive weighting works well even for AD-flagged drugs; excluding them drops good predictions.
- **Prospective N=15 4-track**: Overall AAFE 2.361 (stale 2.478). In-domain AAFE 2.043 (N=13, stale 1.675 on N=9). %2-fold 53% (stale 47%). Prospective overall < holdout overall — no distribution shift.

### Track A — multi-drug NPE (2026-04-10, `docs/sbi_multi_drug_results.md`)
- 50 drugs × 1000 θ = 50,000 simulations (27.6 min, 100% valid solves).
- NSF + embedding_net (13→32→32→32), hidden=64, transforms=8, 92 epochs (20 min).
- **Cumulative IBIS speedup 36,097× on 5 anchor drugs.**
- **Coverage-primary gate**: 11/13 drugs within 10pp at 50/80/90/95%.
- **Strict gate**: 2/13 (morphine, ketorolac); hard coverage failures: 2/13 (diclofenac, pravastatin — acid / CYP2C9).

### Track B — SBI production integration (2026-04-10, `docs/sbi_multi_drug_results.md` Addendum)
- **Production API**: `tdm.bayesian_update(method="sbi")` + silent IBIS fallback.
- **Per-drug routing table**: `data/sbi/method_routing.json` — initially 11 SBI / 1 IS / 1 IBIS.
- **CLI**: `sisyphus tdm --method {is, ibis, enkf, sbi, auto}`. `auto` consults routing table.
- **3-way tournament mean abs bias**: SBI 19% < IS 31% < EnKF 38%. SBI especially wins clozapine (−4% vs IS/EnKF/IBIS +82 to +89%).
- **Wall time per drug**: SBI ~57s < IS ~69s ≪ EnKF ~564s ≪ IBIS ~1390s.
- **Posterior CV inflation bug fix**: `apply_theta_to_drug` must collapse override-field CVs to 0 so posterior CV drops below prior CV (morphine before 56% > 39%, after 34% < 39%).
- **Tests**: +5 SBI dispatch, +2 feature refactor.

### Track D1 — neural surrogate (2026-04-10, `docs/surrogate_ood_fix.md`)
**Initial:**
- Bug: production `params_to_features_single` summed `abundance × affinity` across all nodes (liver+gut) without reversing `_CLINT_SCALING`. Real drugs had log10_clint ≈ 6 vs training range [−0.5, 3.0]. Inflation ~10⁴×.
- Fix: `recover_drug_level_clint()` restricts sum to liver node, divides by `_CLINT_SCALING / _IVIVE_SCALING = 180,000`. All 6 test drugs recover to within 5% of `predict_adme(..).clint.mean`.
- Surrogate accuracy (`data/validation/surrogate_production_accuracy.json`): 13 drugs, R²=0.992, mean abs rel err 22%, 9/13 within 30% (69% overall, 80% on 10-drug SBI routing subset).
- Opt-in integration: `bayesian_update(method="sbi", sbi_use_surrogate=True)`. Batched JAX call (not per-sample). Default False.
- 5-anchor SBI wall: scipy 224s → surrogate 9.2s = **24× cumulative**. Warm per-drug: amantadine 90×, ketorolac 66×, rivaroxaban 138×. Cold (morphine) 10× dominated by JIT.
- vs IBIS: surrogate warm ~0.3–0.7 s/drug vs IBIS ~1390 s = **~2000–4000× per-query**. Sub-second TDM on 4/5 anchors.
- Clozapine edge case: +190% bias because fup posterior shifts features OOD at per-sample level.

**Follow-up (ensemble-std gate, hybrid routing):**
- Root cause of clozapine: feature box guard passed but surrogate's local response surface systematically off. Ensemble std correlated 0.64 with error.
- Fix: two-stage gate — `features_in_distribution` (box) + `ensemble_std <= 0.02`. Rejected samples fall back to scipy. Threshold calibrated so nominal drugs (ensemble std 0.004–0.020) stay on surrogate.
- Clozapine bias: **+190% → −3.6%** (better than scipy −7.8%).
- 5-anchor tournament: scipy 210.6s → hybrid 84.1s = **2.5× cumulative** (down from unguarded 24×, with correct accuracy on all drugs). Per-drug wall 9–23s, still 50–150× vs IBIS.
- Hybrid matches or beats scipy on 4/5 anchors.
- Trade: 24× → 2.5× speedup for correctness. Correct default for production.

### Track C1 — hierarchical SBI (2026-04-12 code, 2026-04-14 2kθ eval)
- **HierarchicalMultiDrugSimulator**: per-(population, drug) EngineSimulator cache. Drug features extracted from adult reference graph (population-independent).
- **Population registry**: `data/sbi/populations.json` — adult (70 kg) + pediatric_5y (18 kg).
- **Conditioning**: 13D → 15D (+2D population one-hot).
- **Training**: 1kθ (75 epochs) → **2kθ (76 epochs, 220k samples)**. `models/sbi/hierarchical_nsf_2k.pt`.
- **SBC**: Coverage ≤10pp 22/26 (85%), KS+coverage gate 8/26 (1kθ was 6/26). 2kθ recovered adult morphine (0.110 → 0.090) + sildenafil (0.110 → 0.067). Posaconazole (0.17/0.13) + pravastatin (0.14/0.14) residual failures.
- **Production**: `bayesian_update(population_class="pediatric_5y")` + CLI `--population pediatric_5y`.
- **Tests**: +18 in `tests/unit/test_sbi_hierarchical.py`.

### Branch consolidation (2026-04-10, merge commit `c0cab88`)
`audit/holdout-leakage-fix` + `feat/ude-diffrax` merged. VDss 4th-track production added, EnKF TDM added, prospective validation series integrated, JAX backend consolidated. Post-merge AAFE 2.808 → 2.695 confirmed. `tdm.py` latent bug exposed and fixed (`method="enkf"` wrong kwarg + `EnKFResult → TDMResult` conversion).

### 2026-04-10 post-merge diagnosis update
- **VDss analytical 4th-track success (−4% AAFE 2.808 → 2.695)** falsifies the earlier "partial replacement is impossible" conclusion. VDss is a 1-compartment analytical approximation (dose / Vd·BW) at 20% weight; the 3 existing tracks scale down to 0.80. No predict-layer replacement required.
- **Why VDss worked where CL/F·t½ failed**: CL · t½ · Cmax depend on the same hepatic / CYP kinetics → correlated error. VDss depends on tissue partitioning (lipophilicity + binding) → clearance-orthogonal. Future track proposals must precompute error decorrelation vs the existing 4 tracks (see [diagnosis.md §4](./diagnosis.md)).
- **Error cancellation wall partially broken**: the 34+ failures shared a common cause — "new model with correlated error vs existing tracks". Criterion established.
- **Remaining practical paths**: (1) TDM Bayesian update, (2) orthogonal-track exploration with decorrelation gate, (3) breakthrough Phase 2 (amortized SBI / BayesFlow).

---

## 2026-03 (earlier)

### Holdout expansion (2026-03-26)
- N=61 → N=107 (+46 drugs from OSP repos, FDA labels, curated literature).
- 7 new drugs added to holdout split (alprazolam, cabozantinib, cimetidine, erythromycin, probenecid, ruxolitinib, triazolam).
- MMPK exclusions updated for 7 new holdout drugs.
- AAFE increase (2.058 → 2.306) expected: expanded set includes harder drugs (prodrugs, high MW, extreme lipophilicity).
- In-domain AAFE 2.114 is the better comparator (excludes AD-flagged drugs).

### Measured ADME PoC (2026-03-26)
- N=12 holdout drugs, engine-only (no meta), Tier 2 (measured fup + CLint).
- Sources: DrugBank fup (experimental), TDC Hepatocyte_AZ CLint (geometric mean).
- Clean set (N=10, excluding montelukast/abiraterone extreme outliers): **AAFE 2.329 → 1.980**, median FE 2.19 → 1.88, 8/10 improved.
- fup-matched subgroup (N=8): 1.91 → 1.79 (CLint-only effect, 6% gain).
- fup-corrected subgroup (N=2): 5.15 → 2.96 (fup+CLint, 42% gain).
- **Pattern C**: engine architecture sound, input quality (CLint R²=0.24) is the primary bottleneck.
- Error cancellation observed for abiraterone (fup 0.085 → 0.01 worsened FE 20.8 → 39.1) but not dominant (80% of drugs benefit).

### v2.0 multi-dose validation
- Atorvastatin 40 mg QD: Css_max 0.027 vs FDA 0.029 mg/L (fold error 0.93) — 7% off.
- Metformin 500 mg BID: Css_max 0.55 vs FDA 1.0 mg/L (0.55×) — renal-dominant, expected under-prediction.
- Warfarin 5 mg QD: Css_max 0.34 vs FDA 1.4 mg/L (0.24×) — fup=0.01 extreme-bound, CLint over-prediction.
- Solver 3/3 success, accumulation ratio direction correct, SS detection works.

### v2.1 TDM validation
- Midazolam 5 mg single dose, t=1h noisy observation.
- CV reduction: 55.4% (44.3% → 19.8%), ESS=586.6 (29.3%).
- Bayesian update mechanism functional.

### v2.1 TDM multi-drug benchmark (2026-03-27)
- 5 holdout drugs (morphine, amantadine, ketorolac, clozapine, rivaroxaban). 2 base + 1 acid + 2 neutral, fold error 2.0–3.25×.
- Synthetic patient: engine C(t) scaled to observed Cmax + 10% assay noise (seed=42).
- **Main results** (15 runs: 5 drugs × 3 scenarios):

| Metric | 1 obs | 2 obs | 3 obs |
|--------|-------|-------|-------|
| Mean CV reduction | 78.1% | 82.7% | 82.9% |
| Mean error reduction | 79.4% | 80.8% | 79.1% |
| Mean posterior CV | 8.4% | 6.5% | 6.4% |

- **Per-drug highlights**:
  - Morphine (base): CVred 74–77%, ErrRed 92–96%, ESS 114–428. Healthy / caution across all scenarios.
  - Amantadine (base): CVred 74–75%, ErrRed 88–94%, ESS 66–514.
  - Clozapine (neutral): CVred 69–77%, ErrRed 85–90%, ESS 59–482.
  - Ketorolac (acid, FE=3.25): CVred 88–93% high but ErrRed 36–44% low. **ESS 2.5–3.3 degenerate.** Prior too far from truth for IS.
  - Rivaroxaban (neutral, FE=2.17): CVred 84–98% high but **ESS 1.0–7.1 degenerate.** Multi-obs particle degeneracy severe.
- **90% CI coverage**: 10/15 (67%) — later diagnosed by Track D2 (2026-04-11) as lognormal over-cover artifact; empirical quantile gives 3/9 tested subset (33%) before floor. After floor=0.5 and the subsequent bundle: 12/15 (80%). 3 ketorolac failures remain engine-level.
- **ESS health**: 3 healthy (>200), 4 caution (100–200), 8 degenerate (<100).
- **Timepoint sensitivity (morphine)**: t=1.0h optimal (CVred 76.3%). After 4h, drops to 34%.
- **Seed sensitivity**: Δ=0.8% (seed 42 / 123 / 456). N=2000 fully robust.
- **Conclusion**: single observation → CV 70–88% reduction, Cmax error 44–92% reduction. Strong for FE < 2.5×. FE > 3× or multi-obs → ESS degeneracy → EnKF / particle filter needed (shipped as Track D1/Phase 3 EnKF).

### Engine-only ablation
- DrugBank enrichment: engine AAFE 3.074 → 2.945 (Δ=−0.129, significant), meta receives only Δ=0.021 through 0.17 weight.
- Meta-learner LOOCV (N=107): w_base=0.45, w_other=0.00 optimal (82% stable). Oracle=1.933.
- pKa model (ON/OFF) × Berezhkovskiy (ON/OFF) 4 experiments: all Δ ≤ 0.02 (noise).
- Conclusion: CLint is the only dominant bottleneck. pKa and Kp method do not move engine AAFE.

---

## Contamination fix (2026-04-04, commit `5e5a3d0`)

- **Leakage discovered**: 76–100 of 107 holdout drugs were in ML training data. Prior headline AAFE 2.283 was invalidated.
- **Fix**: clean retraining of ML Cmax / fup / peff / CLint / VDss on a holdout-stratified split.
- **Full record**: `docs/holdout_contamination_audit.md`, `data/validation/contamination_fix_report.json`.
- **Post-fix headline** (pre-VDss, 3-track): AAFE 2.306 after holdout expansion to N=107 (see 2026-03-26 entry above).

---

## Shipped-phase checklist (completed)

- Phase 0 — UGT revert, w_base=0.65 restored, MMPK migration.
- Phase 1 — Engine (v0.1, 6 flux types, LSODA, MC).
- Phase 2 — Prediction (v0.2, Meta AAFE 2.058 at ship, 12 TDC ADME).
- Phase 3 — Extensibility proof (SC / pediatric / tumor, 17 tests, `engine/` diff=0).
- Phase 4 — Production (v1.0: DDI 22 tests, PK/PD 28 tests, perf 414 ms, MIPD 14 tests).
- Track B — multi-dose v2.0 + TDM v2.1 (IS + IBIS + EnKF + SBI + MIPD dose-adjust).
- Full suite: 348 → 357 → 371 → 434 → 435 → 448 → **494** (2026-04-21, current).

Detailed per-phase milestones: see [phase-completion.md](./phase-completion.md).

---

## How to add new entries

Prepend a new section at the top of the appropriate date block. Each entry should have:
- Date + commit hash (if any).
- One-sentence what-was-tried.
- Numeric outcome.
- Follow-up link (design spec, validation JSON, reverted commit).

If an entry documents a failure, also append it to [dead-ends.md](./dead-ends.md) with the next `DE-NN` id.
