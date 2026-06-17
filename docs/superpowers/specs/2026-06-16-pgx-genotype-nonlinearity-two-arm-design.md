# PGx genotype-nonlinearity validation — two mechanisms (systemic + first-pass)

> **Lineage.** Successor to the single-dose v2.2b spec (`2026-06-15-pgx-cmax-v2b-nonlinear-genotype-design.md`, SUPERSEDED at its §7 spike — single dose + in-vitro `Km` did not robustly engage MM saturation). Reuses the v2.2a Michaelis–Menten flux (`DrugOnGraph.enzyme_km`), the v2.1 EM-anchored `well_stirred` skeleton, the saturable harness (`scripts/validate_pgx_cmax_v2b.py`), the `regimen` multi-dose solver, and the **(A) axial phenotype-scaling fix** (PR #79 — `apply_phenotype_to_graph` resolves axial sub-tanks). Charter: `2026-06-09-engine-as-prior-mipd-charter.md` lineage of the PGx thread.

## 1. Purpose & non-goals

**Purpose.** Demonstrate that the saturable MM engine, with **literature `Km` on the gene fraction (never fitted)**, reproduces genotype-stratified **nonlinear dose-dependence** — a signature no linear or closed-form model can produce — across two mechanistically distinct regimes the engine handles by construction:

- **Arm S (systemic saturation):** at steady state, multi-dose accumulation raises liver `C_u` toward `Km`; the gene-active genotype's clearance falls with dose. `well_stirred` liver + `solve_regimen`. **Primary clean case: phenytoin (CYP2C9).**
- **Arm F (first-pass saturation):** a single oral dose drives the **hepatic-inlet** `C_u` (which `well_stirred` averages away) toward `Km`; the gene-active genotype shows saturable presystemic metabolism (rising F with dose). **Axial `parallel_tube` liver** (`expand_axial`) + the (A) axial phenotype scaling. **Primary clean case: propafenone (CYP2D6)** — the textbook *EM-nonlinear / PM-linear* drug; this exercises the axial unlock directly.

**Non-goals (explicit).**
- **Validation, not productization.** Harness-isolated; **no `predict()` / `reference_man.yaml` / holdout change**. Headline **2.731 bit-identical** (regression-pinned). `enzyme_km` is set only on synthetic skeletons.
- **No fitting, no cherry-picking.** `Km`, `fm`, `fu_mic` are literature/in-vitro and fixed; never tuned to a fold. The Task 0 gate (§4) requires engagement across the **entire** `Km × fu_mic` uncertainty box, foreclosing the single-favorable-`Km` artifact that sank single-dose v2.2b.
- **No mechanism-based inhibition.** Reversible MM only. **Voriconazole is excluded from the clean set** — its nonlinearity is substantially auto-inhibition (TDI/MBI), which v2.2a does not model (→ v2.3). It may appear *only* as a labeled confounded-secondary illustration, never as a validation case.
- **No headline accuracy claim.** The primary deliverable is a **capability-existence demonstration** (§5), not a statistical accuracy delta. Expected clean-N ≈ 2 (one per arm) is a PASS by design.

## 2. Background: why dose-dependence, not the fixed-dose fold

For a drug cleared by gene `G` (fraction `fm`) plus a linear residual `(1−fm)`, the **linear-null** oral AUC genotype fold is exactly `1/(1−fm)` (proven on this skeleton at both high and low extraction — v2.2b spike §Gate-4). The saturable engine can only ever predict a fold **≤ `1/(1−fm)`**: the gene-active (EM) arm saturates as `C_u → Km`, its clearance falls toward the residual, and the fold shrinks *toward 1*. PM (gene off → linear residual, no `Km`) does not saturate.

Two consequences fix the metric:
1. **Fixed-dose fold-compression is not generally discriminating.** Where observed fold ≥ `1/(1−fm)` (e.g. voriconazole ~2.5–4× vs `1/(1−0.6)=2.5×`), saturation moves the fixed-dose prediction *away* from observed. So a single-dose fold comparison cannot be the primary test.
2. **The dose-trend is the robust, theory-guaranteed signature.** As dose rises, the gene-active arm saturates → within-genotype **supra-proportionality** (AUC/Css rises faster than dose) and the genotype **fold shrinks with dose**. Both are correctly-signed for phenytoin and propafenone, both are produced by the saturable engine, and both are **flat / dose-proportional** under the linear-null. This is the primary metric (§5).

## 3. Arms, drug tiers, and controls

| | **Arm S — systemic** | **Arm F — first-pass** |
|---|---|---|
| Engagement | steady-state accumulation (`C_u → Km`) | hepatic-inlet `C_u → Km` |
| Liver model | `well_stirred`, multi-dose (`solve_regimen`) | axial `parallel_tube` (`expand_axial`), single-dose |
| Genotype scaling | `apply_phenotype_to_graph` (non-axial) | (A) axial sub-tank scaling (PR #79) |
| **Primary clean** | **phenytoin** (CYP2C9) | **propafenone** (CYP2D6) |
| Confounded-secondary | voriconazole (CYP2C19, TDI/MBI) — labeled, not a validation case | — |
| At-risk (drop if no clean dose×genotype data) | — | mexiletine (CYP2D6, weak nonlinearity) |
| Negative control | tolbutamide (CYP2C9, clean linear) | metoprolol (CYP2D6, *mild* saturable first-pass → "near-flat" control, caveated) |

Each arm **gates and HALTs independently** (§4): an arm ships if **≥1 clean drug passes**; **0 clean → that arm HALTs and is reported** (not padded). Dropping to 1 clean drug in an arm is a PASS.

## 4. Task 0 — the front-loaded gate (data availability + box robustness; per-arm HALT)

Mandatory, before any scoring harness is built. Two checks per candidate, at the arm's operating regime (steady-state for S, axial first-pass for F):

**4.1 Data-availability.** The locked dataset (§7) must contain, for each *primary* drug, **≥2 dose arms** of genotype-stratified observed exposure (phenytoin dose-Css; propafenone ≥2 oral doses), each with a literature `Km` (+ basis + `fu_mic` + source), `fm`, F, and the EM anchor PK. A primary drug without dose-ranging genotype data is **dropped from primary** (→ at-risk tier). If both primaries in an arm lack it, the arm HALTs.

**4.2 Box-robustness engagement.** Probe the engagement statistic `|Δlog(metric)_{saturable − linear-null}|` (metric = AUC-fold; `Δlog` in natural log, matching the v2.2b spike gate) across the **entire plausible `Km × fu_mic` box** (both ends of the literature `Km` spread × `fu_mic ∈ {0.3, 0.6, 1.0}`), in the arm's regime. **PASS requires `|Δlog| > 0.10` at *every* box corner** (the v2.2b spike threshold) — engagement only at the favorable corner is a FAIL (the cherry-pick foreclosure). Consolidates and supersedes the ad-hoc 2026-06-16 probes (voriconazole steady-state; propafenone/mexiletine axial), which are re-run as the formal gate.

If 4.1 or 4.2 fails for an arm's whole clean set, **HALT that arm and report** — the engine may not reach `C_u ~ Km` robustly in that regime, which reshapes the milestone (exactly as single-dose v2.2b was reshaped).

## 5. Method & metric

### 5.1 Three engines per drug per dose (reused from v2.2b §3)
1. **EM-saturable** — the EM-anchored skeleton (matches observed EM PK by construction; literature `Km` fixed; hepatic CLint split `fm : (1−fm)` between the `Km`-carrying gene enzyme and a no-`Km` `RESIDUAL_HEPATIC` tag).
2. **PM-saturable** — gene scaled to 0 (residual only, linear), saturation active; predicts the genotype contrast.
3. **Linear-null** — the **same** skeleton re-anchored at `Km=∞` (gene CLint `=Vmax/Km` constant, same `fm`, same EM AUC), EM and PM. Isolates *exactly* what the MM flux adds.

Arm S evaluates engines at **steady state** via `solve_regimen` (reusing the single-dose compiled skeleton + params). Arm F evaluates them on the **axial** skeleton at single dose, with genotype applied via the (A) fix.

### 5.2 PRIMARY — pre-registered capability signatures (per drug)
- **P1 within-genotype supra-proportionality.** Statistic: the log–log exposure-vs-dose slope `β = d log(AUC or Css) / d log(dose)` in the gene-active genotype. Observed `β_obs > 1` (supra-proportional). PASS: the **saturable** engine's `β_sat` matches `β_obs` within a pre-registered tolerance (plan-pinned, e.g. `|β_sat − β_obs| < 0.15`) while the **linear-null** gives `β_lin = 1.00` (dose-proportional by construction) and lies outside that tolerance. The discriminating gap is `β_sat − β_lin`.
- **P2 genotype-fold dose-trend.** Statistic: the sign and magnitude of `Δfold = fold(high dose) − fold(low dose)`. Observed `Δfold_obs < 0` (fold shrinks as the gene-active arm saturates). PASS: `Δfold_sat < 0` matching the observed sign (and magnitude where data permit), while the linear-null gives `Δfold_lin = 0` (constant fold across dose).
- **Negative control N1.** tolbutamide / metoprolol show **no** saturation signature (saturable ≈ linear-null ≈ observed; no spurious supra-proportionality or fold dose-trend). A saturable engine that invents nonlinearity on a linear drug FAILS. Metoprolol carries a documented mild-first-pass caveat (its "flat" tolerance is wider).

**Success (capability-existence):** ≥1 clean drug per shipping arm passes P1 **and** P2 with correctly-signed, observation-consistent behavior the linear-null cannot produce, and the negative controls pass N1.

### 5.3 SECONDARY — accuracy delta (supporting context, honest-negative)
Per drug per dose, report `Cmax`/`AUC` fold from the saturable vs linear engine and `|obs_fold − pred_fold|`. Aggregate paired delta (Wilcoxon) reported as **context only** — *not* the headline; ties within the bootstrap CI are reported as such. The fixed-dose fold-compression sub-claim (saturable < linear nearer observed) is asserted **only** for drugs where observed fold < `1/(1−fm)`.

### 5.4 Oracle (C2, reused)
The v2.2a saturation oracle holds on **both** skeletons: engine MM rate = analytic `Vmax·C_u/(Km+C_u)`; the linear engine's oral AUC-fold = analytic `1/(1−fm+fm·a)` (low extraction). Pins the machinery independent of the clinical data.

## 6. `Km` conversion (reused, correctness-critical)
`km_uM_to_unbound_mgL(km_uM, mw, fu_mic) = km_uM × fu_mic × mw / 1000` (literature total-microsomal µM → engine unbound mg/L, the basis of `C_u = fup·c_plasma`). A pure, unit-tested function in `src/sisyphus/validation/pgx_metrics.py` (relocate from the v2b spike script if currently inline). The dataset records the **raw** `Km` + units + basis + `fu_mic` + source per drug (non-circular: `Km`/`fu_mic` are in-vitro, never the clinical fold). Where `fu_mic` is unavailable, treat `Km` as already-unbound with a stated assumption — and the §4.2 box already spans the `fu_mic` range.

## 7. Locked dataset & schema guard
New `data/validation/pgx_genotype_nonlinearity_folds.json` (locked; never refit). Per drug:
- `arm` (`systemic` | `first_pass`), `gene`, `tier` (`clean_primary` | `confounded_secondary` | `at_risk` | `negative_control`), `liver_model` (`well_stirred` | `parallel_tube`), `regimen` (Arm S), flags (`is_mbi`, `is_nonlinear`).
- **≥2 dose rows** for clean-primary drugs, each: dose, observed genotype-stratified endpoint (Css or Cmax/AUC PM/EM + CI), `tmax`/`t½` (EM anchor), oral F (+ gut/hepatic split).
- Raw `Km` (µM + basis + `fu_mic` + source), `fm` (+ source), citations.

**Schema guard** (`tests/regression/`): clean-primary rows require a literature `Km`, ≥2 dose rows, a non-circular `Km`/`fm`, and **reject `is_mbi=true`** (voriconazole may exist only as `tier=confounded_secondary`, excluded from primary scoring). Negative-control rows require `is_nonlinear=false`.

## 8. Testing
- **Unit (`pgx_metrics`):** `km_uM_to_unbound_mgL` worked example; the saturable EM-anchor solver round-trips `(Vmax, CL_r) → EM PK + fm → (Vmax, CL_r)`; the dose-dependence (log–log slope / fold-trend) statistic; the box-robustness probe is monotone in `Km`.
- **Mechanism unit:** on the axial skeleton, the **hepatic-inlet `C_u` exceeds the `well_stirred` `C_u`** for the same drug/dose (the Arm-F premise) — a direct, deterministic check (extends the v2b `_peak_liver_cu` probe).
- **Harness/integration:** Task 0 gate (§4.1/§4.2) as regression pins; P1/P2 on the clean primaries; N1 negative-control flatness; C2 oracle on **both** well_stirred and axial skeletons.
- **Schema guard:** §7.
- **Headline isolation:** importing/running the harness leaves `4track_holdout_predictions.json` untouched; the v2.2a empty-`enzyme_km` bit-identity pin and `test_cached_holdout_aafe_is_2p731` still pass.

## 9. Components
- **Extend** `src/sisyphus/validation/pgx_metrics.py`: `km_uM_to_unbound_mgL` (pure, tested), saturable EM-anchor solver, dose-dependence statistic, box-robustness engagement probe.
- **Extend** `scripts/validate_pgx_cmax_v2b.py`: the multi-dose steady-state path (Arm S, `solve_regimen`); the axial path (Arm F, `expand_axial` + (A) scaling); the three-engine comparison per arm; the Task 0 gate; P1/P2/N1 + secondary scoring; the report.
- **New** `data/validation/pgx_genotype_nonlinearity_folds.json` + schema-guard test.
- **New** `data/validation/pgx_genotype_nonlinearity_2026-06-16.{json,md}` (results + report). Extend `pgx_fm_registry.json` with the saturable layer (per-drug `Km`, regime, tier, signature outcomes).

## 10. Out of scope (→ later)
Mechanism-based / time-dependent inhibition (voriconazole, omeprazole) → v2.3. ECM/intracellular saturation. Production-path `enzyme_km` (a DDI/high-dose registry, separately specced). MIPD genotype prior. Any drug whose only nonlinearity source is transporter saturation or autoinduction.
