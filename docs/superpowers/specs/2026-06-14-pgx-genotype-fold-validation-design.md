---
title: PGx genotype-fold validation (calibration·foundation milestone)
date: 2026-06-14
status: design (awaiting review)
author: Hypatia
supersedes: none
related:
  - src/sisyphus/predict/phenotype.py
  - docs/claude/diagnosis.md (§4 decorrelation gate, §8/§10 walls)
  - src/sisyphus/mipd/ (engine-as-prior; v2 consumer)
---

# PGx genotype-fold validation — calibration·foundation milestone (v1)

## 1. Purpose & non-goals

**Purpose.** Lock the pharmacogenomic **activity-scale + fraction-metabolized (fm)
pipeline** and confirm the engine's genotype response is **physically correct**, as
the foundation for an engine-differentiated v2 (Cmax / nonlinear / MIPD genotype
prior). Concretely: do the Big-3 CYP activity multipliers in `predict/phenotype.py`
+ independently-curated fm reproduce **published within-drug genotype AUC fold-ratios**?

**Why this is worth doing even though the holdout headline is walled.** The SMILES-only
holdout Meta AAFE (2.731) is empirically walled (W1 CLint target-noise, W2
bioavailability-F blindness, W3 fixed-weight meta damping — see `diagnosis.md`). A
**within-drug perturbation ratio** structurally bypasses all three: in `AUC(PM)/AUC(EM)`
the absolute CLint error (W1) and the bioavailability-F error (W2) **cancel** (same
drug, same formulation), and genotype is a pure engine/mechanistic effect with **no ML
track input** (W3 N/A). So this is a *new value axis*, not a headline lever.

**Non-goals (explicit).**
- **Does NOT** claim "the engine does PGx" as a differentiated capability. v1 is mostly
  a **closed-form calibration** (see §4.1); the engine's differentiated value (Cmax,
  nonlinear, sequential gut/hepatic first-pass, multi-dose) is **v2**.
- **Does NOT** fit/tune the activity scales to the benchmark (no parameter fitting;
  systematic deviation is reported as a *finding*, not corrected by fitting).

## 2. Background: the closed-form reduction (the load-bearing fact)

For an orally-dosed, hepatically-metabolized drug, oral AUC is **hepatic-blood-flow
independent** (Rowland/Wilkinson):

```
AUC_po = F_abs · (1 − E_gut) · Dose / (f_u · CL_int,h)
```

So a genotype that scales the gene's fraction of hepatic intrinsic clearance gives a
fold-ratio that is **independent of extraction class and of absolute CLint magnitude**:

```
AUC(variant)/AUC(EM) = 1 / (1 − fm + fm·a)
```

where `a` = activity multiplier (EM/NM = 1) and `fm` = fraction of **total systemic
clearance** via the gene (fraction-metabolized × enzyme-split × (1 − renal/other)).

**Consequences that shape this spec.**
1. Curating in-vivo CL magnitude is **unnecessary** (it cancels) and would be
   ill-conditioned for high-extraction drugs (in-vivo CL → CLint inversion is unstable
   near `CL ≈ Q_h`). Dropped.
2. The AUC-fold test is **near-tautological w.r.t. the engine** — given correct fm and
   a, the engine *must* reproduce the closed form. So v1 tests **(a) the activity-scale
   calibration, (b) the fm curation, (c) the additive-clearance / well-stirred
   adequacy** — *not* the engine's modelling power.
3. The engine is therefore used as a **production-path correctness oracle** (does it
   match the analytical fold?), which is the foundation we need before relying on it in
   v2 where no closed form exists.

### 2.1 The PM = 0 reframe (and a `phenotype.py` finding)

`phenotype.py` uses `PM = 0.10×` as a floor. That floor only makes sense when fm is
assumed **lumped = 1** (no explicit residual). When fm is curated explicitly (fm < 1),
the residual clearance enters via `(1 − fm)`, so a PM activity of 0.10 **double-counts
the residual** and under-predicts the fold. The physically correct value for a true
no-function diplotype is **PM activity = 0**, giving:

```
AUC(PM)/AUC(EM) = 1 / (1 − fm)      (PM, activity = 0)
```

This is a **parameter-free** relation. It also surfaces a real finding: the 0.10 floor
is incompatible with explicit-fm modelling and should be revisited in production (logged
as a v1 output, not changed here unless validation confirms).

## 3. Scope

- **Genes:** CYP2D6, CYP2C19, CYP2C9 (the clearance-actionable "Big 3").
- **Drugs:** oral, **single-dose**, **healthy-volunteer genotype-panel** studies,
  **parent-AUC only**.
- **Phenotypes:** PM (primary; strongest, parameter-free), UM/IM (secondary, exploratory).

## 4. Method

### 4.1 Two computations per pair (analytical + engine)
- **Analytical (the science):** `fold_pred = 1 / (1 − fm + fm·a)` with `a` from CPIC
  conventions (PM = 0, IM = 0.5, UM = 2.0; NM = 1).
- **Engine (production-path oracle):** build a controlled `DrugOnGraph` whose hepatic
  clearance is split into `fm` via `enzyme_affinity[gene]` (phenotype-scaled by
  `apply_phenotype_to_graph`) and `(1 − fm)` via a **non-scaled synthetic
  `RESIDUAL_HEPATIC` enzyme tag** (same liver node → preserves first-pass topology, not
  renal). Run EM and variant; confirm engine oral-AUC fold matches the analytical fold
  to numerical tolerance. **Mismatch = engine bug** (a genuine catch, e.g. a residual
  flow-dependence in the FLUX-1 intrinsic-clearance form).

### 4.2 Primary analysis — PM fm-agreement (parameter-free, best-powered)
For each clean PM pair, compute the **in-vivo-implied fm**:

```
fm_invivo = 1 − 1 / obs_fold_PM
```

and compare against the **independently-curated in-vitro fm** (`fm_invitro`). Report:
- scatter / Bland-Altman of `fm_invitro` vs `fm_invivo`,
- % of pairs agreeing within tolerance (pre-registered, §7),
- regression slope (≈ 1 expected),
- per-pair CIs propagated from `obs_fold` CI.

This tests, with **zero fitted parameters**, whether `(PM-null + additive clearance +
in-vitro fm)` reproduces in-vivo genotype folds. It is non-circular **iff** `fm_invitro`
comes from reaction phenotyping, not from genotype/DDI back-calculation (§5).

### 4.3 Secondary analysis — IM/UM empirical activity (exploratory)
For IM/UM pairs with **high fm (≥ 0.6)** (where back-calc is well-conditioned):

```
a_emp = (1/obs_fold − (1 − fm)) / fm
```

Check whether `a_emp` is consistent with CPIC (IM ≈ 0.5, UM ≈ 2.0). Per-cell N is small,
so this is **exploratory** — a clustering verdict is only issued for (gene, phenotype)
cells with ≥ 3 high-fm pairs; otherwise single-point consistency only. (Low-fm pairs are
excluded from `a_emp`: dividing by small fm amplifies error and can yield nonsensical
negative activity.)

## 5. Benchmark dataset & curation discipline

`data/validation/pgx_genotype_folds.json`, one entry per (drug, gene, phenotype):

```json
{
  "drug": "...", "gene": "CYP2D6", "phenotype": "PM",
  "dose_mg": 0, "formulation": "IR", "route": "oral",
  "obs_auc_fold": 0.0, "obs_auc_fold_ci": [0.0, 0.0],
  "obs_cmax_fold": null,
  "fm_invitro": 0.0, "fm_source_type": "in_vitro_phenotyping",
  "obs_fold_study_design": "genotype_panel_single_dose_healthy",
  "is_prodrug": false, "is_nonlinear": false,
  "citation_fold": "...", "citation_fm": "..."
}
```

**`fm_invitro` is the fraction of TOTAL systemic clearance via the gene** — reaction
phenotyping usually reports fraction of *hepatic metabolism*, so it must be reconciled to
total CL by `× (fraction metabolized) × (1 − f_renal)`. Conflating "fraction of
metabolism" with "fraction of total clearance" corrupts every prediction; the curation
records the reconciliation per drug.

**Curation is model-blind and locked before any engine run** (same discipline as the
holdout reference audit). Hard filters / exclusions:
- **fm source:** `in_vitro_phenotyping` only (rCYP/HLM relative activity factor or
  selective chemical inhibition). **EXCLUDE** `genotype_derived` / `ddi_derived` fm —
  using `fm = 1 − 1/fold` to predict fold is fully circular.
- **obs_fold design:** genotype-panel single-dose healthy only. **EXCLUDE** pop-PK
  covariate analyses and DDI-surrogate ("phenoconversion") folds (the latter is v2).
- **parent-AUC only; EXCLUDE prodrugs / active-moiety drugs** (codeine→morphine,
  clopidogrel, tamoxifen→endoxifen, tramadol …) — their fold is inverted or
  moiety-ambiguous.
- **EXCLUDE nonlinear/saturable** substrates (phenytoin, voriconazole) — the closed
  form breaks; deferred to v2 nonlinear.
- **high-fm (≥ 0.6) and PM** prioritized for the powered analyses.

## 6. Components (single-responsibility)

- `data/validation/pgx_genotype_folds.json` — the locked benchmark.
- `scripts/validate_pgx_genotype_folds.py` — harness: per pair → build controlled
  `DrugOnGraph` (fm-split via `RESIDUAL_HEPATIC` + gene), apply phenotype, run engine,
  emit analytical + engine folds.
- `src/sisyphus/validation/pgx_metrics.py` — pure functions: `fm_invivo`, `a_emp`,
  fm-agreement stats, engine-vs-analytical delta, CI propagation.
- `data/validation/pgx_fold_validation_<date>.json` + short `.md` report.
- `data/validation/pgx_fm_registry.json` — **durable cross-validated Big-3 in-vivo fm
  table** (in-vitro vs in-vivo-PM-derived) for reuse by v2 (Cmax), the MIPD genotype
  prior, and the DDI module (fm drives DDI magnitude).

## 7. Pre-registered pass criteria

Registered **before** running the engine:
- **Primary (PM fm-agreement):** ≥ 70% of clean PM pairs with `|fm_invitro − fm_invivo|
  ≤ 0.15` (absolute), AND OLS slope of `fm_invivo ~ fm_invitro` in [0.7, 1.3]. Below
  this ⇒ either the in-vitro→in-vivo fm transfer or the PM-null assumption fails (a
  finding, not a fitting trigger).
- **Engine regression:** every pair's engine oral-AUC fold within **2%** of the
  analytical fold. Any miss is a flagged engine-correctness defect.
- **Secondary (IM/UM):** descriptive only; report `a_emp` vs CPIC, no pass/fail.

Discrepancy cases (disagreement between in-vitro and in-vivo fm) are the **scientific
output** — candidate mechanisms: active metabolites, residual *3/*3 activity, gut CYP,
in-vitro/in-vivo discordance — enumerated, not hidden.

## 8. Deliverables
1. Locked benchmark JSON (§5).
2. Validation report: PM fm-agreement (primary), IM/UM activity (secondary), engine
   regression, per-pair table, **exclusion log** (every dropped drug + reason).
3. Durable cross-validated Big-3 fm registry (§6) — the lasting asset.
4. A `phenotype.py` finding note on the PM = 0 vs 0.10-floor incompatibility (§2.1).

## 9. Step 0 — feasibility gate (PASSED 2026-06-14)

Gate criterion: ≥ 6 clean PM pairs across Big-3 under the §5 filters. **Result: 10 clean
pairs (9 quantitative-grade — only dextromethorphan is excluded, as extreme), all three
genes ≥ 2 → PASS.** Curated model-blind, fm and fold sourced from independent studies.

| Gene | Drug | fm_invitro | fold_PM | flag |
|---|---|---|---|---|
| CYP2D6 | atomoxetine | 0.90 | 8.1× | — |
| CYP2D6 | nortriptyline | 0.78 | ~4× | — |
| CYP2D6 | desipramine | 0.85 | ~7× | — |
| CYP2D6 | metoprolol | 0.80 | 4.9× | nonlinear first-pass (fold embeds F-rise; hepatic first-pass IS in the closed form, expect wider spread) |
| CYP2D6 | dextromethorphan | 0.90 | ~150× | extreme fold → **directional only, excluded from the quantitative fm-agreement** (fm_invivo super-sensitive to fold noise) |
| CYP2C19 | omeprazole | 0.75 | 7.5× | 40 mg at edge of saturable first-pass (upper-leaning) |
| CYP2C19 | lansoprazole | 0.68 | 4.0× | ~32% CYP3A4 (fm just above threshold) |
| CYP2C9 | celecoxib | 0.78 | 2–4.2× | true *3/*3 (n=2–3) |
| CYP2C9 | flurbiprofen | 0.75 | 2.8× | true *3/*3 (n=2) |
| CYP2C9 | tolbutamide | 0.82 | 6.5× | true *3/*3 (most complete panel); fm partly mass-balance-anchored → fm_confidence MEDIUM |

**Excluded (with reason, logged):** pantoprazole — strong clean fold (~5×) but **every fm
is genotype/PBPK-back-calculated (`fm = 1 − 1/fold`) → circular**, the §5 trap confirmed
empirically; nebivolol (active moiety + racemate CYP2C19 split); propafenone (nonlinear
first-pass + active metabolite); escitalopram / sertraline / diazepam (fm < 0.6 and/or
active metabolite); glimepiride (active M1 + OATP1B1 co-dependence, *3/*3 only n=1);
(S)-warfarin (true *3/*3 but steady-state only, no single-dose HV).

The quantitative fm-agreement (§4.2) uses the 9 non-extreme pairs (metoprolol included,
carrying the nonlinear-first-pass flag); only dextromethorphan is excluded (extreme,
directional-only). **No quantitative scoring was computed at the feasibility stage**
(pre-registration: scored once, after the benchmark JSON is locked).

## 10. Invariant #5 / holdout cleanliness
The genotype benchmark is **independent** of the 107-drug Cmax holdout (it uses published
genotype fold-ratios, not holdout Cmax labels). Activity scales and fm are **drug-
independent** population constants, **fixed at literature values (no fitting)**. Even if a
benchmark drug overlaps the holdout, its holdout Cmax label is never used and no per-drug
tuning occurs. No Invariant #5 exposure.

## 11. Out of scope (YAGNI) + v2 roadmap
**Excluded from v1:** diplotype→activity bioinformatics layer (PharmCAT/PharmVar);
DDI phenoconversion unification; MIPD genotype-prior integration; Cmax-fold as a primary
endpoint; gut-wall enzyme scaling; CYP3A5 / SLCO1B1; activity-scale refitting.

**v2 (engine-differentiated, builds on v1's locked fm registry + scales):** Cmax-fold and
nonlinear/saturable kinetics (where the engine is *not* redundant), sequential gut+hepatic
first-pass, multi-dose accumulation, and genotype as a prior in `mipd.predict_posterior`.

## 12. Risks & honest limitations
- **Small N per (gene, phenotype) cell** → primary is a small-N descriptive validation,
  not a powered hypothesis test. Stated plainly in the report.
- **Single drug-independent activity per genotype is an approximation** (substrate-
  dependent Km/allele effects). The `a_emp` scatter *measures* how good that
  approximation is — a worthy secondary output, not an assumed truth.
- **in-vitro fm ≠ in-vivo fm** in general; the PM test partly validates that transfer,
  so disagreement is multi-causal (enumerated, §7).
- v1's intellectual content is a PGx calibration analysis; its **Sisyphus-specific**
  value is foundation-laying + production-path regression + the durable fm registry.

## 13. Testing
- Unit: `pgx_metrics` functions (fm_invivo, a_emp, CI propagation) against hand-computed
  values; synthetic case `fm=0.9, PM` ⇒ `fold = 10×`.
- Harness correctness pin: a low-extraction synthetic drug ⇒ engine fold equals
  `1/(1−fm+fm·a)` within 2% (regression guard for the engine genotype response).
- Curation guard: schema test rejecting `fm_source_type ∈ {genotype_derived,
  ddi_derived}` and `is_prodrug=true` / `is_nonlinear=true` from the powered set.
