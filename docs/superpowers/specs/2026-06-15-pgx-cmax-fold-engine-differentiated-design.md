# PGx Cmax-fold validation — engine-differentiated milestone (v2.1)

> **Lineage.** Builds on v1 (`2026-06-14-pgx-genotype-fold-validation-design.md`):
> the locked in-vitro fm registry (`data/validation/pgx_fm_registry.json`), the
> CPIC activity scales (PM = 0 reframe), and the controlled-skeleton harness
> (`scripts/validate_pgx_genotype_folds.py`, `src/sisyphus/validation/pgx_metrics.py`).
> v1 validated the **calibration arithmetic**; v2.1 validates the **engine's
> mechanistic first-pass model** — the part v1 explicitly flagged as untested
> (v1 §2 consequence 2: the AUC-fold test is near-tautological w.r.t. the engine).

## 1. Purpose & non-goals

**Purpose.** Test whether the engine's integrated first-pass ODE correctly predicts
how a genotype perturbation reshapes **Cmax** — a quantity with **no closed form** —
beyond what the AUC closed form predicts. Concretely: does the engine reproduce the
observed **peak-to-exposure genotype divergence** `ρ = log(Cmax_fold) − log(AUC_fold)`,
which the closed form predicts to be exactly zero?

**Why this is the right next increment.** v1 proved the calibration (PM fm-agreement,
parameter-free, PASS) and used the engine only as a near-tautological AUC-fold oracle.
The engine's *differentiated* value — the reason a topology-compiled ODE beats a closed
form — lives where the closed form breaks. Cmax under first-pass is the cleanest such
place: it depends on the absolute extraction magnitude `E_h`, the absorption rate `ka`,
and the distribution volume `V`, none of which appear in `1/(1−fm+fm·a)`. A PASS here
means *the engine's first-pass structure is physically right*; a FAIL is a genuine,
honest falsification.

**Non-goals (explicit).**
- **Validation, not productization.** v2.1 tests the engine's first-pass model on an
  isolated harness; it does **not** wire genotype into `predict()` or the production graph.
  CYP2C19 stays a production no-op here — the skeleton injects a synthetic gene tag
  (v1's `_SYNTHETIC_GENE_ABUND` pattern). Shipping genotype as a real prediction input is
  the separate MIPD-prior / CYP2C19-roster tracks (§11). (Headline isolation: §9.)
- **No fitting.** Nothing is tuned to `ρ_obs`; systematic deviation is a reported finding,
  not a correction (discipline: §7).
- **Linear only.** Saturable/nonlinear genotype folds are **v2.2** (§11).

## 2. Background: why Cmax-fold ≠ AUC-fold (the load-bearing mechanism)

For a single oral dose, hepatic genotype, first-order absorption (1-compartment
disposition for the analytic intuition; the engine uses the full graph):

**AUC-fold is exactly the CLint ratio, for any extraction class.** Oral AUC
∝ `1/(fu·CLint)` — the hepatic blood flow `Q` cancels (Rowland/Wilkinson, v1 §2):

```
AUC_fold = AUC(variant)/AUC(EM) = 1 / (1 − fm + fm·a)
```

**Cmax-fold is a product of two factors, neither in the closed form:**

```
Cmax_fold = F_fold × shape_factor
  F_fold      = (Q + fu·CLint_EM) / (Q + fu·CLint_PM)        # ≤ AUC_fold (Q-buffering toward 1)
  shape_factor = exp(−ke_PM·tmax_PM) / exp(−ke_EM·tmax_EM)   # ≥ 1 (PM lowers ke, raises the peak)
  ke = CL_systemic / V ,  tmax = ln(ka/ke)/(ka − ke)
```

Worked cases that fix intuition (and become unit-test pins, §10):
- **Low extraction** (`ke_EM=0.1/h`, `ka=1/h`, PM halves CL): `AUC_fold = 2.0` but
  `Cmax_fold = 1.10`. The integral doubles; the absorption-gated peak barely moves.
- **High extraction** (`E_h,EM=0.9`, `fm=0.9`, PM-null): `AUC_fold = 10` but
  `Cmax_fold ≈ 5.8` (`F_fold≈5.26 × shape≈1.1`).

So `Cmax_fold ≤ AUC_fold` in the absorption-gated regime, and the **gap** is a
non-trivial function of `E_h`, `ka/ke`, and `V`. That gap is the entire scientific object
of v2.1. The engine integrates it; the closed form sets it to zero.

> The sign of `ρ` is **not** asserted as a theorem (a flip-flop / clearance-gated regime
> could differ). `ρ_engine` is a *prediction* to be tested, not a derived constant.

## 3. The metric: peak-to-exposure genotype divergence

```
ρ = log(Cmax_fold) − log(AUC_fold) = log( Cmax_fold / AUC_fold )
```

- **`ρ_obs` is curation-free** — computed from the two reported folds (Cmax-fold and
  AUC-fold) of a genotype-panel study. The quantity under test is **directly measured**,
  model-free. No `fm`, no `a`, no anchoring enters `ρ_obs`.
- **Closed-form null:** `ρ_null = 0` (it equates Cmax-fold and AUC-fold). This is the
  strongest non-mechanistic baseline — "assume the peak tracks the exposure."
- **Engine:** `ρ_engine = log(Cmax_fold_engine) − log(AUC_fold_engine)`, from the
  integrated ODE on the EM-anchored skeleton (§4), with curated-input uncertainty
  forward-propagated (MC) into a band.

Log scale because folds are multiplicative; `ρ` is unit-free and symmetric.

## 4. Method: EM-anchored / PM-predicted controlled skeleton

The maximally-hardened form of v1's controlled skeleton. Absolute-CLint noise (W1) is
quarantined by anchoring the skeleton to the drug's measured **EM arm**, not to
`predict()`'s XGBoost CLint.

### 4.1 Per-pair procedure
1. **Curate EM observables** from the genotype-panel study: `tmax_EM`, `Cmax_EM`,
   `AUC_EM` (and dose), plus an **oral-F decomposition** `F = f_a · (1−E_g) · (1−E_h)`
   from independent literature (total oral F; gut fraction where the gene is intestinal;
   for hepatic-only genes lump `f_a·(1−E_g)` into a single "fraction reaching liver").
2. **Anchor** the skeleton's `ka`, `V`, and `fu·CLint` to the EM observables:
   - `AUC_EM` with the curated F-decomposition ⇒ `fu·CLint` (hence `E_h`, given physiologic `Q`),
   - `tmax_EM` ⇒ `ka` (given `ke = CL_systemic/V`),
   - `Cmax_EM / AUC_EM` shape ⇒ joint `ka`, `V` constraint.
   The solver is identified given the curated F-decomposition (§10 pins identifiability
   on a synthetic case).
3. **Split** hepatic CLint into the gene fraction `fm` (in-vitro, v1 registry) via
   `enzyme_affinity[gene]` and the residual `(1−fm)` via the non-scaled synthetic
   `RESIDUAL_HEPATIC` tag (v1 pattern — same liver node, preserves first-pass topology).
4. **Predict PM:** scale the gene abundance by `a` (PM ⇒ 0) via
   `apply_phenotype_to_graph`, re-integrate, read `Cmax_PM`, `AUC_PM`.
5. **Outputs:** `Cmax_fold_engine = Cmax_PM/Cmax_EM`, `AUC_fold_engine = AUC_PM/AUC_EM`,
   `ρ_engine`.

### 4.2 Internal control (skeleton sanity)
The engine's `AUC_fold_engine` on the anchored skeleton **must** still reproduce the
analytic `1/(1−fm+fm·a)` to v1 tolerance (≤2%). A mismatch means the skeleton is
mis-anchored (or a real engine AUC bug) — it is **not** counted as a Cmax win. This is
v1's oracle re-used as a per-pair guard.

### 4.3 Uncertainty propagation
Curated EM observables and the F-decomposition carry uncertainty (read off published
CIs / digitized SD where available; otherwise a stated default CV). Forward-propagate by
MC through the anchor→predict chain to produce a `ρ_engine` band per pair. `ρ_obs` carries
a band from the published Cmax-fold and AUC-fold CIs.

## 5. Scope & inclusion

- **Genes:** CYP2D6, CYP2C19 (first-pass-rich); CYP2C9 mostly low-extraction → secondary.
- **Phenotype:** **PM primary** (`a=0`, strongest, parameter-free AUC arm). IM/UM exploratory.
- **Primary (powered) inclusion — ALL of:** oral; single-dose; healthy-volunteer genotype
  panel; **reports both Cmax-fold and AUC-fold** (with dispersion); **resolvable
  first-pass** (`ρ_obs` distinguishable from 0 at the pair CI). Candidate pool:
  metoprolol, nebivolol, tramadol, propafenone (CYP2D6); omeprazole (single-dose arm),
  lansoprazole (CYP2C19).
- **Secondary (consistency) set:** low-extraction drugs where `ρ_obs ≈ 0` within CI
  (tolbutamide-class). The engine must **not** predict a large `|ρ_engine|` there —
  falsifiable in the opposite direction.
- **Excluded from the powered set (schema-enforced, §10):** `is_nonlinear=true`
  (saturable / auto-induction: omeprazole multi-dose, voriconazole), `is_prodrug=true`,
  `fm_source_type ∈ {genotype_derived, ddi_derived}` (v1 non-circularity guard),
  single-endpoint studies (Cmax-fold OR AUC-fold missing).

## 6. Pre-registered pass criteria

Fixed **before** running the engine on real pairs.

- **Primary P1 — sign agreement:** across the powered set, `sign(ρ_engine) = sign(ρ_obs)`
  for a **majority** of pairs; binomial sign test reported (not a hard gate at small N,
  but the headline result).
- **Primary P2 — beats the null:** the engine **reduces** the divergence error vs the
  closed-form null: median `|ρ_obs − ρ_engine| < median |ρ_obs − 0|`; paired Wilcoxon
  signed-rank reported. P2 is the core claim ("the engine adds information the closed
  form lacks").
- **Control C1:** every pair passes §4.2 (engine AUC-fold = analytic, ≤2%). Pairs failing
  C1 are quarantined (skeleton/engine issue), not scored.
- **Secondary S1:** on the low-extraction consistency set, `|ρ_engine|` is small
  (within the band of `ρ_obs ≈ 0`).

**Honest outcomes.** If P2 fails (engine no better than ρ=0), the report states the
engine's first-pass structure does **not** add measurable Cmax-fold information at this N
— a real negative, logged to `dead-ends.md`. No input is adjusted to rescue it.

## 7. Curation discipline (non-circularity)

- `ρ_obs` uses only the two **observed** folds — never a model output.
- EM-anchor observables (`tmax/Cmax/AUC`, oral F) describe the **EM population PK** and do
  **not** encode the genotype fold ⇒ anchoring is non-circular.
- `fm` is in-vitro reaction-phenotyping (v1 registry); never genotype/DDI back-calculated.
- `a` is fixed at CPIC convention (PM=0); never fitted.
- Each curated number carries a citation and an uncertainty; curation precedes any engine
  run on that pair (no peeking at `ρ_engine`).

## 8. Components & files

- **Extend** `src/sisyphus/validation/pgx_metrics.py` (pure functions):
  `rho(cmax_fold, auc_fold)`; `anchor_em(tmax, cmax, auc, dose, f_decomp, Q, ...) ->
  (ka, V, clint)` (identifiable EM solver); `rho_band(...)` MC propagation;
  `sign_test(...)`, `wilcoxon_div(...)` stats. No engine import (stays pure).
- **Extend** `scripts/validate_pgx_genotype_folds.py` with a Cmax-fold mode:
  anchor → split → PM-predict → `ρ`, per pair; writes report + registry extension.
- **New benchmark** `data/validation/pgx_cmax_folds.json`: powered + consistency pairs,
  each with EM observables, oral-F decomposition, both observed folds + CIs, `fm` ref,
  flags, citations. Locked (report as-is, never refit).
- **New report** `data/validation/pgx_cmax_fold_validation_2026-06-15.{json,md}`.
- **Registry extension** `data/validation/pgx_fm_registry.json`: add per-drug
  EM-anchored `ka/V/E_h` + `ρ_engine`/`ρ_obs` (durable, engine-differentiated layer).

## 9. Invariants / cleanliness

- **Headline:** no `predict()`/`reference_man.yaml`/holdout touch ⇒ 2.731 bit-untouched.
  A regression test asserts the holdout cache is unchanged by this work (it imports
  nothing from the production predict path).
- **Invariant #5:** benchmark is published genotype folds, not holdout Cmax labels;
  drug-independent population scales, no per-drug fitting. No exposure.
- **No fitting:** §7. **Falsifiable both directions:** §6 (P2 can fail; S1 can fail).

## 10. Testing

- **Unit (`pgx_metrics`):** `rho` against hand values; the two §2 worked cases
  (low-extraction `Cmax_fold≈1.10` / `AUC_fold=2.0`; high-extraction `≈5.8` / `10`) pinned
  through the analytic 1-comp reference; `anchor_em` round-trips `(ka,V,CLint) → observables
  → (ka,V,CLint)` on a synthetic drug within tolerance (identifiability pin); `sign_test`,
  `wilcoxon_div` against hand-computed small cases; `rho_band` monotone in input CV.
- **Harness regression pin:** a synthetic low-extraction drug ⇒ engine `AUC_fold` equals
  analytic within 2% (C1 guard, re-used from v1) AND engine `Cmax_fold` matches the
  1-comp analytic Cmax-fold within tolerance (the new engine-Cmax oracle).
- **Schema guard:** rejects single-endpoint pairs, `is_nonlinear`/`is_prodrug`, and
  circular `fm_source_type` from the powered set; requires both folds + CIs + citations.
- **Holdout-invariance regression:** importing/running the v2.1 harness leaves
  `4track_holdout_predictions.json` untouched (no production-path import).

Every new `pgx_metrics.py` function is written test-first (TDD). The benchmark JSON is
locked **before** the engine runs on it, and the report records the pre-registered §6
criteria verbatim alongside the realized statistics.

## 11. Out of scope (YAGNI) + v2.2 roadmap

**Excluded from v2.1:** nonlinear/saturable kinetics; multi-dose accumulation; CYP2C19
production-graph activation; MIPD genotype-prior; DDI phenoconversion; IM/UM as a powered
endpoint; CYP3A5/SLCO1B1.

**v2.2 (needs new engine capability):** a Michaelis–Menten metabolic `ClearanceFluxSpec`
(`CLint = Vmax/(Km+C)`) → dose-dependent genotype folds (omeprazole/voriconazole), the
strongest "engine is non-redundant" claim. Then: multi-dose genotype accumulation, and
genotype as a prior in `mipd.predict_posterior` (consuming the v2.1 EM-anchored registry).

## 12. Risks & honest limitations

- **Small N per cell** ⇒ P1/P2 are small-N descriptive validations, reported with the
  exact N and the sign/Wilcoxon statistics, not asserted as powered hypothesis tests.
- **EM-anchor identifiability** depends on the curated oral-F decomposition; where total
  oral F is itself uncertain, the `ρ_engine` band widens (propagated, not hidden). Pairs
  with an unidentifiable anchor are dropped from the powered set with a logged reason.
- **1-comp analytic is only the intuition**; the engine is the multi-compartment graph, so
  the §2 closed forms are reference points, not the engine's actual integrator. The §10
  oracle pins the engine against the analytic on a deliberately 1-comp-like synthetic drug.
- **`ρ_obs` inherits study heterogeneity** (assay, population, dose) — stated per pair.
