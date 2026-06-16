# PGx v2.2b — nonlinear (saturable) genotype Cmax/AUC fold validation

> **⚠ SUPERSEDED (2026-06-16) — single-dose premise halted at the §7 spike.** The feasibility
> spike + candidate screen (`data/validation/pgx_cmax_v2b_spike_2026-06-16.md`) showed that at
> single therapeutic dose the engine does **not** robustly engage MM saturation: only
> propafenone qualifies, and only at the low end of its 40× `Km` spread (`Km`-cherry-picking);
> voriconazole is `fu_mic`-borderline; atomoxetine/lansoprazole robustly fail (low `fup`). The
> v2.2a capability is correct (oracle PASS) — the limit is IVIVE of `Km`/exposure at single
> dose. **Pivoted → a multi-dose / steady-state milestone** (accumulation raises `C_u` toward
> `Km`), with its own feasibility gate. The saturable harness and this spec's metric/anchor
> carry forward. See experiment-log 2026-06-16.

> **Lineage.** Consumes the v2.2a Michaelis–Menten flux (`DrugOnGraph.enzyme_km`,
> `2026-06-15-mm-saturable-clearance-design.md`) and reuses v2.1's controlled `well_stirred`
> EM-anchored skeleton, `ρ` metric, and nulls (`2026-06-15-pgx-cmax-fold-engine-differentiated`).
> Feasibility **STRONG PASS**: `data/validation/pgx_cmax_v2b_feasibility_2026-06-15.md` (powered
> set with literature `Km` + dose-ranging folds). This is the validation v2.1 could not run —
> the saturable drugs it had to exclude are exactly this milestone's subjects.

## 1. Purpose & non-goals

**Purpose.** Test whether the engine, with saturable hepatic metabolism (literature `Km` on the
gene fraction), reproduces the **observed genotype Cmax/AUC folds** of nonlinear drugs — and,
where dose arms exist, their **dose-dependence** — better than the *same engine with saturation
turned off*. For these drugs the engine is non-redundant **by construction**: a dose-dependent
genotype fold is something no linear or closed-form model can produce at all.

**Non-goals (explicit).**
- **Validation, not productization.** Harness-isolated; no `predict()`/`reference_man.yaml`
  change. Headline **2.731 bit-identical** (regression-pinned). `enzyme_km` is set only on the
  synthetic skeleton.
- **No fitting.** `Km`, `fm`, `fu_mic` are literature/in-vitro, fixed; never tuned to the fold.
- **No mechanism-based inhibition.** Reversible MM only. Omeprazole (auto-inhibition) is
  excluded → v2.3 MBI capability.
- **`well_stirred` only** (v2.2a scope); no ECM saturation.

## 2. Background: the dose-dependent genotype fold under saturation

For a saturable drug the genotype fold is **dose-dependent**. Low dose (`C_u ≪ Km`): metabolism
is first-order, fold ≈ the linear CLint ratio. High dose: the **EM arm's gene saturates**
(`Vmax/(Km+C_u)` falls) while the PM arm (gene inactive → linear residual) does not, so
`CL_EM → CL_PM` and the **fold shrinks toward 1**. A linear model predicts a flat,
dose-independent fold; the saturation-driven change is the engine's unique signal.

**Single-dose folds also differ (the residual-compensation argument).** Both the saturable and
linear engines are EM-anchored to reproduce the observed EM PK, holding the same intrinsic `fm`.
To match a *saturated* EM, the saturable engine needs a larger intrinsic gene `Vmax` **and** a
larger residual `CL_r`. PM removes the gene → residual-only → the saturable engine's larger
`CL_r` yields a smaller PM AUC, hence a **smaller genotype fold than the linear engine at the
same dose**. So even non-dose-ranging drugs (lansoprazole, atomoxetine) test the capability.

**Two regimes, both engine-handled.** *First-pass saturation* (propafenone, atomoxetine): the
liver sees the whole oral dose, so liver `C_u` during absorption approaches `Km` even when
systemic `C_u ≪ Km`. *Systemic saturation* (voriconazole `C_u ~1.5–6 µM` vs `Km 9.3 µM`;
lansoprazole): systemic `C_u ~ Km`. The engine's saturation acts on the **liver-node `C_u`**, so
both regimes emerge from the ODE; the closed forms cannot.

## 3. Method: saturable EM-anchor + three engines

Per drug, per dose, the v2.1 `well_stirred` controlled skeleton with the hepatic CLint split
`fm`:(1−fm) between the gene enzyme and the non-scaled `RESIDUAL_HEPATIC` tag — **plus** the
gene fraction carrying `enzyme_km` (literature `Km`, converted §5). The residual carries **no**
`Km` (stays linear — handles the shared CYP3A4/FMO3/CYP1A2 fractions the gate flagged).

### 3.1 The saturable EM-anchor (the new inner-solve)
At the reference EM dose, solve the skeleton so the **saturable** EM run reproduces the observed
EM PK, with `Km` fixed at its literature value:
- `ka`, `V` from EM `tmax` and the `Cmax/AUC` shape (as v1);
- intrinsic gene `Vmax` and residual `CL_r` from two constraints: (a) the **intrinsic** (low-dose)
  gene fraction `(Vmax/Km)/((Vmax/Km)+CL_r) = fm`; (b) the saturable EM AUC at the reference dose
  matches observed (self-consistent with the dose's saturation). Identifiable given `Km`, `fm`.

### 3.2 The three engines
1. **EM-saturable** — the anchored skeleton (matches EM by construction).
2. **PM-saturable** — gene scaled to 0 (residual only), saturation active; predict folds.
3. **Linear null** — the **same** skeleton re-anchored with `Km=∞` (v1's linear anchor: gene
   CLint `=Vmax/Km` constant, same `fm`, same EM AUC), EM and PM. This isolates *exactly* what
   the MM flux adds — everything else (skeleton, `fm`, first-pass, EM-anchor target) is identical.

Outputs per drug per dose: `Cmax_fold` and `AUC_fold` from the saturable engine and the linear
engine, vs observed; `ρ = log(Cmax_fold/AUC_fold)` (v2.1) as secondary.

## 4. Metric & pre-registered criteria

- **Primary P1 (single-dose fold accuracy):** across the powered set, the **saturable** engine
  reduces the median fold-prediction error `|obs_fold − pred_fold|` vs the **linear** engine,
  for Cmax and AUC (paired Wilcoxon). The core claim: saturation improves the genotype fold on
  an otherwise-identical skeleton.
- **Primary P2 (dose-dependence, voriconazole + propafenone):** the saturable engine reproduces
  the observed fold at *each* dose (sign + magnitude of the fold's change with dose); the linear
  engine is flat across dose and misses ≥1. Reported per dose-ranging drug.
- **Control C1 (linear anchor):** metoprolol — saturable ≈ linear ≈ observed, **no** spurious
  dose-dependence. A saturable engine that invents saturation on a linear drug fails C1.
- **Control C2 (oracle):** the v2.2a saturation oracle holds on this skeleton (engine MM rate =
  analytic; AUC-fold of the *linear* engine = analytic `1/(1−fm+fm·a)` at low extraction).
- **Secondary:** v2.1's ρ=0 and 1-comp nulls as external context; perhexiline as a
  saturation-physics consistency case (CL/F-based, not a primary fold pair).
- **Honest-negative path:** if P1 ties (saturation no better than linear at this N / `Km`
  uncertainty), report it; log to `dead-ends.md`. No `Km` tuned to fit.

## 5. `Km` conversion (correctness-critical, its own tested function)

Literature `Km` is µM, usually **total microsomal**; the engine's `enzyme_km` is **unbound
mg/L** (the basis of `C_u = fup·c_plasma`). Convert:
```
Km_unbound[mg/L] = Km_total_µM × fu_mic × MW / 1000
```
where `fu_mic` is the microsomal unbound fraction (total→unbound at the enzyme; for `well_stirred`
unbound-at-enzyme ≈ unbound plasma). A pure `km_uM_to_unbound_mgL(km_uM, mw, fu_mic)` function
with a worked-example unit test. The benchmark records the **raw** `Km` + units + basis +
`fu_mic` + source per drug; non-circular (`Km`/`fu_mic` in-vitro, never the clinical fold). Where
`fu_mic` is unavailable, treat `Km` as already-unbound with a stated assumption and a
**sensitivity band** over the plausible range (propafenone 0.12–5.3 µM; voriconazole 9.3±3.6 µM).

## 6. Scope & benchmark

New locked `data/validation/pgx_cmax_v2b_folds.json`:
- **Powered:** voriconazole (CYP2C19, dose-ranging), propafenone (CYP2D6, dose-ranging),
  lansoprazole (CYP2C19), atomoxetine (CYP2D6).
- **Anchor:** metoprolol (CYP2D6, `is_nonlinear=false`).
- **Secondary:** perhexiline (CL/F consistency).

Each pair, per dose: observed Cmax-fold + AUC-fold + CI, raw `Km` (+units/basis/`fu_mic`/source),
`fm` (+source), oral F (+gut/hepatic split), EM `tmax`/`t½`, dose(s), flags, citations.
Dose-ranging pairs carry ≥2 dose rows. **Schema guard:** powered rows require a literature `Km`
(reject "Km not found"), both endpoints, non-circular `Km`/`fm`, and reject MBI drugs
(`is_mbi=true` → omeprazole). Locked; never refit.

## 7. Feasibility spike (front-loaded, before the full harness)

Mandatory de-risking on **one drug (propafenone)** before building the scoring harness:
1. The saturable EM-anchor (§3.1) **converges** and is identifiable.
2. Saturation is **engaged** at therapeutic dose — the saturable fold differs materially from the
   linear fold (else the milestone is underpowered: surface and revisit, do not pad).
3. **Dose-dependence** appears (fold shrinks from 300→400 mg in the engine).
4. The v2.2a oracle (C2) holds on the skeleton.

If the spike fails (2) or (3), halt and report — the engine may not reach liver `C_u ~ Km` at
therapeutic dose, which would reshape the milestone.

## 8. Testing

- **Unit (`pgx_metrics`):** `km_uM_to_unbound_mgL` worked example; the saturable-anchor solver
  round-trips `(Vmax, CL_r) → EM AUC + fm → (Vmax, CL_r)`; dose-dependence stat; sensitivity band
  monotone in `Km`.
- **Harness/integration:** the spike's three checks as regression pins; saturable-fold < linear-
  fold at a saturating dose (§2 residual-compensation); C1 (metoprolol no dose-dependence); C2
  oracle.
- **Schema guard:** §6.
- **Headline isolation:** importing/running the harness leaves `4track_holdout_predictions.json`
  untouched; the v2.2a empty-`enzyme_km` bit-identity pin still holds.

## 9. Components

- **Extend** `src/sisyphus/validation/pgx_metrics.py`: `km_uM_to_unbound_mgL`, the saturable
  EM-anchor solver, the dose-dependence statistic. Pure (numpy/scipy).
- **New** `scripts/validate_pgx_cmax_v2b.py`: the three-engine saturable harness (saturable
  anchor → EM/PM saturable + linear null, per dose) + scoring + report.
- **New** `data/validation/pgx_cmax_v2b_folds.json` + schema guard.
- **New** `data/validation/pgx_cmax_v2b_validation_2026-06-15.{json,md}`; extend
  `pgx_fm_registry.json` with the saturable layer (per-drug `Km`, `Vmax`, regime, fold deltas).

## 10. Out of scope (→ later)

Mechanism-based inhibition (omeprazole) → v2.3; ECM/intracellular saturation; multi-dose
accumulation; production-path `enzyme_km` (a DDI/high-dose registry, separately specced); MIPD
genotype prior.
