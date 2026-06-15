# PGx v2.2b nonlinear-genotype feasibility gate

**Verdict: STRONG PASS.** ≥3 powered pairs with literature in-vitro `Km`, both-endpoint
genotype folds, and genuine MM-saturable kinetics; **≥2 are dose-ranging** (voriconazole,
propafenone) — the sharpest test of saturation, which no linear/closed-form null can reproduce.
This consumes the v2.2a MM flux (`DrugOnGraph.enzyme_km`). Two model-blind Opus curators
(PubMed); non-circular by construction (`Km`/`Vmax` from enzyme kinetics, independent of the
clinical fold). No values fabricated.

## Why this gate passes where v2.1 failed

v2.1 needed *linear* high-first-pass drugs (nearly none exist — high first-pass causes
saturation). v2.2b needs *saturable* drugs, which are abundant and well-characterized: the very
drugs v2.1 had to exclude (propafenone, atomoxetine, nebivolol, voriconazole, omeprazole) are
v2.2b's candidate pool. The engine is **non-redundant by construction** here — a saturable ODE
reproduces dose-dependent genotype folds that `ρ=0` and the 1-comp-linear null (v2.1's Null-1)
*cannot* produce at all. So the validation question sharpens from "does the engine beat a linear
null" (trivially yes) to "does the engine reproduce the **observed magnitude** of the
dose-dependent fold using literature `Km`."

## Powered set

| drug | gene | Cmax_fold | AUC_fold | dose-ranging | in-vitro Km (basis) | fm | oral_F | source(s) |
|---|---|---|---|---|---|---|---|---|
| **voriconazole** | CYP2C19 | 4.4× [3.6–5.4] | 5.6× [4.5–7.0] | **YES** (50 vs 200 mg; UM/EM/PM) | **9.3±3.6 µM** (adult HLM, N-oxidation, total) | major (+CYP3A4/FMO3) | ~0.96 | Zhu 2016 PMID 27530916; Wang 2008 PMID 18496684; **Yanni 2009 PMID 19420130 (Km)**; Wang 2021 (MM pop model) |
| **propafenone** | CYP2D6 | 2.4× | 11× | **YES** (300 vs 400 mg) | **5.3/3.0 µM** (S/R, HLM 5-OH) — also 0.12 µM (Hemeryck); spread flagged | ~0.7–0.85 | 0.05–0.12 EM → ~0.5 PM | Tran 2022 PMID 35890339; **Kroemer 1991 PMID 1857335 (Km)**; Hemeryck 2000 PMID 10917404 |
| **lansoprazole** | CYP2C19 | 1.57× | 3.42× | no | measured (Katsuki 2001) — **numeric Km in table, must extract** | major (+CYP3A4) | ~0.80–0.85 | Hu 2004 PMID 15301728; **Katsuki 2001 PMID 11534791 (Km)** |
| **atomoxetine** | CYP2D6 | ~2.5–3× | ~5× | partial | **2.3 µM** (HLM, 4-OH; CLint 103 µL/min/mg) | ~0.9 | ~0.63–0.94 | **Ring 2002 PMID 11854152 (Km)**; clinical fold from Strattera label / Sauer 2003 — **primary fold+CI to be pulled** |

**Anchor (linear control, is_nonlinear=false):** metoprolol — CYP2D6, Cmax 2.3× / AUC 4.9×
(Blake 2013 PMID 23665868); the engine must NOT predict dose-dependence here.

**Secondary (CL/F-powered, not paired Cmax+AUC):** perhexiline — gold-standard saturation
physics (**Km 3.3±1.5 µM**, Sørensen 2003 PMID 12814462; dose-ranging CL/F drop, Inglis 2007
PMID 17429312), but the clinical genotype contrast is CL/F + metabolite-ratio, not paired
single-dose Cmax+AUC+CI. Use as a saturation-physics consistency case, not a primary fold pair.

**Excluded:** nebivolol (no discrete CYP2D6 `Km`; mixed-enzyme + active metabolites); omeprazole
(oral nonlinearity is **mechanism-based auto-inhibition**, not reversible MM — a pure
`1/(1+C_u/Km)` term cannot represent time-dependent enzyme loss; defer to an MBI capability,
v2.3).

## Design implications to carry into the v2.2b spec

1. **Shared-enzyme split is mandatory.** None of the saturable drugs is 100% cleared by the
   gene (voriconazole +CYP3A4/FMO3; propafenone +CYP1A2/3A4; atomoxetine ~0.9). The MM setup
   must apply `Km` to the **gene fraction (fm)** and keep the **residual linear** (the v2.1
   `RESIDUAL_HEPATIC` split — residual tag carries no `enzyme_km` ⇒ stays linear, exactly what
   v2.2a's per-tag contract supports).
2. **`Km` unit/basis conversion (correctness-critical, a curation step).** Literature `Km` is
   µM, often total-substrate, microsomal; the engine's `enzyme_km` is **unbound mg/L**. Convert
   `µM → mg/L` via MW and total→unbound via `fu_mic`/`fu_p`, documented per drug. This
   conversion is a non-circular but error-prone step — pin it with a worked example + a unit test.
3. **The metric gains a dose axis.** v2.1's `ρ=log(Cmax_fold/AUC_fold)` carries over, but the
   primary v2.2b test is whether the engine reproduces the **dose-dependence** of the
   genotype fold (fold at dose₁ vs dose₂), which the linear nulls cannot produce. Voriconazole
   and propafenone supply the dose contrast; metoprolol is the dose-independent control.
4. **Engine config:** `well_stirred` + per-gene `enzyme_km` (v2.2a), EM-anchored / PM-predicted
   skeleton (v2.1), dense Cmax grid. Headline 2.731 untouched (harness-isolated, no `predict()`
   change).
5. **Propafenone `Km` spread** (0.12 vs 5.3 µM, ~40×) and the voriconazole shared-FMO3 fraction
   are the two largest sensitivity sources — report `ρ_engine` bands over the `Km` range.

Curated data preserved for the spec/plan regardless of path.
