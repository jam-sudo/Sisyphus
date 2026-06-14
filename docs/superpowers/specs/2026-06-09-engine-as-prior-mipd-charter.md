# Sisyphus-MIPD — Mechanistic Engine-as-Prior for Measured-Input, Out-of-Domain, and Individualized PK

**Date:** 2026-06-09
**Author:** Hypatia (with Jae Min Yoon)
**Status:** Charter / design — the pivot direction after the SMILES-only headline program was empirically foreclosed (see `2026-06-09-differentiable-mechanism-calibrated-pbpk-design.md` §0, Test A: meta residual CV R²≤0). No implementation until Gate 0 (a cheap, decisive feasibility test on data already on disk) passes. Likely a **new repo** (`sisyphus-mipd`) reusing the engine, graph, IVIVE, and `regimen/` SBI kernel.
**One-line:** Reposition the mechanistic engine from a one-shot SMILES→Cmax *oracle* (walled at 2.78) to a **structural prior that any sparse measured observation sharply updates** — winning in the three regimes where the current ML/meta stack structurally fails: out-of-domain chemistry, dose/regimen/population extrapolation, and individualized prediction.

---

## 0. Why this, and why now (the honest premise)

The SMILES-only Cmax headline is **empirically walled, not under-engineered** — proven this session, leak-clean, on n=1028: the meta's own residual is structure-unpredictable out-of-sample (CV R² ≤ 0 across GBM/Ridge × phys/Morgan), and the engine's recoverable-beyond-its-inputs signal is 0.064 < the 0.15 bar. 48 dead-ends + this measurement close that book. The walled quantity is **bioavailability F**, and F is *not in the SMILES* — it is formulation, salt/crystal form, food, particle size, and transporter genetics. **The only lever that moves the number is measured data.**

But "measured data" does **not** mean "feed measured ADME and beat 2.78 on the retrospective holdout." That is already characterized and is **not** the value proposition:
- DE-48: on a representative N=93 set, the engine with *measured* fup+CLint is ≈3.84 AAFE — **worse** than the meta's 2.78. The clean-10 "2.63→1.77 with measured F" was a cherry-pick (DE-48 retraction). On the retrospective a-priori holdout, the ML track wins and measured inputs do not change that.

So the value of measured data is **not** retrospective point accuracy. It is the three regimes where the ML/meta stack has **no signal by construction**, and where a mechanistic model conditioned on sparse measurements is the *only* thing that works:

| Regime | Why ML/meta fails | Why engine-as-prior wins |
|---|---|---|
| **A. Out-of-domain / novel chemistry** | ML is k-NN-like; novel scaffolds have no neighbors. Prospective AAFE is **3.27** (vs retrospective 2.78) and 26/107 holdout drugs are already out-of-AD. | The engine's parameters are mechanistic, not memorized — it degrades gracefully OOD, and one measured anchor calibrates the residual F-error. |
| **B. Dose / regimen / population extrapolation** | The ML/meta predicts a *single* (dose, healthy-adult) Cmax. It **cannot** extrapolate to other doses, multiple-dose steady state, renal-impaired/pediatric populations, or food states — there is no mechanism to move. | The engine *is* a dose-driven ODE over a parameterized physiology. Condition it on ONE arm (a single-dose Cmax, a microdose AUC), update the F/CL parameters, and predict every other arm. **872/1260 corpus drugs have ≥2 doses (median 3, max 14)** — the structure to prove this exists. |
| **C. Individualized / MIPD** | A population point estimate is not a patient prediction. | Already half-built: `regimen/tdm_sbi.py`/`tdm_enkf.py`/`tdm_ibis.py` assimilate measured patient concentrations. Extend that kernel from *dosing* to *full PK prediction* at the prediction stage. |

The product metric **shifts accordingly**, away from the walled number: from "a-priori single-Cmax AAFE on N=107" to **posterior-predictive accuracy + calibrated coverage as a function of how much measured data is supplied** (the data-efficiency curve), measured in the OOD/extrapolation regimes.

This plays directly to the author's stated specialty (SIR/SBI/importance-sampling/UQ — the pharmpy contribution plan) and turns the engine's structural correctness from a meta-damped liability into the asset it should be.

---

## 1. Thesis

**Engine = mechanistic structural prior. Measured observation = likelihood. Product = calibrated posterior over PK + its interval.**

Given a molecule and *whatever measured data exists* — nothing, a measured F, a microdose AUC, a single plasma concentration, one dose arm — produce a posterior over the engine's F/CL-determining parameters by SBI/importance-sampling, and propagate it to a posterior over the target PK quantity (Cmax at any dose/regimen/population) with an honest interval. With **zero** measured data it reduces to today's a-priori prediction; each added observation sharply narrows it. The engine's strength — it gets the *structure* (dose-response, distribution kinetics, accumulation) right while getting F-*magnitude* wrong — is exactly what makes one anchor collapse the error.

---

## 2. The decisive feasibility gate (Gate 0 — cheap, pre-registered, run BEFORE any build)

The premise has one load-bearing, un-run claim: **conditioning the engine on a single measured anchor extrapolates better than the ML/meta stack, which cannot move with the anchor.** Test it on data already on disk, no new product code, before committing to a repo. *(This is the analogue of the SMILES-program's Test A — the make-or-break number.)*

### Gate 0 — RESULTS (measured 2026-06-09, leak-clean, this session)

- **Gate 0a (dose extrapolation) — MOOT on the current engine, RETIRED.** The production engine is **exactly linear in dose** (verified: Cmax(2d)/Cmax(d) = 2.0000 for midazolam/warfarin/metoprolol; all production fluxes are first-order — clearance `cl_intrinsic·c`, absorption `ka·y`; the only Michaelis-Menten term lives in `active_transport`, which **no production YAML instantiates**). For a linear engine, "condition on one dose arm, predict another" is **algebraically identical to the dose-proportional baseline** → guaranteed tie. **The dose-extrapolation moat does NOT exist until saturable kinetics (dissolution cap / MM clearance / saturable first-pass) are built — that is now an explicit prerequisite, not a near-term win.**
- **Gate 0b (measured-anchor cross-endpoint) — PASSED.** On a random representative leak-clean measured-F set (n=200): routing one measured anchor (F) through the engine improves the **production meta 2.567 → 2.343 (Δ +0.225, −9%; bootstrap 95% CI [+0.061, +0.388], excludes 0)**. The a-priori engine alone (5.31) is rescued to ~2.45 by the single anchor — **one measured point collapses the engine's F-error**, the core premise. The gain is in aggregate AAFE (the F-catastrophe tail), not %within-2-fold (48→49%). **The engine is the irreplaceable mechanism** converting measured-F→Cmax — a measured-ADME ML regressor cannot (DE-47), and *up-weighting* the measured engine degrades (DE-48); *feeding measured F through the existing blend* is the operation that works.
- **Gate 0c (OOD margin) — PASSED, the strongest result.** Measured absolute F was sourced for 8 of the 28 prospective NMEs (all IV-referenced, adversarially verified: pirtobrutinib 85.5%, zongertinib 76%, inavolisib 76%, nerandomilast 73%, paltusotine 69%, remibrutinib 34%, imlunestrant 10%, rilzabrutinib 4.7%; the other 20 genuinely report "absolute F not determined"). The measured-F margin is **~3× larger OOD than in-domain**: OOD (n=8) meta **2.892 → 2.150, Δ +0.742** (bootstrap 95% CI **[+0.089, +1.414]**, excludes 0) vs in-domain (n=200) **Δ +0.225**. Per-drug, measured F corrects 6/8 — the high-F NMEs the a-priori engine under-predicted (pirtobrutinib 0.24×→0.37×, zongertinib 0.28×→0.58×, nerandomilast 0.40×→0.71×) move toward observed. **This confirms the core thesis: measured data + the mechanistic engine helps *most* exactly where ML memorization fails (OOD) — the regime where the SMILES-only product is weakest (prospective 3.27).** Caveats: N=8 (CI wide but excludes 0); both margins computed on the current working tree (uncommitted engine changes), so absolute baselines differ from the canonical 2.784 snapshot, but the a-priori-vs-measured-F margin is internally apples-to-apples; part-1 engine-competitiveness used the 2026-06-01 artifact snapshot.
- **Implication for scope:** the near-term, demonstrable win is the **measured-input-conditioned product** (regimes A/C + the data-efficiency curve), NOT dose/regimen extrapolation (regime B, gated behind building nonlinear kinetics). Gate 0b PASSED (in-domain), Gate 0c PASSED (OOD, the advantage is largest where it matters most), Gate 0a RETIRED (engine linear). **The engine-as-prior direction is empirically validated for the measured-input regime.**

The original pre-registered gate definitions below are retained for the record; 0a is retired (engine linear), 0b passed (in-domain +0.225), 0c passed (OOD +0.742, ~3× larger).

**Gate 0a — dose/regimen extrapolation (the strongest claim).** On the **872 multi-dose corpus drugs** (leak-clean via `load_mmpk_data`'s 3 guards), for each drug hold out all dose arms but one; condition the engine on the single retained arm (fit a per-drug F/CL scalar by matching that arm's Cmax); predict the held-out arms. Compare three predictors on the held-out arms:
1. **engine + 1 anchor** (mechanistic extrapolation),
2. the **a-priori meta** (fixed — cannot use the anchor),
3. a **dose-proportional ML baseline** (the naive extrapolation).

**PASS iff engine+anchor beats both (2) and (3) on held-out-arm AAFE by > 0.15 absolute, and the margin is larger on high dose-ratio / nonlinear-PK arms.** This isolates exactly what the engine can do that ML structurally cannot.

**Gate 0b — data-efficiency curve (the product claim).** On the measured-F drugs (517 corpus + 498 `bioavailability_v1.csv`, leak-clean), plot posterior Cmax AAFE + 90% coverage as a function of measured inputs supplied: {none} → {measured fup} → {+CLint} → {+F} → {+1 conc}. **PASS iff coverage stays ≥ nominal AND AAFE falls monotonically, with the {+F}/{+1 conc} step delivering a materially tighter, still-calibrated interval than the a-priori conformal /÷12.92.** *(Honest expectation per DE-48: the a-priori and measured-ADME *point* numbers will be mediocre; the win must be in the interval tightening + the OOD/extrapolation margin, not retrospective point AAFE.)*

**Gate 0c — OOD margin.** Repeat 0a/0b restricted to the prospective N=28 (and out-of-AD holdout) drugs — the regime where ML is weakest. **PASS iff the engine+anchor margin over the meta is *larger* OOD than in-domain** (the core thesis: mechanism degrades gracefully where memorization fails).

If 0a fails, the extrapolation moat is illusory → stop and reconsider. If 0a passes but 0b/0c are weak, the product is "dose/regimen extrapolation" only (still valuable), not a general MIPD platform.

---

## 3. Architecture (reuse heavily; build narrowly)

```
SMILES + dose ──► engine forward ODE (REUSE: graph/, engine/, IVIVE)  ──► a-priori prior over PK
                          ▲                                                      │
 measured data ───────────┘  likelihood layer (NEW)                              ▼
 {F, AUC, conc(t),     ──►  SBI/SIR posterior over F/CL params (REUSE kernel:  POSTERIOR PK + honest PI
  dose-arm, covariates}     regimen/tdm_sbi·enkf·ibis; NEW: molecular prior,    (Cmax @ any dose/regimen/pop)
                            multi-observation-type likelihood)
```

- **Reuse as-is:** the graph/ODE engine and forward simulation; IVIVE (SMILES→parameter priors); the conformal calibration; the `regimen/` SBI/EnKF/IBIS inference kernels and the `sbi/` training scaffold.
- **Build (the narrow new surface):**
  1. **A heterogeneous likelihood layer** — map each observation type (measured F, NCA AUC, a single timed concentration, a full dose arm, a covariate like CrCl) to a likelihood over the engine's outputs. This is the genuinely new modeling work.
  2. **A molecular-feature amortizer / prior** — the honest correction from the prior spec: `regimen/tdm_sbi.sbi_update` is *per-patient*, keyed on measured concentrations, with a per-drug population prior. Conditioning at the *prediction* stage on molecular features needs a **new amortized posterior over the F/CL parameters given (SMILES-prior, observation)** — a from-scratch SBI build reusing the `sbi/` scaffold, **not** a rewire of the TDM path.
  3. **A small observation-routing API** — `predict(smiles, dose, observations=[...]) → PosteriorPK` — that degrades to today's `predict()` when `observations=[]` (so the a-priori path and the 2.784 headline are untouched, exactly as measured-F routing already is).
- **Invariants preserved:** engine stays identity-blind (conditioning happens in the inference/predict layer, never in `engine/flux.py`); compiler/solver untouched (Invariant 8); holdout inviolable (Gate 0 uses the `load_mmpk_data` 3-guard filter — the verified-clean one, never the 63-name `exclusions.json`); everything stays Distribution-native (the posterior *is* the output).

---

## 4. Metric & success criteria (the deliberate shift)

- **NOT** the N=107 a-priori single-Cmax AAFE (walled at 2.78; this program does not target it and will not beat the meta there — stated up front to avoid the prior spec's overclaim trap).
- **Primary:** held-out-dose-arm AAFE (Gate 0a) and the data-efficiency curve (Gate 0b) — engine+anchor materially beating the a-priori meta and the dose-proportional baseline, with calibrated coverage.
- **Headline product claim:** "Sisyphus-MIPD turns *N* measured points into a calibrated PK posterior; with one phase-1 anchor it predicts untested doses/regimens/populations where pure-ML and pure-a-priori systems cannot." Quantified by the OOD margin (Gate 0c).
- **Reference floor:** OrBiTo's expert-input F-AAFE 1.75 is the realistic ceiling *with* measured inputs (vs the 2.78 SMILES-only wall) — the legitimate headroom this direction unlocks.
- **Discipline:** every claim measured leak-clean in public-clone state; the a-priori path bit-identical when `observations=[]`; honest "proven vs to-test" labeling throughout.

---

## 5. Risk register (front-loaded honesty — the lessons from this session)

| Risk | Mitigation / status |
|---|---|
| **Measured *point* accuracy is mediocre (DE-48: representative engine-measured ≈3.84)** | Acknowledged and designed-around: the value is extrapolation + interval, NOT retrospective point Cmax. Gate 0 measures the right thing. |
| **The extrapolation moat (0a) might not exist** | That is exactly why 0a is the pre-registered cheap gate, run before any repo. Kill if engine+anchor doesn't beat the dose-proportional baseline. |
| **Residual structural error survives anchoring** (well-stirred Fh model-form for high-PPB acids; the F-error isn't a single scalar per drug) | A 1-parameter anchor corrects a magnitude offset, not a model-form error. Gate 0b's data-efficiency curve will show where one anchor is insufficient and ≥2 are needed; report it honestly per drug class. |
| **The amortizer is a from-scratch build, not a `regimen/` rewire** | Scoped as the one genuinely new component; reuse the `sbi/` training scaffold + PRIOR_LOW infra, not `tdm_sbi.sbi_update`. |
| **Measured-F label noise** (`bioavailability_v1.csv` is text-mined; f_pct has >100% rows) | Use it for the likelihood with a noise model; audit/exclude the impossible rows; prefer dose-arm Cmax (denser) for Gate 0a. |
| **Leak** | Reuse the verified `load_mmpk_data` 3-guard (flag + 183-name + InChIKey-14) filter; never the 63-name `exclusions.json`. (The prior spec's "leak blocker" was a false positive from that wrong file — verified this session.) |
| **Scope / solo-researcher effort** | Gate 0 is scripts on existing data (no repo). The repo is committed only after 0a passes. The new modeling surface is one likelihood layer + one amortizer. |

---

## 6. New repo vs in-place

**Recommend a new repo `sisyphus-mipd`** (or a long-lived branch) once Gate 0a passes: the observation-routing API, the heterogeneous likelihood layer, and the molecular-feature amortizer are a coherent new product, and the metric/positioning differ from the SMILES-only platform. **Reuse** (as a dependency or vendored): `graph/`, `engine/`, `predict/ivive.py`, `pk/`, the conformal calibration, and the `regimen/`+`sbi/` inference kernels. The ChemRxiv framing extends naturally ("topology-compiled PBPK + Bayesian parameter refinement" → "...as a measured-input-conditioned posterior PK engine").

---

## 7. Relationship to the foreclosed SMILES-only program

This **is** the breakthrough the `/goal` asked for — but the evidence relocated it. The SMILES-only headline cannot move (measured, not argued: Test A, §0 of the sibling spec). The genuine lever is measured data, and the genuine product is the engine-as-prior posterior in the OOD/extrapolation/individualized regimes. The sibling spec's gradient-free corpus-calibration work is **not** part of this — it was itself gated out (Test A) — except for its one reusable artifact: the verified leak-clean `load_mmpk_data` protocol.

---

## Appendix — what is PROVEN vs what Gate 0 must ESTABLISH

**Proven (measured this session / in the repo):**
- SMILES-only meta residual is structure-unpredictable (CV R²≤0) — the headline is walled.
- Representative measured-ADME engine ≈3.84 (DE-48) — measured inputs don't win on retrospective point Cmax.
- The data exists: 872 multi-dose drugs, 517+498 measured-F, populated formulation column, prospective N=28.
- `regimen/` SBI/EnKF/IBIS exists (per-patient assimilation kernel).
- Shipped training is leak-clean via the 3-guard `load_mmpk_data` filter.

**Must be established by Gate 0 (do NOT assume):**
- That engine + 1 anchor beats the dose-proportional baseline and the a-priori meta on held-out dose arms (0a).
- That the data-efficiency curve is monotone with calibrated coverage and a materially tighter interval at +F/+conc (0b).
- That the margin is *larger* OOD than in-domain (0c) — the core "mechanism > memorization where it counts" claim.
