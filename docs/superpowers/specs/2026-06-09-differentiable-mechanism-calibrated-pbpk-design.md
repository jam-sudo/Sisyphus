# Sisyphus v2 — Gradient-Free, OrBiTo-Anchored, Corpus-Calibrated PBPK

**Date:** 2026-06-09 (hardened against 5 adversarial reviews + a live artifact audit, then a red-team pass on the synthesis)
**Author:** Hypatia (with Jae Min Yoon)
**Status:** Design — charter for a *staged spike*, not a rebuild. No new repo, no engine/solver changes, and no segmented-gut work until Test A, Test B, and the leak-guard CI all pass. No implementation until approved.
**One-line:** Port Omega's one un-ported lever — **gradient-free calibration of ~3–6 identifiable global/class-conditioned physical constants on a holdout-clean corpus, co-calibrated with whatever replaces the fixed meta** — and *prove it earns a rebuild before building one*. Honest primary target: **enter the OrBiTo band (AAFE 2.08–2.74) at its top (≈2.70–2.74)**; 2.55–2.65 is a *stretch contingent on the unproven dissolution lever*. The engine-F-fix is a strict precondition for any gate work.

> **What changed from the 2026-06-09 "Differentiable, End-to-End Mechanism-Calibrated" draft, in one breath:** differentiability is *deleted* from the critical path (the forward map is provably non-differentiable — `np.clip(kp,0.01,50)` at three IVIVE sites, a 5-bin particle-radius step function, dict-keyed FM, registry lookups); the segmented-gut "rebuild" is *re-scoped* to the one thing actually missing (a dissolution flux + regional gut-wall CYP3A — the 8-segment lumen already exists in `reference_man.yaml`); the headline target moves from Omega's likely-contaminated 2.215 to the externally-verified OrBiTo band; the learned gate is *demoted* from the thesis to a downstream, precondition-gated probe; and a **hard leak blocker** (≥23 holdout drugs leak into the corpus even with both current guards) is fixed before any number is computed.

> **⚠ Honest-arithmetic caveat (the synthesis red-team's verdict, kept visible):** the only *directly-supported* calibration move (between-class debias) is worth a small amount — below the ~0.15 stack-drift/CI resolution floor on the meta. The path to even 2.70 depends on the global-median null surviving the meta (DE-43 says it largely won't) **plus** the *unproven, pre-gated* dissolution within-class lever. **This program's expected headline gain is small and may be zero; its decisive test (Test A) was bearish a priori (DE-28 R²=−0.09).**

---

## 0. EMPIRICAL OUTCOME — Gate 0 RAN AND FAILED (measured 2026-06-09, leak-clean, this session)

**The decisive pre-registered kill-test was executed on data already on disk. The SMILES-only program is HALTED at Gate 0. The build phases below were NOT triggered.** All numbers below are from direct measurement (csv + rdkit + sklearn on the conda base env), independently reproducing the review's load-bearing claims:

- **Test A — FAILED its kill criterion.** On the leak-clean corpus (n=1028, holdout InChIKeys excluded, scaffold 5-fold GroupKFold):
  - **The META residual is NOT structure-predictable out-of-sample: CV R² ≤ 0 for every model** (GBM phys −0.089, GBM Morgan −0.133, GBM Morgan+phys −0.137, Ridge −0.016 to −0.459). "Correcting" the meta with a structure-learned residual makes it *worse* (3.286 → 3.42–4.17). **The headline 2.784 is structurally walled — no SMILES-only lever (calibration, ML, post-hoc) can move it.** This cleanly reconfirms DE-28/DE-23 on n=1028.
  - The ENGINE residual is only weakly predictable (Morgan+phys R²=0.134; phys-only 0.070). The pre-registered gate metric **GAP = R²_total − R²_circular = 0.134 − 0.070 = 0.064 < 0.15 → KILL.** Even fully exploited it moves the engine track 3.87 → 3.45, which the meta then damps to nothing.
- **Verified (independent re-measurement, matches the review):** OOF signed-residual r(eng,ml)=**0.781**, r(ml,meta)=**0.982** (the learned gate inherits co-calibration — no orthogonal axis); engine bias **sign-flips** holdout −0.419 (0.38× under) vs corpus +0.136 (1.37× over) — corpus→holdout calibration transfer is genuinely broken; holdout meta >3× tail **18/18 OVER, 0 under**; engine >5× catastrophes **8/8 OVER**.
- **The "corpus leak" hard blocker was a FALSE POSITIVE — RETRACTED.** It audited against `mmpk_sisyphus_holdout_exclusions.json` (63 names), which the training does **not** use. The shipped training (`scripts/ml_cmax_improvement.py::load_mmpk_data`) applies **three** guards — `in_holdout` flag + the full **183-name** holdout+train list + **InChIKey-14** — and **0 holdout structures survive into the training set** (verified). The shipped 2.784 is leak-clean. *(The one durable lesson: any future corpus calibration must reuse `load_mmpk_data`'s exact 3-guard filter, never roll its own against the 63-name exclusions file.)*
- **Class-debias headroom (corrected):** per-class median debias moves the *engine* track corpus AAFE 3.866 → 3.761 (Δ≈0.10, driven by neutrals 0.19) — larger than the review's "≤0.05," but it is an *engine-track* move, meta-damped, and **sign-flip-blocked from transferring to the holdout**. Net headline value ≈ 0.

**VERDICT:** There is **no SMILES-only algorithmic breakthrough**. The meta's error is structurally unpredictable; the engine's recoverable-beyond-current-inputs signal is sub-threshold and meta-damped. **The only path to a materially better number is a measured F-orthogonal data modality (Direction 1 / "Leap D", §6 fallback) — a data-acquisition program, not a model change.** The §1–§8 calibration/structure design below is retained as the *record of what was tested and why it was foreclosed*, and as the reusable leak-clean protocol — not as a build that should proceed.

---

## 1. Why the current number is walled (the problem)

The a-priori SMILES-only Cmax headline is **AAFE 2.784** (N=107 holdout). 48 enumerated dead-ends prove this is walled, not under-engineered. The wall is **three multiplicative layers** (diagnosis §I):

- **W1 — CLint target noise.** R²≈0.24 is intrinsic to the hepatocyte assay (interlab CV up to 99.8%); foundation models lose to Morgan+XGB. *But the label-noise floor is only AAFE≈1.18 (N=14 replicate pairs, medium confidence; 3–16% of error variance) — the system is MODEL-limited, with ~0.70 AAFE of nominal headroom down to the OrBiTo floor (2.08).*
- **W2 — bioavailability-F structural error (the binding wall).** `F = fa·Fg·Fh`. On the holdout the engine error is **overwhelmingly unidirectional UNDER-prediction** (mean signed log −0.419 ≈ 0.38×; verified: base 40 under/15 over, neutral 27/6, zwitterion 10/1, acid 7/1). DE-42 proved a flat `ka` scalar nulls the *median* but cannot reduce *per-drug dispersion*. FLUX-1 fixed the worst over-extraction instance and *regressed* the headline — the wrong formula was load-bearing calibration **because it was shipped without recalibrating the meta**.
- **W3 — the meta damps and penalizes the engine.** A fixed-weight geometric mean with an anti-engine disagreement rule (>10× → down-weight the *engine*) attenuates a physically-correct ±15–17% engine move to ±2–3% headline (derived effective pass-through 0.053/0.169 under penalty).

**The unlock the dead-ends missed:** every one of the 48 attacks moved *one* front and let the fixed meta absorb it. W2 and W3 are multiplicative — paracellular physics (−5% engine) became −1.1% headline; FLUX-1's correct physics regressed. **No single-front attack can win. The engine fix and the meta replacement must be co-calibrated.** This co-calibration discipline — never ship structure without recalibrating the blend — is the spec's spine and is the one strategy the 48 dead-ends never tried. *(All 5 reviewers endorse this as the genuine contribution; the synthesis red-team notes its benefit is asserted, not yet bounded — Test B must adjudicate it, see §6.)*

**The historical lead, now correctly caveated:** the predecessor **Omega** calibrated 5 global constants on its MMPK corpus (Optuna, 180 trials) and Sisyphus *never ported corpus calibration at all* — its engine constants (`_CLINT_SCALING=10.8` ivive.py:37, absorption `2.88` flux.py:426, gut CYP3A4 `1.38e7` reference_man.yaml:100, GFR, `_IVIVE_SCALING`) are hand/literature-set and never corpus-fit. **The technique gap is real and is the spec's load-bearing bet.** Omega's *number* (2.215) is **not** a target: it predates the 2026-04-04 holdout-leak fix (which invalidated Sisyphus's own pre-fix 2.283 when 76–100/107 holdout drugs leaked into ML training), used a different corpus/holdout, and is almost certainly contamination-era. **The honest external anchor is OrBiTo expert-harmonised AAFE 2.08–2.74**; Sisyphus at 2.784 already sits just outside the top of that band, fully automated and SMILES-only.

---

## 2. Thesis (what changes) — and what does NOT

**Calibrate a small set of identifiable global + compound-class-conditioned physical constants θ on a holdout-clean corpus, gradient-free, co-calibrated with a recalibrated blend — and gate every expensive step behind a cheap kill-test.** Three changes, each scoped to what the evidence supports:

| Wall | Current | v2 — corrected scope | Evidence-bounded value |
|---|---|---|---|
| W2 between-class bias | one global scalar (DE-42 dead) | **class-conditioned hepatic-extraction / fu constants in `predict/ivive.py`** (acid/base/neutral) | **between-class median trim only, ≤0.05 AAFE (sub-threshold).** Per-class-median debiasing moves corpus engine AAFE only acid 3.95→3.92, base 3.60→3.53, neutral 4.06→3.86. Within-class dispersion dominates — class knobs cannot touch it. |
| W2 within-class fa dispersion | single linear `ka`, **8-segment lumen ALREADY present**; missing = dissolution + regional gut-wall CYP3A | **add a dissolution flux (Noyes-Whitney + pH-dependent S(pH)) + regionalize the gut wall** | **unproven, pre-gated, correctly directed only for the low-solubility OVER-tail.** A solubility/dose-number cap *lowers* fa for high-dose/low-sol drugs (it cannot raise fa above the first-order value) → it helps **over-predictors** (lenacapavir-class, EXTREME_LIPOPHILIC), not the under-tail. In-repo solubility = `−logP+0.5` (DE-45) → SMILES-fed dissolution is logP-circular; build ONLY if a pKa-aware-S(pH) probe shows dispersion correlates with the F-residual via the *dose-number* term, not logP. |
| W3 | fixed geometric mean, anti-engine | **recalibrated blend whose weights are themselves corpus-calibrated constants (part of θ)**; a *learned* feature-conditional gate is a downstream, precondition-gated option only | **gate is downstream of the engine F-fix.** Engine standalone = 4.458 > ML 3.01 > meta 2.784. A gate trained on today's engine will correctly *reproduce* the damping (DE-43/48). The corpus oracle ceiling (best-pick eng\|ml = 2.45, optimal blend = 2.30 on the OOF surface) is the *most* a gate can reach; fixed reweighting is already near-exhausted (best global eng/ml blend 2.94 holdout). |
| Calibration | none (hand-set) | **gradient-free (CMA-ES / Optuna-TPE) point estimate; SBI for the θ posterior** | Omega's actual method. **~3–6** actively-calibrated constants *after the identifiability prune determines the rank* (not 10–15, and possibly fewer than 5 — see §3.2). |
| N=107 underpowering | every meta experiment overfit | **all gate decisions on the corpus via nested CV; N=107 touched exactly once** | The corpus is *harder* than the holdout (OOF engine 3.86 / ml 3.30 / meta 3.28) and signed bias *flips* (corpus over-predicts, holdout under-predicts) — so corpus→holdout transfer is an explicit, tested, **possibly program-killing** risk (Gate 0.5), not an assumption. |

**Deleted from the original spec (do NOT build):** the differentiable forward map / diffrax adjoint (§3.1 of the prior draft); the "7-segment gut rebuild" framing; the 2.215 target; the "new repo Sisyphus2" as a Phase-0 commitment; the learned gate as a load-bearing thesis element.

---

## 3. Architecture

### 3.1 Calibration method: GRADIENT-FREE (differentiability is an optional later accelerator only)

The forward map `SMILES-features → ADME → engine ODE → Cmax` is **provably non-differentiable in θ**: `np.clip(kp,0.01,50)` at ivive.py:409/458/520 (zero/undefined gradient exactly at the bounds, which fire for the high-Kp bases and high-PPB acids — the W2 failure classes), the 5-bin `_estimate_particle_radius` step function (ivive.py:566–590), `_DEFAULT_FM` dict-keyed on `compound_type`, and registry lookups. For ~3–6 constants, **CMA-ES or Optuna-TPE is sufficient and robust** (Omega used Optuna, 180 trials), and an SBI/SNPE posterior gives θ uncertainty (Invariant 2: θ become calibrated *Distributions*). **`engine/compiler.py`, `engine/solver.py`, the JAX RHS, and `DrugOnGraph` are untouched (Invariant 8).** A diffrax adjoint is reserved as a *Phase-4 efficiency item only* if gradient-free proves too slow — never on the critical path.

### 3.2 The calibrated constant set θ — FEW, PHYSICAL, GLOBAL/CLASS, and IDENTIFIABILITY-PRUNED

Candidate pool (all enter via `predict/ivive.py` or a predict-layer config object — never as a branch in `engine/flux.py`; see Invariant-1 clause §5):
- CLint hepatocyte→whole-liver scaling (`_CLINT_SCALING=10.8`)
- Absorption scaling (the `2.88` constant; gut-transit scaling)
- Gut CYP3A4 abundance scaling (the FLUX-1 re-anchor)
- **Class-conditioned hepatic-extraction multipliers** (acid/base/neutral) — `compound_type` already lives in `ivive.py` (`_FM_ADJUSTMENTS`, `_compute_kp_rodgers_rowland`).
- **A single global acid-class fu-correction constant** — replaces the DE-37 data-blocked *per-drug* `lookup_hepatic_fu_correction` registry with ONE class constant (Invariant-6-legal, closes the paywall block as a class trim — **not** as a per-drug fix).
- Renal GFR/secretion scaling; a global Kp clip-bound.

**Hard identifiability prune (mandatory pre-calibration, REQUIRED before any optimizer run, and BEFORE committing to a constant count).** The corpus has **NO `fup` and NO `CLint` columns** (verified: `mmpk_expanded_full.csv` columns are name/canon_smiles/dose_mg/cmax_mg_L/log_cmax_per_dose/n_studies/tmax_h/thalf_h/auc_ng_h_ml/f_pct/formulation/is_prodrug/in_holdout) and **only 27% of rows have `f_pct`** (1023/3806). θ is therefore **Cmax-supervised only**, so:
- `fu,inc/fup` correction is **non-identifiable separately from CLint-scaling** from Cmax alone → **drop the separate fu,inc/fup target; keep only the one acid-class fu constant**.
- `_CLINT_SCALING`, gut-CYP3A, and class-extraction move first-pass F in overlapping ways (FLUX-1's ×0.652 gut re-anchor already had to compensate for the D-2 Qgut entanglement). Compute the Jacobian/Fisher information of corpus-Cmax wrt the candidate θ; **report the RANK of the sensitivity matrix and drop/merge any pair with |sensitivity-vector corr| > 0.8** (CLint-scaling vs gut-CYP3A vs Qgut are the prime suspects — they may collapse to a single free first-pass-F direction, in which case the identifiable count is 2–3, not 5–8, and the §8 target must be revised down accordingly).
- The binding limit on *how many you can distinguish* is the ~12% per-drug stack-drift noise floor + the N=107 final-test CI [2.41, 3.25], which resolves only effects > ~0.15 AAFE. **Cap the actively-calibrated set at the identifiable rank (expected ~3–6, possibly fewer);** the rest stay literature-fixed with Bayesian priors centered at current values.

### 3.3 The blend: recalibrated weights first; a learned gate is downstream of the engine F-fix

**Stage 1 (default):** replace the fixed geometric mean + anti-engine penalty with a **simple recalibrated weighting whose weights are corpus-calibrated constants (part of θ)**. This captures the co-calibration benefit (move structure and blend together) **without** the gate's overfit risk. *Caveat: the best global fixed eng/ml blend is already AAFE 2.94 holdout / 3.26 corpus — fixed reweighting is near-exhausted, so Stage 1's standalone value is small; its purpose is to let θ and the weights move together (Test B adjudicates whether that joint move beats frozen weights within the prior-σ trust region — see §6).*

**Stage 2 (option, precondition-gated):** a small, constrained learned gate `g(features)→simplex weights over {engine, ml, vdss, clf}`, features = {AD flags, `compound_type`, engine/ml disagreement, predicted-F, CLint uncertainty}, trained on the OOF corpus. **The anti-engine rule is deleted only if the engine F-fix has demonstrably landed.**

**The precondition, stated as a hard ordering (correction #3, all reviewers):** the engine standalone AAFE is **4.458** (worse than ML 3.01 and meta 2.784). A gate cannot manufacture engine correctness; trained on today's engine it will *learn to down-weight it* (DE-43/48 — damping is the correct behavior). **Build the learned gate ONLY after the engine-F-fix has driven the engine's *corpus* AAFE below the ML track (~3.0 on the OOF surface).** Until then, Stage-1 recalibrated weights are the blend. *(The up-weighting opportunity is real but small: 17/107 holdout drugs are engine-uniquely-right while ML is wrong — 11–14/17 bases/zwitterions, e.g. budesonide eng 1.43× vs ml 28.1× — vs only 1/107 ml-uniquely-right; on the corpus 184/1029 engine-uniquely-right vs 175 ml. The gate's value is the oracle-minus-meta gap, corpus 3.28→2.45 ceiling, not the full headroom.)*

### 3.4 Absorption physics (the within-class W2 substrate) — re-scoped to the ACTUAL gap, correctly directed

`reference_man.yaml` (lines 206–361) **already defines** 8 GI lumen segments (stomach/duodenum/jejunum1,2/ileum1,2,3/colon) with per-segment `ka_fraction` (1.0→0.05 taper), per-segment `transit_rate` (4.0→0.074 /h), AND a parallel Renkin paracellular route on every segment. `AbsorptionFluxSpec.apply` (flux.py:411–430) is `ka = 2.88·peff·ka_frac/radius·y[lumen]` — pure first-order; the existing topology already disperses fa 0.03–0.96 continuously via Peff. **The lumen segmentation is NOT new and must not be re-built.**

The two genuine gaps:
1. **Dissolution dynamics (D-6).** Absorption has no solid↔dissolved state split and **dose never enters absorption** — a 600 mg BCS-II tablet absorbs the same *fraction* as a 1 mg microdose. A solubility-capped dissolution flux is the only proposed move that introduces a dose/solubility *nonlinearity* in the ODE. **Direction (red-team-corrected):** a Noyes-Whitney rate with a dissolved pool capped at `S(pH)·V_lumen` can only *reduce* fa for high-dose/low-solubility drugs — it **cannot raise fa** above the current first-order value. So its headline effect is to **LOWER predicted Cmax for low-solubility OVER-predictors** (e.g. lenacapavir 38× over, EXTREME_LIPOPHILIC; the classic dose-number-limited BCS-II/IV class). This is *correctly directed* for part of the 18/18 OVER tail — the original draft's "fixes UNDER" framing was wrong. `S(pH)` via Henderson-Hasselbalch subsumes the un-run DE-45 pKa-aware-solubility variant.
2. **Regional gut-wall CYP3A (Fg dispersion).** The wall is a single `gut_wall` node with one CYP3A4 pool (reference_man.yaml:95–100). Split it into regional wall sub-nodes / regional CYP3A fractions (a builder + YAML change, identity-blind). For high-first-pass bases this *raises* gut extraction → also lowers Cmax on over-predicted bases.

**This whole sub-section is gated behind Gate 3a (§6).** It is the *only* arm that can plausibly move the headline OVER-tail, but it is unproven and logP-circularity-prone — hence pre-gated.

---

## 4. Calibration & evaluation protocol (the load-bearing discipline)

- **Objective:** a **robust point loss (Huberized |log-fold| — AAFE-native) on corpus Cmax**, NOT the meta's conformal/posterior NLL. The label-noise model the NLL needs is under-identified (the 1.18 floor rests on N=14 replicate pairs; corpus median `n_studies`=1, and **536/1260 drugs (43%) have ≥2 studies** — i.e. most drugs are single-study), so an NLL with mis-specified aleatoric variance would over-trust single-study labels and can *degrade* point AAFE. **NLL / interval coverage are reported as a SECONDARY metric only.** F is an *auxiliary regularizer / sanity check* on the engine's IV-reference solve, never the primary target (it is 27%-complete, text-mined in `bioavailability_v1.csv`, with 30/1023 `f_pct>100%` rows that are pure noise).
- **Optimizer:** CMA-ES or Optuna-TPE for the θ point estimate; SBI/SNPE for the θ posterior in the full build (reuse the `sbi/` *training scaffold* `scripts/sbi_train_*.py`, NOT `regimen/tdm_sbi.sbi_update` — that is a per-patient assimilation tool keyed on measured concentrations and is structurally unrelated to corpus θ-calibration).
- **Splits:** **nested CV on the corpus**; the inner CV never sees the test fold's θ. **N=107 holdout LOCKED, evaluated exactly once after Phase 4**, with a pre-registered single AAFE + bootstrap CI and **no option to iterate** (Invariant 5).
- **Conformal:** split-conformal is re-run ONCE on the *frozen* predictor (as today), AFTER θ is locked — never inside the calibration loop (it calibrates residuals of the final predictor; an in-loop conformal term leaks).
- **Regularization:** Bayesian priors on θ centered at current literature values; report θ drift between the engine-only fit (§6 Stage 1a) and any joint step — if θ moves > 1 prior-σ in the joint step, the loss is exploiting the blend, not improving physics → revert to frozen θ. *(Synthesis red-team note: this guard, if honored, forbids exactly the region where co-calibration would add value beyond fixed-reweighting — so Test B must report whether any co-calibration benefit exists WITHIN the prior-σ region, else co-calibration is DE-26 in new clothing.)*

### 4.0 LEAK-PROOF DATA PROTOCOL (a hard blocker — fix BEFORE any number is computed)

**Audited finding (this session, on-disk):** the corpus contains holdout drugs that the *named* guards miss. Of the 107 holdout drugs, **98 appear in `mmpk_expanded_full.csv` by name**; the `mmpk_sisyphus_holdout_exclusions.json` file covers **63**, the `in_holdout=True` column flags **48 rows**, and **even applying BOTH guards, ≥23 holdout drugs still leak** (acamprosate, alvimopan, atovaquone, brincidofovir, budesonide, carbinoxamine, dalfampridine, dapagliflozin, lamivudine, mefenamic acid, nilotinib, paroxetine, penicillamine, pilocarpine, ponatinib, progesterone, quinine, ramelteon, rifabutin, rivaroxaban, selegiline, ulipristal, upadacitinib). Name-matching also misses salt/stereo/isomeric variants, so the true count is ≥23. **As-written, every Gate-1/Phase-3 corpus number would be holdout-contaminated — the exact pre-2026-04-04 mistake the project already paid for (Invariant 5 violated at the data-loading layer).**

**Mandatory remediation, gated as a CI test before Gate 1:**
1. Regenerate the exclusion set **by InChIKey** (canonicalize both corpus SMILES and holdout SMILES via RDKit; match on first-block InChIKey, not name string).
2. Add a CI assertion `len(corpus_inchikey ∩ holdout_inchikey) == 0` and report the post-clean N in the spec.
3. Add a **scaffold-neighbor** exclusion: class-conditioned constants leak through scaffold-near holdout drugs; drop corpus drugs within a Tanimoto/scaffold neighborhood of any holdout drug used to fit a class constant.
4. **Apply the InChIKey audit to `oof_predictions.csv` too** — it has NO name/InChIKey column, so "OOF by construction" is a raw-SMILES guard only (the weak guard rejected elsewhere). Add a name/InChIKey column and assert empty holdout intersection before any gate trains on it.
5. **Kill criterion:** if the cleaned corpus loses > 40% of drugs to holdout + scaffold-neighbor exclusion, N may be too small to fit even ~3–6 constants without overfit — re-evaluate whether the corpus advantage over N=107 survives before proceeding.

**The same InChIKey leak-clean applies to Test A** (`bioavailability_v1.csv` shares **48 drugs with the holdout by name** — its R² is inflated unless leak-excluded).

---

## 5. Invariant preservation (load-bearing walls stay up)

1. **Engine identity-blind.** **Explicit binding clause:** all θ enter via `predict/ivive.py` or a predict-layer config object (scaling `enzyme_affinity` / `fu_correction` BEFORE `DrugOnGraph`); `engine/flux.py`, `engine/compiler.py`, `engine/solver.py` are **string-blind and untouched**. *Verification gate: the random-organ-rename test must still pass, bit-identically, post-calibration.* Concretely: the data-blocked per-drug `hepatic_fu_correction` registry is replaced by ONE global acid-class fu constant in θ — the cleanest Invariant-6-legal attack on the acid arm. ✔
2. **All parameters Distribution.** θ become calibrated Distributions (SBI posterior). ✔
3. **Compile once.** Topology fixed; θ flow as parameters. ✔
4. **Flow conservation build-time.** Extended to the regionalized gut wall in the builder (new validation). ✔
5. **Holdout inviolable.** §4.0 InChIKey leak-clean + nested-CV-on-corpus + single locked-holdout touch. ✔
6. **No drug-specific branches.** Class-conditioning is the strongest allowed generalization; no `if drug==X`. ✔
7/8. **20-files / hard-no-touch.** No new repo until Gate 1 passes; `compiler.py`/`solver.py`/JAX-RHS/`DrugOnGraph` untouched (gradient-free → no adjoint, no clip-smoothing). ✔

---

## 6. Phased plan with go/no-go gates — kill-first, rebuild-last

**Everything through Gate 1 is a 2-script spike on the live pipeline. No new repo, no engine change, no segmented gut until Gate 3a.**

### Test A (Gate 0 — the program ceiling; run FIRST; one afternoon) — THE DECISIVE EXPERIMENT
**Bound the SMILES-reducible fraction of the engine F-residual.** This decides whether v2 is worth building at all. On the InChIKey-leak-cleaned measured-F subset (~**445 unique drugs after name-exclusion**; report the exact post-InChIKey + scaffold-neighbor count, do not assume), regress the engine F-residual (engine-F minus measured-F via the IV-reference solve) on **two feature sets** and report the **GAP**, out-of-bag / scaffold-CV with a bootstrap CI:
- set 1 = the engine's *existing* inputs {logP, TPSA, pKa, MW, predicted Peff/solubility} → R²_circular (DE-44/45 trap: descriptors re-explaining the engine's own inputs is *not* new signal; dose-number was logP in a costume, corr 0.912).
- set 2 = set 1 + candidate new axes (pKa-aware S(pH), formulation column) → R²_total.
- **SMILES-reducible-but-not-yet-used ceiling = R²_total − R²_circular.**

**Kill criterion (pre-registered):** if (R²_total − R²_circular) < 0.15 OOB, the SMILES-reducible F-fraction is too small to justify the rebuild → **STOP the F-dispersion program and fall back to Direction (1)** (engine-as-prior / measured-modality, below). Report the bound as the program's headline deliverable *regardless of outcome*. **Prior is bearish (DE-28 SMILES→F scaffold-CV R²=−0.09), so this gate is a genuine coin-flip-or-worse and must be honored.**

### Test B (Gate 1 — the calibration ceiling under W3; one script)
**Gradient-free calibration of the ~2–3 identifiable highest-leverage existing constants** (`_CLINT_SCALING`, the `2.88` absorption constant, gut-CYP3A — *after the §3.2 identifiability prune confirms how many are separable*) on the cleaned corpus, current (non-segmented) engine, perturbed via a **config object — NOT by editing `ivive.py`**, with Stage-1 recalibrated blend weights co-moved. Optuna/CMA-ES, ~200 evals.

**Kill criterion (pre-registered, HOLDOUT-meta not corpus-engine):** Go **iff nested-CV corpus *meta* AAFE (through the recalibrated blend) improves by > 0.15 absolute (> 2× the stack-drift floor)** AND a corpus→holdout-style **transfer slice** (a held-back corpus slice matched to holdout label-provenance: single-study, FDA-label-style) also improves. A corpus-*engine* AAFE drop is NOT sufficient (curve-fitting; the meta may eat it on the holdout — the FLUX-1/DE-43 pattern).
**Named co-calibration sub-gate:** report the nested-CV corpus meta improvement under **co-moved weights** vs **frozen weights**, AND whether the delta is achieved **without θ exceeding 1 prior-σ**. If the only gains come from θ drifting past the guard, co-calibration is curve-fitting the blend (DE-26 in new clothing) → label it as such and fall back to correctness-only.

### Gate 0.5 (transfer probe — cheap, run alongside Test B — CAN KILL THE WHOLE PROGRAM)
**The engine signed bias FLIPS between surfaces:** holdout engine mean signed log = −0.419 (0.38× UNDER) vs corpus OOF over-prediction (neutral ~1.6×, base ~1.3×); acids holdout ~0.23× (4× under) vs corpus ~0.79–0.93× (near-unbiased). A constant tuned to remove corpus over-prediction will push the holdout *further* under — **the synthesis red-team judges this regression near-certain, not a coin-flip.** Fit the simplest global Cmax-bias scalar on a **single-study/label-provenance-matched** corpus subset, apply to the locked holdout once, **require holdout AAFE non-regression AND signed-bias reduction.**
**Pre-registered program-level consequence:** if the sign-flip survives provenance-matching (the bias scalar regresses the holdout), the **corpus-calibration spine is broken for headline transfer** → the program falls back to **correctness-only work BEFORE Gate 1 spends further effort** (the corpus advantage over N=107 is then illusory for the headline). State plainly: **Gate 0.5 can kill the whole spec, not merely tune the CV scheme.**

### Phase 2 (the gate, EXPECTED-NULL diagnostic — only if Test A, Test B, Gate 0.5 pass)
Learned conditional gate on the cleaned OOF corpus, engine substrate unchanged. **Its job is to quantify the optimal damping given the CURRENT engine — the baseline a post-dissolution engine must beat — NOT to win.** Pre-registered: **ship only if nested-CV corpus meta beats `meta_oof` (3.284) by closing > 30% of the gap to the best-pick oracle (2.454), i.e. reaches ≤ 3.03, AND no candidate gate feature exceeds |Spearman ρ| > 0.3 reliance on a log-obs-correlated feature.** The OOF abs-error correlation eng,ml = 0.600 (signed 0.781) and only 35% of corpus rows are gate-decidable, so the gate has limited material to arbitrate; if no feature reaches |ρ|>0.3 with per-drug meta error, the gate reduces to fixed weights (DE-26) → demote to "optional probe," keep Stage-1 recalibrated weights.

### Gate 3a + Phase 3 (dissolution / regional gut-wall — the within-class bet, last and gated)
Build the dissolution flux first, then regionalize the gut wall. **Pre-registered kill (the un-run DE-45 guard):** on a corpus dev fold, the dissolution model must (a) cut **within-class** engine AAFE *dispersion* (per-class-median-debiased AAFE) by ≥ 15%, AND (b) the dose-number/S(pH) term — **not logP** — must carry that reduction (partial correlation controlling for logP and CLint), using **pKa-aware S(pH), not the logP-only `−logP+0.5`.** If dispersion does not move or is logP-collinear, the "dissolution disperses F" thesis is falsified cheaply and Phase 3 stops — the gut work buys *correctness*, not holdout AAFE (the FLUX-1 prior).
**Direction (red-team-corrected):** dissolution capping LOWERS Cmax for low-solubility/high-dose **over-predictors** (lenacapavir-class) and regional gut-CYP3A raises extraction on over-predicted bases — both *correctly directed* for the 18/18 OVER headline tail. Gate-3a ship criterion: on the 107-holdout, co-calibrated through the recalibrated blend, must move Meta AAFE below 2.70 AND **reduce** the >3× OVER tail (currently 18) without growing the under-tail. This is now the *only* arm with a plausible headline mechanism — but it is unproven and pre-gated.

### Phase 4 (lock)
SBI θ-posterior; conformal recalibration on the frozen predictor; **single** holdout evaluation (pre-registered AAFE + CI, must exclude 2.78); external/prospective re-score.

### Fallback — Direction (1): engine-as-prior (if Test A or Gate 0.5 kill fires)
If Test A's kill fires (SMILES-reducible F < 0.15) OR Gate 0.5 shows transfer-failure, the SMILES-only headline program is capped. The honest fallback is **engine-as-prior + a measured F-orthogonal data modality (Leap D)** — measured solubility/permeability/transporter kinetics or measured F, which is how the field reaches F-AAFE 1.75 (OrBiTo expert input). **Be explicit:** this needs a NEW amortizer over molecular features (reuse the `sbi/` training scaffold + PRIOR_LOW infra) — the per-patient `regimen/tdm_sbi.py`/`tdm_enkf.py`/`tdm_ibis.py` machinery conditions on *measured concentrations for one patient* and does NOT support corpus θ-calibration; it shares a library, not a problem. This fallback is a data-acquisition program, not an algorithm change, and must be gated against the DE-23 (Δ-model, holdout 3.5–8.5 overfit) and DE-30 (UDE, CV R²<0) precedents.

---

## 7. Risk register

| Risk | Status / Mitigation |
|---|---|
| **Corpus holdout-leak (≥23 drugs by name, more by InChIKey)** | **HARD BLOCKER.** §4.0 InChIKey rebuild + CI assertion + scaffold-neighbor exclusion + OOF-surface audit BEFORE Gate 1. |
| **Corpus↔holdout bias SIGN-FLIP (near-certain transfer failure)** | **Can kill the program.** Gate 0.5 provenance-matched transfer probe with a pre-registered program-level fallback (correctness-only), not just a CV-scheme change. |
| **F-arm true N is ~445 (not 1,020); subclasses tiny (high-PPB acid n=6–8)** | Report per-subclass calibrated-constant CIs; if a class constant's CI spans the no-op (1.0) at n<10, fold it into the global constant (do NOT ship the acid arm as class-conditioned — DE-37 data-limited). |
| **θ non-identifiability (CLint-scaling ⟷ gut-CYP3A ⟷ Qgut; fu,inc ⟷ CLint)** | Mandatory Jacobian/Fisher prune (§3.2) reporting the **rank** before committing to a count; drop fu,inc target (no fup column); cap actively-calibrated θ at the identifiable rank (~3–6, possibly 2–3). |
| **Class-conditioning headroom ≤0.05 AAFE (sub-threshold)** | Acknowledged; class knobs = between-class trim below the resolution floor; the headline bet rests on Gate-3a dissolution + the meta replacement, not class constants. |
| **Co-calibration benefit asserted, not bounded; prior-σ guard forbids its region** | Test B's named co-calibration sub-gate adjudicates whether any benefit exists within the trust region; if not, label as DE-26 and drop. |
| **Gate inherits co-calibration (OOF signed corr 0.781–0.982)** | Gate is precondition-gated (engine corpus-AAFE < ML first) and ships only on the Phase-2 nested-CV + feature-reliance test; else Stage-1 recalibrated weights. |
| **Differentiability brittle/unnecessary** | DELETED from critical path; gradient-free primary; adjoint reserved as optional Phase-4 accelerator only. |
| **Missing Cmax-relevant mechanisms** | Logged as candidate structure, not silently assumed irreducible: enterohepatic recirculation (`cl_int_bile` is a sink-only at flux.py:325 — no bile→duodenum return edge; a clean identity-blind edge, Cmax/AUC lever); food/FaSSIF solubilization (0 code hits — the reason F has a ~0.3–0.4 AAFE formulation/food floor); the corpus `formulation` column (tablet/capsule/solution/ER) as a measured handle to BOUND idiosyncratic-F variance in Test A rather than assume it. |
| **Scope creep back to a rebuild** | No new repo until Gate 1 passes; Gates 0/1 are 2 scripts on the live pipeline. |

---

## 8. Success criteria

- **Primary (evidence-supported):** holdout Meta AAFE **enters the OrBiTo band at its top, ≈2.70–2.74**, with the bootstrap CI not worsening vs 2.78. This is what the verified between-class trim + global-median null + the correctly-directed dissolution/gut-CYP3A OVER-tail moves can *plausibly* reach.
- **Stretch (contingent):** 2.55–2.65 (≈5–8% relative) — **only if** the unproven dissolution within-class lever clears Gate 3a AND Gate 0.5 transfer holds. Do not treat as a baseline expectation.
- **Honest floor note:** the label-noise floor (AAFE 1.18, N=14) means a *perfect* model still scores ~2.56–2.75 on this holdout, so anything tighter than ~2.5 on N=107 is inside the current CI [2.41, 3.25] and unprovable. **The realistic ceiling of this SMILES-only program is OrBiTo-band entry, not a leap.**
- **Secondary:** %within-2-fold above 50% (now 43.9%); within-class log-fold dispersion (σ_within per class) compressed; the leak fix and EHC/dissolution shipped as correctness regardless of headline.
- **Reference (NON-binding):** Omega's 2.215 is a caveated historical figure (likely contamination-era, different holdout), recorded only to motivate the technique gap — NEVER a go/no-go target. In-domain 1.8–2.0 is a stretch contingent on a measured modality (Leap D), not achievable SMILES-only.
- **Discipline:** all gains in public-clone state (not dev artifacts; [[correctness-over-benchmark]]), holdout touched once, every θ physically interpretable with a reported CI.

---

## Appendix — why each move is NOT a re-run of a dead-end, with a kill criterion

**A. Gradient-free corpus calibration of θ (the spine).**
- *Closest dead-end:* none in `dead-ends.md` — Sisyphus has NO corpus/global calibration (verified: only `calibrate_conformal` + single-transporter-abundance scripts exist). Distinct from DE-05/17/19/29/35 (E2E neural / UDE): θ is ~3–6 *interpretable physical constants*, not a structure→residual net (cannot have negative R² by construction); distinct from Omega only in being gradient-free + co-calibrated with the blend.
- *Kill:* Test B — nested-CV corpus *meta* improvement < 0.15 absolute, OR Gate 0.5 transfer regresses the holdout → fall back to calibration-only / correctness work. *Residual risk:* the W3-damping (DE-43) and the sign-flip (Gate 0.5) are the real killers and are orthogonal to the calibration-vs-neural distinction.

**B. Class-conditioned hepatic-extraction / fu constants (between-class trim).**
- *Closest dead-end:* DE-25/27 (class-aware meta weights, 1765-cell sweep, TIE 2.277) and DE-42 (flat absorption scalar). Distinct: this calibrates a *physical* extraction constant in `ivive.py` on a *corpus*, not a meta weight on N=107; but the value is bounded to ≤0.05 AAFE (sub-threshold, verified).
- *Kill:* after fitting, measure σ_within(log fold) per class on a leak-clean fold. If class-conditioning only shifts class means (not within-class dispersion), it is DE-42 at the class level → fold into the global constant. The acid arm is n=8 holdout / n=6–8 corpus and DE-37 data-blocked; ship as class-conditioned only if its CI excludes 1.0 at n≥10.

**C. Recalibrated blend weights as θ (Stage 1).**
- *Closest dead-end:* DE-23/24/25/26 (post-hoc meta variants, all r>0.986). Distinct: weights are corpus-calibrated *jointly with the engine constants* (co-calibration), not a post-hoc restack of fixed tracks. But the best GLOBAL fixed eng/ml blend is already AAFE 2.94 holdout / 3.26 corpus — fixed reweighting is near-exhausted (W3).
- *Kill:* Test B co-calibration sub-gate — if co-moved weights do not beat frozen weights within the prior-σ trust region, Stage 1 is DE-26 in new clothing → the engine-F-fix (Gate 3a) is the only remaining lever.

**D. Learned confidence-conditional gate (Stage 2, downstream).**
- *Closest dead-end:* DE-23/24/25/41/44/47/48. Distinct: corpus-scale + features conditioned on AD/class/disagreement, and *only built after the engine F-fix*. NOT distinct on the core objection: the OOF surface has signed residual corr eng,ml=0.781 / ml,meta=0.982 — co-calibration is a shared-error-mode artifact, not an N=107 artifact, so corpus scale defeats the *overfit* objection but NOT the *no-orthogonal-conditioning-signal* objection (DE-41).
- *Kill:* Phase-2 — must close >30% of the meta_oof→oracle gap on nested CV AND no feature with |Spearman ρ|>0.3 reliance on a log-obs-correlated signal AND engine corpus-AAFE already < ML. Else demote to optional probe.

**E. Dissolution + regional gut-wall (within-class fa/Fg dispersion, the OVER-tail lever).**
- *Closest dead-end:* DE-42 (flat ka scalar), DE-45 (dose-number = logP, corr 0.912). Distinct: introduces the *only* dose/solubility nonlinearity in the ODE (a flat scalar cannot disperse; a solubility-capped dissolution flux can) and uses pKa-aware S(pH) (the un-run DE-45 variant). The 8-segment lumen is NOT being rebuilt — only the wall regionalization + dissolution flux are new. **Correctly directed: it LOWERS Cmax for low-sol over-predictors (not raises fa for unders).**
- *Kill:* Gate 3a — within-class dispersion must drop ≥15% on a corpus dev fold AND the S(pH)/dose-number term (not logP, partial-correlation-controlled) must carry it. Plus the holdout-direction guard: must *reduce* the 18/18 OVER tail. Framed as the only plausible headline arm, but unproven.

**F. Differentiable forward map (DELETED).**
- *Closest dead-end:* none — it is *removed* because the forward map is non-differentiable (3× `np.clip`, 5-bin radius lookup, dict-keyed FM) and would force touching `solver.py` (Invariant 8). Reserved only as an optional Phase-4 accelerator. No kill criterion needed — it is not on the critical path.

**G. Direction (1) engine-as-prior + measured modality (fallback).**
- *Closest dead-end:* DE-23 (Δ-model, holdout 3.5–8.5), DE-30 (UDE, CV R²<0), DE-48 (measured-regime routing degrades, engine-measured N=93 ≈3.84 > meta 2.78). Distinct: a measured *F-orthogonal* input channel (Leap D) is the field-accepted escape (OrBiTo F-AAFE 1.75), not a reweight of the current four tracks. Honest cost: a data-acquisition program; the `regimen/` SBI machinery is per-patient and does NOT transfer — a new molecular-feature amortizer is required.
- *Kill / trigger:* this is the *fallback*, triggered by Test A's kill (SMILES-reducible F < 0.15) or Gate 0.5 transfer-failure. Gate the build against DE-23/30 before relying on it.
