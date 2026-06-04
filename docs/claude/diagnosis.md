---
last_updated: 2026-06-03
parent: ../../CLAUDE.md
charter: Why Sisyphus's Cmax accuracy ceiling exists, what broke it, and which paths remain open
---

# Accuracy Ceiling Diagnosis

**Short version:** Meta AAFE 2.698 (4-track; was 2.695 at 2026-04-20 — the 2.695→2.698 drift is numerics-stack + metric-neutral UGT/prodrug cycles per [CLAUDE.md](../../CLAUDE.md), the qualitative ceiling model is unchanged) is the ceiling under the current architecture. The ceiling is a combined **CLint target-noise floor** (R²≈0.24 is intrinsic, not engineering-limited) + **pipeline error-cancellation** (the 4 tracks are co-calibrated on a specific error profile; partial replacements destroy the balance). VDss analytical 4th track proved that **orthogonal** tracks can still be added, so the ceiling is not absolute — but "orthogonal" now has an empirical test (error decorrelation with existing tracks) that must precede any track proposal.

Before proposing any accuracy improvement, read [dead-ends.md](./dead-ends.md) first — 41 enumerated attempts are documented. Most new proposals that "haven't been tried" are variants of something already reverted.

---

## 1. The CLint R²=0.24 floor

- **XGBoost v1** on TDC Hepatocyte_AZ (1,213 compounds): R²=0.24.
- **v2** augmented to ~3,700 compounds: marginal improvement — target noise dominates.
- **ChEMBL expansion** (539 unique new compounds, 2026-03-27): scaffold CV R² 0.279→0.333 (+0.054). Engine AAFE +0.099, Meta AAFE +0.038 — **homogeneous data expansion destroys error cancellation**.
- **Foundation model shootout** (MoLFormer, ChemBERTa, Uni-Mol, frozen embedding + Ridge/MLP/XGB, 2026): Morgan FP + XGB (R²=0.205) dominates every alternative. CLint R²≈0.20 is a **target-noise ceiling, not a representation ceiling**.
- **BDE features** (ALFABET, 978 compounds): r=+0.033 vs log10(CLint) — zero correlation. Hepatocyte CLint integrates kcat + Km + enzyme complement; C-H BDE captures only the kcat component.

Consequence: measured CLint would raise the ceiling, predicted CLint cannot (at current data scale / target noise).

## 2. Error cancellation

Cmax = f(fup, CLint, Peff, Kp, ...). Each ADME predictor has its own error, and the production pipeline is calibrated on the **joint** error profile inherited from Omega. Partial ADME replacements that improve one component's R² in isolation **destroy the joint calibration** and worsen pipeline AAFE.

Evidence:
- **ALL-ON** experiment (pKa + Berezhkovskiy + expanded CLint simultaneously): engine AAFE 2.945→3.016 (+0.072), meta AAFE 2.058→2.135 (+0.077). The individual harms sum.
- **Full predict replacement** (2026-03-30): CLint +0.033, fup +0.042, VDss +0.057 in R². Engine AAFE +0.165, Meta +0.023 worse.
- **Post-hoc meta-learner tournament**: 33 methods tested, all have error correlation r > 0.986 with the baseline. Mathematically, Engine + ML post-hoc combinations cannot break 2.277 (pre-VDss baseline).
- **ADME fup override** (2026-04-11): DrugBank measured fup prioritized over XGBoost. Principled, empirically harmful: engine AAFE +0.306. **35th error-cancellation failure.** Reverted.

## 3. Measured ADME PoC (Pattern C)

12 holdout drugs with measured fup + CLint, engine-only (no meta). Clean set (N=10, excluding montelukast/abiraterone extreme outliers).

**Reconciled 2026-06-02.** The earlier "AAFE 2.329 → **1.980**" figures are a *stale engine state*. `scripts/measured_adme_poc.py` is byte-unchanged, but the engine evolved underneath it (the 2026-05-01 `realize_means` hardening; clopidogrel prodrug routing, 2026-05-20 B-03; the OATP/non_cyp registries). Re-running the unchanged PoC **today** gives clean-10 **2.81 → 2.69**, not 1.98. Current reproducible numbers:
- **Production `predict(measured_adme=fup+clint)`, engine-only** (`scripts/run_measured_adme_benchmark.py`): clean-10 SMILES **2.63 → measured 2.33** (~11% gain). This is the honest current floor; it *beats* the leaner PoC path (2.69) because production prodrug-routes clopidogrel (FE 10.25 → 3.35; that one drug moves the PoC clean-10 2.69 → 2.40) and applies the registries.
- The legacy fup-matched (1.91→1.79) / fup-corrected (5.15→2.96) subgroup splits were tied to the stale state and are not re-derived.

**Conclusion (refined 2026-06-02):** measured ADME helps *modestly* (~11%), not dramatically — the engine carries significant **structural** error that correct fup/CLint do not remove. Critically, the engine-only measured path is **not** error-cancellation-free: alprazolam's FE *worsens* 2.67 → 6.04 when given the correct measured fup (0.20 vs predicted 0.028) — the wrong predicted fup was *compensating* for a separate engine error (cf. abiraterone 20.8 → 39.1 under measured fup). So "input quality is the bottleneck" is only partly true: engine structural error and internal compensation are real, and the measured-input path is a **probe of engine structure**, not a guaranteed clean win.

## 4. The VDss exception — when a new track *is* OK

2026-04-10: VDss analytical 4th track dropped Meta AAFE from 2.808 to **2.695 (−4.0%)**. The original "partial replacement is impossible" conclusion is **falsified in the limit**.

**Why VDss worked where CL/F · t½ failed:**
- CL · t½ · Cmax all depend on hepatic clearance / CYP-dominant kinetics → errors correlate across tracks.
- VDss depends on tissue partitioning (lipophilicity + tissue binding) → **clearance-orthogonal** error component.
- dose / (Vd · BW) 1-compartment analytical Cmax at 20% weight scales down the 3 existing tracks to 0.80 and adds uncorrelated signal.

**Decorrelation criterion (for future track proposals):** measure per-drug error correlation (Pearson r on log Cmax residuals) between the candidate track and the existing 4 tracks. Only consider tracks with |r| < 0.5 against all 4 existing tracks. This gate precedes any integration work.

## 5. Direct CL/F · t½ — confirmed negative path

- **Direct CL/F 3rd track** (IVIVE bypass, 2026-03-27): CV R²=0.232, analytical 1-cpt Cmax. LOOCV w_clf = 0.00 (both base and other regions). Standalone AAFE 3.133 (ML 2.336 wins). Meta AAFE Δ = −0.005 (noise). Oracle 3-track 1.788 (28/107 drugs CL/F is best individually) — but no fixed weight unlocks it.
- **Post-VDss direct CL/F · t½ predictors** (6 variants): all negative, `data/validation/post_vdss_negative_results.json`. Falsifies "IVIVE bypass is what made VDss work" — the real reason is **decorrelation**, not bypass.

## 6. Remaining practical paths (ranked)

1. **TDM Bayesian update** (IBIS + EnKF + SBI dual/triple method, shipped) — individual-patient accuracy. CV −55% on single-obs, Cmax bias +52% → +2% on morphine with P6 SBI reweighting. Production-ready today.
2. **Additional orthogonal track exploration** — candidates: renal clearance analytical, formulation-aware dissolution, tissue-specific partitioning. **Gate**: precompute error decorrelation vs the 4 existing tracks. No track ships without passing the gate.
3. **Breakthrough path Phase 2 (amortized SBI / BayesFlow) for population-level calibration** — requires larger data scale; not executed.
4. **Engine refinement for OATP-class substrates** — ECM (Extended Clearance Model) closed-form hepatocyte QSSA, planned 2026-04-20: breaks the flow-limited MM saturation that blocks Phase 2A/2B validation. See `docs/superpowers/specs/2026-04-20-oatp-ecm-hepatic-clearance-design.md`.

## 7. Population-level AAFE ≤1.7 evaluation

Population-level AAFE 1.7 is **unreachable from SMILES alone** under the current CLint R²=0.24 ceiling. Per-patient accuracy is reachable today via TDM Bayesian update (CV −55% from a single observation). To improve the population ceiling we need either (a) measured CLint routing on a per-drug basis, or (b) a new in-vitro data source that changes the CLint prediction floor.

---

## 8. Novel-drug failure mode — bioavailability (F), not CLint (2026-06-01)

The 2026-06-01 prospective expansion (N=28; prospective Meta AAFE 3.21 > retrospective 2.698) exposed a failure mode the CLint-centric story above does **not** capture. The engine's worst prospective errors are catastrophic **under**-predictions of low-clearance, high-exposure 2025 NMEs (mirdametinib 30×, sevabertinib 18×). An IV-vs-oral decomposition localises the error:

- Engine **CL_systemic is ≈ correct** (mirdametinib 4.8 vs literature CL/F 4.6 L/h; the drug *is* low-clearance and the engine knows it).
- Engine **bioavailability F is catastrophically low** (mirdametinib F=0.08, sevabertinib F=0.05) vs implied real F ≈ 0.5–1.0. The entire 12–88× Cmax gap is in the absorption / first-pass (F) model, not clearance.
- On the prospective new-16, corr(engine_F, |log10 fold|) = −0.54 (lower predicted F → worse). CLint is **not** the differentiator (median CLint ≈ equal for under- vs over-predicted). The engine track (5.10) is much worse than ML (3.40) on the new drugs.

**Refinement to §1:** for *novel / out-of-distribution* drugs the binding constraint is the **F (fa·fg·fh) absorption model**, not the CLint R²=0.24 floor — CL_systemic was right where Cmax was 30× off. The CLint ceiling governs the *retrospective, in-distribution* set; the *prospective* gap is an absorption-model extrapolation problem.

**No predict-time AD signal recovers it** (see dead-ends.md DE-41): low predicted-F and engine↔ML divergence both correlate ≈0 with error on the holdout (the engine predicts low F for nearly everything — median 0.18 — co-calibrated, so it is not discriminative). The per-drug F error is structural, consistent with the ~30% PI coverage. A real lever would be **measured-F routing** or an **absorption-model recalibration**, not an AD flag.

**2026-06-03 update — the absorption-recalibration lever is now tested and foreclosed (dead-ends.md DE-42).** A measurement-only `F = fa·Fg·Fh` decomposition (10 measured-fup+CLint PoC drugs) localises the *median* under-call to **fa** (fa median bias 0.55; `ka = 2.88·Peff·ka_fraction/radius` ~6%/segment loses to gut transit ~3.85/h; non-CYP3A acids prove it — empty `metabolized_gut` sink yet suppressed F). But `ka` is **linear** in the ODE, so an absorption recalibration is a flat scalar: it nulls the median engine-F/lit-F (0.46→1.0; engine N=107 3.831→3.336) yet cannot reduce per-drug dispersion (all candidate refinements plateau at geomean fold-error ~1.43; the best on the full holdout scored 3.405, *worse* than the scalar) and flips the engine to a 30-drug `>3×`-over tail (meta-regression risk HIGH). The per-drug residual is **bidirectional first-pass**, not absorption: CYP3A first-pass *over*-extraction for bases (alprazolam/carbamazepine/quinine cap at F ≈ 0.5 vs lit 0.8–0.9 even at fa=1) ⊕ well-stirred-`fu` *under*-extraction for high-PPB acids (diclofenac/febuxostat/etodolac — the §3 / DE-37 / B-11 hepatic-fu lever). These oppose, so no single knob reduces the spread. **Measured-F routing remains the only un-foreclosed F lever; an absorption recalibration does not raise the ceiling.**

**2026-06-04 — FLUX-1: the "cap at F ≈ 0.5" symptom was a structural double-count bug; fixing it is correct physics but REGRESSES the headline.** The bidirectional-first-pass note above names the over-extraction mode precisely: "CYP3A first-pass over-extraction for bases ... **cap at F ≈ 0.5** vs lit 0.8–0.9 *even at fa=1*." That `F ≈ 0.5` ceiling is the literal fingerprint of a flow-limitation double-count: the clearance flux applied the whole-organ `CL_h` (which already embeds blood flow `Q`) to the perfusion-compartment outlet `c_out`, while a *separate* convective edge carried `Q·c_out` — so the realized extraction was `E = fup·CLint/(Q + 2·fup·CLint) → 0.5` instead of the canonical `→ 1.0` (extra factor of 2; triple-verified by topology, algebra, and an empirical engine probe E≈0.495 on both `well_stirred` and `extended` paths). **The fix** (apply the *intrinsic* clearance `fup·CLint·c_out` in all four clearance models; the convective edge supplies flow limitation) lifts the ceiling and is mathematically correct. **But it makes the headline WORSE** (canonical public-clone, macOS estimate: Meta 2.762→2.784; an earlier 2.698→2.625 "improvement" was a developer-state artifact). The reason is the textbook lesson of this whole document: the wrong formula was *load-bearing as calibration* — fixing it removed a compensating error, so 22 holdout drugs worsened vs 17 improved (high-first-pass actives like selegiline 20.2×→9.4× correct, but well-predicted drugs like carbinoxamine 0.86→0.38 break). **This refines — does not overturn — DE-43:** the engine first-pass arm was *wrong* (a fixable structural bug, not the irreducible floor DE-41/42 implied), but the fixed-weight meta still damped the engine's large move (±15-17%) to a small headline move (±2-3%), so the engine remains not a *headline* lever — now confirmed by a regression rather than a non-improvement. **Correctness vs accuracy are different axes:** "mathematically correct" = matches the well-stirred equation the engine claims to implement; it does not imply better Cmax. Per the user's call, correct physics ships anyway (cache/goldens pending canonical-env regen). See experiment-log.md 2026-06-04 and spec `2026-06-03-flux1-extraction-double-count-design.md`. FLUX-1 is a *structural* fix, distinct from the *recalibration* lever DE-42 forecloses for the absorption (`fa`) arm.

**2026-06-03 — the prospective set is also co-calibration-locked; the meta damps the engine to ~18% everywhere (dead-ends.md DE-43).** The natural follow-up — "the *prospective* N=28 set (3.21) isn't in the meta co-calibration, so a first-pass lever foreclosed retrospectively should help it" — was tested and **falsified**. A `F = fa·Fg·Fh` decomposition shows the prospective catastrophes (mirdametinib, sevabertinib, kinase inhibitors) are **fa-first** (absorption starved: low Peff or low-solubility → `particle_radius=50µm` → `ka ≪` gut transit) then **gut-CYP3A `Fg`-second** — *not* the fa-saturated pure-CYP3A mode. Two levers (absorption scalar 5.25×; gut-CYP3A 0.5×) each improve the **engine** track on prospective (4.11→3.75 / 4.11→4.00; mirdametinib engine fold 58→13) — but the fixed-weight **meta-learner damps the change to ~18–19% pass-through, identically on prospective and retrospective**, so the production meta nets neutral-to-negative (absorption scalar: prospective −0.069 *vs* retrospective +0.082 = **net −0.012**; gut-CYP3A: +0.020, inside the N=28 CI). **The meta is robust to engine errors by construction, which symmetrically blocks engine improvements — on every benchmark.** This is the unifying mechanism behind the error-cancellation dead-ends (§2): the engine is structurally not a headline lever, retrospective *or* prospective. The only remaining levers are per-drug **measured-F routing** (a capability, not a global move) or a **meta-architecture change** (AD-gated engine weighting — itself likely foreclosed per the meta-learner dead-ends DE-23/24/25 and the non-generalizing AD signal DE-41, and N=28-underpowered to validate).

---

## See also

- [dead-ends.md](./dead-ends.md) — the 41 enumerated failed attempts in one place.
- [experiment-log.md](./experiment-log.md) — chronological record of experiments, successes and failures.
- `docs/breakthrough_path.md` — UDE roadmap (Phase 1 falsified; Phase 2 / 3 pending).
- `docs/holdout_contamination_audit.md` — the 2026-04-04 leakage discovery and fix (AAFE 2.283 → invalidated).
