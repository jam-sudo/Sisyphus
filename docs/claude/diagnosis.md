---
last_updated: 2026-06-01
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

12 holdout drugs with measured fup + CLint, engine-only (no meta). Clean set (N=10, excluding montelukast/abiraterone extreme outliers):
- AAFE 2.329 → **1.980** (measured ADME), median FE 2.19 → **1.88**, 8/10 improved.
- fup-matched subgroup (N=8): 1.91 → 1.79 (CLint-only effect, 6% gain).
- fup-corrected subgroup (N=2): 5.15 → 2.96 (fup+CLint, 42% gain).

**Conclusion:** engine architecture is sound. Minor systematic bias exists but is not dominant. Input quality is the bottleneck. Error cancellation happens in some drugs (abiraterone fup 0.085→0.01 worsened FE 20.8→39.1) but is not the dominant pattern — 80% of drugs benefit from measured data.

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

**No predict-time AD signal recovers it** (see dead-ends.md DE-41): low predicted-F and engine↔ML divergence both correlate ≈0 with error on the holdout (the engine predicts low F for nearly everything — median 0.18 — co-calibrated, so it is not discriminative). The per-drug F error is structural, consistent with the ~30% PI coverage. A real lever would be **measured-F routing** or an **absorption-model recalibration** (untested), not an AD flag.

---

## See also

- [dead-ends.md](./dead-ends.md) — the 41 enumerated failed attempts in one place.
- [experiment-log.md](./experiment-log.md) — chronological record of experiments, successes and failures.
- `docs/breakthrough_path.md` — UDE roadmap (Phase 1 falsified; Phase 2 / 3 pending).
- `docs/holdout_contamination_audit.md` — the 2026-04-04 leakage discovery and fix (AAFE 2.283 → invalidated).
