---
last_updated: 2026-04-21
parent: ../../CLAUDE.md
charter: Chronological log of Sisyphus experiments (successes, negatives, infrastructure). Latest first.
---

# Experiment Log

Reverse-chronological. Top-level [CLAUDE.md](../../CLAUDE.md) carries only the **current** headline numbers; this file is the history. For the authoritative failed-experiment list (with do-not-retry gating), see [dead-ends.md](./dead-ends.md). For the why-accuracy-is-bounded analysis, see [diagnosis.md](./diagnosis.md).

---

## 2026-04 (current session)

### OATP ECM hepatic clearance (2026-04-20, design + plan written)
- **Status:** spec committed, plan committed, implementation pending.
- **Problem:** OATP Phase 2A/2B blocked by flow-limited MM saturation at abundance 1e11 — abundance sweep shows Cmax invariant across [1e9, 1e11], phenotype scaling has zero directional effect, 4/5 statins stall on LSODA and Kvaerno5.
- **Design:** ECM (Shitara 2006, Watanabe 2009, Varma 2014) via QSSA on the hepatocyte; closed-form CL_h with active + passive uptake, passive efflux, metabolism, biliary clearance. Non-OATP drugs get defaults `PS_passive = PS_eff = 1e6 L/h, CL_int_bile = 0` that reduce ECM to well-stirred algebraically (<0.1% deviation).
- **Spec:** `docs/superpowers/specs/2026-04-20-oatp-ecm-hepatic-clearance-design.md`.
- **Plan:** `docs/superpowers/plans/2026-04-20-oatp-ecm-hepatic-clearance.md` (12 TDD tasks).
- **Diagnostics:** `data/validation/oatp_phase2a_stiff_diagnosis.json`, `oatp_abundance_sweep.json`.

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
