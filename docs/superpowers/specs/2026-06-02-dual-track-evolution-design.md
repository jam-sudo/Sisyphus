---
title: Dual-Track Evolution — Measured-Input Engine Path + SMILES Frontier Push
date: 2026-06-02
status: draft v2 (revised after 2026-06-02 Opus adversarial spec-review; awaiting user re-review)
parent: ../../../CLAUDE.md
charter: The next evolution of Sisyphus along two tracks — a measured-input engine path (additive, zero-regression) and a renewed push on the SMILES-only frontier — grounded in the DE-41 bioavailability-F diagnosis and a *corrected* reading of the 2026-06-02 decorrelation-gate experiment.
---

# Dual-Track Evolution

> **Revision note (v2).** A three-verifier Opus review of v1 found 3 blockers + 4 majors, all code-verified. v1's "DE-42 geometric ceiling" overstated the evidence (the gate's `vdss`/`clf` columns were reconstructed, not production tracks — the renal `r=0.999` was a `dose/Vdss`-vs-`dose/Vdss` tautology); v1's `kp_method`/SP2 Kp-injection plan was unreachable (the engine never reads `kp_method`, and no `kp_overrides` channel reaches `predict()`); and v1's "clean engine-only signal" framing was false for the default `predict().pk` (the CLF and VDss tracks read `adme.peff`/`adme.vdss`). This v2 corrects all of it. Verifier evidence is cited inline as `file:line`.

## 0. Thesis

The SMILES-only retrospective headline (**Meta AAFE 2.698**, N=107) is **likely near its ceiling, but that ceiling is not yet proven**. What *is* verified: every permeability/absorption-shaped SMILES proxy correlates with the **real engine track** (absorption-peff `r=0.807`, ion-Fabs `r=0.587` vs the genuine stored engine residuals, `orthogonal_track_gate.py:137-138`) — i.e. the lipophilicity axis is genuinely occupied by the engine. What is **not** yet established: redundancy with the **VDss** axis (the gate compared two self-authored `dose/Vdss` formulas — a tautology) and with the **CLF** axis (live-regenerated, w=0.00). So the geometric-ceiling argument holds for the *engine/lipophilicity* axis only; the full claim requires a corrected gate (§6.0).

The frontier is the **engine**, attacked through two complementary regimes:

- **Track 2 — measured-input engine path.** An *additive, bit-identical-when-unused* route that feeds measured ADME into the engine. Consumed via `result.engine_pk` / an `--engine-only` benchmark, it is a genuine engine-only signal with no meta-learner to re-absorb a bias-correction. Documented headroom: Pattern C engine-only **1.980** vs SMILES engine **3.831** (`scripts/measured_adme_poc.py:74-133`).
- **Track 1 — SMILES frontier push.** First *fix the decorrelation gate* (§6.0) so the ceiling claim is real; then (1A) recalibrate the engine absorption/F model — the actual prospective failure (DE-41) — hard-gated on 2.698 non-regression; and (1B) hunt the one plausibly-orthogonal new signal the corrected geometry leaves open: **gut transporter recognition**.

**Unifying point:** the measured-`peff` override (Track 2, SP2) and the absorption/F recalibration (Track 1, 1A) attack the *same* engine `ka` mechanism (`rhs_jax.py:377`) — Track 2 is the error-cancellation-free test-bed where it is *proven* before any risky reintroduction to the SMILES meta.

## 1. Current state (verified 2026-06-02)

- Headline **2.698** / N=107 (`4track_holdout_predictions.json`: `overall.meta.aafe=2.69825`, `in_domain.n=79`). Untouched.
- The holdout artifact stores per-drug **only** `eng/ml/meta` (+folds/flags) — **no per-drug `vdss` or `clf`** (`orthogonal_track_gate.py:126-128,169-172` self-document the reconstruction). It is named "4track" but holds 3 real per-drug tracks.
- **DE-42 (corrected reading):** absorption-peff and ion-Fabs genuinely fail vs the *real engine* track; the renal-`vdss` `r=0.999` is a **tautology** (proxy vs proxy), and `clf` was regenerated against a w=0.00 track. **Not** yet a verified 4-track ceiling. Artifact `scripts/orthogonal_track_gate.py` is untracked and unreviewed.
- **DE-41:** prospective N=28 = 3.21; failure = bioavailability-F under-prediction (engine CL_systemic correct); no predict-time AD signal recovers it.
- **Pattern C:** engine-only AAFE 1.980 (N=10, measured fup+CLint). Floor is engine *structural* error, not the CLint R²=0.24 noise floor.
- **Plumbing reality (verified):** `predict()` always calls `predict_adme(profile)` (predict.py:180) then `build_drug_on_graph(...)` (predict.py:213, and a 2nd time at 293-300 on the phenotype back-solve path). No measured-input or `kp_overrides` parameter exists anywhere; `MeasuredADMEInput` does not exist.
- **`kp_method` reality (verified):** the engine **never reads** `drug.kp_method` (grep empty in `engine/`); Kp enters solely via `drug.kp_overrides` → `compiler._build_kp_map` (compiler.py:182-200) → `rhs_jax` `node_kp`. `_KP_FUNCTIONS` (ivive.py:485-488) has only `rodgers_rowland`/`poulin_theil`; **`"provided"` silently aliases to Rodgers-Rowland and computes Kp**. `build_drug_on_graph` has **no `kp_overrides` parameter** (ivive.py:598-608). The only real provided-Kp path is the YAML loader `compounds.py:56-87`, which `predict()` does not use. **⇒ The line-738 `kp_method` assignment is dead metadata, not a PK bug, and fixing it enables nothing on its own.**
- **Track perturbation reality (verified):** ML/XGBoost Cmax track is SMILES-only (predict.py:376, models.py:44-63) — adme-inert. CLF track reads `adme.peff.mean` (predict.py:395→clf_predictor.py:125-129) and `engine_tmax` (predict.py:393). VDss track reads `adme.vdss.mean` (predict.py:413, fixed 20% weight, ensemble.py:40). The `fup/clint` args into `meta.combine` are **dead** (ensemble.py body uses only `compound_type`, :117). The engine's internal Vss is **emergent from `kp_overrides` × tissue volumes** — no scalar Vss input; `adme.vdss` feeds only the analytical VDss track.

## 2. Invariants honored

1. **Engine identity-blind** — no new string matching in `engine/`.
2. **Holdout inviolable for training/tuning** — the measured benchmark and the corrected gate are *evaluation* (split-stratified, InChIKey-audited); no parameter is fit against holdout Cmax. The gate's oracle-blend test (§6.0) reads holdout residuals diagnostically, exactly as the VDss track was validated.
3. **SMILES-only path bit-identical when `measured_adme=None`** — guarded by an exact-float test **and** a 107/107 holdout rerun. The override must mutate the `adme` name so the **second** build at predict.py:293-300 also sees substituted values (or stays untouched when None).
4. **No drug-specific branches.**
5. **Hard no-touch respected** — we *add* paths; we do not modify `engine/compiler.py`, `engine/solver.py`, or any existing `DrugOnGraph` field.
6. **All inputs are `Distribution`** — measured values wrapped `Distribution(mean, cv)`.

## 3. Decomposition & sequencing

```
SP1 (measured-input path)  →  SP2 (measured-peff absorption test on engine path)  →  SP3 (Track-1 frontier)
                                                                                      6.0 fix the gate → 1A absorption recalib → 1B transporter signal
[deferred] SP4: kp_overrides injection channel → Vss/Kp (DE-33) correction
```

Track 2 first: it is the engine-only test-bed. The Vss/Kp (DE-33) work moves to a **deferred SP4** because it needs a `kp_overrides` channel that does not exist (B2/B3); SP2 instead uses the **already-wired** measured-`peff` lever.

## 4. SP1 — Additive measured-input path  *(Track 2 foundation)*

**Contract.** New frozen dataclass `MeasuredADMEInput` in `predict/adme.py` (after `ADMEProperties`, ~line 59): value fields `fup, clint, peff, vdss, rbp, solubility` (`float | None = None`) + paired `*_cv` (defaults `fup_cv=0.15, clint_cv=0.20, peff_cv=0.25, vdss_cv=0.20, rbp_cv=0.15, solubility_cv=0.30`). `__post_init__` enforces: **atomic `fup`+`clint` pair** (see rationale below), **CV floor 0.10**, all values `> 0`.

**Atomic-pair rationale (corrected — engine-IVIVE grounds, not DE-31 meta-cancellation).** In the engine's well-stirred hepatic extraction, `fup` and `clint` *co-determine* CL_int (CL_h = f(fup·CLint·…)); supplying a measured `clint` against a *predicted* `fup` (or vice-versa) mixes a measured and a model term inside one clearance expression and distorts the engine's extraction. The pairing protects the **engine** IVIVE coupling. (DE-31 — automatic DrugBank-fup substitution into the *meta* — is a different, also-real failure but is not the justification here.)

**Wiring.** One keyword-only param on `predict()` (after `kp_method`): `measured_adme: MeasuredADMEInput | None = None`. A single `if measured_adme is not None:` block between predict.py:180 and :213 that **mutates the `adme` name** via `dataclasses.replace(adme, **overrides)` so both downstream builds (213 and 293-300) see it. `predict_adme()` always runs (ML/VDss never see `None`). Append a `measured_adme:overrides=[...]` warning tag.

**Track-perturbation honesty (corrected).** The override is **not** engine-confined on the default `result.pk` (meta): `peff` perturbs the **CLF** track, `vdss` perturbs the **VDss** track. The clean engine-only signal is `result.engine_pk` / the `--engine-only` benchmark — **that is the reported measured-input surface** (§9). The ML/XGBoost track is adme-inert; the `fup/clint` meta args are dead. So `fup`+`clint` overrides move only the engine track (plus a small indirect CLF coupling via engine-Tmax); `peff`/`vdss`/`rbp`/`solubility` additionally move CLF/VDss on the meta surface and are therefore documented as **engine-only-meaningful** (use `engine_pk`).

**Bit-identical guarantee.** `measured_adme is None` ⇒ block skipped ⇒ both builds use the unmodified `adme`. Tests (`tests/regression/test_measured_adme_passthrough.py`): exact-float `predict(s,d).pk.cmax.mean == predict(s,d, measured_adme=None).pk.cmax.mean`; present-changes-cmax; warning-tag; `ValueError` on partial (fup-only). **Plus** a `scripts/run_engine_benchmark.py` rerun asserting 107/107 holdout bit-identity.

**`kp_method` field hygiene (optional, downgraded).** `ivive.py:738` hardcodes the *metadata* field; the engine ignores it, so this is **cosmetic, not a PK fix**. If shipped, it is provably bit-identical (default computes identical `kp_overrides`; the field is never read by the ODE). It does **not** enable measured-Kp injection — that is SP4. Lowest priority; may be dropped.

**Benchmark harness** (separate, never merged into the headline): `data/validation/measured_adme_benchmark.json` (N≥20, per-drug `holdout_member`/`training_member` from InChIKey audit; **every curation value verified against a cited source — agent-supplied numbers are unverified**), `scripts/run_measured_adme_benchmark.py` (`--engine-only` default, side-by-side SMILES-only vs measured AAFE, split-stratified). CI gets a **smoke test only** (constructs / None-bit-identical / partial `ValueError`); the live AAFE run is human-triggered, logged to `experiment-log.md`. Reference: engine-only ~1.98 on the measured subset.

## 5. SP2 — Measured-`peff` absorption correction on the engine path  *(Track 2, re-scoped)*

DE-41's prospective failure is bioavailability-F (engine `ka = 2.88·Peff·ka_fraction/radius`, `rhs_jax.py:377`). Measured `peff` (and `solubility`) **already flow into the engine** through `MeasuredADMEInput`→`build_drug_on_graph`→`ka` — **no new plumbing required**. SP2 measures, on the `--engine-only` benchmark, how much supplying measured `peff` closes the F-gap for drugs whose predicted `peff` is wrong. Ship/kill: improves measured-input engine AAFE without breaking any SP1 bit-identity test. (This replaces v1's unreachable Vss/Kp-via-`provided` plan.)

## 6. SP3 — Track 1 frontier

### 6.0 Fix the decorrelation gate (prerequisite — the rigorous "push harder")
Before any Track-1 ceiling claim: (1) regenerate the holdout artifact to **persist per-drug production `vdss` and `clf` Cmax** from the same run that produced `eng/ml/meta`; (2) re-run the gate correlating candidates against those **real** residuals (engine/ml are already valid); (3) replace the structurally-guaranteed `r` with an **oracle-blend test** — does adding the candidate at *any* weight move the 2.698 meta? (the genuine redundancy test the project used for VDss/DE-15/DE-24). Only then is "the orthogonal space is exhausted" an evidenced claim. Review and track `scripts/orthogonal_track_gate.py` as part of this.

### 6A. Absorption/F recalibration (prospective-targeted)
Recalibrate **regional Peff** / **gut Fg saturation** in the engine `ka`/gut path. **HARD GATE: retrospective 107/107 must not regress (2.698 ± bootstrap noise)**; success measured on prospective N=28. Honest odds: a global SMILES-meta recalibration is likely a DE-40-style no-op (error-cancellation) — but the same recalibration validated first on the SP1/SP2 engine-only path is where it can land. Overlaps directly with SP2's measured-`peff` lever.

### 6B. New orthogonal input signal — gut transporter recognition
**Gut transporter substrate recognition** (P-gp/BCRP efflux; PEPT1/OATP uptake) is structurally specific, not a smooth function of logP/TPSA — the one axis the corrected geometry plausibly leaves open, and it maps onto the DE-41 F-failure (novel kinase inhibitors are efflux substrates). Data: TDC `Pgp_Broccatelli` (acquire; confirm availability). Two gated forms: (i) **engine F-modifier** (gut efflux reduces `fa`) validated on the engine path first; (ii) **decorrelated meta-track candidate** scored through the **fixed** gate (§6.0) with the oracle-blend test — **|r|<0.5 vs the real four tracks AND a positive oracle move before any wiring**. Honest odds: efflux substrate-ness carries some logP dependence and may still correlate.

## 7. Risks & kill criteria

| Risk | Trigger | Action |
|---|---|---|
| Bit-identity breaks | any `measured_adme=None` test fails, or holdout ≠ 107/107 | **STOP** — architecture violation |
| Gate still invalid | §6.0 not completed before a ceiling/track claim | block the claim; no orthogonal-track work proceeds |
| Track-1 6A regresses headline | 107 retrospective AAFE > 2.698 + noise | revert; log dead-end |
| Transporter signal redundant | gate \|r\|≥0.5 **or** oracle move ≈0 | document; engine-F-modifier form may still be valid |
| Curation leak | benchmark drug in training set | drop/flag; split-stratified reporting exposes it |
| SP4 scope creep | building `kp_overrides` channel before SP1/SP2 land | keep SP4 deferred |

## 8. This-session scope

Per user direction (push SMILES harder + hunt a new signal; Track 2 plumbing + benchmark + first bias-correction): **SP1** (measured-input path + bit-identity guard + benchmark) → **SP2** (measured-`peff` engine-only absorption test) → **SP3 §6.0** (fix the gate — the prerequisite for the Track-1 "push harder"). 6A/6B and SP4 are queued behind their gates. Each ships as its own logical PR, committed as `jam-sudo`, **no Claude co-author trailer**.

## 9. Resolved defaults
- Measured number reported **engine-only** (`result.engine_pk` / `--engine-only`) — the only genuinely engine-confined surface.
- Benchmark **allows holdout members** with split-stratified reporting (evaluation, not training).
- `MeasuredADMEInput` ships **all override fields**, but `peff/vdss/rbp/solubility` are documented as **engine-only-meaningful** (they perturb CLF/VDss on the meta surface).
- `kp_method` field-hygiene fix is **optional/cosmetic**, decoupled from any Kp-injection claim.
- Vss/Kp (DE-33) correction is **deferred to SP4**, behind a real `kp_overrides` channel (`build_drug_on_graph` param + `"provided"` skip-branch + `predict()`/`MeasuredADMEInput` plumbing).
