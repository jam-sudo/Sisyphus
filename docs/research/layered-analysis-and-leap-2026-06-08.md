---
last_updated: 2026-06-08
charter: A full layer-by-layer (data / engine / ML+meta / validation-UQ) analysis of Sisyphus and an honest, gate-tested assessment of where a genuine performance leap is — and is not — available. Re-examines foreclosed methodology. Scientific-mathematical rigor is the binding constraint.
---

# Sisyphus — Layered Analysis & Performance-Leap Assessment (2026-06-08)

> Method: five independent deep-dives (data, engine, ML/meta, validation/UQ, plus a decisive
> decorrelation experiment), each grounded in `dead-ends.md` (44 prior entries) and `diagnosis.md`, each
> required to pass the **error-decorrelation gate** (|r|<0.5 on per-drug log-Cmax residuals vs every existing
> track — the only empirically-proven path, the VDss exception) before any claim of a lever. Four new
> candidate tracks were tested against that gate this session. All four were falsified. The convergence is the
> result.

---

## 0. Bottom line up front

1. **The headroom is real, but it is not label noise.** The clinical-Cmax *label-noise floor* is **AAFE ≈ 1.18**
   (band 1.18–1.5), measured from 14 clean same-drug/same-dose study replicates. Label noise is only **3–16%**
   of the model's error variance. The dead-ends.md line-32 claim ("residual ≈ experimental + formulation +
   inter-patient variability") is **quantitatively false for the bulk of the error**. Sisyphus at 2.78 is
   **model-limited, not label-limited** — there is ~0.70 AAFE of genuine model-side headroom down to the
   OrBiTo commercial floor (2.08).

2. **But every channel to that headroom is foreclosed — and the reason is now unified.** The single dominant
   recoverable error mode is **bioavailability-F structural error** (absorption `fa` ⊕ first-pass `Fg`/`Fh`),
   and it is **shared (correlated) across all four tracks**. This is *why* the decorrelation gate keeps
   failing: any new track that also lacks an F mechanism re-makes the same directional errors on the same
   drugs. It is *why* the meta damps the engine to ~18% (DE-43). It is *why* SMILES→F regressors fail (DE-28).
   It is *why* an absorption recalibration is a flat scalar (DE-42). **The error, not the input, is what must
   decorrelate — and F-error is everywhere.**

3. **A "leap" expressed as a lower SMILES-only headline AAFE is not available.** This is now exhaustively
   confirmed: 44 prior dead-ends + 4 fresh gate failures this session, against an externally-verified
   commercial ceiling (OrBiTo 2.08–2.74; Sisyphus 2.78 already inside the band). Continuing to chase the 2.78
   number is the one move the evidence forbids.

4. **The genuine, rigor-preserving leaps are reframes of "performance," and they are real:**
   (a) **correctness** as the durable product (FLUX-1/RBP shipped; D-1 transporter-unit bug; reference-data
   fixes; ECM/OATP albumin uptake); (b) ~~the measured-input regime as a second operating point~~ —
   **FALSIFIED 2026-06-08 (DE-48): measured-regime routing *degrades* accuracy** (engine-measured on a
   representative N=93 is ~3.84, *worse* than the SMILES meta 2.78; the "2.33 floor" was a clean-10 artifact;
   the meta correctly damps the engine even with measured inputs — DE-43 in the measured regime); (c)
   **calibrated, conditional honesty** (conformal PI shipped + per-subclass advisory flags); (d) the only path
   to a materially higher number is a **new F-orthogonal measured data modality** (in-vitro
   permeability/dissolution/transporter kinetics, or measured F) — a data-acquisition program, which is exactly
   the field-accepted escape (OrBiTo's expert-harmonised F reaches AAFE 1.75).

---

## I. The unified cross-layer diagnosis

Sisyphus's accuracy is bounded by **three nested walls**, in order of how deep they sit:

| Wall | What it is | Confirmed by |
|---|---|---|
| **W1 — CLint target-noise** | hepatocyte CLint R²≈0.24 is intrinsic (interlab CV up to 99.8%) | DE-14/44, Bowman-Benet 2019, external literature review |
| **W2 — bioavailability-F structural error** | the engine's `fa·Fg·Fh` map is wrong in a **bidirectional**, per-drug way no single knob reconciles; shared across tracks | DE-41/42/43, FLUX-1, this session's 4 gate failures |
| **W3 — meta co-calibration** | the fixed-weight geometric blend damps any single-track change to ~18% (r>0.986 co-calibration + a disagreement penalty that *down-weights the engine on exactly the drugs a fix targets*) | DE-23/24/25/26/43, this session's meta-math derivation |

The label-noise result (this session) adds the crucial fourth fact: **W4 does not exist** — there is no
label-noise wall at 2.0 to hit. So the 2.78 is W1⊕W2 propagated through W3, not an irreducible property of the
target. The headroom is real; the walls are what block it.

The deepest, most actionable insight: **W2 is the binding wall, and W3 is what makes it un-attackable from any
single track.** F-error is recoverable in principle (the field reaches F AAFE 1.75 with expert input), but in
Sisyphus it (i) cannot be learned from SMILES (DE-28, circular with logP/the same molecular inputs), (ii)
cannot be recalibrated (DE-42, `ka` linear → flat scalar), (iii) cannot propagate through the engine (DE-43,
meta-damped), and (iv) cannot be added as a regressor track (this session, fails the gate because it shares the
F-blindness). The only inputs that break the symmetry are **measured/experimental** and **F-orthogonal**.

---

## II. Layer-by-layer findings

### II.1 Data layer
- **Corpus:** scoring reference `clinical_pk.json` (331 drugs, 177 Cmax; 107-drug holdout; 72% Cmax are FDA-label
  analytic reconstructions, single-value curation). Measured-ADME+Cmax corpus `mmpk_pbpk_features.csv`
  (**1,128 drugs, all with measured fup+clint+dose+cmax_obs** — verified measured, not predicted).
- **Label-noise floor = AAFE 1.18** (14 clean replicate pairs; between-study geomean fold 1.261 → σ_label 0.090;
  external anchor: FDA intra-subject Cmax CV 21.7%±8.8%). Variance decomposition (σ_total=0.557 at AAFE 2.784):
  label noise is 2.6–15.7% of variance → a *perfect* model would still score AAFE 2.56–2.75. **Headroom to the
  label floor is tiny; headroom to OrBiTo (2.08) is real and model-side (~0.70 AAFE).** *(Confidence: high on
  direction, medium on the point — N=14 replicates.)*
- **Confirmed reference-data errors (correctness fixes, primary-source-adjudicated):** oxybutynin ref
  0.001→**0.008** mg/L (FDA Ditropan single-dose ~8 ng/mL, ~8× decimal/unit slip; defensible 8–12 ng/mL band); selegiline
  0.001→**~0.0022** (~2×). Both currently score as model *over*-predictions purely because the label is too low.
  *Net AAFE effect is small and bidirectional* (a bulk relabel = −0.011, error-cancellation) — these are
  correctness fixes, **not** a headline lever, and must be done by primary-source adjudication only (never
  toward the model; Invariant #5/#8).

### II.2 Engine / physics layer
- **FLUX-1 (E-cap-at-0.5 double-count) and RBP (`fu_b=fup/Rb`) fixes are in-tree and verified canonical**
  (`E = fu_b·CLint/(Q+fu_b·CLint)`, →1.0). These were the two known textbook deviations; both closed.
- **Open structural items (correctness, meta-damped unless surfaced as a new track):**
  - **D-1 — `ActiveTransportFluxSpec` Michaelis-Menten output has no MW/time unit closure** (`flux.py:726-736`;
    the code comment admits "rate is in arbitrary units… IVIVE handles unit conversion"). **Latent but
    load-bearing**: it is the class through which any OAT/OCT/MATE renal-secretion or gut-P-gp track must flow.
    Genuine bug; fix is a dedicated transporter `ivive_scaling` with explicit `×MW×60/1e6`.
  - **D-2 — gut first-pass uses full villous blood flow** (58.5 L/h), not the permeability-limited **Qgut**
    (Yang 2007, ~3–18 L/h). Systematically under-extracts gut CYP3A; currently calibration-entangled (the
    ×0.652 gut-CYP3A re-anchor compensates). Touch with care (DE-42 bidirectional-first-pass contributor).
  - **D-6 — absorption has no solubility/dose-number cap** (pure first-order `ka·A_lumen`); a 600 mg BCS-II
    dose absorbs the same *fraction* as a 1 mg microdose. Real gap, but the SMILES-only fix is circular (below).
  - In-flight working-tree work: `ParacellularAbsorptionFluxSpec` (Renkin pore-sieving) — an orthogonal
    *absorption-physics* route; engine-track (meta-damped) unless surfaced separately.

### II.3 ML / meta-learner layer
- **The meta is a fixed-weight, compound-type-binary geometric blend in log10 space** (`ensemble.py`): weights
  are module constants (base: 0.60/0.40/0.00; other: 0.35/0.50/0.15; `_W_VDSS=0.20`), regime selected by a
  single `compound_type=="base"` branch. No fit, no per-drug routing.
- **The ~18% damping is fully derived:** pass-through of an engine move equals its renormalized effective weight
  = **0.28 (other) / 0.48 (base)** after VDss ×0.80, dropping to **0.053 / 0.169** under the >10× disagreement
  penalty — which fires on *exactly the catastrophic drugs an engine fix targets*. The damping is a consequence
  of co-calibration (r>0.986), **not a tunable knob**.
- **The ML and CL/F tracks are `predict(smiles, dose)` — structurally blind to measured ADME**
  (`models.py:44`, `clf_predictor.py:72`). In the measured-input regime, measured fup/clint reach *only* the
  engine track. **This is the one structural gap that is not foreclosed** (see §V.2).

### II.4 Validation / uncertainty layer
- **Dispersion is concentrated in two subclasses** (confirms diagnosis §9 "a few hard subclasses"):
  high-PPB **acids** (AAFE 6.62, **0%** within 2-fold, n=6–8) and high-first-pass **bases** (AAFE 3.21, owns
  **37.5%** of the >3-fold tail, n=29). The >3-fold tail is **bidirectional: 18 over / 22 under** — two opposing
  first-pass modes (base over-extraction ⊕ acid under-extraction) that defeat any single scalar.
- **Subclass→lever:** acids → hepatic-fu (**data-blocked**, DE-37, paywalled supp tables); bases → FLUX-1
  (**shipped, meta-damped**); OATP/transporter → ECM albumin-uptake (the one open mechanistic lever, but
  barely present in the holdout — its value is *production breadth*).
- **Conditional UQ (CQR / Mondrian-by-subclass) is NOT a deliverable at N=107.** The raw 5.8× spread ratio is
  real but every Levene/Kruskal test is non-significant (p=0.28); the acid signal is **location-bias, not
  scale** (CQR exploits scale). The marginal split-conformal PI (q90=1.111, /÷12.92, 0.953 coverage) is already
  near-optimal. *Open in principle with measured labels + N≫107; not on the present holdout.*

---

## III. Four decisive experiments this session (gate results)

| # | Candidate | Decisive number | Verdict |
|---|---|---|---|
| 1 | **Label-noise floor** | AAFE 1.18 (14 replicate pairs); noise = 3–16% of variance | model-limited, not label-limited |
| 2 | **Dose-number / dissolution SMILES track** | corr(log Do, logP)=**+0.912**; candidate residual vs ML r=**+0.544** | **NO-GO** (= logP; DE-44 circularity) → DE-45 |
| 3 | **Renal-secretion SMILES track** | only 23/107 drugs score; vs-meta residual r=+0.011 | **NO-GO** (no signal) → DE-46 |
| 4 | **Measured-ADME-aware ML track** | N=93 overlap; CV AAFE **3.79** (>3.01, >2.33); residual r=**+0.69/+0.78/+0.79** vs eng/ml/meta | **NO-GO** (fails gate; F-blindness shared) → DE-47 |

Experiment 4 is the most important: it falsified the *leading* leap candidate by the *same* gate that killed
dose-number, at high power (N=93), with clean provenance. **measured fup/clint were nearly inert in a flat
regressor** (importance 0.045/0.053 vs dose 0.49) — the measured information is only usable *mechanistically*
(through the engine, 2.33), and a regressor re-learns the shared F-error. This is the empirical heart of W2.

---

## IV. Methodology re-examination (re-examining the foreclosed)

Per the mandate, each prior failure was re-checked for a *methodological* gap (not just confirmed):

- **CLint-side levers (DE-08…21, 44):** correctly foreclosed. The methodology was sound; the floor is real
  (W1). *No gap.*
- **Post-hoc meta variants (DE-23…26):** correctly foreclosed. They recombine the *same tracks* (r>0.986).
  *Gap found, but closed:* the only escape would be a conditioning signal *orthogonal* to the tracks' own
  outputs — and DE-41 showed every such signal (predicted-F, divergence) is r≈0 with error on the holdout. The
  one weakly-significant external signal (`is_acid`, rho=+0.22) is **location-bias**, which class-aware weights
  (DE-25/27) already swept to a tie. *Closed.*
- **Absorption/F recalibration (DE-42/43):** correctly foreclosed. *Methodological subtlety confirmed:* the
  failure is that `ka` is linear (flat scalar) **and** the residual is bidirectional first-pass. *No gap.*
- **Measured-input (the open frontier):** here a real **methodological gap exists** — the ML/CLF tracks are
  SMILES-blind, so measured inputs are under-exploited. But this session shows the gap is **not** closable by a
  measured-ADME *regressor* track (Exp 4); it is closable only by **measured-regime routing** (trust the
  measured-engine more when measured ADME is present — §V.2), which is a *capability* extension of the shipped
  measured-F routing, not a headline move.

**Net methodological conclusion:** the prior team's foreclosures are correct. The one genuine gap (SMILES-blind
tracks in the measured regime) is real but yields a *capability*, not a SMILES-headline leap. There is no
methodological error hiding a free lever.

---

## V. The leap, honestly assessed

### V.0 What a leap on the 2.78 number would require — and why it's unavailable
A track with F-orthogonal error and |r|<0.5 vs the existing four. VDss was the last such track in the SMILES
input space (distribution-orthogonal). Every remaining SMILES-derived axis (dissolution, renal, F, CLint) is
either circular with the molecular inputs the engine already eats, or shares the F-error. **The SMILES input
space is decorrelation-exhausted.** Therefore a higher SMILES-only number requires a *new F-orthogonal input
channel*, which by definition is not SMILES.

### V.1 Leap A — Correctness as the product (ship; durable; not headline)
Per correctness-over-benchmark. Concrete, ready:
- **Reference-data fixes** (oxybutynin ~8×, selegiline 2×) — primary-source-adjudicated.
- **D-1 transporter-MM unit closure** — a real latent bug; prerequisite for any transporter physics.
- **ECM/OATP albumin-mediated uptake re-anchor** — the one un-foreclosed *mechanistic* lever (Li/Benet 2020,
  ~1.9–2.0 fold, no empirical scaling), already FLUX-1-deferred (the xfailed pravastatin/pitavastatin
  statins). Production-breadth value; near-zero on the 107-holdout (only pravastatin, inviolable).

### V.2 Leap B — Measured-regime routing — ❌ FALSIFIED (gate-tested 2026-06-08; DE-48)
**This was originally proposed here as the leading feasible leap. The decisive gate killed it — recorded
honestly.** The premise rested on a "~2.33 engine-measured floor," but **2.33 is a hand-curated clean-10 PoC
artifact**. On the honest high-power set (**N=93** holdout drugs with measured fup+clint) the reproducible
**engine-measured AAFE is ~3.84** (verified 3 independent ways + by an adversarial skeptic), which is **worse**
than both the dragged measured-meta (~2.9) and the SMILES-only meta (**2.78**). Up-weighting the engine when
measured ADME is present therefore **degrades** accuracy monotonically (V1 engine=1.0 → 3.84; best reweight
w=0.3 → 2.89; *every* variant > 2.78). Mechanism: measured high-CLint blows up first-pass actives in the engine
(selegiline 126×, acamprosate 100×, methylphenidate 64×, progesterone 49×, abiraterone 40× over), and the
ML/VDss tracks were correctly **damping exactly those** — i.e. **DE-43 in the measured regime: the meta's
engine-damping is a feature, not a bug.** The inversion holds on the obs-agree subset (N=52: engine 3.92 vs
meta 3.05) and on the clean-10 only because that subset was selected to exclude the engine's structural-failure
classes (NSAIDs/acids, P-gp/transporter drugs). **Verdict: ship NO measured-regime routing; keep production
weights in the measured regime.** *(Caveat: measured on a working tree with uncommitted engine changes —
flux.py/builder.py/types.py — so the exact 3.84 is non-canonical and should be CI-regenerated; the inversion
margin (3.84 vs 2.78) is far too large to flip.)* This **retracts** the original Leap B and its "2.33 floor /
≤2.0 operating point" claim. The measured-input regime is **not** a better operating point on a representative
set; the only measured lever that survives is a **new F-orthogonal data modality** (Leap D), not a reweight.

### V.3 Leap C — Conditional honesty (product-quality at capped AAFE)
- **Per-subclass advisory flags** ("high-dispersion: high-PPB acid → AAFE-6.6 tail"; "high-first-pass base").
  Costs nothing on AAFE; tells the user *which* predictions sit in the dispersion tail vs the well-behaved
  core. Rigorous, honest, immediately shippable.
- Conformal PI is already calibrated; CQR is *not* worth building at N=107 (§II.4).

### V.4 Leap D — The only path to a higher *number*: a new F-orthogonal data modality
The honest "what would it actually take." To move the population Cmax number itself, acquire an input that
carries F information **decoupled** from the molecular descriptors the engine already uses: measured aqueous /
pH-dependent solubility + crystal form + particle size (breaks the dose-number=logP circularity), measured
Caco-2/PAMPA permeability, measured transporter kinetics, or measured F. This is a data-acquisition program,
not an algorithm change — and it is exactly how the field reaches F AAFE 1.75 (OrBiTo expert-harmonised). It is
the only thing in this analysis that could raise the population ceiling, and it is honest about its cost.

### V.5 One residual SMILES-only test worth a single afternoon (low probability, not yet run)
The dose-number gate used the repo's **logP-only** solubility (`log10 S = −logP + 0.5`), so it was circular by
construction. A **pKa-aware pH-dependent** dose number (Henderson-Hasselbalch: `S(pH)=S0·(1+10^(pH−pKa))` at
gastric pH for weak bases) carries an ionization term **not** reducible to logP. Probability it clears the
gate: low (the engine already ingests pKa for ionization; the holdout has few high-dose weak-base BCS-II
drugs), but it is the one un-run variant and costs one afternoon. Pre-register: PASS only if |r|<0.5 vs all
four tracks AND the pH-solubility term (not logP) carries the signal.

---

## VI. Concrete action plan (ranked; each with its gate)

| Rank | Action | Type | Gate / guard | Effort |
|---|---|---|---|---|
| ~~—~~ | ~~Measured-regime routing~~ — **❌ FALSIFIED (DE-48): degrades accuracy, do not build** | — | gate FAILED: engine-measured N=93 ~3.84 > meta 2.78; meta correctly damps the engine | — |
| 1 | **Reference fix: oxybutynin** (0.001→0.008; FDA single-dose ~8× error) + flag **selegiline** (uncertain — defensible ~0.002, but 0.001 within the noisy low tail; leave or annotate) | Correctness (Leap A) | primary-source only, never toward the model; **needs holdout-reference sign-off + CI regen** (touches the inviolable yardstick) | S |
| 2 | **D-1 transporter-MM unit closure** | Correctness (Leap A) | identity-blind random-rename invariance; **latent — 0 production YAML uses `active_transport`, zero blast radius**; coordinate with in-flight paracellular work (flux.py is dirty) | S |
| 3 | **Per-subclass advisory flags** (acid/high-PPB, high-first-pass base) | Product honesty (Leap C) | zero AAFE effect; pure reporting | S |
| 4 | **ECM/OATP albumin-uptake re-anchor** (un-xfail the statins) | Correctness/breadth (Leap A) | non-holdout OATP substrate for re-anchor; holdout bit-identical | M |
| 5 | **pKa-aware pH-solubility dose-number gate** | Falsification (Leap V.5) | pre-registered |r|<0.5 AND pH-term carries signal; else → next DE | S |
| — | **New F-orthogonal measured data modality** | Strategic (Leap D) | a data program; the only population-ceiling lever | L |

**Do NOT pursue** (re-confirmed this session): any SMILES→{CLint, F, dose-number, renal} regressor as a headline
lever; any reweighting/restacking of the current four tracks; CQR/Mondrian conditional intervals at N=107;
hepatic-fu for acids (data-blocked); selling any measured-ADME work as a move on the 2.78 SMILES headline.

---

## VII. Proposed canonical updates (for review)

1. **dead-ends.md** — append **DE-45** (dose-number/dissolution SMILES track, NO-GO, =logP), **DE-46**
   (renal-secretion SMILES track, NO-GO, no signal), **DE-47** (measured-ADME-aware ML regressor track, NO-GO,
   fails decorrelation gate r=0.69–0.79 — *the error not the input must decorrelate*), **DE-48**
   (measured-regime engine up-weighting/routing, NO-GO — engine-measured on representative N=93 is ~3.84 >
   meta 2.78; the meta correctly damps the engine; "2.33 floor" was a clean-10 artifact; this retracts Leap B).
   *(Appended this session; see dead-ends.md.)*
2. **diagnosis.md** — add a **§10 "The label-noise floor"**: the ceiling is model-limited (label floor AAFE
   ≈1.18), refining the line-32 "residual ≈ experimental variability" claim; and record W2 (bioavailability-F
   blindness as the shared, binding wall) as the unifying mechanism behind the decorrelation-gate failures.
   *(Medium confidence on the 1.18 point estimate, N=14 — flagged.)*
3. **No top-metrics-table change** — nothing here moves the 2.784 headline (correctly).

---

*See also: `dead-ends.md` (DE-45/46/47 appended), `diagnosis.md` §1/§4/§8/§9, the FLUX-1 spec, and
external-pbpk-benchmark-bar / correctness-over-benchmark.*
