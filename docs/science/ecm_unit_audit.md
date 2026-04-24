# ECM Unit Audit — Extended Clearance Model

**Scope**: dimensional consistency and provenance of every symbol in the hepatic Extended Clearance Model (ECM) at `src/sisyphus/engine/flux.py:291-334` (`ClearanceFluxSpec(model="extended")`). Traces the chain from OATP1B1 kinetic inputs to an organ-level clearance value in L/h, and makes explicit which parameters are biological measurements, which are calibrated effective values, and which are dimensional scaffolds.

**Why this doc exists**: the ECM drives DE-33 (OATP1B1 non-statin underprediction, V3 valsartan/glimepiride 2.5× Mode C). Any architectural fix for DE-33 must reason about the unit chain. Today those pieces live in five separate places; this file consolidates them without changing code.

Related files referenced below:
- Engine: `src/sisyphus/engine/flux.py:291-334`
- Physiology: `data/physiology/reference_man.yaml` (liver node, `ivive_scaling`)
- Kinetics: `data/transporters/oatp1b1.json`
- V3 test result: `data/validation/oatp_generalization_result_v3.json`
- Specs: `docs/superpowers/specs/2026-04-20-oatp-ecm-hepatic-clearance-design.md`, `2026-04-21-ecm-generalization-test-design.md` (amendment v2.1 commit `0d78c38`)
- Memory: `project_ecm_generalization_test.md`

---

## 1. Symbol inventory — `flux.py:291-334`

ECM branch of `ClearanceFluxSpec.apply`:

```python
elif self.model == "extended":
    src = self.source_name
    ivive = params.node_param(src, "ivive_scaling")

    # PS_active: sum over transporters at the source node
    ps_active = 0.0
    for tag, abundance in params.node_transporters(src).items():
        jmax = params.drug_transporter_jmax(tag)
        km = params.drug_transporter_km(tag)
        if jmax <= 0 or km <= 0 or abundance <= 0:
            continue
        ps_active += abundance * jmax / km
    ps_active *= ivive

    ps_passive = params.drug_param("ps_passive")
    ps_eff     = params.drug_param("ps_eff")
    cl_int_bile = params.drug_param("cl_int_bile")

    ps_inf = ps_active + ps_passive

    # Metabolism (same pattern as well-stirred)
    cl_int_metab = 0.0
    for tag, abundance in params.node_enzymes(src).items():
        affinity = params.drug_enzyme_affinity(tag)
        if affinity > 0 and abundance > 0:
            cl_int_metab += abundance * affinity * ivive
    cl_int_h = cl_int_metab + cl_int_bile

    fup = params.drug_param("fup")
    q   = params.total_inflow(src)

    num = q * fup * ps_inf * cl_int_h
    den = q * (ps_eff + cl_int_h) + fup * ps_inf * cl_int_h
    clh = num / den
    ...
    rate = clh * c_out
```

Each symbol, its origin, its declared or derived units, and its classification:

| Symbol | Source | Units | Classification |
|---|---|---|---|
| `abundance` (OATP1B1) | `reference_man.yaml` liver.transporters.OATP1B1.mean = 5.0e5 | **mg-protein** (see §2) | **calibrated** |
| `jmax` | `oatp1b1.json` per-drug `jmax_pmol_per_min_per_mg` | pmol/min/mg-protein | biological (pravastatin) / calibrated (valsartan, see §3) |
| `km` | `oatp1b1.json` per-drug `km_uM` | µM = pmol/µL | biological |
| `ivive` | `reference_man.yaml` liver.ivive_scaling = 6e-5 | (L/h) ÷ (µL/min) = 60/1e6 | **placeholder** (unit-conversion constant) |
| `ps_active` (after `×= ivive`) | derived | L/h | derived |
| `ps_passive` | per-drug `DrugOnGraph.ps_passive` | L/h | biological (when literature-based) |
| `ps_eff` | per-drug `DrugOnGraph.ps_eff` | L/h | biological |
| `cl_int_bile` | per-drug `DrugOnGraph.cl_int_bile` | L/h | biological |
| `cl_int_metab` | enzyme × affinity × ivive, per-node | L/h | derived |
| `fup` | `DrugOnGraph.fup` | dimensionless (0,1) | predict-layer (XGBoost), occasionally biological when DrugBank-verified |
| `q` | `total_inflow(liver)` from flow conservation | L/h | biological (ICRP Reference Man blood flow) |
| `clh` | num / den | L/h | derived |
| `c_out` | `amount * rbp / (volume * kp)` | mg/L | derived |
| `rate` | `clh * c_out` | mg/h | derived |

### Classification key

- **biological** — parameter has a measurable, literature-traceable value. Changing the code's numeric would make it diverge from a real-world measurement.
- **calibrated** — numeric value was set by fitting to observed pipeline output (usually pravastatin Cmax in Phase-1 commissioning). The value carries "effective" semantics: it is not directly measured, but it absorbs residual bias in upstream parameters.
- **placeholder** — value whose magnitude is forced by a unit-conversion convention, not measurement. Changing it requires changing the unit convention of downstream symbols.

---

## 2. `abundance = 5.0e5` — effective mg-protein

`reference_man.yaml` liver node, post-Achour-merge (commit `2275932`):

```yaml
transporters:
  # OATP1B1 independent lognormal — Achour 2021 empirical r<0.3 vs CYPs
  OATP1B1: {mean: 5.0e5, cv: 0.484}
```

The comment labels this "independent" relative to CYPs — meaning OATP1B1 is sampled from its own lognormal rather than the Achour 5-CYP correlation matrix (Achour 2021, empirical Pearson r < 0.3 with the CYPs).

### Biological scale check

Hepatic OATP1B1 expression is typically reported in pmol-transporter per mg-membrane-protein. A useful cross-check converts to total transporter count in liver:

- MPPGL (microsomal protein per gram liver) ≈ 99 mg/g liver (Barter 2007)
- Reference-man liver mass ≈ 1500 g
- Total hepatic protein ≈ 99 × 1500 ≈ 1.5×10⁵ mg ≈ 1.5e5 mg-protein
- OATP1B1 expression ≈ 2–10 pmol/mg membrane protein (Prasad 2014, Kumar 2018)
- Total transporter count ≈ 3×10⁵ – 1.5×10⁶ pmol

**Interpretation**: `abundance = 5.0e5` is inside the range of plausible total hepatic OATP1B1 transporter count (pmol-transporter). It is NOT MPPGL × liver-weight (which is mg-protein), despite the YAML comment `# 60/1e6: converts uL/min -> L/h (MPPGL * organ_wt already in abundance)` implying otherwise.

This is a **calibrated** value: it was set to make the pravastatin pipeline match observed Cmax within tolerance at Phase-1 commissioning. The number is dimensionally consistent with hepatic transporter biology in either mg-protein or pmol interpretation (the difference is absorbed by `jmax`'s reported units, which historically varied across assay formats). The `ivive_scaling` then provides the time/volume conversion regardless of which interpretation applies.

### Before Achour: 1.0e11

Prior to the P4.5 Achour-correlated-abundance merge, `liver.transporters.OATP1B1 = 1.0e11` (see `project_ecm_generalization_test.md` and commits predating `2275932`). That value carried the explicit label "calibrated on pravastatin in Phase 1 — absorbs residual bias" and had NO plausible biological interpretation (six orders of magnitude above total hepatic transporter count). The Achour merge deliberately brought the numeric back into a biologically plausible range while keeping pravastatin Cmax invariant. Because the downstream QSSA denominator `q × ps_eff + fup × ps_inf × cl_int_h` has `ps_inf × cl_int_h` in a saturating position, the pravastatin hepatic clearance was not sensitive to whether `ps_active` was 10⁴ or 10⁸ L/h — flow-limited saturation does the work either way. Shrinking `abundance` × 10⁵ meant re-interpretation without re-fitting.

---

## 3. Pravastatin numerical reconstruction

Given the values above, the intrinsic OATP1B1 uptake clearance for pravastatin is:

```
ps_active (pravastatin)
  = abundance × (jmax / km) × ivive
  = 5.0e5 [mg-protein]
      × (228 [pmol/min/mg-protein] / 13.6 [pmol/µL])
      × (6e-5 [(L/h) / (µL/min)])
  = 5.0e5 × 16.76 [µL/min/mg] × 6e-5 [(L/h)/(µL/min)]
  = 502.9 L/h
```

Units check:
- `(pmol/min/mg) / (pmol/µL) = µL/min/mg` ✓
- `[mg-protein] × [µL/min/mg-protein] = µL/min` ✓
- `[µL/min] × [(L/h) / (µL/min)] = L/h` ✓

This is the raw intrinsic uptake clearance — much larger than human hepatic blood flow (~90 L/h). The QSSA formula (Shitara 2013) handles the flow limit:

```
num = q × fup × ps_inf × cl_int_h
den = q × (ps_eff + cl_int_h) + fup × ps_inf × cl_int_h
clh = num / den
```

When `ps_active` is very large, `ps_inf ≈ ps_active`, the numerator scales with `q × ps_inf × cl_int_h`, and the denominator's `fup × ps_inf × cl_int_h` term dominates as `ps_active → ∞`. In that limit:

```
clh → (q × fup × ps_inf × cl_int_h) / (fup × ps_inf × cl_int_h) = q
```

— hepatic uptake saturates on liver blood flow, which is the classical flow-limited ECM behavior. This is why pravastatin's predicted Cmax is not sensitive to `abundance` once `ps_active` is in the "big" regime.

---

## 4. Flat-CLuptake scaling (amendment v2.1) and its Km-invariance

Amendment v2.1 (`docs/superpowers/specs/2026-04-21-ecm-generalization-test-design.md`, commit `0d78c38`) introduced a Jmax-scaling rule for OATP1B1 substrates whose primary Jmax is not available in the literature. From `data/transporters/oatp1b1.json`:

> Amendment v2.1: valsartan promoted from blocked_drugs to drugs via flat-CLuptake Jmax scaling — `Jmax_val = (Jmax_prava/Km_prava) × Km_val = (228/13.6) × 1.39 = 23.3 pmol/min/mg, CV 0.70 (widened to absorb scaling uncertainty)`.

The rule:

```
Jmax_val := (Jmax_prava / Km_prava) × Km_val
```

Now substitute into the engine formula:

```
ps_active_val
  = abundance × (Jmax_val / Km_val) × ivive
  = abundance × ((Jmax_prava / Km_prava) × Km_val / Km_val) × ivive
  = abundance × (Jmax_prava / Km_prava) × ivive
  = ps_active_prava
```

`Km_val` appears once in the numerator (as Jmax_val) and once in the denominator; the ratio cancels. **The intrinsic OATP1B1 uptake clearance under flat-CLuptake scaling is independent of `Km_val` by construction.**

### Numerical demonstration (valsartan, Km = 1.39 µM)

```
Jmax_val_scaled = (228 / 13.6) × 1.39 = 23.3 pmol/min/mg
ps_active_val = 5.0e5 × (23.3 / 1.39) × 6e-5 = 502.9 L/h   ← identical to pravastatin
```

### Why this matters for DE-33

The 2026-04-22 TransPortal Km audit (dead-end #12 in `project_remaining_priorities.md`) found that UCSF TransPortal's multi-assay geometric-mean Km for valsartan is 5.70 µM, versus the 1.39 µM from Yamashiro 2006 via Niemi 2009. A substantial (4×) difference. The intuitive hope: update `Km_val` → different valsartan prediction → potentially close the 2.5× underprediction gap.

The audit concluded NO: under flat-CLuptake scaling, `ps_active_val` is invariant to `Km_val`. The V3 underprediction is not a Km-sourcing problem. Km updates alone cannot fix DE-33.

---

## 5. V3 ECM test interpretation

Frozen V3 run (`data/validation/oatp_generalization_result_v3.json`, commit `7aa49ae`):

| Drug | Dose (mg) | Observed Cmax (mg/L) | Predicted point (mg/L) | PI90 | log10(FE) | Fold error |
|---|---|---|---|---|---|---|
| glimepiride | 1 | 0.243 | 0.0948 | [0.0874, 0.1009] | −0.409 | 0.39× (underpredict) |
| valsartan | 20 | 4.02 | 1.94 | [1.80, 2.06] | −0.316 | 0.48× (underpredict) |

Median |log10 FE| = 0.363 — below the 0.50 Mode-B systematic-bias threshold, so formally Mode C (both fail the PI but no declared systematic bias). Both drugs underpredict in the same direction. The PI is narrow because the only MC-propagated parameters are Jmax/Km/fup (the engine is nearly deterministic once ps_active saturates on flow).

### Mechanical explanation

Under flat-CLuptake scaling, valsartan and glimepiride inherit **pravastatin's intrinsic OATP1B1 uptake clearance** (§4). But pravastatin's *absolute* hepatic extraction is flow-limited; the *shape* of hepatic uptake saturation depends on `ps_inf`, `ps_eff`, and metabolism paths. For drugs with different ps_eff / cl_int_metab (valsartan has minimal hepatic metabolism; glimepiride has CYP2C9 but different magnitude), the full QSSA output is **not** a pure pravastatin-scale of hepatic CL. The ~2-2.5× underprediction is the residual of forcing a statin-calibrated effective-abundance onto substrates whose ps_eff / metabolism ratios diverge from pravastatin's.

**This is not a Km or Jmax problem alone.** It is an architectural consequence of single-reference calibration.

---

## 6. DE-33 architectural options

From `project_ecm_generalization_test.md` (§ DE-33 remaining paths), with pre/con from this unit-chain perspective:

### Option 1 — Real per-drug Jmax from primary literature

**Mechanism**: replace flat-CLuptake with substrate-specific Jmax measurements (HEK293-OATP1B1 uptake assays).

- **Pro**: breaks the Km-invariance trap. `Jmax_val / Km_val` becomes a real per-substrate ratio, not a pravastatin-cloned constant.
- **Con**: Jmax varies ~10× across assay formats (HEK293 vs hepatocyte vs oocyte). For valsartan, Yamashiro 2006 is paywalled (15+ source attempts logged in prior `oatp1b1.json` commits). Literature Jmax for non-statin OATP1B1 substrates is sparse and assay-format-sensitive.
- **Unit impact**: none — same formula, different numeric `jmax` per drug.

### Option 2 — Replace flat-CLuptake with a different scaling methodology

Candidates:
- Km-conditioned CLuptake regression (fit `CL_uptake(log10 Km, logP, ...)` across statins + known OATP1B1 substrates)
- Tissue-abundance-weighted uptake (separate abundance per-substrate based on binding-site competition)

- **Pro**: eliminates the single-reference-calibration bias. Could deliver a statistical model that generalizes beyond statins.
- **Con**: requires a labelled dataset of primary-source Jmax values — same scarcity as Option 1, but now as a training set rather than per-drug lookup.
- **Unit impact**: changes the engine's `ps_active` formula. Touches `ClearanceFluxSpec(model="extended")`. Not invariant-safe.

### Option 3 — Drop non-statin substrates from active ECM set

**Mechanism**: revert to statin-only ECM. Document in DE-33 that non-statin OATP1B1 substrates remain an open research question. Valsartan and glimepiride exit the generalization test set.

- **Pro**: zero risk of contaminating statin performance. Acknowledges the architectural limit honestly.
- **Con**: V3 ECM test becomes N=0 for non-statin OATP1B1. The hypothesis "ECM generalizes across OATP1B1 substrates" stays untested until Option 1 or 2 lands.
- **Unit impact**: none — substrate list change in `oatp1b1.json`.

### Option 4 — Add OATP1B3 and NTCP as parallel transporters

**Mechanism**: replicate the OATP1B1 pattern for additional hepatic uptake transporters; let aggregate `ps_active` across multiple transporters carry non-statin substrates whose OATP1B1 affinity is modest but whose aggregate hepatic uptake is correctly modeled.

- **Pro**: physiologically motivated — valsartan and glimepiride are known OATP1B3 + NTCP substrates too. More transporter paths → less reliance on a single-substrate calibration.
- **Con**: each new transporter requires abundance (per-node), Jmax/Km (per-drug, per-transporter), and calibration. Compounds the bookkeeping. Opens new DrugOnGraph fields.
- **Unit impact**: same formula, more terms in the `ps_active` sum. Not invariant-breaking but high bookkeeping cost.

### Recommendation (informational, not a decision)

Option 3 has the lowest cost and clearest acceptance criteria; it should be adopted as the current stance pending Option 1/2/4 research. Options 1, 2, and 4 each require a dedicated spec + plan cycle and a labelled dataset; they are 2026Q3+ scope.

---

## 7. Recomputing the canonical numerics

To verify this document against the live engine after any change:

```python
# reference_man.yaml values
abundance  = 5.0e5
ivive      = 6e-5
# oatp1b1.json pravastatin
jmax_prava = 228.0
km_prava   = 13.6
ps_active_prava = abundance * (jmax_prava / km_prava) * ivive
# Should print 502.9 L/h (±0.1 for float precision)
print(f"ps_active (pravastatin) = {ps_active_prava:.1f} L/h")
```

If this number changes, one of the three source files has changed. Update the table in §1 and the narrative in §3.

---

## 8. Cross-references

- `docs/superpowers/specs/2026-04-20-oatp-ecm-hepatic-clearance-design.md` — ECM architecture design, QSSA derivation.
- `docs/superpowers/specs/2026-04-21-ecm-generalization-test-design.md` — V3 test spec, amendment v2/v2.1.
- `docs/superpowers/specs/2026-04-22-iv-cmax-observation-design.md` — V3 IV-Cmax observation routing (predates this doc but is a prerequisite for V3 numeric interpretation).
- `project_ecm_generalization_test.md` (memory) — DE-33 context, TransPortal Km audit (dead-end #12).
- `project_remaining_priorities.md` — DE-33 in dead-ends list, current active work.
- `src/sisyphus/engine/flux.py:291-334` — authoritative engine source.
- `data/transporters/oatp1b1.json` — per-drug kinetic source of record + v2.1 amendment notes.
