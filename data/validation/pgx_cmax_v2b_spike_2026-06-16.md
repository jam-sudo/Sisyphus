# PGx v2.2b saturable harness feasibility SPIKE — propafenone (CYP2D6)

**Date:** 2026-06-16
**Branch:** `feat/pgx-cmax-v2b-nonlinear`
**Harness:** `scripts/validate_pgx_cmax_v2b.py` (Task-1 spike; ONE drug, propafenone)
**Engine:** v2.2a saturable Michaelis–Menten (`DrugOnGraph.enzyme_km`), `well_stirred`
liver edge (forced via `dataclasses.replace`), engine-measured E_h anchoring.
**Isolation:** no `predict()` / `reference_man.yaml` / holdout change. Headline 2.731 untouched.

## VERDICT: **HALT**

Gate 2 (the decisive saturation-engagement gate) **FAILS**. At a therapeutic propafenone
dose the engine's first-pass **peak unbound liver concentration reaches only ~17 % of Km**,
so the saturable factor `1/(1+C_u/Km)` barely deviates from 1 and the saturable genotype
fold is statistically indistinguishable from the linear (Km=∞) null. This is the
HALT outcome anticipated in the task brief — it reshapes the v2.2b milestone (the engine
does not reach liver C_u ~ Km at therapeutic dose), it is **not** a bug to paper over.

## Inputs (propafenone, CYP2D6)

| quantity | value |
|---|---|
| fm (CYP2D6) | 0.80 |
| MW | 341.4 |
| dose (EM anchor / high) | 300 mg / 400 mg |
| oral F (EM) ⇒ E_h target | 0.10 ⇒ **0.90** |
| tmax / t½ (EM) | 2.5 h / 5.5 h |
| **Km_unbound** = 5.3 µM × MW/1000 × fu_mic(0.5) | **0.90471 mg/L** |

## Anchor convergence (Gate 1 = PASS)

| anchor | cltot | peff | E_h (engine) | tmax | Cu_peak (mg/L) |
|---|---|---|---|---|---|
| SAT EM @300 | 3.74e7 | 20.0 (default) | 0.9000 | 0.55 h | 0.1550 |
| LIN EM @300 | 3.52e7 | 20.0 (default) | 0.9000 | 0.55 h | 0.1459 |
| LIN EM @lowE | 1.65e6 | — | 0.3000 | — | — |

The E_h bisection (`brentq` over `log cltot ∈ [1e1, 1e9]`) converges cleanly and is
monotone in cltot; both SAT and LIN EM runs reproduce the target E_h=0.90 to 4 decimals
(rel-err 0 % « 5 % tolerance). **The tmax bisection is NOT bracketable** on this `well_stirred`
skeleton — the maximum achievable tmax is ~1.05 h (at peff→0.1), well short of the 2.5 h
reference, so peff defaults to 20. This is itself informative: a slower (more realistic
tmax) absorption would *lower* the first-pass C_u peak, making saturation even weaker. The
peff=20 default is the most saturation-favorable realistic operating point.

## Folds (PM/EM)

| run | Cmax-fold | AUC-fold | Cu_peak(EM) | Cu_peak/Km |
|---|---|---|---|---|
| **sat_300** | 3.7955 | **4.6051** | 0.1550 | **0.171** |
| **lin_300** | 4.2036 | **4.9999** | 0.1459 | 0.161 |
| **sat_400** | 3.6548 | **4.4769** | 0.2148 | **0.237** |
| lin_lowE_300 | 1.6167 | 5.0002 | — | — |

Effect directions are **physically correct but sub-threshold**:
- saturable AUC-fold (4.605) < linear (5.000) — saturation in the EM arm raises EM exposure,
  compressing the PM/EM ratio. Correct sign.
- 300→400 mg saturable AUC-fold shrinks 4.605→4.477 — more saturation at higher dose.
  Correct sign.

## The four gates

| Gate | criterion | value | result |
|---|---|---|---|
| **1 converge** | finite cltot/peff; EM E_h within 5 % of 0.90 | E_h=0.9000 (0 % err) | **PASS** |
| **2 saturation engaged** | `|Δlog AUC_fold (sat−lin)| > 0.10` | **0.0823** | **FAIL** |
| **3 dose-dependence** | `|Δlog AUC_fold (300−400)| > 0.03` & shrinks | 0.0282 (shrinks ✓) | **FAIL** |
| **4 oracle highE** | linear AUC-fold ≈ 1/(1−fm)=5.0 within 2 % | rel-err 0.0000 | **PASS** |
| **4 oracle lowE (0.30)** | same, low-extraction control | rel-err 0.0000 | **PASS** |

> Gate-4 note: the oral-AUC linear identity `AUC_fold = 1/(1−fm) = 5.0` holds **exactly**
> here, at *both* E_h=0.90 (high) and E_h=0.30 (low). The task anticipated it might break at
> high extraction; on this `well_stirred` skeleton with renal/gut clearance zeroed it does
> not — the engine's oral AUC-fold is a clean `1/(CLint_PM/CLint_EM)`. The oracle is sound.

## Root-cause diagnostic (the HALT signal)

Peak unbound liver concentration vs Km (the quantity that must approach Km for saturation):

```
Cu_peak(EM) @300 mg = 0.1550 mg/L  ⇒  Cu/Km = 0.171   (saturable factor ≈ 0.85)
Cu_peak(EM) @400 mg = 0.2148 mg/L  ⇒  Cu/Km = 0.237   (saturable factor ≈ 0.81)
```

peff scan @300 mg (sat drug) — C_u/Km never approaches 1 at any physiological peff:

| peff | tmax | Cu_peak | Cu/Km |
|---|---|---|---|
| 0.1 | 1.05 | 0.0034 | 0.004 |
| 1.0 | 1.00 | 0.0148 | 0.016 |
| 5.0 | 0.80 | 0.0566 | 0.063 |
| 20.0 | 0.55 | 0.1550 | 0.171 |
| 100.0 | 0.30 | 0.3463 | 0.383 |

Even at an unphysiological peff=100 (tmax=0.30 h) the peak liver C_u reaches only 0.38×Km.
At the therapeutic operating point C_u/Km ≈ 0.17. With `1/(1+0.17) ≈ 0.85`, the saturable
factor moves the fold by only ~8 % in log-space — below the 0.10 engagement gate.

## Interpretation & milestone impact

The v2.2a saturable machinery works correctly (the factor is applied, effects are
correctly-signed and dose-dependent), but for propafenone at therapeutic dose the engine's
**first-pass liver C_u (~0.15 mg/L) sits an order of magnitude below the literature
Km (0.905 mg/L)**. Michaelis–Menten saturation is therefore not materially engaged, and a
saturable genotype Cmax/AUC fold cannot be distinguished from the linear null at N=1.

Implications for the v2.2b milestone:
- A nonlinear-genotype validation on **propafenone** is **not powered** — the drug does not
  reach its own Km in first-pass at clinical doses in this skeleton.
- The milestone needs either (a) candidate drugs with a genuinely **low Km relative to
  first-pass liver C_u** (low-Km, high-dose, high-fm substrates) re-curated against this same
  C_u/Km feasibility probe, or (b) a scope change away from saturable-genotype Cmax-folds.
- Before building the full v2.2b benchmark/scoring, **run this C_u/Km probe per candidate** —
  any candidate with peak C_u/Km ≪ 1 will fail Gate 2 the same way.

**Recommendation:** HALT v2.2b on the propafenone basis; re-curate the candidate list with an
explicit `Cu_peak/Km ≳ 0.5` feasibility filter before resuming.
