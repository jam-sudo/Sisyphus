# PGx genotype-nonlinearity two-arm validation — results (2026-06-17)

**Spec:** `docs/superpowers/specs/2026-06-16-pgx-genotype-nonlinearity-two-arm-design.md`
**Plan:** `docs/superpowers/plans/2026-06-16-pgx-genotype-nonlinearity-two-arm.md`
**Harness:** `scripts/validate_pgx_cmax_v2b.py` · **Tests:** `tests/integration/test_pgx_arm_sign_mechanism.py` (10 passed, 1 xfail)
**Isolation:** no `predict()` / `reference_man.yaml` / holdout change. Headline Meta AAFE **2.731 bit-identical** (holdout cache byte-unchanged + `test_mm_headline_bit_identity` + `test_cached_holdout_aafe_is_2p731` pass after the harness runs). Built on the merged (A) axial phenotype fix (PR #79).

## Verdict: first-pass arm SHIPS; systemic arm DEAD-ENDS (DE-49)

A capability-existence study (per the user's scope call): does the v2.2a saturable MM engine reproduce genotype-stratified nonlinear dose-dependence a linear model cannot? Two ultrathink passes pre-corrected the metric to the **log–log slope `β = d log(exposure)/d log(dose)`** per genotype and the **cross-term `Δβ = β_PM − β_EM`** (the linear-null gives `β≡1`, `Δβ=0` by construction).

### ✅ First-pass arm (axial `parallel_tube` liver) — robust, ships

| signal | result | meaning |
|---|---|---|
| `Δβ` (CYP2D6, single-dose axial) | **≈ −0.32** (β_EM≈1.34, β_PM≈1.02) | EM-nonlinear / PM-linear → fold **converges**. Robust across the Km×cltot box (PM≈0 activity pins `β_PM≡1`). |
| hepatic-inlet `C_u` | axial > `well_stirred` | the axial liver resolves the high inlet conc that `well_stirred` averages away — the engagement mechanism. |
| propafenone EM P1 vs Siddoway 1987 | engine `β_EM > 1` (supra-proportional); linear-null `β=1` | reproduces the **direction** of Siddoway's ~10× concentration rise over a 3× dose increase (β_obs≈2.1). Magnitude not claimed (Km/`fup` uncertain). |

The first-pass arm is the piece tied to the axial unlock (the (A)-fix payoff) and is corroborated by the one citable clinical saturation signature.

### ❌ Systemic arm (`well_stirred` steady-state) — two independent dead-ends

1. **Data-availability HALT (§4.1).** No citable ≥2-dose genotype-stratified exposure grid exists. Phenytoin: dose-*requirement* at constant Css only (van der Weide 2001, PMID 11434505) or paywalled per-genotype Vmax (Mamiya 1998, PMID 9860067); the `*3/*3` activity fraction was **not found and not invented**. Propafenone: clean EM/PM cross only at a single 300 mg dose (Tran 2022, PMC9324789).
2. **Sign non-invariance (refutes spec §5.2).** A synthetic Km×clearance sweep (fixed cltot/kp/PM-activity/doses, varying only Km):

| Km (mg/L) | regime | `Δβ` (systemic) | sign |
|---|---|---|---|
| 2.0 | deep saturation | **< 0** | convergence |
| 5.0 | moderate | ~ +0.4 (pocket) | divergence |
| 10.0 | mild saturation | **> 0** | divergence |

The systemic `Δβ` sign **flips with Km**; "systemic always diverges" is false. Worse, the phenytoin-relevant *low-Km* regime gives the **wrong sign** vs phenytoin's clinical divergence — untestable here because the data HALTed.

### ⚠ Oracle correction
The `1/(1−fm)` oral-AUC genotype-fold identity is **well_stirred-only** (fold=10.0 at fm=0.9, matches). The axial `parallel_tube` liver structurally deviates (≈3.98 vs 5.0 at fm=0.8) because `E=1−exp(−fu·CLint/Q)` does not telescope to the closed form — a real topology property, shipped as a strict `xfail`. Spec §5.4 corrected.

## Data-quality catches (literature audit)
- propafenone `fup` ≈ **0.10** (≈90% protein-bound), **not** the 0.30 first assumed — corrected in the P1 test.
- Kroemer 1989 (5.3 µM) / Hemeryck (0.12 µM) propafenone `Km` could **not** be confirmed in accessible sources; the 40× span is retained in the box probe.

## Discipline
No fold magnitudes fit; no genotype-activity fraction invented; the crossed-grid HALT recorded honestly in `data/validation/pgx_genotype_nonlinearity_folds.json`. Headline 2.731 untouched throughout. Full negative logged as **DE-49** (`docs/claude/dead-ends.md`).
