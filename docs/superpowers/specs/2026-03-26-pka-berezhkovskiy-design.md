# pKa Prediction Model + Berezhkovskiy Correction

**Date:** 2026-03-26
**Status:** Approved
**Goal:** Improve engine AAFE (2.945) by fixing the two highest-leverage mechanistic inputs: pKa accuracy and Kp protein-binding correction.

## Context

Engine-only ablation revealed DrugBank enrichment improves engine AAFE (3.074 → 2.945, Δ=-0.129). The meta-learner attenuates this to Δ=-0.021, confirming Ceiling 3. However, LOOCV shows w_other=0.00 is optimal because engine (2.945) is still worse than ML (2.206).

To make the engine competitive with ML, we need engine AAFE closer to 2.2. Two high-leverage interventions identified:

1. **pKa model**: Current defaults (acid=4.5, base=9.0) have ±2-3 unit error. Kp depends exponentially on pKa via `10^(pH - pKa)`. 1 unit pKa error → 10x ionization ratio error → cascading Kp error across all 16 tissues.

2. **Berezhkovskiy correction**: Already implemented in `ivive.py:416-432` but not activated. Corrects Kp overestimation for highly protein-bound drugs (fup < 0.05).

## Step 1: pKa Prediction Model

### Training Data

- Source: DrugBank `data/drugbank/drugs.csv`
- Fields: `pka_acidic` (9,974 values), `pka_basic` (10,987 values), `canonical_smiles`
- Values: ChemAxon calculated (not experimental, but MAE ~0.5 vs experimental)
- Holdout exclusion: 61 holdout drugs removed by InChIKey matching
- Expected training set: ~9,500-10,500 per model (after holdout + invalid SMILES removal)

### Models

Two independent XGBoost regressors:
- `pka_acidic_model`: SMILES → strongest acidic pKa
  - Train only on molecules that HAVE an acidic pKa value
  - Target: raw pKa float (no log transform needed, range ~0-14)
- `pka_basic_model`: SMILES → strongest basic pKa
  - Train only on molecules that HAVE a basic pKa value

Features: Morgan fingerprint (2048 bits, radius 2) + 9 RDKit descriptors (same as existing ADME models, via `descriptors.py:compute_features()`).

Hyperparameters: XGBoost defaults with `n_estimators=500, max_depth=6, learning_rate=0.1, subsample=0.8`. Same as existing fup/CLint models for consistency.

Validation: 5-fold Murcko scaffold split CV. Report MAE and R² per fold.

### compound_type Classification

When pKa model provides predictions (DrugBank lookup misses):
```
if acidic_pka < 7.0 and basic_pka > 7.0:  → "zwitterion", pka = acidic_pka
elif acidic_pka < 7.0:                     → "acid", pka = acidic_pka
elif basic_pka > 7.0:                      → "base", pka = basic_pka
else:                                      → "neutral", pka = None
```

This replaces the current SMARTS-based heuristic for molecules not in DrugBank.

### Integration (chemistry.py)

3-tier fallback:
1. DrugBank ChemAxon pKa (exact match) — highest quality
2. XGBoost pKa model prediction — trained on ChemAxon values
3. Default values (acid=4.5, base=9.0) — only if model fails to load

Lazy singleton pattern for model loading (same as existing `_model_cache` in adme.py).

### Files

- `scripts/train_pka_model.py` — training script with holdout exclusion + CV reporting
- `models/adme/xgboost_pka_acidic.json` — trained model (gitignored)
- `models/adme/xgboost_pka_basic.json` — trained model (gitignored)
- `src/sisyphus/predict/chemistry.py` — add model loading + fallback path
- `tests/unit/test_chemistry.py` — test pKa model fallback + compound_type classification

## Step 2: Berezhkovskiy Correction Activation

### Change

In `build_drug_on_graph()` (ivive.py:558), change default kp_method:
```python
kp_method: str = "rodgers_rowland"  →  kp_method: str = "berezhkovskiy"
```

The implementation already exists:
- `_apply_bz_correction(kp, fup)` at ivive.py:416-432
- `_compute_all_kp()` handles `kp_method="berezhkovskiy"` at ivive.py:465-474
- fup is already passed through at ivive.py:600-603

### Effect

For a drug with fup=0.01, Kp=100:
- R&R: Kp = 100
- Berezhkovskiy: Kp = 100 / (1 + 99 × 0.01) = 50.3

For a drug with fup=0.5, Kp=10:
- R&R: Kp = 10
- Berezhkovskiy: Kp = 10 / (1 + 9 × 0.5) = 1.8

Highly bound drugs (warfarin fup=0.01, amiodarone fup=0.04) are most affected.

### Files

- `src/sisyphus/predict/ivive.py` — one-line default change
- Existing tests should still pass (Berezhkovskiy is a superset of R&R)

## Step 3: Evaluation Protocol

### Experiments

All measured via engine-only benchmark (`scripts/run_engine_ablation.py` or variant):

| Experiment | pKa Model | Berezhkovskiy | Expected |
|-----------|-----------|---------------|----------|
| Baseline | OFF (defaults) | OFF (R&R) | 2.945 (known) |
| Exp 1 | ON | OFF | pKa isolated effect |
| Exp 2 | OFF | ON | BZ isolated effect |
| Exp 3 | ON | ON | Combined effect |

### Success Criteria

**Primary (pKa model quality):**
- 5-fold CV MAE < 1.5 pKa units (current defaults: MAE ~2.5 units)
- 5-fold CV R² > 0.5

**Secondary (downstream engine impact):**
- Engine AAFE Δ > 0.1 in any experiment vs baseline
- If Exp 3 Engine AAFE < 2.7: strong signal, proceed to meta-learner re-evaluation

**Failure criteria:**
- pKa CV MAE > 2.0: model not better than structural heuristics
- All experiments Δ ≤ 0.05: pKa/BZ not actual bottlenecks at engine level

## Invariant Compliance

| Rule | Status | Notes |
|------|--------|-------|
| engine/ untouched | ✅ | Only predict/ and ivive.py modified |
| DrugOnGraph fields unchanged | ✅ | pka, compound_type, kp_method are existing fields |
| Holdout not in training | ✅ | InChIKey exclusion in training script |
| No Cmax loss fudging | ✅ | pKa model trained on ChemAxon pKa, not Cmax |
| No drug-specific branches | ✅ | XGBoost is general SMILES → pKa function |
| Not in "failed attempts" list | ✅ | pKa model was never attempted |
| Performance ≤ 500ms | ✅ | XGBoost inference ~1ms overhead |
| 20 files per directory | ✅ | predict/ goes from 5 to 5 files (modification only) |
