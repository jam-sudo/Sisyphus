# Phase 2: SMILES → PK + Uncertainty Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development

**Goal:** End-to-end SMILES → PK prediction with MC uncertainty. Holdout AAFE ≤ 2.5.

**Architecture:** predict layer (SMILES → DrugOnGraph) + engine (graph ODE) + ml (direct Cmax XGBoost) + pk (endpoints) + pipeline (meta-learner combining engine+ML). MC uncertainty via N=1000 parameter samples.

**Tech Stack:** Python 3.10+, RDKit, XGBoost, scikit-learn, numpy, scipy

**Acceptance:** Holdout AAFE ≤ 2.5, 90% PI coverage ≥ 80%, deterministic ≤ 2s

---

## Task 1: Data Migration from Omega

Migrate reference data and pre-trained XGBoost models from Omega.

**Files to create:**
- `data/reference/clinical_pk.json` — copy from Omega (290 drugs with Cmax/AUC/t½)
- `data/reference/holdout.json` — copy from Omega (76 train + 100 holdout)
- `data/reference/adme_reference.csv` — copy from Omega (153 compounds)
- `models/direct_pk/xgboost_cmax_v2.json` + `meta_v2.json`
- `models/meta_learner/xgboost_meta.json`
- `models/adme/xgboost_clint.json`
- `models/adme/xgboost_fup.json`

## Task 2: Feature Engineering (ml/features.py)

Morgan fingerprints (2048 bits, radius=2) + 9 RDKit descriptors → 2057-element feature vector.

## Task 3: Chemistry Module (predict/chemistry.py)

SMILES → MolecularProfile (mw, logP, pKa, TPSA, HBA/HBD, compound_type, AD check).

## Task 4: ADME Prediction (predict/adme.py)

Load XGBoost models, predict fup, CLint, Peff, solubility, RBP. All outputs as Distribution.

## Task 5: IVIVE + Kp (predict/ivive.py)

CLint → per-enzyme affinity decomposition. Berezhkovskiy/R&R Kp calculation. Build DrugOnGraph.

## Task 6: Direct Cmax Predictor (ml/models.py)

XGBoost v2 Cmax model wrapper. SMILES → Cmax prediction.

## Task 7: Meta-Learner (ml/ensemble.py)

Combine engine Cmax + ML Cmax → final Cmax. 12-feature XGBoost.

## Task 8: MC Uncertainty (engine/uncertainty.py)

Implement UncertaintyEngine.propagate(). N=1000 samples.

## Task 9: Pipeline + Validation (pipeline/ + validation/)

Orchestrator + reference loader + AAFE metrics + benchmark runner.

## Task 10: CLI + Holdout Benchmark

Entry point + run holdout AAFE, target ≤ 2.5.
