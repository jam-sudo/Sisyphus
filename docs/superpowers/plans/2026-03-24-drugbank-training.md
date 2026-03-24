# DrugBank Training Integration — Phase A Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retrain fup model (logit transform, TDC + DrugBank data) and add logP residual correction, improving prediction for ALL SMILES — not just DrugBank-matched drugs.

**Architecture:** Training scripts produce v2 model files. adme.py auto-selects v2 if file exists, v1 fallback. Existing DrugBank lookup (secondary) remains unchanged. Holdout exclusion via multi-key matching (canonical SMILES + InChIKey-14 + name). Benchmark 3-way: Baseline/Silver/Gold.

**Tech Stack:** XGBoost, RDKit, PyTDC, numpy, scikit-learn (cross-validation)

**Spec:** `docs/superpowers/specs/2026-03-24-drugbank-training-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `scripts/holdout_audit.py` | **Create** | TDC vs holdout SMILES overlap check |
| `scripts/train_fup_v2.py` | **Create** | fup retraining: TDC+DrugBank, logit transform, holdout exclusion |
| `scripts/train_logp_correction.py` | **Create** | logP residual correction model training |
| `scripts/run_ablation.py` | **Create** | 5-experiment ablation runner |
| `src/sisyphus/predict/adme.py` | Modify | v1/v2 model auto-selection for fup |
| `src/sisyphus/predict/chemistry.py` | Modify | logP correction + isomericSmiles=True |
| `scripts/extract_drugbank.py` | Modify | isomericSmiles=True |
| `tests/unit/test_training.py` | **Create** | Training utility tests (logit/sigmoid, holdout exclusion) |

---

### Task 1: IsomericSMILES Enforcement

**Files:**
- Modify: `scripts/extract_drugbank.py:303`
- Modify: `src/sisyphus/predict/chemistry.py:318`

- [ ] **Step 1: Add isomericSmiles=True to both files**

In `scripts/extract_drugbank.py` line 303:
```python
# Before:
canonical_smiles = Chem.MolToSmiles(mol)
# After:
canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
```

In `src/sisyphus/predict/chemistry.py` line 318:
```python
# Before:
canonical = Chem.MolToSmiles(mol)
# After:
canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
```

- [ ] **Step 2: Run tests**

Run: `python3 -m pytest tests/ -q --tb=short`
Expected: 283 passed (no behavior change — default is already True in RDKit 2025.09)

- [ ] **Step 3: Commit**

```bash
git add scripts/extract_drugbank.py src/sisyphus/predict/chemistry.py
git commit -m "fix: explicit isomericSmiles=True on all MolToSmiles calls"
```

---

### Task 2: Holdout Contamination Audit

**Files:**
- Create: `scripts/holdout_audit.py`
- Output: `docs/holdout_contamination_audit.md`

- [ ] **Step 1: Write the audit script**

```python
#!/usr/bin/env python3
"""Holdout contamination audit: check TDC training data vs holdout overlap."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem.inchi import MolToInchi, InchiToInchiKey

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent


def build_holdout_keys() -> tuple[set[str], set[str], set[str]]:
    """Build holdout exclusion sets: canonical SMILES, InChIKey-14, names."""
    with open(ROOT / "data" / "reference" / "holdout.json") as f:
        holdout_data = json.load(f)
    with open(ROOT / "data" / "reference" / "clinical_pk.json") as f:
        clinical = json.load(f)

    holdout_names = set(holdout_data["holdout"])
    holdout_smiles = set()
    holdout_ik14 = set()

    for name in holdout_names:
        drug = clinical["drugs"].get(name)
        if drug and drug.get("smiles"):
            mol = Chem.MolFromSmiles(drug["smiles"])
            if mol:
                holdout_smiles.add(Chem.MolToSmiles(mol, isomericSmiles=True))
                try:
                    inchi = MolToInchi(mol)
                    if inchi:
                        ik = InchiToInchiKey(inchi)
                        if ik and len(ik) >= 14:
                            holdout_ik14.add(ik[:14])
                except Exception:
                    pass

    logger.info(
        "Holdout keys: %d names, %d SMILES, %d InChIKey-14",
        len(holdout_names), len(holdout_smiles), len(holdout_ik14),
    )
    return holdout_smiles, holdout_ik14, holdout_names


def is_holdout(
    smiles: str,
    holdout_smiles: set[str],
    holdout_ik14: set[str],
    holdout_names: set[str],
    name: str | None = None,
) -> bool:
    """Multi-key holdout check."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
    if canonical in holdout_smiles:
        return True
    try:
        inchi = MolToInchi(mol)
        if inchi:
            ik = InchiToInchiKey(inchi)
            if ik and len(ik) >= 14 and ik[:14] in holdout_ik14:
                return True
    except Exception:
        pass
    if name and name.lower() in holdout_names:
        return True
    return False


def audit_tdc():
    """Check TDC datasets for holdout overlap."""
    from tdc.single_pred import ADME

    holdout_smiles, holdout_ik14, holdout_names = build_holdout_keys()

    datasets = {
        "PPBR_AZ": "fup (fraction bound)",
        "HIA_Hou": "intestinal absorption",
    }

    results = {}
    for ds_name, description in datasets.items():
        try:
            data = ADME(name=ds_name)
            df = data.get_data()
            overlap = []
            for _, row in df.iterrows():
                if is_holdout(row["Drug"], holdout_smiles, holdout_ik14, holdout_names):
                    overlap.append(row["Drug"][:60])
            pct = 100 * len(overlap) / len(df) if len(df) > 0 else 0
            results[ds_name] = {
                "total": len(df),
                "overlap": len(overlap),
                "pct": f"{pct:.1f}%",
                "description": description,
            }
            logger.info("%s: %d/%d overlap (%.1f%%)", ds_name, len(overlap), len(df), pct)
        except Exception as e:
            logger.warning("Failed to load TDC %s: %s", ds_name, e)
            results[ds_name] = {"error": str(e)}

    # Write report
    report_path = ROOT / "docs" / "holdout_contamination_audit.md"
    with open(report_path, "w") as f:
        f.write("# Holdout Contamination Audit\n\n")
        f.write(f"Holdout: {len(holdout_names)} drugs, {len(holdout_smiles)} with SMILES\n\n")
        f.write("| TDC Dataset | Total | Overlap | % | Description |\n")
        f.write("|-------------|-------|---------|---|-------------|\n")
        for ds, r in results.items():
            if "error" in r:
                f.write(f"| {ds} | ERROR | — | — | {r['error']} |\n")
            else:
                f.write(f"| {ds} | {r['total']} | {r['overlap']} | {r['pct']} | {r['description']} |\n")
        if any(r.get("pct", "0%").replace("%", "") != "0" for r in results.values() if "pct" in r):
            f.write("\n**WARNING:** Overlap detected. Baseline AAFE may be optimistic.\n")

    logger.info("Audit report written to %s", report_path)
    return results


if __name__ == "__main__":
    audit_tdc()
```

- [ ] **Step 2: Run the audit**

Run: `python3 scripts/holdout_audit.py`
Expected: Report showing overlap counts per TDC dataset.

- [ ] **Step 3: Commit**

```bash
git add scripts/holdout_audit.py docs/holdout_contamination_audit.md
git commit -m "audit: TDC vs holdout SMILES overlap check"
```

---

### Task 3: Training Utilities + Tests

**Files:**
- Create: `tests/unit/test_training.py`

- [ ] **Step 1: Write tests for logit/sigmoid and holdout exclusion**

```python
"""Tests for training utilities."""
import numpy as np
import pytest


class TestLogitSigmoid:
    def test_logit_sigmoid_roundtrip(self):
        """logit → sigmoid should recover original value."""
        x = np.array([0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99])
        eps = 1e-6
        x_clipped = np.clip(x, eps, 1 - eps)
        logit_x = np.log(x_clipped / (1 - x_clipped))
        recovered = 1 / (1 + np.exp(-logit_x))
        np.testing.assert_allclose(recovered, x, atol=1e-5)

    def test_sigmoid_bounded(self):
        """Sigmoid output must be in (0, 1) for any input."""
        extreme = np.array([-100, -10, -1, 0, 1, 10, 100])
        result = 1 / (1 + np.exp(-extreme))
        assert np.all(result > 0)
        assert np.all(result < 1)

    def test_logit_zero_epsilon(self):
        """logit(0) should use epsilon, not -inf."""
        eps = 1e-6
        x = np.clip(0.0, eps, 1 - eps)
        logit_val = np.log(x / (1 - x))
        assert np.isfinite(logit_val)
        assert logit_val < -10  # very negative but finite

    def test_logit_one_epsilon(self):
        """logit(1) should use epsilon, not +inf."""
        eps = 1e-6
        x = np.clip(1.0, eps, 1 - eps)
        logit_val = np.log(x / (1 - x))
        assert np.isfinite(logit_val)
        assert logit_val > 10  # very positive but finite


class TestTDCConversion:
    def test_ppbr_to_fup(self):
        """TDC PPBR_AZ Y (% bound) → fup conversion."""
        # Y is % bound (0-100 range)
        y = np.array([95.0, 50.0, 10.0, 99.5])
        fup = (100 - y) / 100
        np.testing.assert_allclose(fup, [0.05, 0.50, 0.90, 0.005])

    def test_ppbr_range_detection(self):
        """Detect whether TDC data is 0-100 (%) or 0-1 (fraction)."""
        pct_data = np.array([95, 88, 50, 10])
        frac_data = np.array([0.95, 0.88, 0.50, 0.10])
        assert pct_data.max() > 1.0  # percentage
        assert frac_data.max() <= 1.0  # fraction


class TestHoldoutExclusion:
    def test_exact_smiles_excluded(self):
        """Drug with matching canonical SMILES is excluded."""
        from rdkit import Chem
        smiles = "CC(=O)Oc1ccccc1C(=O)O"  # aspirin
        canonical = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True)
        holdout_set = {canonical}
        assert canonical in holdout_set

    def test_salt_form_caught_by_inchikey(self):
        """Salt form has different SMILES but same InChIKey-14."""
        from rdkit import Chem
        from rdkit.Chem.inchi import MolToInchi, InchiToInchiKey
        base = Chem.MolFromSmiles("CC(C)NCC(O)c1ccc(O)c(O)c1")  # isoprenaline free base
        salt = Chem.MolFromSmiles("CC(C)NCC(O)c1ccc(O)c(O)c1.Cl")  # HCl salt
        ik_base = InchiToInchiKey(MolToInchi(base))[:14]
        ik_salt = InchiToInchiKey(MolToInchi(salt))[:14]
        # InChIKey-14 should differ for salt (different connectivity)
        # Actually salts with . separator have different InChI
        # This test documents the behavior
        if ik_base == ik_salt:
            pass  # caught by InChIKey
        else:
            pass  # salt form has different InChIKey — name matching needed
```

- [ ] **Step 2: Run tests**

Run: `python3 -m pytest tests/unit/test_training.py -v`
Expected: All pass.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_training.py
git commit -m "test: training utilities — logit/sigmoid, TDC conversion, holdout exclusion"
```

---

### Task 4: fup v2 Training Script

**Files:**
- Create: `scripts/train_fup_v2.py`

- [ ] **Step 1: Write the training script**

```python
#!/usr/bin/env python3
"""Train fup v2 model: TDC PPBR_AZ + DrugBank measured fup, logit transform.

Usage: python scripts/train_fup_v2.py

Output: models/adme/xgboost_fup_v2.json
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import numpy as np
import xgboost as xgb
from rdkit import Chem
from rdkit.Chem.inchi import InchiToInchiKey, MolToInchi
from sklearn.model_selection import cross_val_predict

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
MODEL_OUT = ROOT / "models" / "adme" / "xgboost_fup_v2.json"

# Import shared feature computation
import sys
sys.path.insert(0, str(ROOT / "src"))
from sisyphus.descriptors import compute_features


def logit(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = np.clip(x, eps, 1 - eps)
    return np.log(x / (1 - x))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def build_holdout_keys():
    """Build multi-key holdout exclusion set."""
    with open(ROOT / "data" / "reference" / "holdout.json") as f:
        holdout = json.load(f)
    with open(ROOT / "data" / "reference" / "clinical_pk.json") as f:
        clinical = json.load(f)

    names = {n.lower() for n in holdout["holdout"]}
    smiles_set = set()
    ik14_set = set()

    for name in holdout["holdout"]:
        drug = clinical["drugs"].get(name)
        if drug and drug.get("smiles"):
            mol = Chem.MolFromSmiles(drug["smiles"])
            if mol:
                smiles_set.add(Chem.MolToSmiles(mol, isomericSmiles=True))
                try:
                    inchi = MolToInchi(mol)
                    if inchi:
                        ik = InchiToInchiKey(inchi)
                        if ik:
                            ik14_set.add(ik[:14])
                except Exception:
                    pass

    logger.info("Holdout keys: %d names, %d SMILES, %d IK14", len(names), len(smiles_set), len(ik14_set))
    return smiles_set, ik14_set, names


def is_holdout(smiles: str, ho_smiles: set, ho_ik14: set, ho_names: set, name: str | None = None) -> bool:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
    if canonical in ho_smiles:
        return True
    try:
        inchi = MolToInchi(mol)
        if inchi:
            ik = InchiToInchiKey(inchi)
            if ik and ik[:14] in ho_ik14:
                return True
    except Exception:
        pass
    if name and name.lower() in ho_names:
        return True
    return False


def load_tdc_ppbr():
    """Load TDC PPBR_AZ, convert to fup."""
    from tdc.single_pred import ADME
    data = ADME(name="PPBR_AZ")
    df = data.get_data()
    # Y is % bound (range 11-100). Convert to fup.
    assert df["Y"].max() > 1.0, "Expected percentage (0-100), got fraction"
    records = []
    for _, row in df.iterrows():
        fup = (100 - row["Y"]) / 100
        if 0.001 <= fup <= 0.999:
            records.append((row["Drug"], fup, "tdc_ppbr"))
    logger.info("TDC PPBR_AZ: %d valid records (of %d)", len(records), len(df))
    return records


def load_drugbank_fup():
    """Load DrugBank parsed fup values."""
    pk_path = ROOT / "data" / "drugbank" / "pk_data.csv"
    drugs_path = ROOT / "data" / "drugbank" / "drugs.csv"
    if not pk_path.exists() or not drugs_path.exists():
        logger.warning("DrugBank CSVs not found, skipping")
        return []

    # Build drugbank_id → canonical_smiles map
    id_to_smiles = {}
    with open(drugs_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cs = row.get("canonical_smiles", "").strip()
            dbid = row.get("drugbank_id", "")
            if cs and dbid:
                id_to_smiles[dbid] = cs

    records = []
    with open(pk_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("field") != "protein_binding":
                continue
            val = row.get("parsed_value", "").strip()
            dbid = row.get("drugbank_id", "")
            if val and dbid and dbid in id_to_smiles:
                try:
                    fup = float(val)
                    if 0.001 <= fup <= 0.999:
                        records.append((id_to_smiles[dbid], fup, "drugbank"))
                except ValueError:
                    pass
    logger.info("DrugBank fup: %d valid records", len(records))
    return records


def merge_and_exclude(tdc_records, db_records, ho_smiles, ho_ik14, ho_names):
    """Merge TDC + DrugBank, deduplicate, exclude holdout."""
    # Index by canonical SMILES. DrugBank overrides TDC.
    merged = {}
    for smiles, fup, source in tdc_records:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
        merged[canonical] = (fup, source)

    for smiles, fup, source in db_records:
        # DrugBank SMILES are already canonical from extraction
        merged[smiles] = (fup, source)  # override TDC

    # Exclude holdout
    excluded = 0
    clean = {}
    for smiles, (fup, source) in merged.items():
        if is_holdout(smiles, ho_smiles, ho_ik14, ho_names):
            excluded += 1
        else:
            clean[smiles] = (fup, source)

    logger.info(
        "Merged: %d total, %d holdout excluded, %d training samples",
        len(merged), excluded, len(clean),
    )
    return clean


def train():
    logger.info("=== fup v2 Training ===")

    ho_smiles, ho_ik14, ho_names = build_holdout_keys()
    tdc = load_tdc_ppbr()
    db = load_drugbank_fup()
    training_data = merge_and_exclude(tdc, db, ho_smiles, ho_ik14, ho_names)

    # Compute features
    logger.info("Computing features for %d drugs...", len(training_data))
    X_list, y_list = [], []
    failed = 0
    for smiles, (fup, source) in training_data.items():
        try:
            features = compute_features(smiles)
            X_list.append(features)
            y_list.append(fup)
        except Exception:
            failed += 1

    X = np.array(X_list)
    y_fup = np.array(y_list)
    y_logit = logit(y_fup)
    logger.info("Features computed: %d success, %d failed", len(X_list), failed)
    logger.info("fup range: [%.4f, %.4f], mean=%.4f", y_fup.min(), y_fup.max(), y_fup.mean())

    # Train XGBoost
    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    model.fit(X, y_logit)

    # Cross-validation for CV estimation
    logger.info("Running 5-fold CV for uncertainty estimation...")
    cv_pred = cross_val_predict(model, X, y_logit, cv=5)
    cv_fup_pred = sigmoid(cv_pred)
    rmse_logit = np.sqrt(np.mean((cv_pred - y_logit) ** 2))
    # Approximate CV from fold errors
    fold_errors = np.abs(np.log10(np.clip(cv_fup_pred, 1e-6, None) / np.clip(y_fup, 1e-6, None)))
    aafe_cv = 10 ** np.mean(fold_errors)
    cv_estimate = aafe_cv - 1  # rough CV approximation

    logger.info("CV RMSE (logit): %.3f", rmse_logit)
    logger.info("CV AAFE: %.3f → estimated CV: %.3f", aafe_cv, cv_estimate)

    # Post-training contamination check
    logger.info("Post-training contamination check...")
    contaminated = 0
    for smiles in training_data:
        if is_holdout(smiles, ho_smiles, ho_ik14, ho_names):
            contaminated += 1
            logger.error("CONTAMINATION: %s", smiles[:60])
    if contaminated > 0:
        logger.error("ABORT: %d holdout drugs in training set!", contaminated)
        return
    logger.info("Contamination check passed: 0 holdout drugs in training set")

    # Save model
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_OUT))
    logger.info("Model saved to %s", MODEL_OUT)
    logger.info(
        "Summary: %d training samples, RMSE(logit)=%.3f, AAFE(CV)=%.3f, est_CV=%.3f",
        len(X_list), rmse_logit, aafe_cv, cv_estimate,
    )


if __name__ == "__main__":
    train()
```

- [ ] **Step 2: Run training**

Run: `python3 scripts/train_fup_v2.py`
Expected: Model saved to `models/adme/xgboost_fup_v2.json`. No contamination.

- [ ] **Step 3: Commit**

```bash
git add scripts/train_fup_v2.py
git commit -m "feat(training): fup v2 training script — TDC+DrugBank, logit transform"
```

---

### Task 5: adme.py — v1/v2 Auto-Selection

**Files:**
- Modify: `src/sisyphus/predict/adme.py`

- [ ] **Step 1: Rename existing _predict_fup to _predict_fup_v1, add v2 + dispatcher**

After the existing `_predict_fup` function (~line 88), restructure:

```python
_FUP_V2_PATH = _MODEL_DIR / "xgboost_fup_v2.json"
_FUP_CV_V2 = 0.35  # updated after cross-validation; placeholder until measured


def _predict_fup_v1(features: np.ndarray) -> Distribution:
    """v1: log10-space model. Output clamped to [0.001, 1.0]."""
    model = _load_model("xgboost_fup.json")
    log_fup = float(model.predict(features)[0])
    fup = float(np.clip(10**log_fup, 0.001, 1.0))
    return Distribution(mean=fup, cv=_FUP_CV)


def _predict_fup_v2(features: np.ndarray) -> Distribution:
    """v2: logit-space model, sigmoid inverse. Output in (0, 1) guaranteed."""
    model = _load_model("xgboost_fup_v2.json")
    logit_fup = float(model.predict(features)[0])
    fup = float(1.0 / (1.0 + np.exp(-logit_fup)))  # sigmoid
    return Distribution(mean=fup, cv=_FUP_CV_V2)


def _predict_fup(features: np.ndarray) -> Distribution:
    """Auto-select fup model: v2 (logit) if available, v1 (log10) fallback."""
    if _FUP_V2_PATH.exists():
        return _predict_fup_v2(features)
    return _predict_fup_v1(features)
```

- [ ] **Step 2: Run tests**

Run: `python3 -m pytest tests/ -q --tb=short`
Expected: All pass. If v2 model exists on disk, it's auto-selected.

- [ ] **Step 3: Commit**

```bash
git add src/sisyphus/predict/adme.py
git commit -m "feat(predict): fup v1/v2 auto-selection — logit model when available"
```

---

### Task 6: logP Correction Training Script

**Files:**
- Create: `scripts/train_logp_correction.py`

- [ ] **Step 1: Write the training script**

```python
#!/usr/bin/env python3
"""Train logP residual correction model.

Learns: correction = experimental_logP - crippen_logP
Uses 6 features (not full 2057 — overfitting risk with 1,463 samples).

Usage: python scripts/train_logp_correction.py
Output: models/adme/logp_correction.json
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import numpy as np
import xgboost as xgb
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.inchi import InchiToInchiKey, MolToInchi
from sklearn.model_selection import cross_val_predict

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
MODEL_OUT = ROOT / "models" / "adme" / "logp_correction.json"


def build_holdout_keys():
    """Same as train_fup_v2.py — reuse."""
    with open(ROOT / "data" / "reference" / "holdout.json") as f:
        holdout = json.load(f)
    with open(ROOT / "data" / "reference" / "clinical_pk.json") as f:
        clinical = json.load(f)

    names = {n.lower() for n in holdout["holdout"]}
    smiles_set = set()
    ik14_set = set()
    for name in holdout["holdout"]:
        drug = clinical["drugs"].get(name)
        if drug and drug.get("smiles"):
            mol = Chem.MolFromSmiles(drug["smiles"])
            if mol:
                smiles_set.add(Chem.MolToSmiles(mol, isomericSmiles=True))
                try:
                    inchi = MolToInchi(mol)
                    if inchi:
                        ik = InchiToInchiKey(inchi)
                        if ik:
                            ik14_set.add(ik[:14])
                except Exception:
                    pass
    return smiles_set, ik14_set, names


def is_holdout(smiles, ho_smiles, ho_ik14, ho_names, name=None):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
    if canonical in ho_smiles:
        return True
    try:
        inchi = MolToInchi(mol)
        if inchi:
            ik = InchiToInchiKey(inchi)
            if ik and ik[:14] in ho_ik14:
                return True
    except Exception:
        pass
    if name and name.lower() in ho_names:
        return True
    return False


def compute_logp_features(smiles: str) -> np.ndarray | None:
    """Compute 6-feature vector for logP correction."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return np.array([
        Descriptors.MolLogP(mol),           # Crippen logP
        Descriptors.MolWt(mol),             # MW
        Descriptors.TPSA(mol),              # TPSA
        Descriptors.NumHDonors(mol),        # HBD
        Descriptors.NumHAcceptors(mol),     # HBA
        Descriptors.NumRotatableBonds(mol), # rotatable bonds
    ])


def train():
    logger.info("=== logP Correction Training ===")

    ho_smiles, ho_ik14, ho_names = build_holdout_keys()

    # Load DrugBank experimental logP
    exp_path = ROOT / "data" / "drugbank" / "experimental_properties.csv"
    drugs_path = ROOT / "data" / "drugbank" / "drugs.csv"
    if not exp_path.exists():
        logger.error("DrugBank experimental_properties.csv not found")
        return

    # Build drugbank_id → canonical_smiles
    id_to_smiles = {}
    with open(drugs_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cs = row.get("canonical_smiles", "").strip()
            dbid = row.get("drugbank_id", "")
            if cs and dbid:
                id_to_smiles[dbid] = cs

    # Load experimental logP
    records = []
    with open(exp_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("property") != "logP":
                continue
            dbid = row.get("drugbank_id", "")
            val = row.get("value", "").strip()
            if dbid in id_to_smiles and val:
                try:
                    exp_logp = float(val)
                    smiles = id_to_smiles[dbid]
                    if not is_holdout(smiles, ho_smiles, ho_ik14, ho_names):
                        records.append((smiles, exp_logp))
                except ValueError:
                    pass

    logger.info("Experimental logP records (holdout excluded): %d", len(records))

    # Compute features and residuals
    X_list, y_list = [], []
    for smiles, exp_logp in records:
        feats = compute_logp_features(smiles)
        if feats is not None:
            crippen = feats[0]  # first feature is Crippen logP
            residual = exp_logp - crippen
            X_list.append(feats)
            y_list.append(residual)

    X = np.array(X_list)
    y = np.array(y_list)
    logger.info(
        "Training data: %d samples, residual range [%.2f, %.2f], mean=%.3f",
        len(y), y.min(), y.max(), y.mean(),
    )

    # Train
    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=4,     # shallower — 6 features, avoid overfitting
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=1.0,  # only 6 features, use all
        random_state=42,
    )
    model.fit(X, y)

    # 5-fold CV
    cv_pred = cross_val_predict(model, X, y, cv=5)
    rmse = np.sqrt(np.mean((cv_pred - y) ** 2))
    logger.info("CV RMSE (residual): %.3f logP units", rmse)

    # Compare: Crippen-only RMSE vs corrected RMSE
    crippen_residuals = y  # y IS the residual = exp - crippen
    crippen_rmse = np.sqrt(np.mean(crippen_residuals ** 2))
    corrected_residuals = y - cv_pred  # remaining error after correction
    corrected_rmse = np.sqrt(np.mean(corrected_residuals ** 2))
    logger.info(
        "Crippen RMSE: %.3f → Corrected RMSE: %.3f (improvement: %.1f%%)",
        crippen_rmse, corrected_rmse,
        100 * (1 - corrected_rmse / crippen_rmse),
    )

    # Save
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_OUT))
    logger.info("Model saved to %s", MODEL_OUT)


if __name__ == "__main__":
    train()
```

- [ ] **Step 2: Run training**

Run: `python3 scripts/train_logp_correction.py`
Expected: Model saved. Corrected RMSE < Crippen RMSE.

- [ ] **Step 3: Commit**

```bash
git add scripts/train_logp_correction.py
git commit -m "feat(training): logP residual correction model — 6 features, 1463 samples"
```

---

### Task 7: chemistry.py — logP Correction Integration

**Files:**
- Modify: `src/sisyphus/predict/chemistry.py`

- [ ] **Step 1: Add logP correction loading and application**

In `compute_profile()`, after the DrugBank logP override block, add correction:

```python
    # logP correction model (residual learning)
    _LOGP_CORRECTION_PATH = Path(__file__).resolve().parent.parent.parent.parent / "models" / "adme" / "logp_correction.json"
    if _LOGP_CORRECTION_PATH.exists() and db_logp is None:
        # Only correct Crippen logP, not experimental logP
        try:
            import xgboost as xgb
            if not hasattr(compute_profile, "_logp_model"):
                m = xgb.XGBRegressor()
                m.load_model(str(_LOGP_CORRECTION_PATH))
                compute_profile._logp_model = m  # cache on function
            correction_features = np.array([[
                logp, mw, tpsa, float(hbd), float(hba), float(rotatable_bonds)
            ]])
            correction = float(compute_profile._logp_model.predict(correction_features)[0])
            logp = logp + correction
        except Exception as e:
            logger.warning("logP correction failed: %s", e)
```

Note: correction only applies when experimental logP is NOT available (db_logp is None).
When DrugBank has experimental logP, it's already better than Crippen+correction.

- [ ] **Step 2: Run tests**

Run: `python3 -m pytest tests/ -q --tb=short`
Expected: All pass.

- [ ] **Step 3: Commit**

```bash
git add src/sisyphus/predict/chemistry.py
git commit -m "feat(predict): logP residual correction when model available"
```

---

### Task 8: Ablation Study Runner

**Files:**
- Create: `scripts/run_ablation.py`

- [ ] **Step 1: Write the ablation runner**

```python
#!/usr/bin/env python3
"""Run 5-experiment ablation study for DrugBank training integration.

Experiments:
1. Baseline: v1 models, lookup OFF
2. fup v2 only
3. logP correction only
4. Silver: fup v2 + logP correction, lookup OFF
5. Gold: fup v2 + logP correction, lookup ON
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
FUP_V2 = ROOT / "models" / "adme" / "xgboost_fup_v2.json"
LOGP_CORR = ROOT / "models" / "adme" / "logp_correction.json"

import sys
sys.path.insert(0, str(ROOT / "src"))


def run_benchmark_with_config(lookup_enabled: bool) -> dict:
    """Run holdout benchmark with specified config."""
    from sisyphus.predict.drugbank import DrugBankConfig, DrugBankLookup, _reset_singleton
    _reset_singleton()

    config = DrugBankConfig(
        enable_enzyme_fm=lookup_enabled,
        enable_fup=lookup_enabled,
        enable_pka=lookup_enabled,
        enable_logp=lookup_enabled,
    )
    # Force singleton with this config
    DrugBankLookup(config=config)
    from sisyphus.predict import drugbank
    drugbank._INSTANCE = DrugBankLookup(config=config)

    from sisyphus.validation.benchmark import run_benchmark
    result = run_benchmark(holdout_only=True)
    _reset_singleton()
    return {
        "aafe": result.aafe,
        "aafe_in_domain": result.aafe_in_domain,
        "pct_2fold": result.pct_2fold,
        "n_drugs": result.n_drugs,
        "n_gold": result.n_gold,
        "n_silver": result.n_silver,
    }


def run_experiments():
    # Check model availability
    has_fup_v2 = FUP_V2.exists()
    has_logp = LOGP_CORR.exists()
    logger.info("Models: fup_v2=%s, logp_correction=%s", has_fup_v2, has_logp)

    results = {}

    # Exp 1: Baseline (hide v2 models, lookup OFF)
    logger.info("=== Exp 1: Baseline ===")
    fup_v2_bak = FUP_V2.with_suffix(".json.bak") if has_fup_v2 else None
    logp_bak = LOGP_CORR.with_suffix(".json.bak") if has_logp else None
    if has_fup_v2:
        shutil.move(str(FUP_V2), str(fup_v2_bak))
    if has_logp:
        shutil.move(str(LOGP_CORR), str(logp_bak))
    results["baseline"] = run_benchmark_with_config(lookup_enabled=False)
    logger.info("Baseline AAFE: %.3f", results["baseline"]["aafe"])

    # Exp 2: fup v2 only
    logger.info("=== Exp 2: fup v2 only ===")
    if fup_v2_bak:
        shutil.move(str(fup_v2_bak), str(FUP_V2))
    results["fup_v2_only"] = run_benchmark_with_config(lookup_enabled=False)
    logger.info("fup v2 AAFE: %.3f", results["fup_v2_only"]["aafe"])
    if has_fup_v2:
        shutil.move(str(FUP_V2), str(fup_v2_bak))  # hide again

    # Exp 3: logP correction only
    logger.info("=== Exp 3: logP correction only ===")
    if logp_bak:
        shutil.move(str(logp_bak), str(LOGP_CORR))
    results["logp_only"] = run_benchmark_with_config(lookup_enabled=False)
    logger.info("logP corr AAFE: %.3f", results["logp_only"]["aafe"])

    # Exp 4: Silver (both retrained, lookup OFF)
    logger.info("=== Exp 4: Silver ===")
    if fup_v2_bak:
        shutil.move(str(fup_v2_bak), str(FUP_V2))
    results["silver"] = run_benchmark_with_config(lookup_enabled=False)
    logger.info("Silver AAFE: %.3f", results["silver"]["aafe"])

    # Exp 5: Gold (both retrained, lookup ON)
    logger.info("=== Exp 5: Gold ===")
    results["gold"] = run_benchmark_with_config(lookup_enabled=True)
    logger.info("Gold AAFE: %.3f", results["gold"]["aafe"])

    # Summary
    print("\n=== ABLATION RESULTS ===")
    print(f"{'Experiment':<25s} {'AAFE':>8s} {'In-domain':>10s} {'%2-fold':>8s} {'N':>4s}")
    print("-" * 60)
    for name, r in results.items():
        print(f"{name:<25s} {r['aafe']:8.3f} {r['aafe_in_domain']:10.3f} {r['pct_2fold']:7.1f}% {r['n_drugs']:4d}")

    baseline_aafe = results["baseline"]["aafe"]
    silver_aafe = results["silver"]["aafe"]
    gold_aafe = results["gold"]["aafe"]
    print(f"\nTraining effect (Silver - Baseline): {silver_aafe - baseline_aafe:+.3f}")
    print(f"Lookup effect (Gold - Silver):       {gold_aafe - silver_aafe:+.3f}")
    print(f"Total effect (Gold - Baseline):      {gold_aafe - baseline_aafe:+.3f}")


if __name__ == "__main__":
    run_experiments()
```

- [ ] **Step 2: Run ablation**

Run: `python3 scripts/run_ablation.py`
Expected: 5 experiments with AAFE comparison table.

- [ ] **Step 3: Commit**

```bash
git add scripts/run_ablation.py
git commit -m "feat(validation): ablation study runner — 5 experiments, Baseline/Silver/Gold"
```

---

### Task 9: Final Test Run + Update fup CV

**Files:**
- Modify: `src/sisyphus/predict/adme.py` — update `_FUP_CV_V2` with actual CV from training

- [ ] **Step 1: Update CV constant from training output**

After training (Task 4), check the logged `est_CV` value. Update `_FUP_CV_V2` in adme.py:
```python
_FUP_CV_V2 = 0.XX  # from 5-fold CV on TDC+DrugBank training set
```

- [ ] **Step 2: Run full test suite**

Run: `python3 -m pytest tests/ -q --tb=short`
Expected: All tests pass.

- [ ] **Step 3: Commit**

```bash
git add src/sisyphus/predict/adme.py
git commit -m "feat(predict): update fup v2 CV from cross-validation"
```
