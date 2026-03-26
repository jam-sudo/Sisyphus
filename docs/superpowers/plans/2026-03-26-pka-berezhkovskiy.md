# pKa Model + Berezhkovskiy Correction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve engine AAFE (2.945 → target <2.7) by training a pKa prediction model from DrugBank data and activating Berezhkovskiy Kp correction for protein-bound drugs.

**Architecture:** Two independent mechanistic improvements to the `predict/` layer. pKa model replaces crude defaults (4.5/9.0) with XGBoost trained on 9,974+ DrugBank ChemAxon values. Berezhkovskiy correction is a one-line activation of already-implemented code in `ivive.py`. Both feed directly into Kp calculation, which has exponential sensitivity to pKa.

**Tech Stack:** XGBoost, RDKit, scikit-learn (scaffold splitting), numpy, existing `descriptors.py`

---

### Task 1: pKa Training Script — Data Loading & Holdout Exclusion

**Files:**
- Create: `scripts/train_pka_model.py`

- [ ] **Step 1: Create training script with data loading**

```python
#!/usr/bin/env python3
"""Train XGBoost pKa models from DrugBank ChemAxon data.

Two models:
  - pka_acidic: SMILES → strongest acidic pKa (N~9,974)
  - pka_basic:  SMILES → strongest basic pKa (N~10,987)

Holdout drugs excluded via 3-key matching (SMILES, InChIKey-14, name).
5-fold Murcko scaffold split CV with MAE/R² reporting.

Output: models/adme/xgboost_pka_acidic.json, xgboost_pka_basic.json
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger(__name__)

DRUGBANK_DRUGS_CSV = ROOT / "data" / "drugbank" / "drugs.csv"
HOLDOUT_JSON = ROOT / "data" / "reference" / "holdout.json"
CLINICAL_PK_JSON = ROOT / "data" / "reference" / "clinical_pk.json"
OUTPUT_DIR = ROOT / "models" / "adme"


def _canonical_smiles(smiles: str) -> str | None:
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def _inchikey14(smiles: str) -> str | None:
    from rdkit import Chem
    from rdkit.Chem.inchi import InchiToInchiKey, MolToInchi
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    inchi = MolToInchi(mol)
    if inchi is None:
        return None
    ik = InchiToInchiKey(inchi)
    return ik[:14] if ik and len(ik) >= 14 else None


def load_holdout_keys() -> set[str]:
    """Load holdout identifiers for 3-key exclusion (SMILES, InChIKey-14, name)."""
    keys: set[str] = set()
    for path in [HOLDOUT_JSON, CLINICAL_PK_JSON]:
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)
        drugs = data if isinstance(data, list) else data.get("drugs", [])
        for d in drugs:
            if d.get("in_holdout", False) or path == HOLDOUT_JSON:
                name = d.get("name", "").strip().lower()
                smiles = d.get("smiles", "").strip()
                if name:
                    keys.add(name)
                cs = _canonical_smiles(smiles) if smiles else None
                if cs:
                    keys.add(cs)
                ik = _inchikey14(smiles) if smiles else None
                if ik:
                    keys.add(ik)
    log.info("Holdout keys: %d identifiers", len(keys))
    return keys


def is_holdout(smiles: str, name: str, holdout_keys: set[str]) -> bool:
    if name.strip().lower() in holdout_keys:
        return True
    cs = _canonical_smiles(smiles)
    if cs and cs in holdout_keys:
        return True
    ik = _inchikey14(smiles)
    if ik and ik in holdout_keys:
        return True
    return False


def load_pka_data(holdout_keys: set[str]) -> tuple[list, list, list, list, list]:
    """Load DrugBank pKa data, returning (smiles, acidic, basic, names, types).

    Returns separate lists; entries may have acidic=None or basic=None.
    Holdout drugs are excluded.
    """
    smiles_list: list[str] = []
    acidic_list: list[float | None] = []
    basic_list: list[float | None] = []
    names: list[str] = []
    excluded = 0

    with open(DRUGBANK_DRUGS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            smi = row.get("canonical_smiles", "").strip()
            name = row.get("name", "").strip()
            if not smi:
                continue

            pka_a = row.get("pka_acidic", "").strip()
            pka_b = row.get("pka_basic", "").strip()

            # Must have at least one pKa
            a_val: float | None = None
            b_val: float | None = None
            try:
                if pka_a:
                    a_val = float(pka_a)
                if pka_b:
                    b_val = float(pka_b)
            except ValueError:
                continue

            if a_val is None and b_val is None:
                continue

            if is_holdout(smi, name, holdout_keys):
                excluded += 1
                continue

            smiles_list.append(smi)
            acidic_list.append(a_val)
            basic_list.append(b_val)
            names.append(name)

    log.info("Loaded %d drugs with pKa (%d holdout excluded)", len(smiles_list), excluded)
    return smiles_list, acidic_list, basic_list, names, []


def main():
    holdout_keys = load_holdout_keys()
    smiles_list, acidic_list, basic_list, names, _ = load_pka_data(holdout_keys)
    log.info("Total: %d, with acidic: %d, with basic: %d",
             len(smiles_list),
             sum(1 for a in acidic_list if a is not None),
             sum(1 for b in basic_list if b is not None))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run to verify data loading works**

Run: `python3 scripts/train_pka_model.py`
Expected: Log output showing ~11,000+ drugs loaded, ~X holdout excluded.

- [ ] **Step 3: Commit**

```bash
git add scripts/train_pka_model.py
git commit -m "feat(predict): pKa training script — data loading and holdout exclusion"
```

---

### Task 2: pKa Training Script — Feature Computation & Scaffold Split CV

**Files:**
- Modify: `scripts/train_pka_model.py`

- [ ] **Step 1: Add feature computation and scaffold-split CV training**

Append to `main()` in `scripts/train_pka_model.py`, replacing the placeholder ending:

```python
def compute_all_features(smiles_list: list[str]) -> tuple[np.ndarray, list[bool]]:
    """Compute Morgan FP + RDKit descriptors for all SMILES. Returns (features, valid_mask)."""
    from sisyphus.descriptors import compute_features
    features = []
    valid = []
    for i, smi in enumerate(smiles_list):
        try:
            feat = compute_features(smi)
            features.append(feat)
            valid.append(True)
        except Exception:
            features.append(np.zeros(2057))
            valid.append(False)
        if (i + 1) % 2000 == 0:
            log.info("  Features: %d/%d", i + 1, len(smiles_list))
    return np.array(features), valid


def scaffold_split_indices(smiles_list: list[str], n_folds: int = 5, seed: int = 42):
    """Generate n-fold scaffold split indices."""
    from rdkit import Chem
    from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles

    scaffold_to_indices: dict[str, list[int]] = {}
    for i, smi in enumerate(smiles_list):
        try:
            mol = Chem.MolFromSmiles(smi)
            scaffold = MurckoScaffoldSmiles(mol=mol, includeChirality=False) if mol else ""
        except Exception:
            scaffold = ""
        scaffold_to_indices.setdefault(scaffold, []).append(i)

    rng = np.random.default_rng(seed)
    scaffolds = list(scaffold_to_indices.keys())
    rng.shuffle(scaffolds)

    folds: list[list[int]] = [[] for _ in range(n_folds)]
    for i, scaffold in enumerate(scaffolds):
        folds[i % n_folds].extend(scaffold_to_indices[scaffold])

    return folds


def train_and_evaluate(
    X: np.ndarray, y: np.ndarray, smiles_list: list[str], label: str
) -> xgb.XGBRegressor:
    """Train XGBoost with 5-fold scaffold CV, report MAE/R², return model trained on all data."""
    folds = scaffold_split_indices(smiles_list, n_folds=5)

    all_preds = np.zeros(len(y))
    all_true = np.zeros(len(y))
    fold_maes: list[float] = []
    fold_r2s: list[float] = []

    for fold_idx in range(5):
        test_idx = folds[fold_idx]
        train_idx = [i for f_idx in range(5) if f_idx != fold_idx for i in folds[f_idx]]

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = xgb.XGBRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1,
        )
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)],
                  verbose=False)

        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        fold_maes.append(mae)
        fold_r2s.append(r2)

        all_preds[test_idx] = preds
        all_true[test_idx] = y_test

        log.info("  Fold %d: MAE=%.3f, R²=%.3f (N_train=%d, N_test=%d)",
                 fold_idx + 1, mae, r2, len(train_idx), len(test_idx))

    overall_mae = mean_absolute_error(all_true, all_preds)
    overall_r2 = r2_score(all_true, all_preds)
    log.info("%s CV: MAE=%.3f (±%.3f), R²=%.3f (±%.3f)",
             label, np.mean(fold_maes), np.std(fold_maes),
             np.mean(fold_r2s), np.std(fold_r2s))
    log.info("%s overall: MAE=%.3f, R²=%.3f", label, overall_mae, overall_r2)

    # Train final model on ALL data
    final_model = xgb.XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1,
    )
    final_model.fit(X, y, verbose=False)
    return final_model


def main():
    holdout_keys = load_holdout_keys()
    smiles_list, acidic_list, basic_list, names, _ = load_pka_data(holdout_keys)

    log.info("Computing features for %d molecules...", len(smiles_list))
    X_all, valid = compute_all_features(smiles_list)

    # --- Acidic pKa model ---
    acid_mask = [i for i, (a, v) in enumerate(zip(acidic_list, valid))
                 if a is not None and v]
    X_acid = X_all[acid_mask]
    y_acid = np.array([acidic_list[i] for i in acid_mask])
    smi_acid = [smiles_list[i] for i in acid_mask]
    log.info("Acidic pKa: %d samples", len(acid_mask))

    acid_model = train_and_evaluate(X_acid, y_acid, smi_acid, "pKa_acidic")

    # --- Basic pKa model ---
    base_mask = [i for i, (b, v) in enumerate(zip(basic_list, valid))
                 if b is not None and v]
    X_base = X_all[base_mask]
    y_base = np.array([basic_list[i] for i in base_mask])
    smi_base = [smiles_list[i] for i in base_mask]
    log.info("Basic pKa: %d samples", len(base_mask))

    base_model = train_and_evaluate(X_base, y_base, smi_base, "pKa_basic")

    # --- Save models ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    acid_path = OUTPUT_DIR / "xgboost_pka_acidic.json"
    base_path = OUTPUT_DIR / "xgboost_pka_basic.json"
    acid_model.save_model(str(acid_path))
    base_model.save_model(str(base_path))
    log.info("Saved: %s, %s", acid_path, base_path)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run training script**

Run: `python3 scripts/train_pka_model.py`
Expected: 5-fold CV results for both acidic and basic pKa. Target MAE < 1.5 units, R² > 0.5.

- [ ] **Step 3: Commit**

```bash
git add scripts/train_pka_model.py
git commit -m "feat(predict): pKa training — scaffold CV + model save"
```

---

### Task 3: Integrate pKa Model into chemistry.py

**Files:**
- Modify: `src/sisyphus/predict/chemistry.py`
- Test: `tests/unit/test_features_chemistry.py`

- [ ] **Step 1: Write failing tests for pKa model fallback**

Add to `tests/unit/test_features_chemistry.py`:

```python
class TestPkaModelFallback:
    """Test pKa prediction model integration in chemistry.py."""

    def test_compound_type_aspirin_acid(self):
        """Aspirin (carboxylic acid) should be classified as acid."""
        profile = compute_profile("CC(=O)Oc1ccccc1C(=O)O")
        assert profile.compound_type == "acid"

    def test_compound_type_metformin_base(self):
        """Metformin (biguanide) should be classified as base."""
        profile = compute_profile("CN(C)C(=N)NC(=N)N")
        assert profile.compound_type == "base"

    def test_pka_not_none_for_ionizable(self):
        """Ionizable drugs should have a non-None pKa."""
        profile = compute_profile("CC(=O)Oc1ccccc1C(=O)O")  # aspirin
        assert profile.pka is not None

    def test_pka_numeric_range(self):
        """Predicted pKa should be in reasonable range."""
        profile = compute_profile("CC(=O)Oc1ccccc1C(=O)O")  # aspirin
        if profile.pka is not None:
            assert 0.0 < profile.pka < 14.0

    def test_neutral_compound_caffeine(self):
        """Caffeine has no strong ionizable groups at pH 7.4."""
        profile = compute_profile("Cn1c(=O)c2c(ncn2C)n(C)c1=O")
        # Caffeine is weakly basic but typically classified as neutral
        assert profile.compound_type in ("neutral", "base")
```

- [ ] **Step 2: Run tests to verify baseline behavior**

Run: `python3 -m pytest tests/unit/test_features_chemistry.py::TestPkaModelFallback -v`
Expected: Tests should pass with current code (SMARTS fallback + DrugBank lookup).

- [ ] **Step 3: Add pKa model loading to chemistry.py**

Add after the logP correction block (around line 350) in `src/sisyphus/predict/chemistry.py`. Replace the existing pKa block:

```python
    # pKa: DrugBank ChemAxon → XGBoost model → fallback SMARTS
    db_pka = db.get_pka(canonical)
    if db_pka is not None:
        pka, compound_type = _classify_from_pka(db_pka[0], db_pka[1])
    else:
        # Try XGBoost pKa model
        pka, compound_type = _predict_pka_from_model(canonical, mol, logp)
```

And add the `_predict_pka_from_model` function earlier in the file:

```python
# ---------------------------------------------------------------------------
# pKa prediction model (XGBoost, trained on DrugBank ChemAxon values)
# ---------------------------------------------------------------------------

_PKA_MODEL_DIR = Path(__file__).resolve().parent.parent.parent.parent / "models" / "adme"


def _load_pka_models() -> tuple | None:
    """Load acidic and basic pKa XGBoost models. Returns (acid_model, base_model) or None."""
    acid_path = _PKA_MODEL_DIR / "xgboost_pka_acidic.json"
    base_path = _PKA_MODEL_DIR / "xgboost_pka_basic.json"
    if not acid_path.exists() or not base_path.exists():
        return None
    try:
        import xgboost as xgb
        if not hasattr(_load_pka_models, "_cache"):
            acid_model = xgb.XGBRegressor()
            acid_model.load_model(str(acid_path))
            base_model = xgb.XGBRegressor()
            base_model.load_model(str(base_path))
            _load_pka_models._cache = (acid_model, base_model)  # type: ignore[attr-defined]
            logger.info("Loaded pKa models: %s, %s", acid_path.name, base_path.name)
        return _load_pka_models._cache  # type: ignore[attr-defined]
    except Exception as e:
        logger.warning("pKa model loading failed: %s", e)
        return None


def _predict_pka_from_model(
    canonical_smiles: str, mol, logp: float
) -> tuple[float | None, str]:
    """Predict pKa from XGBoost model, fall back to SMARTS heuristic."""
    models = _load_pka_models()
    if models is None:
        return _estimate_pka_type(mol, logp)

    try:
        from sisyphus.descriptors import compute_features
        features = compute_features(canonical_smiles).reshape(1, -1)

        acid_model, base_model = models
        acidic_pred = float(acid_model.predict(features)[0])
        basic_pred = float(base_model.predict(features)[0])

        return _classify_from_pka(acidic_pred, basic_pred)
    except Exception as e:
        logger.warning("pKa prediction failed: %s, using SMARTS fallback", e)
        return _estimate_pka_type(mol, logp)
```

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/unit/test_features_chemistry.py -v`
Expected: All tests pass. If pKa models are present, they're used; otherwise SMARTS fallback.

- [ ] **Step 5: Run full test suite**

Run: `python3 -m pytest tests/ -x -q`
Expected: 348 tests pass (no regressions).

- [ ] **Step 6: Commit**

```bash
git add src/sisyphus/predict/chemistry.py tests/unit/test_features_chemistry.py
git commit -m "feat(predict): integrate XGBoost pKa model with 3-tier fallback"
```

---

### Task 4: Activate Berezhkovskiy Correction

**Files:**
- Modify: `src/sisyphus/predict/ivive.py:558`
- Test: `tests/unit/test_adme_ivive.py`

- [ ] **Step 1: Write test for Berezhkovskiy Kp reduction**

Add to `tests/unit/test_adme_ivive.py`:

```python
class TestBerezhkovskiyCorrection:
    """Berezhkovskiy correction reduces Kp for highly bound drugs."""

    def test_bz_reduces_kp_for_low_fup(self):
        """For fup=0.01, Berezhkovskiy should substantially reduce Kp."""
        from sisyphus.predict.ivive import _apply_bz_correction
        kp_rr = 100.0
        kp_bz = _apply_bz_correction(kp_rr, fup=0.01)
        # Kp_bz = 100 / (1 + 99*0.01) = 50.3
        assert kp_bz < kp_rr
        assert abs(kp_bz - 50.25) < 1.0

    def test_bz_minimal_for_high_fup(self):
        """For fup=0.9, correction should be minimal."""
        from sisyphus.predict.ivive import _apply_bz_correction
        kp_rr = 5.0
        kp_bz = _apply_bz_correction(kp_rr, fup=0.9)
        # Kp_bz = 5 / (1 + 4*0.9) = 5/4.6 = 1.09 — larger effect actually
        assert kp_bz < kp_rr

    def test_bz_kp_one_unchanged(self):
        """Kp=1.0 should remain 1.0 regardless of fup."""
        from sisyphus.predict.ivive import _apply_bz_correction
        assert _apply_bz_correction(1.0, 0.5) == 1.0

    def test_default_kp_method_is_berezhkovskiy(self):
        """build_drug_on_graph should use berezhkovskiy by default."""
        import inspect
        from sisyphus.predict.ivive import build_drug_on_graph
        sig = inspect.signature(build_drug_on_graph)
        default = sig.parameters["kp_method"].default
        assert default == "berezhkovskiy"
```

- [ ] **Step 2: Run tests to see the default check fail**

Run: `python3 -m pytest tests/unit/test_adme_ivive.py::TestBerezhkovskiyCorrection -v`
Expected: `test_default_kp_method_is_berezhkovskiy` FAILS (currently "rodgers_rowland").

- [ ] **Step 3: Change default kp_method**

In `src/sisyphus/predict/ivive.py`, line 558, change:

```python
    kp_method: str = "rodgers_rowland",
```
to:
```python
    kp_method: str = "berezhkovskiy",
```

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/unit/test_adme_ivive.py::TestBerezhkovskiyCorrection -v`
Expected: All 4 tests PASS.

- [ ] **Step 5: Run full test suite**

Run: `python3 -m pytest tests/ -x -q`
Expected: 348+ tests pass. Some existing engine validation tests may shift slightly due to Kp change — if any test fails, investigate whether the change is expected (Berezhkovskiy lowers Cmax for high-binding drugs) and update expected values.

- [ ] **Step 6: Commit**

```bash
git add src/sisyphus/predict/ivive.py tests/unit/test_adme_ivive.py
git commit -m "feat(predict): activate Berezhkovskiy Kp correction as default"
```

---

### Task 5: Evaluation — Engine-Only Ablation with pKa + Berezhkovskiy

**Files:**
- Create: `scripts/run_pka_bz_evaluation.py`

- [ ] **Step 1: Write evaluation script**

```python
#!/usr/bin/env python3
"""Evaluate pKa model + Berezhkovskiy correction impact on engine AAFE.

Runs 4 experiments:
  Baseline: pKa model OFF, BZ OFF (rodgers_rowland)
  Exp 1:    pKa model ON,  BZ OFF
  Exp 2:    pKa model OFF, BZ ON
  Exp 3:    pKa model ON,  BZ ON  (new default)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger(__name__)

_PKA_ACID = ROOT / "models" / "adme" / "xgboost_pka_acidic.json"
_PKA_BASE = ROOT / "models" / "adme" / "xgboost_pka_basic.json"


def run_experiment(pka_on: bool, bz_on: bool) -> dict:
    """Run benchmark with specified pKa/BZ configuration."""
    # Reset caches
    from sisyphus.predict import chemistry
    if hasattr(chemistry._load_pka_models, "_cache"):
        del chemistry._load_pka_models._cache

    from sisyphus.predict import adme
    adme._model_cache.clear()

    if hasattr(chemistry.compute_profile, "_logp_model"):
        del chemistry.compute_profile._logp_model

    from sisyphus.predict.drugbank import _reset_singleton
    _reset_singleton()

    # Toggle pKa model: rename files to hide
    import shutil
    acid_bak = base_bak = None
    if not pka_on:
        if _PKA_ACID.exists():
            acid_bak = _PKA_ACID.with_suffix(".json.off")
            shutil.move(str(_PKA_ACID), str(acid_bak))
        if _PKA_BASE.exists():
            base_bak = _PKA_BASE.with_suffix(".json.off")
            shutil.move(str(_PKA_BASE), str(base_bak))

    # Toggle BZ: monkey-patch default kp_method
    from sisyphus.predict import ivive
    original_default = "berezhkovskiy"
    if not bz_on:
        # Temporarily override build_drug_on_graph to use rodgers_rowland
        _orig_build = ivive.build_drug_on_graph.__wrapped__ if hasattr(
            ivive.build_drug_on_graph, "__wrapped__") else None
        import functools
        _real_build = ivive.build_drug_on_graph

        @functools.wraps(_real_build)
        def _patched_build(*args, kp_method="rodgers_rowland", **kwargs):
            return _real_build(*args, kp_method=kp_method, **kwargs)

        ivive.build_drug_on_graph = _patched_build

    try:
        from sisyphus.validation.benchmark import run_benchmark
        result = run_benchmark(holdout_only=True)
    finally:
        # Restore pKa models
        if acid_bak and acid_bak.exists():
            shutil.move(str(acid_bak), str(_PKA_ACID))
        if base_bak and base_bak.exists():
            shutil.move(str(base_bak), str(_PKA_BASE))
        # Restore BZ
        if not bz_on:
            ivive.build_drug_on_graph = _real_build

    return {
        "meta_aafe": result.aafe,
        "meta_pct2": result.pct_2fold,
        "engine_aafe": result.engine_aafe,
        "engine_pct2": result.engine_pct_2fold,
        "n": result.n_drugs,
        "n_engine": result.n_engine,
    }


def fmt(val, w=10):
    return f"{val:{w}.3f}" if val is not None else f"{'N/A':>{w}}"


def main():
    print("=" * 75)
    print("pKa + BEREZHKOVSKIY EVALUATION")
    print("=" * 75)

    experiments = [
        ("Baseline (OFF/OFF)", False, False),
        ("Exp 1 (pKa ON/BZ OFF)", True, False),
        ("Exp 2 (pKa OFF/BZ ON)", False, True),
        ("Exp 3 (pKa ON/BZ ON)", True, True),
    ]

    results = {}
    for name, pka_on, bz_on in experiments:
        print(f"\nRunning {name} ...")
        results[name] = run_experiment(pka_on, bz_on)
        print(f"  Engine AAFE: {fmt(results[name]['engine_aafe'])}, "
              f"Meta AAFE: {fmt(results[name]['meta_aafe'])}")

    # Summary table
    print("\n" + "=" * 75)
    print(f"{'Experiment':<28s} {'Engine AAFE':>12s} {'Meta AAFE':>12s} {'Δ Engine':>10s}")
    print("-" * 75)

    baseline_eng = results["Baseline (OFF/OFF)"]["engine_aafe"]
    for name, r in results.items():
        eng = r["engine_aafe"]
        delta = (eng - baseline_eng) if eng and baseline_eng else None
        print(f"{name:<28s} {fmt(eng)} {fmt(r['meta_aafe'])} "
              f"{f'{delta:>+10.3f}' if delta is not None else f'{'N/A':>10}'}")

    print()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Train pKa models first (if not yet done)**

Run: `python3 scripts/train_pka_model.py`

- [ ] **Step 3: Run evaluation**

Run: `python3 scripts/run_pka_bz_evaluation.py`
Expected: 4 rows of results. Success if any experiment shows Engine AAFE Δ > 0.1 vs baseline.

- [ ] **Step 4: Commit**

```bash
git add scripts/run_pka_bz_evaluation.py
git commit -m "feat(validation): pKa + Berezhkovskiy evaluation script"
```

---

### Task 6: Update CLAUDE.md with Results

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update session state with pKa CV metrics and engine AAFE results**

After running evaluation, update these CLAUDE.md sections:
- `Current Metrics`: update Engine AAFE if improved
- `확정된 진단`: revise if pKa model breaks the ceiling
- `다음 할 것`: add pKa/BZ results, next steps (ChEMBL expansion if warranted)

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update session state with pKa + Berezhkovskiy results"
```
