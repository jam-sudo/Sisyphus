#!/usr/bin/env python3
"""CLint 3-Class Classification — Phase 0 complete pipeline.

Phase 0A: Class definition (Low<10, Med 10-50, High>50 µL/min/10⁶ cells)
Phase 0B: XGBoost classifier training (5-fold scaffold CV)
Phase 0C: Engine integration via probability-weighted MC mixture (Option 2)
Phase 0D: Holdout N=107 benchmark
Phase 0E: Error cancellation gate

Usage:
    python3 scripts/clint_classification.py
"""
from __future__ import annotations

import json
import logging
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

from sisyphus.descriptors import compute_features

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TDC_HEP_TAB = ROOT / "data" / "training" / "clearance_hepatocyte_az.tab"
HOLDOUT_JSON = ROOT / "data" / "reference" / "holdout.json"
CLINICAL_PK_JSON = ROOT / "data" / "reference" / "clinical_pk.json"
MODEL_DIR = ROOT / "models" / "adme"
OUTPUT_DIR = ROOT / "data" / "validation"
CLASSIFIER_PATH = MODEL_DIR / "xgboost_clint_classifier.json"

# ---------------------------------------------------------------------------
# Class definitions
# ---------------------------------------------------------------------------
# Low: CLint < 10 µL/min/10⁶ cells — stable, long half-life
# Medium: 10 ≤ CLint ≤ 50
# High: CLint > 50 — rapid metabolism, short half-life
CLASS_BOUNDARIES = (10.0, 50.0)
CLASS_NAMES = ["Low", "Medium", "High"]
# Representative values for MC mixture (geometric center of each class range)
# Low: [0.1, 10) → geometric mean ~ 5 µL/min/10⁶ cells
# Medium: [10, 50] → geometric mean ~ 22 (use 25 for round number)
# High: (50, ∞) → representative 80 (approx median of High class in TDC)
CLASS_REPRESENTATIVES = {0: 5.0, 1: 25.0, 2: 80.0}


def clint_to_class(clint: float) -> int:
    """Convert CLint value to class label (0=Low, 1=Medium, 2=High)."""
    if clint < CLASS_BOUNDARIES[0]:
        return 0
    elif clint <= CLASS_BOUNDARIES[1]:
        return 1
    else:
        return 2


# ---------------------------------------------------------------------------
# RDKit helpers
# ---------------------------------------------------------------------------
from rdkit import Chem
from rdkit.Chem.inchi import MolToInchi, InchiToInchiKey
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles


def _canonical_smiles(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol, isomericSmiles=True) if mol else None


def _inchikey_prefix(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    inchi = MolToInchi(mol)
    if inchi is None:
        return None
    ik = InchiToInchiKey(inchi)
    return ik[:14] if ik else None


# ---------------------------------------------------------------------------
# Holdout exclusion
# ---------------------------------------------------------------------------
def build_holdout_keys() -> dict:
    with open(HOLDOUT_JSON) as f:
        hd = json.load(f)
    with open(CLINICAL_PK_JSON) as f:
        cp = json.load(f)

    all_names = hd.get("holdout", []) + hd.get("train", [])
    drugs = cp.get("drugs", {})
    ho_smiles, ho_ik, ho_names = set(), set(), set(n.lower() for n in all_names)

    for name in all_names:
        entry = drugs.get(name) or drugs.get(name.replace(" ", "_"))
        if not entry:
            continue
        smi = entry.get("smiles", "")
        if not smi:
            continue
        cs = _canonical_smiles(smi)
        if cs:
            ho_smiles.add(cs)
        ik = _inchikey_prefix(smi)
        if ik:
            ho_ik.add(ik)

    return {"smiles": ho_smiles, "ik": ho_ik, "names": ho_names}


def is_holdout(smiles: str, holdout_keys: dict) -> bool:
    cs = _canonical_smiles(smiles)
    if cs and cs in holdout_keys["smiles"]:
        return True
    ik = _inchikey_prefix(smiles)
    if ik and ik in holdout_keys["ik"]:
        return True
    return False


# ---------------------------------------------------------------------------
# Scaffold split
# ---------------------------------------------------------------------------
def scaffold_split(smiles_list: list[str], n_folds: int = 5, seed: int = 42):
    s2i: dict[str, list[int]] = {}
    for i, smi in enumerate(smiles_list):
        try:
            mol = Chem.MolFromSmiles(smi)
            sc = MurckoScaffoldSmiles(mol=mol, includeChirality=False) if mol else ""
        except Exception:
            sc = ""
        s2i.setdefault(sc, []).append(i)
    rng = np.random.default_rng(seed)
    scaffolds = list(s2i.keys())
    rng.shuffle(scaffolds)
    folds = [[] for _ in range(n_folds)]
    for i, sc in enumerate(scaffolds):
        folds[i % n_folds].extend(s2i[sc])
    return folds


# ═══════════════════════════════════════════════════════════════════════════
# Phase 0A + 0B: Classification Model
# ═══════════════════════════════════════════════════════════════════════════
def train_classifier():
    log.info("=" * 60)
    log.info("Phase 0A+0B: CLint 3-Class Classification")
    log.info("=" * 60)

    # Load TDC Hepatocyte_AZ
    df = pd.read_csv(TDC_HEP_TAB, sep="\t")
    df.columns = ["ID", "SMILES", "Y"]
    df["canon"] = df["SMILES"].apply(_canonical_smiles)
    df = df.dropna(subset=["canon"]).reset_index(drop=True)

    # Deduplicate by canonical SMILES (mean if duplicates)
    df = df.groupby("canon").agg(Y=("Y", "mean")).reset_index()

    # Holdout exclusion
    ho_keys = build_holdout_keys()
    mask = df["canon"].apply(lambda s: is_holdout(s, ho_keys))
    n_excl = mask.sum()
    df = df[~mask].reset_index(drop=True)
    log.info("Holdout exclusion: %d removed, %d remain", n_excl, len(df))

    # Assign classes
    df["class"] = df["Y"].apply(clint_to_class)

    # Class distribution
    class_dist = df["class"].value_counts().sort_index()
    log.info("Class distribution (10/50 cutoff):")
    for c in range(3):
        n = class_dist.get(c, 0)
        log.info("  %s (class %d): %d (%.1f%%)", CLASS_NAMES[c], c, n, n / len(df) * 100)

    # Per-class CLint stats (for MC mixture σ)
    class_stats = {}
    for c in range(3):
        vals = df[df["class"] == c]["Y"].values
        log_vals = np.log(np.maximum(vals, 0.1))
        class_stats[c] = {
            "n": len(vals),
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "log_std": float(np.std(log_vals)),  # σ for LogNormal sampling
        }
        log.info(
            "  Class %d (%s): N=%d, mean=%.1f, median=%.1f, log_std=%.3f",
            c, CLASS_NAMES[c], class_stats[c]["n"],
            class_stats[c]["mean"], class_stats[c]["median"],
            class_stats[c]["log_std"],
        )

    # Compute features
    log.info("Computing features ...")
    X_list, y_list, smiles_list = [], [], []
    failed = 0
    for _, row in df.iterrows():
        try:
            feat = compute_features(row["canon"])
            X_list.append(feat)
            y_list.append(row["class"])
            smiles_list.append(row["canon"])
        except Exception:
            failed += 1

    X = np.array(X_list, dtype=np.float64)
    y = np.array(y_list, dtype=int)
    log.info("Features: %s (failed: %d)", X.shape, failed)

    # 5-fold scaffold CV
    log.info("5-fold scaffold CV ...")
    folds = scaffold_split(smiles_list)
    all_preds = np.zeros(len(y), dtype=int)
    all_proba = np.zeros((len(y), 3), dtype=np.float64)
    fold_accs = []

    xgb_params = dict(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=4,
        verbosity=0,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
    )

    for fi in range(5):
        test_idx = folds[fi]
        train_idx = [j for f in range(5) if f != fi for j in folds[f]]
        model = xgb.XGBClassifier(**xgb_params)
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[test_idx])
        proba = model.predict_proba(X[test_idx])
        all_preds[test_idx] = preds
        all_proba[test_idx] = proba
        acc = accuracy_score(y[test_idx], preds)
        fold_accs.append(acc)
        log.info("  Fold %d: accuracy=%.3f (N_train=%d, N_test=%d)",
                 fi + 1, acc, len(train_idx), len(test_idx))

    # Overall metrics
    overall_acc = accuracy_score(y, all_preds)
    kappa = cohen_kappa_score(y, all_preds)
    cm = confusion_matrix(y, all_preds)
    report = classification_report(y, all_preds, target_names=CLASS_NAMES, output_dict=True)

    log.info("Overall accuracy: %.3f (%.3f ± %.3f per fold)", overall_acc,
             np.mean(fold_accs), np.std(fold_accs))
    log.info("Cohen's kappa: %.3f", kappa)
    log.info("Confusion matrix:\n%s", cm)

    # Print results
    print()
    print("=" * 60)
    print("  CLint 3-CLASS CLASSIFICATION RESULTS")
    print("=" * 60)
    print()
    print(f"  N compounds:     {len(y)}")
    print(f"  Overall accuracy: {overall_acc:.3f}")
    print(f"  Cohen's kappa:    {kappa:.3f}")
    print(f"  Fold accuracies:  {', '.join(f'{a:.3f}' for a in fold_accs)}")
    print()
    print("  Per-class metrics:")
    print(f"  {'Class':<10s} {'Precision':>10s} {'Recall':>8s} {'F1':>6s} {'Support':>8s}")
    print("  " + "-" * 48)
    for c, name in enumerate(CLASS_NAMES):
        r = report[name]
        print(f"  {name:<10s} {r['precision']:>10.3f} {r['recall']:>8.3f} "
              f"{r['f1-score']:>6.3f} {r['support']:>8.0f}")
    print()
    print("  Confusion matrix (rows=true, cols=predicted):")
    print(f"  {'':>10s}", end="")
    for name in CLASS_NAMES:
        print(f" {name:>8s}", end="")
    print()
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name:>10s}", end="")
        for j in range(3):
            print(f" {cm[i, j]:>8d}", end="")
        print()

    # Train final model on all data
    log.info("Training final classifier on all data ...")
    final_model = xgb.XGBClassifier(**xgb_params)
    final_model.fit(X, y)
    final_model.save_model(str(CLASSIFIER_PATH))
    log.info("Classifier saved to %s", CLASSIFIER_PATH)

    return {
        "n": len(y),
        "accuracy": float(overall_acc),
        "kappa": float(kappa),
        "fold_accs": [float(a) for a in fold_accs],
        "confusion_matrix": cm.tolist(),
        "report": {k: v for k, v in report.items() if k in CLASS_NAMES},
        "class_stats": class_stats,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Phase 0C+0D: Engine Integration + Holdout Benchmark
# ═══════════════════════════════════════════════════════════════════════════
def run_benchmark(class_stats: dict):
    """Run holdout benchmark with classification mixture vs regression baseline."""
    log.info("=" * 60)
    log.info("Phase 0C+0D: Engine Integration + Holdout Benchmark")
    log.info("=" * 60)

    from sisyphus.core import Distribution
    from sisyphus.predict import adme as adme_module
    from sisyphus.validation.benchmark import run_benchmark as _run_benchmark

    # --- Baseline benchmark (current regression model) ---
    log.info("Running BASELINE benchmark (regression CLint) ...")
    logging.getLogger("sisyphus").setLevel(logging.WARNING)
    result_baseline = _run_benchmark(holdout_only=True)
    logging.getLogger("sisyphus").setLevel(logging.INFO)

    log.info(
        "Baseline: Engine AAFE=%.3f, Meta AAFE=%.3f, %%2-fold=%.1f%%",
        result_baseline.engine_aafe or 0,
        result_baseline.aafe,
        result_baseline.pct_2fold,
    )

    # --- Classification mixture benchmark ---
    # Monkey-patch _predict_clint to use classification mixture
    log.info("Loading classifier ...")
    classifier = xgb.XGBClassifier()
    classifier.load_model(str(CLASSIFIER_PATH))

    # Compute class-specific log-normal parameters from training data stats
    class_mu = {}
    class_sigma = {}
    for c in range(3):
        rep = CLASS_REPRESENTATIVES[c]
        sigma = class_stats[str(c) if str(c) in class_stats else c]["log_std"]
        # µ for LogNormal: log(representative)
        class_mu[c] = np.log(rep)
        class_sigma[c] = sigma

    log.info("MC mixture parameters:")
    for c in range(3):
        log.info(
            "  Class %d (%s): representative=%.1f, log_mu=%.3f, log_sigma=%.3f",
            c, CLASS_NAMES[c], CLASS_REPRESENTATIVES[c], class_mu[c], class_sigma[c],
        )

    # Store original _predict_clint
    original_predict_clint = adme_module._predict_clint

    def _predict_clint_mixture(features: np.ndarray) -> Distribution:
        """Classification-based CLint prediction using probability-weighted MC mixture.

        Instead of a single point estimate, generates a mixture distribution:
        - Get P(Low), P(Med), P(High) from classifier
        - Sample CLint from LogNormal distributions weighted by class probabilities
        - Return Distribution with mixture mean and CV

        This avoids Jensen's inequality issues with the non-linear well-stirred model.
        """
        proba = classifier.predict_proba(features)[0]  # [P_low, P_med, P_high]

        # MC mixture: weighted sampling from 3 LogNormal distributions
        n_mc = 1000
        rng = np.random.default_rng(42)
        samples = []
        for c in range(3):
            n_c = max(1, int(round(proba[c] * n_mc)))
            s = rng.lognormal(class_mu[c], class_sigma[c], size=n_c)
            samples.extend(s.tolist())

        samples = np.array(samples)
        samples = np.clip(samples, 0.1, 10000)  # floor/ceiling

        mean_clint = float(np.mean(samples))
        cv_clint = float(np.std(samples) / mean_clint) if mean_clint > 0 else 1.0

        return Distribution(mean=mean_clint, cv=cv_clint)

    # Patch
    adme_module._predict_clint = _predict_clint_mixture
    adme_module._model_cache.clear()

    log.info("Running CLASSIFICATION benchmark (mixture CLint) ...")
    logging.getLogger("sisyphus").setLevel(logging.WARNING)
    result_class = _run_benchmark(holdout_only=True)
    logging.getLogger("sisyphus").setLevel(logging.INFO)

    log.info(
        "Classification: Engine AAFE=%.3f, Meta AAFE=%.3f, %%2-fold=%.1f%%",
        result_class.engine_aafe or 0,
        result_class.aafe,
        result_class.pct_2fold,
    )

    # Restore original
    adme_module._predict_clint = original_predict_clint
    adme_module._model_cache.clear()

    # --- Comparison ---
    engine_delta = (result_class.engine_aafe or 0) - (result_baseline.engine_aafe or 0)
    meta_delta = result_class.aafe - result_baseline.aafe
    pct2_delta = result_class.pct_2fold - result_baseline.pct_2fold

    print()
    print("=" * 70)
    print("  HOLDOUT BENCHMARK COMPARISON (N=107)")
    print("=" * 70)
    print()

    def _fmt(v):
        return f"{v:.3f}" if v is not None else "N/A"

    header = f"{'Metric':<22s} {'Baseline (regr)':>15s}  {'Classification':>15s}  {'Delta':>8s}"
    print(header)
    print("-" * 70)
    for key, label in [
        ("engine_aafe", "Engine AAFE"),
        ("ml_aafe", "ML AAFE"),
        ("aafe", "Meta AAFE"),
        ("pct_2fold", "%2-fold"),
        ("aafe_in_domain", "In-domain AAFE"),
    ]:
        bv = getattr(result_baseline, key)
        cv = getattr(result_class, key)
        if key == "pct_2fold":
            print(f"{label:<22s} {bv:>14.1f}%  {cv:>14.1f}%  {cv - bv:>+7.1f}%")
        else:
            d = f"{cv - bv:+.3f}" if bv is not None and cv is not None else "N/A"
            print(f"{label:<22s} {_fmt(bv):>15s}  {_fmt(cv):>15s}  {d:>8s}")

    print()

    return {
        "baseline": {
            "engine_aafe": result_baseline.engine_aafe,
            "ml_aafe": result_baseline.ml_aafe,
            "meta_aafe": result_baseline.aafe,
            "pct_2fold": result_baseline.pct_2fold,
            "in_domain_aafe": result_baseline.aafe_in_domain,
        },
        "classification": {
            "engine_aafe": result_class.engine_aafe,
            "ml_aafe": result_class.ml_aafe,
            "meta_aafe": result_class.aafe,
            "pct_2fold": result_class.pct_2fold,
            "in_domain_aafe": result_class.aafe_in_domain,
        },
        "delta": {
            "engine_aafe": float(engine_delta),
            "meta_aafe": float(meta_delta),
            "pct_2fold": float(pct2_delta),
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# Phase 0E: Error Cancellation Gate
# ═══════════════════════════════════════════════════════════════════════════
def evaluate_gate(benchmark: dict) -> str:
    meta_delta = benchmark["delta"]["meta_aafe"]
    engine_delta = benchmark["delta"]["engine_aafe"]

    print("=" * 70)
    print("  ERROR CANCELLATION GATE")
    print("=" * 70)
    print()
    print(f"  Engine AAFE delta: {engine_delta:+.3f}")
    print(f"  Meta AAFE delta:   {meta_delta:+.3f}")
    print()

    if meta_delta < -0.001:
        decision = "KEEP"
        print("  DECISION: KEEP — meta AAFE improved")
    elif meta_delta > 0.01:
        decision = "REVERT"
        if engine_delta < 0:
            print("  DECISION: REVERT — error cancellation detected")
            print("  Engine improved but meta worsened.")
        else:
            print("  DECISION: REVERT — classification worse than regression")
    else:
        decision = "NOISE"
        print(f"  DECISION: NOISE — meta AAFE change {meta_delta:+.3f} < 0.01")
        print("  Classification not meaningfully different from regression.")

    print()
    return decision


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Phase 0A + 0B
    class_result = train_classifier()

    # Phase 0C + 0D
    benchmark = run_benchmark(class_result["class_stats"])

    # Phase 0E
    decision = evaluate_gate(benchmark)

    # Save all results
    output = {
        "classification": class_result,
        "benchmark": benchmark,
        "decision": decision,
    }
    output_path = OUTPUT_DIR / "clint_classification.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    log.info("Results saved to %s", output_path)

    print(f"  Results saved to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
