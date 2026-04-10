"""Phase 4: Holdout benchmark with docking-enriched CLint model.

Monkey-patches _predict_clint to use the docking-enriched model,
then runs the standard holdout benchmark.

Usage:
    python scripts/benchmark_clint_docking.py
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import xgboost as xgb
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

log = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
CACHE_DIFFDOCK = ROOT / "data" / "docking" / "cache_diffdock"
CACHE_VINA = ROOT / "data" / "docking" / "cache"


def load_docking_meta() -> dict:
    """Load metadata about the enriched CLint model."""
    meta_path = ROOT / "models" / "adme" / "clint_docking_meta.json"
    if not meta_path.exists():
        log.error("No clint_docking_meta.json — run train_clint_docking.py first")
        sys.exit(1)
    with open(meta_path) as f:
        return json.load(f)


def inchikey_14(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        import hashlib
        return hashlib.md5(smiles.encode()).hexdigest()[:14]
    return Chem.inchi.MolToInchiKey(mol)[:14]


DOCK_FEATURE_NAMES = [
    "docking_score",
    "heme_fe_distance_min",
    "pose_centroid_fe_dist",
    "n_heavy_atoms",
]


def get_docking_features(smiles: str, cyps: list[str], cache_dir: Path) -> np.ndarray:
    """Get docking features for a SMILES from cache."""
    key14 = inchikey_14(smiles)
    features = []
    for cyp in cyps:
        cache_path = cache_dir / f"{key14}_{cyp}.json"
        if cache_path.exists():
            with open(cache_path) as f:
                data = json.load(f)
            for fn in DOCK_FEATURE_NAMES:
                val = data.get(fn, float("nan"))
                features.append(float(val) if val is not None else float("nan"))
        else:
            features.extend([float("nan")] * len(DOCK_FEATURE_NAMES))
    return np.array(features, dtype=np.float64)


def patch_clint_predictor(meta: dict):
    """Monkey-patch the CLint predictor to use docking-enriched model."""
    from sisyphus.predict import adme
    from sisyphus.descriptors import compute_features

    # Load enriched model
    model_path = ROOT / "models" / "adme" / "xgboost_clint_docking.json"
    if not model_path.exists():
        log.error("No xgboost_clint_docking.json — run train_clint_docking.py first")
        sys.exit(1)

    enriched_model = xgb.XGBRegressor()
    enriched_model.load_model(str(model_path))

    cyps = meta["cyps"]
    tool = meta["tool"]
    cache_dir = CACHE_DIFFDOCK if tool == "diffdock" else CACHE_VINA
    n_dock_features = meta["n_dock_features"]

    log.info(f"Patching CLint predictor: tool={tool}, CYPs={cyps}, n_dock_features={n_dock_features}")

    from sisyphus.graph.types import Distribution

    _CLINT_CV = 1.0

    def _predict_clint_enriched(features_2d: np.ndarray) -> Distribution:
        """Enriched CLint predictor with docking features."""
        # features_2d is (1, 2057) — base features
        # We need to append docking features

        # Get SMILES from the calling context — we use a hack:
        # reconstruct features to get fingerprint bits, then look up from cache by InChIKey
        # This is impractical; instead, we use a thread-local variable set by predict_adme

        smiles = getattr(_predict_clint_enriched, "_current_smiles", None)
        if smiles is None:
            # Fallback: use base model
            base_model = adme._load_model("xgboost_clint.json")
            log_clint = float(base_model.predict(features_2d)[0])
            clint = max(10**log_clint, 0.1)
            return Distribution(mean=clint, cv=_CLINT_CV)

        dock_feats = get_docking_features(smiles, cyps, cache_dir)
        enriched_feats = np.concatenate([features_2d[0], dock_feats]).reshape(1, -1)

        log_clint = float(enriched_model.predict(enriched_feats)[0])
        clint = max(10**log_clint, 0.1)
        return Distribution(mean=clint, cv=_CLINT_CV)

    # Patch predict_adme to set current SMILES before CLint call
    original_predict_adme = adme.predict_adme

    def patched_predict_adme(profile):
        _predict_clint_enriched._current_smiles = profile.smiles
        result = original_predict_adme(profile)
        _predict_clint_enriched._current_smiles = None
        return result

    # Apply patches
    adme._predict_clint = _predict_clint_enriched
    adme.predict_adme = patched_predict_adme

    log.info("CLint predictor patched successfully")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    meta = load_docking_meta()

    print("=" * 70)
    print("  Phase 4: Holdout Benchmark — Docking-Enriched CLint")
    print("=" * 70)
    print(f"  Tool: {meta['tool']}")
    print(f"  CYPs: {meta['cyps']}")
    print(f"  CLint CV R² (base):     {meta['metrics_base']['overall_r2']:.4f}")
    print(f"  CLint CV R² (enriched): {meta['metrics_enriched']['overall_r2']:.4f}")
    print(f"  ΔR²: {meta['delta_r2']:+.4f}")
    print()

    # Patch CLint
    patch_clint_predictor(meta)

    # Clear model cache to force reload
    from sisyphus.predict import adme
    adme._model_cache.clear()

    # Run benchmark
    from sisyphus.validation.benchmark import run_benchmark

    log.info("Running holdout benchmark with enriched CLint model...")
    result = run_benchmark(holdout_only=True)

    print()
    print("=" * 70)
    print("  BENCHMARK RESULTS")
    print("=" * 70)
    print()
    print(f"  {'Metric':<25s} {'Baseline':>12s}  {'Enriched':>12s}")
    print("  " + "-" * 55)
    print(f"  {'N drugs':25s} {'107':>12s}  {result.n_drugs:>12d}")
    print(f"  {'Engine AAFE':25s} {'3.415':>12s}  {result.engine_aafe:>12.3f}" if result.engine_aafe else "")
    print(f"  {'Meta AAFE':25s} {'2.283':>12s}  {result.aafe:>12.3f}")
    print(f"  {'%2-fold':25s} {'54.2%':>12s}  {result.pct_2fold:>11.1f}%")
    print(f"  {'In-domain AAFE':25s} {'2.100':>12s}  {result.aafe_in_domain:>12.3f}")
    print(f"  {'In-domain N':25s} {'83':>12s}  {result.n_in_domain:>12d}")
    print()

    # Delta analysis
    delta_meta = result.aafe - 2.283
    delta_engine = (result.engine_aafe - 3.415) if result.engine_aafe else None

    print(f"  Meta AAFE delta: {delta_meta:+.3f}")
    if delta_engine is not None:
        print(f"  Engine AAFE delta: {delta_engine:+.3f}")

    if delta_meta < -0.05:
        print("\n  PASS: Meta AAFE improved by >0.05")
    elif delta_meta < 0:
        print("\n  MARGINAL: Meta AAFE slightly improved")
    else:
        print("\n  FAIL: Meta AAFE did not improve")
        print("  Docking features failed to improve holdout prediction.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
