# Model Manifest Schema

Every XGBoost model artifact loaded by the pipeline has a sibling `<model>.meta.json` manifest file. The manifest captures training provenance and the feature-schema fingerprint needed to detect silent incompatibilities.

## File location

Next to the model artifact:

```
models/adme/xgboost_fup_v2.json        ← model
models/adme/xgboost_fup_v2.meta.json   ← manifest
```

Exception: the existing `models/direct_pk/meta.json` covers `xgboost_cmax.json` (historical co-located style). New artifacts use the sibling `.meta.json` pattern.

## Required fields

```json
{
  "version": "string — model version tag (e.g. v1, v2_mw)",
  "target": "string — what the model predicts, with units",
  "trained_on": {
    "dataset_path": "string or null",
    "sha256": "string or 'unknown_legacy'"
  },
  "feature_schema": {
    "name": "string — canonical pipeline name",
    "n_features": "integer",
    "sha256": "string — hash of compute_features(CANONICAL_SMILES).tobytes()",
    "description": "string — brief"
  },
  "trained_at": "ISO-8601 timestamp or 'unknown_legacy'",
  "n_drugs_original": "integer or null",
  "n_drugs_excluded": "integer or null",
  "holdout_version": "string or 'unknown_legacy'",
  "holdout_metric": {
    "name": "string (e.g. AAFE, scaffold_cv_r2, train_r2)",
    "value": "number or null"
  },
  "hyperparameters": "object or {}",
  "retrained_reason": "string or 'unknown_legacy'"
}
```

All keys are REQUIRED. Fields that cannot be recovered from historical context use the literal string `"unknown_legacy"` (or `null` for numeric fields where documented). This makes missing provenance explicit and searchable.

## `feature_schema.sha256`

The reproducible fingerprint of the feature-vector construction pipeline. Computed as:

```python
import hashlib
from sisyphus.descriptors import compute_features

CANONICAL_SMILES = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"  # caffeine
features = compute_features(CANONICAL_SMILES)      # numpy ndarray
sha256 = hashlib.sha256(features.tobytes()).hexdigest()
```

**Why hash?** `n_features: 2057` alone does not catch changes in feature ORDER or IDENTITY. Swapping Morgan radius from 2 → 3 keeps the count but changes the bits. Adding a new RDKit descriptor also often keeps the same count by replacing another. Hashing the byte-representation of the feature vector on a canonical input catches all such drifts.

**Canonical inputs** currently used (hashes computed under the `requirements-lock.txt` environment — rdkit 2026.3.1, numpy 2.2.6):

| Feature pipeline | Canonical SMILES | Canonical hash |
|---|---|---|
| `compute_features_v1` (2048 Morgan r=2 + 9 RDKit descriptors, 2057 total, float64) | `CN1C=NC2=C1C(=O)N(C(=O)N2C)C` (caffeine) | `b41dddd78533661b8f4aed86bdb7c5f55d15ed3cd2ad20b39e9100557eda631e` |
| `logp_corr_6` (logp + mw + tpsa + hbd + hba + rot. bonds, 6 total, float64, shape (1, 6)) | same | `a8da0c08b6425b5d5967e38d60fe0130119ad66282a1fac2016bcbf3e9ab0e6d` |

A manifest whose `feature_schema.sha256` does not match the hash computed at load time emits a warning (the loaded model may produce garbage if the feature extractor has evolved).

**Environment coupling**: the canonical hashes are stable within the pinned RDKit + numpy versions. Because Morgan fingerprint bits and some `Descriptors.*` outputs can change across RDKit versions, the manifests ship hashes computed under the lockfile env — that is the environment CI and fresh installs both use. When upgrading RDKit, recompute the canonical hashes in a fresh venv, update this table, and regenerate the manifests.

## Registry behavior

`sisyphus.ml.registry.verify_model_artifact` enforces the runtime schema:

- **Manifest missing or incomplete**: raise `ValueError`; the model cannot load.
- **Artifact hash or feature hash mismatch**: raise `ValueError`; the model cannot load.
- **New model manifest**: write a sibling `<model>.meta.json` with all required fields and matching hashes.

`load_manifest` and `validate_manifest` remain available for offline diagnostics.

## Legacy mini-manifests

Two models already had partial manifests in a different shape:

- `models/adme/xgboost_vdss_v2_meta.json`
- `models/adme/xgboost_bioavailability_meta.json`
- `models/direct_pk/meta.json`

Their existing fields (n_train, scaffold_cv_r2, source, etc.) are preserved by mapping to the unified schema:

- `n_train` → `n_drugs_original`
- `scaffold_cv_r2` → `holdout_metric: {name: "scaffold_cv_r2", value: …}`
- `source` → `trained_on.dataset_path` (string only; sha256 unknown → `"unknown_legacy"`)

The original mini-manifest files are deleted after migration to avoid divergence. The `models/direct_pk/meta.json` is RENAMED to `models/direct_pk/xgboost_cmax.meta.json` to match the sibling-pattern.

## Regeneration

```bash
# Recompute a canonical feature hash (for schema drift checks)
python3 -c "import hashlib; from sisyphus.descriptors import compute_features; \
  print(hashlib.sha256(compute_features('CN1C=NC2=C1C(=O)N(C(=O)N2C)C').tobytes()).hexdigest())"
```

Update this doc's canonical-hash table if `compute_features` changes.
