# External Integrations

**Analysis Date:** 2026-03-24

## APIs & External Services

**None.** Sisyphus is a fully offline, self-contained computational platform. No API calls, no network requests, no cloud services. All data is local, all computation is local.

## Data Sources

### DrugBank 5.1 (Extracted)

- **Purpose:** Drug property enrichment — experimental logP, measured fup, ChemAxon pKa, CYP/UGT substrate annotations
- **Format:** Pre-extracted CSV files in `data/drugbank/`
- **Source:** DrugBank full database XML (1.8 GB), extracted via `scripts/extract_drugbank.py`
- **Access pattern:** Lazy-loaded singleton `DrugBankLookup` class in `src/sisyphus/predict/drugbank.py`
- **Lookup:** 2-tier — canonical SMILES exact match, InChIKey-14 fallback (via RDKit InChI conversion)
- **Scale:** 15,485 small molecules, 1,774 enzyme annotations, 1,439 parsed fup values, 6,277 experimental properties
- **Feature flags:** `DrugBankConfig` dataclass allows disabling individual enrichment types for ablation studies

**Data files:**

| File | Rows | Content |
|------|------|---------|
| `data/drugbank/drugs.csv` | 15,485 | Master: SMILES, InChIKey, MW, calculated pKa (acidic + basic), PSA |
| `data/drugbank/enzyme_annotations.csv` | 5,873 | Drug -> enzyme -> action (substrate/inhibitor/inducer) |
| `data/drugbank/pk_data.csv` | 10,769 | PK text fields + parsed numeric (half-life, protein binding, Vd, CL) |
| `data/drugbank/experimental_properties.csv` | 6,277 | Measured logP, solubility, melting point |
| `data/drugbank/transporter_annotations.csv` | ~1,156 | Drug -> transporter -> action |
| `data/drugbank/extraction_summary.json` | 1 | Extraction metadata and quality metrics |

**CYP enzyme normalization** (DrugBank names to Sisyphus tags):
```python
# src/sisyphus/predict/drugbank.py lines 21-31
"Cytochrome P450 3A4" -> "CYP3A4"
"Cytochrome P450 2D6" -> "CYP2D6"
"Cytochrome P450 1A2" -> "CYP1A2"
"Cytochrome P450 2C9" -> "CYP2C9"
"Cytochrome P450 2E1" -> "CYP2E1"
"Cytochrome P450 3A5" -> "CYP3A4"   # merged into same family
"Cytochrome P450 2C19" -> "CYP2C9"  # merged into 2C subfamily
"Cytochrome P450 2C8"  -> "CYP2C9"  # merged into 2C subfamily
```

**UGT enzyme normalization** (only isoforms with abundance in `reference_man.yaml`):
```python
# src/sisyphus/predict/drugbank.py lines 35-42
"UDP-glucuronosyltransferase 2B7" -> "UGT2B7"
"UDP-glucuronosyltransferase 1A1" -> "UGT1A1"
"UDP-glucuronosyltransferase 1A4" -> "UGT1A4"
"UDP-glucuronosyltransferase 1A9" -> "UGT1A9"
```

### ICRP Reference Man Physiology

- **Purpose:** Body graph definition — 34 nodes (organs, blood pools, GI tract), edges (blood flow, transit, absorption), enzyme abundances
- **Format:** YAML
- **Files:**
  - `data/physiology/reference_man.yaml` — primary 70 kg adult male graph
  - `data/physiology/pediatric_5y.yaml` — pediatric overlay
  - `data/physiology/sc_overlay.yaml` — subcutaneous injection route overlay
  - `data/physiology/tumor_overlay.yaml` — tumor compartment overlay
- **Access:** `src/sisyphus/graph/builder.py` via `build_from_yaml()`, loaded by pipeline at `src/sisyphus/pipeline/predict.py` line 98

### Clinical PK Reference Data

- **Purpose:** Benchmark validation — observed Cmax and AUC for known drugs
- **Format:** JSON
- **File:** `data/reference/clinical_pk.json` — 291 drugs with SMILES, dose, route, Cmax, AUC
- **Access:** `src/sisyphus/validation/reference.py` via `load_reference()`
- **Split:** Holdout set defined in `data/reference/holdout.json` — inviolable (Invariant 5)

### ADME Measured Reference

- **Purpose:** Validation of ADME predictions against measured values
- **Format:** CSV with columns: name, smiles, mw, logP, fup, rbp, clint_3a4, peff
- **File:** `data/reference/adme_reference.csv`

### TDC (Therapeutics Data Commons) Datasets

- **Purpose:** Training data for XGBoost ADME models
- **Format:** Tab-delimited (.tab) with columns: Drug_ID, Drug (SMILES), Y (target value)
- **Files:**
  - `data/caco2_wang.tab` — Caco-2 permeability (Peff model training)
  - `data/ppbr_az.tab` — Plasma protein binding ratio (fup model training)
- **Access:** Used by training scripts in `scripts/` (e.g., `scripts/train_fup_v2.py`, `scripts/train_peff.py`)

### Compound YAML Configs

- **Purpose:** Curated drug configurations with known properties for engine validation
- **Format:** YAML
- **Files:**
  - `data/compounds/midazolam.yaml` — CYP3A4 substrate, benchmark drug
  - `data/compounds/warfarin.yaml` — CYP2C9 substrate, benchmark drug
  - `data/compounds/caffeine.yaml` — CYP1A2 substrate, benchmark drug
  - `data/compounds/propranolol.yaml` — basic drug, benchmark
  - `data/compounds/midazolam_sc.yaml` — subcutaneous route variant
- **Access:** `src/sisyphus/compounds.py` via `load_compound()`

## External Library Dependencies

### RDKit (Chemistry)

- **Criticality:** Hard dependency for predict layer and descriptors
- **Used for:**
  - SMILES parsing and canonicalization (`Chem.MolFromSmiles`, `Chem.MolToSmiles`)
  - Molecular descriptors: MW, LogP, TPSA, HBD, HBA, RotatableBonds, RingCount, FractionCSP3, MolMR
  - Morgan fingerprints (2048-bit, radius 2) via `AllChem.GetMorganFingerprintAsBitVect`
  - SMARTS pattern matching for pKa estimation and prodrug detection
  - InChI/InChIKey generation for DrugBank fallback lookup
- **Import locations:**
  - `src/sisyphus/descriptors.py` — `from rdkit import Chem; from rdkit.Chem import AllChem, Descriptors`
  - `src/sisyphus/predict/chemistry.py` — `from rdkit import Chem; from rdkit.Chem import Descriptors`
  - `src/sisyphus/predict/drugbank.py` — `from rdkit import Chem; from rdkit.Chem.inchi import InchiToInchiKey, MolToInchi`
  - `scripts/extract_drugbank.py` — `from rdkit import Chem, rdBase`

### XGBoost (ML)

- **Criticality:** Hard dependency for predict and ml layers
- **Used for:** Loading and running 8 pre-trained models (JSON format)
- **Import locations:**
  - `src/sisyphus/predict/adme.py` — ADME property prediction (fup, CLint, RBP, VDss, Peff, logP correction)
  - `src/sisyphus/ml/models.py` — Direct Cmax prediction
  - `src/sisyphus/predict/chemistry.py` — logP residual correction model (lazy import)
- **Model format:** XGBoost JSON (`xgb.XGBRegressor.load_model()`)

### SciPy (Numerical)

- **Criticality:** Hard dependency for engine layer
- **Used for:** ODE integration via `scipy.integrate.solve_ivp` with LSODA method
- **Import location:** `src/sisyphus/engine/solver.py` — sole usage
- **Solver configuration:**
  - Full solve: rtol=1e-8, atol=1e-10, 500 time points
  - MC solve: rtol=1e-4, atol=1e-6, adaptive grid (no t_eval)

### NumPy (Numerical)

- **Criticality:** Hard dependency for all layers
- **Used for:** Array operations, random number generation (`np.random.Generator`), Distribution sampling, trapezoid integration, linspace, clip, log, exp
- **Compatibility:** Handles both numpy 1.x (`np.trapz`) and 2.0+ (`np.trapezoid`) via `getattr` fallback in `src/sisyphus/engine/solver.py` line 142

### PyYAML (Configuration)

- **Criticality:** Hard dependency for graph and compounds layers
- **Used for:** Loading physiology YAML and compound YAML via `yaml.safe_load()`
- **Import locations:**
  - `src/sisyphus/graph/builder.py`
  - `src/sisyphus/compounds.py`

## Data Storage

**Databases:**
- None. All data is file-based (JSON, CSV, YAML, TSV).

**File Storage:**
- Local filesystem only. All paths resolved relative to source files.
- Pattern: `Path(__file__).resolve().parent.parent...` for relative resolution

**Caching:**
- In-memory caching only:
  - XGBoost model cache: `_model_cache` dict in `src/sisyphus/predict/adme.py` line 65
  - DrugBank singleton: module-level `_INSTANCE` in `src/sisyphus/predict/drugbank.py` line 257
  - logP correction model: cached as function attribute `compute_profile._logp_model` in `src/sisyphus/predict/chemistry.py` line 342

## Authentication & Identity

**None.** No auth required — offline system with no external service connections.

## Monitoring & Observability

**Error Tracking:**
- None (no Sentry, no external error reporting)

**Logs:**
- Python `logging` module throughout (`logging.getLogger(__name__)`)
- CLI configures log level: `--verbose` sets DEBUG, default is WARNING
- Log format: `%(levelname)s %(name)s: %(message)s`

## CI/CD & Deployment

**Hosting:**
- Local development only — no deployment target configured

**CI Pipeline:**
- None configured. No `.github/` directory, no CI config files.

**Tests:**
- pytest with `testpaths = ["tests"]`
- Three test directories: `tests/unit/`, `tests/integration/`, `tests/benchmark/`
- Run: `pytest` (all), `pytest --run-slow` (including slow-marked tests)

## Environment Configuration

**Required env vars:**
- None. All configuration is file-based.

**Secrets location:**
- No secrets management. DrugBank data is pre-extracted; no API keys needed at runtime.
- DrugBank source XML (`full database.xml`) referenced by extraction script but not required for runtime.

## Webhooks & Callbacks

**Incoming:**
- None

**Outgoing:**
- None

## Path Resolution Pattern

All internal path resolution uses the same pattern — relative to source file location:

```python
# Example from src/sisyphus/predict/adme.py line 28
_MODEL_DIR = Path(__file__).resolve().parent.parent.parent.parent / "models" / "adme"

# Example from src/sisyphus/pipeline/predict.py line 21
_PHYSIOLOGY_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "physiology"

# Example from src/sisyphus/validation/reference.py line 19
_DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "reference"
```

This pattern assumes `src/sisyphus/` is 4 levels below the repository root. It works for both editable installs and direct execution, but would break if the package were installed as a wheel to site-packages (since `data/` and `models/` directories would not be alongside the installed package).

---

*Integration audit: 2026-03-24*
