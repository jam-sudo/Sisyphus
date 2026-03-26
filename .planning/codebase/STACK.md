# Technology Stack

**Analysis Date:** 2026-03-24

## Languages

**Primary:**
- Python 3.10+ — entire codebase (`src/sisyphus/`, `scripts/`, `tests/`)

**Secondary:**
- YAML — physiology definitions (`data/physiology/*.yaml`), compound configs (`data/compounds/*.yaml`)
- JSON — ML model serialization (`models/**/*.json`), reference data (`data/reference/*.json`)
- CSV — DrugBank extracts (`data/drugbank/*.csv`), ADME reference (`data/reference/adme_reference.csv`)
- TSV (tab-delimited) — TDC datasets (`data/caco2_wang.tab`, `data/ppbr_az.tab`)

## Runtime

**Environment:**
- Python 3.10+ (requires-python `>=3.10` in `pyproject.toml`; runtime detected as 3.10.12)
- No `.python-version` file — version not pinned beyond pyproject.toml floor

**Package Manager:**
- pip + hatchling build backend (`pyproject.toml` line 1-3)
- No lockfile present (no `uv.lock`, `poetry.lock`, `Pipfile.lock`, or `requirements.txt`)
- Editable install expected: `pip install -e ".[ml,chem,dev]"`

## Frameworks

**Core:**
- NumPy `>=1.24` — array operations, ODE state vectors, Monte Carlo sampling (used in every layer)
- SciPy `>=1.10` — ODE integration via `solve_ivp` with LSODA method (`src/sisyphus/engine/solver.py`)
- PyYAML `>=6.0` — physiology YAML parsing (`src/sisyphus/graph/builder.py`, `src/sisyphus/compounds.py`)

**ML:**
- XGBoost `>=2.0` — ADME property prediction (fup, CLint, RBP, VDss, Peff) and direct Cmax prediction (`src/sisyphus/predict/adme.py`, `src/sisyphus/ml/models.py`)
- LightGBM `>=4.0` — declared in optional deps but not imported anywhere in current source
- scikit-learn `>=1.3` — declared in optional deps but not directly imported in production code

**Chemistry:**
- RDKit (no version pin) — SMILES parsing, molecular descriptors, Morgan fingerprints, InChIKey generation (`src/sisyphus/descriptors.py`, `src/sisyphus/predict/chemistry.py`, `src/sisyphus/predict/drugbank.py`)

**Experimental/Scripts only:**
- PyTorch — used only in `scripts/train_e2e.py` for E2E differentiable PBPK (Phase 2B experiment, negative result)
- Saved model: `models/e2e_mlp.pt` (not loaded by production pipeline)

**Testing:**
- pytest `>=7.0` — test runner, configured in `pyproject.toml`
- Custom markers: `slow` (run with `--run-slow`)

**Linting/Formatting:**
- ruff `>=0.4` — linter + formatter
  - Line length: 100
  - Target: Python 3.10
  - Rules: E, F, I, W, UP (pycodestyle, pyflakes, isort, warnings, pyupgrade)

## Key Dependencies

**Critical (production pipeline requires all):**
- `numpy` — every module imports it; array backbone for ODE, MC, and PK extraction
- `scipy` — `solve_ivp` is the sole ODE integrator; no fallback exists
- `pyyaml` — physiology and compound YAML loading; graph cannot build without it
- `xgboost` — 7 ADME models + 1 direct PK model; predict and ml layers fail without it
- `rdkit` — SMILES validation, descriptor computation, InChIKey lookup; predict layer hard-depends on it

**Infrastructure:**
- `logging` (stdlib) — structured logging throughout; no external log framework
- `argparse` (stdlib) — CLI entry point (`src/sisyphus/cli.py`)
- `json` (stdlib) — reference data, model metadata, DrugBank extraction summary
- `csv` (stdlib) — DrugBank CSV parsing (`src/sisyphus/predict/drugbank.py`)

**Declared but unused in production:**
- `lightgbm` — in `[project.optional-dependencies].ml` but no import found
- `scikit-learn` — in `[project.optional-dependencies].ml` but no direct import in `src/sisyphus/`

## Configuration

**Environment:**
- No `.env` files present
- No environment variables required — all configuration is via YAML files, CLI args, and hardcoded constants
- Paths resolved relative to source file locations (e.g., `Path(__file__).resolve().parent.parent...`)

**Build:**
- `pyproject.toml` — single source of project metadata, build config, tool config
- Build backend: hatchling
- Wheel packages: `src/sisyphus` (src layout)
- Entry point: `sisyphus = "sisyphus.cli:main"`

**Ruff (linter):**
- Config in `pyproject.toml` under `[tool.ruff]` and `[tool.ruff.lint]`
- Line length 100, Python 3.10 target
- Selected rules: E (pycodestyle), F (pyflakes), I (isort), W (warnings), UP (pyupgrade)

**Pytest:**
- Config in `pyproject.toml` under `[tool.pytest.ini_options]`
- Test paths: `tests/`
- Markers: `slow`

## Platform Requirements

**Development:**
- Python 3.10+
- RDKit (binary distribution via `pip install rdkit` or conda)
- XGBoost 2.0+ (pip installable)
- All pre-trained models in `models/` directory (8 XGBoost JSON files + 1 meta-learner)
- DrugBank extracted data in `data/drugbank/` (5 CSV files from `scripts/extract_drugbank.py`)
- Reference clinical PK data in `data/reference/` (JSON + CSV)

**Production:**
- Same as development — no separate deployment configuration
- No Docker, no CI/CD pipeline (no `.github/`, no `Dockerfile`)
- CLI interface only: `sisyphus predict --smiles "..." --dose 500`

## Key Configuration Files

| File | Purpose | Notable Settings |
|------|---------|-----------------|
| `pyproject.toml` | Project metadata, dependencies, tool config | hatchling build, ruff lint, pytest config |
| `data/physiology/reference_man.yaml` | 34-node ICRP Reference Man body graph | Cardiac output, organ volumes, enzyme abundances |
| `data/physiology/pediatric_5y.yaml` | Pediatric (5-year) overlay | Extension proof for Phase 3 |
| `data/physiology/sc_overlay.yaml` | Subcutaneous injection overlay | Extension proof for Phase 3 |
| `data/physiology/tumor_overlay.yaml` | Tumor compartment overlay | Extension proof for Phase 3 |
| `data/reference/clinical_pk.json` | 291 drugs with observed Cmax, AUC | Holdout/training split flags |
| `data/reference/holdout.json` | Holdout drug list (inviolable) | Train/holdout partition |
| `data/reference/adme_reference.csv` | Measured ADME for validation drugs | fup, rbp, CLint, Peff |
| `data/drugbank/extraction_summary.json` | DrugBank 5.1 extraction metadata | 15,485 small molecules, 1,774 enzyme annotations |

## ML Model Assets

| Model File | Purpose | Training Data |
|------------|---------|---------------|
| `models/adme/xgboost_fup.json` | Fraction unbound (v1, log10-space) | TDC + augmented |
| `models/adme/xgboost_fup_v2.json` | Fraction unbound (v2, logit-space) | TDC + augmented; R^2=0.41 |
| `models/adme/xgboost_clint.json` | Intrinsic clearance | TDC Hepatocyte_AZ; R^2=0.24 |
| `models/adme/xgboost_rbp.json` | Blood:plasma ratio | 50 compounds; R^2=-0.08 (poor) |
| `models/adme/xgboost_vdss.json` | Volume of distribution | TDC |
| `models/adme/xgboost_peff.json` | Permeability (Caco-2 -> Peff) | TDC Caco2_Wang; R^2=0.71 |
| `models/adme/logp_correction.json` | logP residual correction | Trained on Crippen-vs-exp residuals |
| `models/direct_pk/xgboost_cmax.json` | Direct Cmax prediction | 1,128 MMPK drugs, 2057 features |
| `models/meta_learner/xgboost_meta.json` | Meta-learner (not currently loaded) | N/A |
| `models/e2e_mlp.pt` | PyTorch E2E MLP (experimental) | Phase 2B negative result; unused |

## Feature Vector Specification

All XGBoost models consume the same 2057-element feature vector computed by `src/sisyphus/descriptors.py`:
- Morgan fingerprint: 2048 bits (radius=2)
- 9 normalized RDKit descriptors: LogP, TPSA/200, MW/600, HBA/10, HBD/5, RotBonds/15, RingCount/5, FractionCSP3, MolMR/150

---

*Stack analysis: 2026-03-24*
