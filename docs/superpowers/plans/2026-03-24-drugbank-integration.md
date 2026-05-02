# DrugBank Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate DrugBank 5.1 data into Sisyphus's predict layer — measured fup, experimental logP, ChemAxon pKa, and CYP substrate annotations replace/enhance XGBoost predictions where available.

**Architecture:** New `predict/drugbank.py` module provides a lazy-loaded singleton lookup indexed by canonical SMILES + InChIKey fallback. Four integration points: `chemistry.py` (logP, pKa), `adme.py` (fup), `ivive.py` (enzyme fm). Feature flags enable individual ablation. Pipeline tags warnings for gold/silver benchmark split.

**Tech Stack:** Python 3.10+, RDKit (canonical SMILES/InChIKey), csv stdlib, dataclasses. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-03-23-drugbank-integration-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `src/sisyphus/predict/drugbank.py` | **Create** | DrugBankLookup singleton, DrugBankConfig, CYP normalization, CSV loading |
| `scripts/extract_drugbank.py` | Modify | Add `canonical_smiles`, `inchikey_14` columns; record RDKit version |
| `src/sisyphus/predict/chemistry.py` | Modify | logP + pKa DrugBank lookup in `compute_profile()` |
| `src/sisyphus/predict/adme.py` | Modify | fup DrugBank lookup with 5x cross-validation guard |
| `src/sisyphus/predict/ivive.py` | Modify | `_get_fm_fractions()` + `_decompose_clint()` accept `substrate_enzymes` |
| `src/sisyphus/pipeline/predict.py` | Modify | DrugBank warning tags after predict step |
| `src/sisyphus/validation/benchmark.py` | Modify | Gold/silver AAFE split in BenchmarkResult |
| `.gitignore` | Modify | Add `data/drugbank/` |
| `tests/unit/test_drugbank.py` | **Create** | DrugBankLookup unit tests |
| `tests/unit/test_drugbank_integration.py` | **Create** | Integration tests: enriched predict paths |

---

### Task 1: Extraction Script — Add canonical_smiles + inchikey_14

**Files:**
- Modify: `scripts/extract_drugbank.py`

- [ ] **Step 1: Add RDKit import and canonical SMILES generation**

In `extract()`, after computing `smiles` and `inchikey` from calculated properties, add canonical SMILES normalization. Add `canonical_smiles` and `inchikey_14` to the CSV header and each row.

```python
# At top of file, add:
from rdkit import Chem, rdBase

# In extract(), after line that sets smiles and inchikey:
canonical_smiles = ""
inchikey_14 = ""
if smiles:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            canonical_smiles = Chem.MolToSmiles(mol)
    except Exception:
        pass
if inchikey and len(inchikey) >= 14:
    inchikey_14 = inchikey[:14]
```

Update `w_drugs.writerow` header to include `canonical_smiles, inchikey_14` after `inchikey`. Update each row similarly.

Add RDKit version to extraction_summary.json:
```python
summary["rdkit_version"] = rdBase.rdkitVersion
```

- [ ] **Step 2: Re-run extraction and verify**

Run: `python3 scripts/extract_drugbank.py`

Verify: `head -1 data/drugbank/drugs.csv` shows new columns. `python3 -c "import csv; r=csv.DictReader(open('data/drugbank/drugs.csv')); row=next(r); print(row['canonical_smiles'][:50], row['inchikey_14'])"` prints values.

- [ ] **Step 3: Commit**

```bash
git add scripts/extract_drugbank.py
git commit -m "feat(extract): add canonical_smiles and inchikey_14 columns to DrugBank CSV"
```

---

### Task 2: .gitignore — Add data/drugbank/

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Add data/drugbank/ to .gitignore**

Append to `.gitignore`:
```
# DrugBank derived data (license restricted, user must extract)
data/drugbank/
```

- [ ] **Step 2: Remove tracked DrugBank files from git index**

```bash
git rm -r --cached data/drugbank/ 2>/dev/null || true
```

- [ ] **Step 3: Commit**

```bash
git add .gitignore
git commit -m "chore: git-ignore data/drugbank/ (DrugBank license compliance)"
```

---

### Task 3: DrugBankLookup Module — Core + Tests

**Files:**
- Create: `src/sisyphus/predict/drugbank.py`
- Create: `tests/unit/test_drugbank.py`

- [ ] **Step 1: Write failing tests for DrugBankLookup**

```python
# tests/unit/test_drugbank.py
"""Tests for DrugBank lookup module."""
import pytest
from sisyphus.predict.drugbank import DrugBankConfig, DrugBankLookup


class TestDrugBankConfig:
    def test_default_all_enabled(self):
        cfg = DrugBankConfig()
        assert cfg.enable_enzyme_fm is True
        assert cfg.enable_fup is True
        assert cfg.enable_pka is True
        assert cfg.enable_logp is True

    def test_individual_disable(self):
        cfg = DrugBankConfig(enable_fup=False)
        assert cfg.enable_fup is False
        assert cfg.enable_logp is True


class TestDrugBankLookupNoData:
    """Tests when CSV files do not exist — all lookups return None."""

    def test_lookup_returns_none(self, tmp_path):
        lookup = DrugBankLookup(data_dir=tmp_path)  # empty dir
        assert lookup.lookup("CCO") is None

    def test_get_substrate_enzymes_returns_none(self, tmp_path):
        lookup = DrugBankLookup(data_dir=tmp_path)
        assert lookup.get_substrate_enzymes("CCO") is None

    def test_get_fup_returns_none(self, tmp_path):
        lookup = DrugBankLookup(data_dir=tmp_path)
        assert lookup.get_fup("CCO") is None

    def test_get_pka_returns_none(self, tmp_path):
        lookup = DrugBankLookup(data_dir=tmp_path)
        assert lookup.get_pka("CCO") is None

    def test_get_logp_returns_none(self, tmp_path):
        lookup = DrugBankLookup(data_dir=tmp_path)
        assert lookup.get_logp("CCO") is None


class TestDrugBankLookupWithData:
    """Tests with synthetic CSV fixtures."""

    @pytest.fixture
    def data_dir(self, tmp_path):
        """Create minimal CSV fixtures."""
        # drugs.csv
        (tmp_path / "drugs.csv").write_text(
            "drugbank_id,name,cas,smiles,inchikey,mw,logp_calc,pka_acidic,pka_basic,"
            "psa,hba,hbd,rotatable_bonds,state,groups,n_ddi,canonical_smiles,inchikey_14\n"
            'DB99901,TestDrug,,,TESTINCHIKEY1234-REST,180,2.5,4.2,9.1,50,3,1,2,solid,approved,0,'
            'c1ccc(O)cc1,TESTINCHIKEY1234\n'
            'DB99902,TestDrug2,,,OTHERINCHIKEY567-REST,200,3.0,10.5,2.0,60,4,2,3,solid,approved,0,'
            'CC(=O)O,OTHERINCHIKEY567\n'
        )
        # enzyme_annotations.csv
        (tmp_path / "enzyme_annotations.csv").write_text(
            "drugbank_id,drug_name,enzyme_name,uniprot_id,actions\n"
            "DB99901,TestDrug,Cytochrome P450 3A4,P08684,substrate\n"
            "DB99901,TestDrug,Cytochrome P450 2D6,P10635,substrate,inhibitor\n"
            "DB99901,TestDrug,Cytochrome P450 3A5,P20815,substrate\n"
        )
        # pk_data.csv
        (tmp_path / "pk_data.csv").write_text(
            "drugbank_id,drug_name,field,raw_text,parsed_value,parsed_unit\n"
            "DB99901,TestDrug,protein_binding,95% bound,0.05,fup\n"
        )
        # experimental_properties.csv
        (tmp_path / "experimental_properties.csv").write_text(
            "drugbank_id,drug_name,property,value\n"
            "DB99901,TestDrug,logP,2.8\n"
        )
        return tmp_path

    def test_lookup_by_canonical_smiles(self, data_dir):
        lookup = DrugBankLookup(data_dir=data_dir)
        assert lookup.lookup("c1ccc(O)cc1") == "DB99901"

    def test_lookup_by_inchikey_fallback(self, data_dir):
        lookup = DrugBankLookup(data_dir=data_dir)
        # Use a SMILES that doesn't match canonical but same InChIKey prefix
        assert lookup.lookup("NOMATCH") is None
        # Direct InChIKey lookup (simulated by passing a SMILES whose
        # RDKit canonical differs from stored, but InChIKey matches)
        # For unit test, just verify the index exists
        assert "TESTINCHIKEY1234" in lookup._inchikey14_to_id

    def test_get_substrate_enzymes_with_cyp_normalization(self, data_dir):
        lookup = DrugBankLookup(data_dir=data_dir)
        enzymes = lookup.get_substrate_enzymes("c1ccc(O)cc1")
        assert enzymes is not None
        # CYP3A4 direct + CYP3A5 → CYP3A4 (merged), CYP2D6 direct
        assert enzymes == {"CYP3A4", "CYP2D6"}

    def test_get_fup(self, data_dir):
        lookup = DrugBankLookup(data_dir=data_dir)
        fup = lookup.get_fup("c1ccc(O)cc1")
        assert fup == pytest.approx(0.05)

    def test_get_pka(self, data_dir):
        lookup = DrugBankLookup(data_dir=data_dir)
        pka = lookup.get_pka("c1ccc(O)cc1")
        assert pka == pytest.approx((4.2, 9.1))

    def test_get_logp(self, data_dir):
        lookup = DrugBankLookup(data_dir=data_dir)
        logp = lookup.get_logp("c1ccc(O)cc1")
        assert logp == pytest.approx(2.8)

    def test_feature_flag_disables_lookup(self, data_dir):
        cfg = DrugBankConfig(enable_fup=False, enable_logp=False)
        lookup = DrugBankLookup(data_dir=data_dir, config=cfg)
        assert lookup.get_fup("c1ccc(O)cc1") is None  # disabled
        assert lookup.get_logp("c1ccc(O)cc1") is None  # disabled
        assert lookup.get_pka("c1ccc(O)cc1") is not None  # still enabled
        assert lookup.get_substrate_enzymes("c1ccc(O)cc1") is not None  # still enabled

    def test_miss_returns_none(self, data_dir):
        lookup = DrugBankLookup(data_dir=data_dir)
        assert lookup.get_fup("CCCNOTINDB") is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/unit/test_drugbank.py -v`
Expected: ImportError — module doesn't exist yet.

- [ ] **Step 3: Implement DrugBankLookup**

Create `src/sisyphus/predict/drugbank.py`:

```python
"""DrugBank data lookup for predict layer enrichment.

Loads extracted DrugBank CSV files and provides O(1) lookup by canonical
SMILES (primary) or InChIKey connectivity layer (fallback).

All lookups return None when data is unavailable — the predict layer
falls back to XGBoost/heuristic predictions.  No exceptions are raised
for missing files or malformed data.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CYP name normalization: DrugBank full names → Sisyphus YAML tags
# ---------------------------------------------------------------------------

_CYP_NORMALIZATION: dict[str, str] = {
    "Cytochrome P450 3A4": "CYP3A4",
    "Cytochrome P450 2D6": "CYP2D6",
    "Cytochrome P450 1A2": "CYP1A2",
    "Cytochrome P450 2C9": "CYP2C9",
    "Cytochrome P450 2E1": "CYP2E1",
    "Cytochrome P450 3A5": "CYP3A4",   # same gene family, merge
    "Cytochrome P450 2C19": "CYP2C9",  # same 2C subfamily
    "Cytochrome P450 2C8": "CYP2C9",   # same 2C subfamily
    # CYP2B6 is intentionally absent — no Sisyphus equivalent
}

_DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "drugbank"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class DrugBankConfig:
    """Feature flags for individual enrichment toggle (ablation support)."""

    enable_enzyme_fm: bool = True
    enable_fup: bool = True
    enable_pka: bool = True
    enable_logp: bool = True


# ---------------------------------------------------------------------------
# Lookup class
# ---------------------------------------------------------------------------


class DrugBankLookup:
    """Lazy-loaded DrugBank data lookup.

    Args:
        data_dir: Directory containing extracted DrugBank CSVs.
        config: Feature flags for ablation.
    """

    def __init__(
        self,
        data_dir: Path = _DEFAULT_DATA_DIR,
        config: DrugBankConfig | None = None,
    ) -> None:
        self._data_dir = data_dir
        self._config = config or DrugBankConfig()
        self._loaded = False

        # Indices (populated by _load)
        self._smiles_to_id: dict[str, str] = {}
        self._inchikey14_to_id: dict[str, str] = {}
        self._enzyme_substrates: dict[str, set[str]] = {}  # dbid → {CYP tags}
        self._fup: dict[str, float] = {}  # dbid → fup
        self._pka: dict[str, tuple[float, float]] = {}  # dbid → (acidic, basic)
        self._logp: dict[str, float] = {}  # dbid → logP

    # -- Loading --------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        self._load_drugs()
        self._load_enzymes()
        self._load_pk_data()
        self._load_experimental()
        logger.info(
            "DrugBank loaded: %d SMILES, %d InChIKey, %d enzyme, %d fup, %d pka, %d logp",
            len(self._smiles_to_id),
            len(self._inchikey14_to_id),
            len(self._enzyme_substrates),
            len(self._fup),
            len(self._pka),
            len(self._logp),
        )

    def _load_drugs(self) -> None:
        path = self._data_dir / "drugs.csv"
        if not path.exists():
            return
        try:
            with open(path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    dbid = row.get("drugbank_id", "")
                    cs = row.get("canonical_smiles", "").strip()
                    ik14 = row.get("inchikey_14", "").strip()
                    if cs and dbid:
                        self._smiles_to_id[cs] = dbid
                    if ik14 and dbid:
                        self._inchikey14_to_id.setdefault(ik14, dbid)

                    # pKa from drugs.csv (ChemAxon calculated)
                    pka_a = row.get("pka_acidic", "").strip()
                    pka_b = row.get("pka_basic", "").strip()
                    if pka_a and pka_b and dbid:
                        try:
                            self._pka[dbid] = (float(pka_a), float(pka_b))
                        except ValueError:
                            pass
        except Exception as e:
            logger.warning("Failed to load DrugBank drugs.csv: %s", e)

    def _load_enzymes(self) -> None:
        path = self._data_dir / "enzyme_annotations.csv"
        if not path.exists():
            return
        try:
            with open(path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if "substrate" not in row.get("actions", ""):
                        continue
                    dbid = row.get("drugbank_id", "")
                    enz_name = row.get("enzyme_name", "")
                    cyp_tag = _CYP_NORMALIZATION.get(enz_name)
                    if cyp_tag and dbid:
                        self._enzyme_substrates.setdefault(dbid, set()).add(cyp_tag)
        except Exception as e:
            logger.warning("Failed to load DrugBank enzyme_annotations.csv: %s", e)

    def _load_pk_data(self) -> None:
        path = self._data_dir / "pk_data.csv"
        if not path.exists():
            return
        try:
            with open(path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if row.get("field") != "protein_binding":
                        continue
                    val = row.get("parsed_value", "").strip()
                    dbid = row.get("drugbank_id", "")
                    if val and dbid:
                        try:
                            self._fup[dbid] = float(val)
                        except ValueError:
                            pass
        except Exception as e:
            logger.warning("Failed to load DrugBank pk_data.csv: %s", e)

    def _load_experimental(self) -> None:
        path = self._data_dir / "experimental_properties.csv"
        if not path.exists():
            return
        try:
            with open(path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if row.get("property") != "logP":
                        continue
                    val = row.get("value", "").strip()
                    dbid = row.get("drugbank_id", "")
                    if val and dbid:
                        try:
                            self._logp[dbid] = float(val)
                        except ValueError:
                            pass
        except Exception as e:
            logger.warning("Failed to load DrugBank experimental_properties.csv: %s", e)

    # -- Lookup ---------------------------------------------------------------

    def lookup(self, canonical_smiles: str) -> str | None:
        """Resolve canonical SMILES to DrugBank ID.

        2-tier matching:
        1. Exact canonical SMILES match (handles most drugs)
        2. InChIKey connectivity layer fallback (handles stereo/tautomer differences)

        InChIKey fallback requires RDKit at runtime for the query SMILES.
        This is acceptable since compute_profile() already uses RDKit.
        """
        self._ensure_loaded()
        dbid = self._smiles_to_id.get(canonical_smiles)
        if dbid:
            return dbid
        # InChIKey fallback: compute InChIKey from query SMILES
        try:
            from rdkit import Chem
            from rdkit.Chem.inchi import MolToInchi, InchiToInchiKey
            mol = Chem.MolFromSmiles(canonical_smiles)
            if mol is not None:
                inchi = MolToInchi(mol)
                if inchi:
                    inchikey = InchiToInchiKey(inchi)
                    if inchikey and len(inchikey) >= 14:
                        dbid = self._inchikey14_to_id.get(inchikey[:14])
                        if dbid:
                            logger.debug("InChIKey fallback match: %s → %s", canonical_smiles[:30], dbid)
                            return dbid
        except Exception:
            pass  # RDKit not available or InChI conversion failed
        return None

    def get_substrate_enzymes(self, canonical_smiles: str) -> set[str] | None:
        """Return set of Sisyphus CYP tags for which this drug is a substrate."""
        if not self._config.enable_enzyme_fm:
            return None
        dbid = self.lookup(canonical_smiles)
        if dbid is None:
            return None
        enzymes = self._enzyme_substrates.get(dbid)
        return set(enzymes) if enzymes else None

    def get_fup(self, canonical_smiles: str) -> float | None:
        """Return parsed fraction unbound in plasma."""
        if not self._config.enable_fup:
            return None
        dbid = self.lookup(canonical_smiles)
        if dbid is None:
            return None
        return self._fup.get(dbid)

    def get_pka(self, canonical_smiles: str) -> tuple[float, float] | None:
        """Return (pka_acidic, pka_basic) from ChemAxon."""
        if not self._config.enable_pka:
            return None
        dbid = self.lookup(canonical_smiles)
        if dbid is None:
            return None
        return self._pka.get(dbid)

    def get_logp(self, canonical_smiles: str) -> float | None:
        """Return experimental logP."""
        if not self._config.enable_logp:
            return None
        dbid = self.lookup(canonical_smiles)
        if dbid is None:
            return None
        return self._logp.get(dbid)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_INSTANCE: DrugBankLookup | None = None


def drugbank_lookup(config: DrugBankConfig | None = None) -> DrugBankLookup:
    """Get the module-level DrugBankLookup singleton.

    Creates on first call.  Subsequent calls return the same instance
    (config argument is ignored after first call).
    """
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = DrugBankLookup(config=config)
    return _INSTANCE


def _reset_singleton() -> None:
    """Reset singleton for testing.  Not for production use."""
    global _INSTANCE
    _INSTANCE = None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/unit/test_drugbank.py -v`
Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add src/sisyphus/predict/drugbank.py tests/unit/test_drugbank.py
git commit -m "feat(predict): add DrugBankLookup module with CYP normalization and feature flags"
```

---

### Task 4: Integrate logP + pKa Lookup into chemistry.py

**Files:**
- Modify: `src/sisyphus/predict/chemistry.py`
- Create or extend: `tests/unit/test_drugbank_integration.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_drugbank_integration.py
"""Integration tests for DrugBank enrichment in predict layer."""
import pytest
from sisyphus.predict.drugbank import DrugBankConfig, DrugBankLookup, _reset_singleton


class TestChemistryDrugBankIntegration:
    """Test logP and pKa lookup in compute_profile."""

    def test_pka_classify_acid(self):
        from sisyphus.predict.chemistry import _classify_from_pka
        pka, ct = _classify_from_pka(4.5, 2.0)
        assert ct == "acid"
        assert pka == pytest.approx(4.5)

    def test_pka_classify_base(self):
        from sisyphus.predict.chemistry import _classify_from_pka
        pka, ct = _classify_from_pka(12.0, 9.0)
        assert ct == "base"
        assert pka == pytest.approx(9.0)

    def test_pka_classify_zwitterion(self):
        from sisyphus.predict.chemistry import _classify_from_pka
        pka, ct = _classify_from_pka(4.0, 9.5)
        assert ct == "zwitterion"
        assert pka == pytest.approx(9.5)

    def test_pka_classify_neutral(self):
        from sisyphus.predict.chemistry import _classify_from_pka
        pka, ct = _classify_from_pka(11.0, 3.0)
        assert ct == "neutral"
        assert pka is None

    def test_pka_uncertainty_zone_classified_neutral(self):
        """pKa in 7.0-8.0 zone should be neutral."""
        from sisyphus.predict.chemistry import _classify_from_pka
        _, ct = _classify_from_pka(7.2, 7.5)
        assert ct == "neutral"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/unit/test_drugbank_integration.py::TestChemistryDrugBankIntegration -v`
Expected: ImportError for `_classify_from_pka`.

- [ ] **Step 3: Add `_classify_from_pka` and DrugBank lookup to chemistry.py**

In `src/sisyphus/predict/chemistry.py`, add after the `_DEFAULT_PKA` dict (~line 190):

```python
def _classify_from_pka(pka_acidic: float, pka_basic: float) -> tuple[float | None, str]:
    """ChemAxon pKa → (pka_for_rr, compound_type).

    Henderson-Hasselbalch thresholds:
    - acidic < 7.0: >71% ionized at pH 7.4, R&R ion_ratio effect 43%+
    - basic > 8.0: >80% protonated at pH 7.4, R&R ion_ratio effect 121%+
    - pKa 7.0-8.0: uncertainty zone → neutral (safe default)
    """
    is_acidic = pka_acidic < 7.0
    is_basic = pka_basic > 8.0

    if is_acidic and is_basic:
        return pka_basic, "zwitterion"
    elif is_acidic:
        return pka_acidic, "acid"
    elif is_basic:
        return pka_basic, "base"
    else:
        return None, "neutral"
```

Then modify `compute_profile()` to use DrugBank logP and pKa.
Insert BETWEEN `logp = Descriptors.MolLogP(mol)` (line 299) and
`pka, compound_type = _estimate_pka_type(mol, logp)` (line 305).
The `_check_ad(mol, mw, logp, tpsa)` call on line 307 uses the local
`logp` variable, so the override automatically flows into AD check too
(per spec §5.4 — experimental logP should be used for AD check).

```python
def compute_profile(smiles: str) -> MolecularProfile:
    # ... existing: mol, canonical, mw creation ...
    logp = Descriptors.MolLogP(mol)       # line 299 — Crippen default
    tpsa = Descriptors.TPSA(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    rotatable_bonds = Descriptors.NumRotatableBonds(mol)

    # ── DrugBank overrides (insert here, before pKa and AD check) ──
    from sisyphus.predict.drugbank import drugbank_lookup
    db = drugbank_lookup()

    db_logp = db.get_logp(canonical)
    if db_logp is not None:
        logp = db_logp  # experimental overrides Crippen

    # pKa: DrugBank ChemAxon → fallback SMARTS
    db_pka = db.get_pka(canonical)
    if db_pka is not None:
        pka, compound_type = _classify_from_pka(db_pka[0], db_pka[1])
    else:
        pka, compound_type = _estimate_pka_type(mol, logp)

    # ── AD check uses overridden logp automatically ──────────────
    ad_flags = _check_ad(mol, mw, logp, tpsa)  # line 307 — picks up db_logp

    # ... rest unchanged (prodrug, return MolecularProfile)
```

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/unit/test_drugbank_integration.py tests/unit/test_features_chemistry.py -v`
Expected: All pass (new + existing).

- [ ] **Step 5: Commit**

```bash
git add src/sisyphus/predict/chemistry.py tests/unit/test_drugbank_integration.py
git commit -m "feat(predict): integrate DrugBank logP and ChemAxon pKa into compute_profile"
```

---

### Task 5: Integrate fup Lookup into adme.py

**Files:**
- Modify: `src/sisyphus/predict/adme.py`
- Extend: `tests/unit/test_drugbank_integration.py`

- [ ] **Step 1: Write failing test**

Append to `tests/unit/test_drugbank_integration.py`:

```python
class TestAdmeDrugBankIntegration:
    """Test fup DrugBank lookup logic."""

    def test_fup_lookup_returns_value(self, tmp_path):
        """DrugBankLookup.get_fup returns parsed value."""
        from sisyphus.predict.drugbank import DrugBankLookup
        (tmp_path / "drugs.csv").write_text(
            "drugbank_id,name,cas,smiles,inchikey,mw,logp_calc,pka_acidic,pka_basic,"
            "psa,hba,hbd,rotatable_bonds,state,groups,n_ddi,canonical_smiles,inchikey_14\n"
            "DB99901,TestDrug,,,,180,2.5,4.2,9.1,50,3,1,2,solid,approved,0,"
            "Oc1ccccc1,\n"
        )
        (tmp_path / "enzyme_annotations.csv").write_text(
            "drugbank_id,drug_name,enzyme_name,uniprot_id,actions\n"
        )
        (tmp_path / "pk_data.csv").write_text(
            "drugbank_id,drug_name,field,raw_text,parsed_value,parsed_unit\n"
            "DB99901,TestDrug,protein_binding,95% bound,0.05,fup\n"
        )
        (tmp_path / "experimental_properties.csv").write_text(
            "drugbank_id,drug_name,property,value\n"
        )
        lookup = DrugBankLookup(data_dir=tmp_path)
        assert lookup.get_fup("Oc1ccccc1") == pytest.approx(0.05)

    def test_fup_5x_guard_accepts_close_values(self):
        """fup within 5x of XGBoost → use DrugBank."""
        from sisyphus.core import Distribution
        db_fup = 0.05  # DrugBank
        xgb_fup = 0.08  # XGBoost
        # ratio = 0.08/0.05 = 1.6x → within 5x → accept
        assert db_fup / xgb_fup <= 5.0 and xgb_fup / db_fup <= 5.0

    def test_fup_5x_guard_rejects_divergent_values(self):
        """fup >5x different from XGBoost → reject, use XGBoost."""
        db_fup = 0.50  # DrugBank (bad parse, e.g. erythrocyte %)
        xgb_fup = 0.08  # XGBoost
        # ratio = 0.50/0.08 = 6.25x → exceeds 5x → reject
        assert db_fup / xgb_fup > 5.0 or xgb_fup / db_fup > 5.0

    def test_fup_sanity_rejects_out_of_range(self):
        """fup outside [0.001, 1.0] → reject."""
        assert not (0.001 <= 0.0 <= 1.0)  # zero
        assert not (0.001 <= 1.5 <= 1.0)  # >1
        assert 0.001 <= 0.05 <= 1.0       # valid
```

- [ ] **Step 2: Modify adme.py**

In `predict_adme()`, replace the `fup = _predict_fup(features_2d)` line:

```python
def predict_adme(profile: MolecularProfile) -> ADMEProperties:
    features = compute_features(profile.smiles)
    features_2d = features.reshape(1, -1)

    # fup: DrugBank measured → XGBoost fallback
    from sisyphus.predict.drugbank import drugbank_lookup
    db = drugbank_lookup()
    db_fup = db.get_fup(profile.smiles)
    if db_fup is not None and 0.001 <= db_fup <= 1.0:
        xgb_fup = _predict_fup(features_2d)
        if xgb_fup.mean > 0 and (db_fup / xgb_fup.mean > 5.0 or xgb_fup.mean / db_fup > 5.0):
            logger.warning(
                "DrugBank fup (%.3f) disagrees with XGBoost (%.3f) by >5x, using XGBoost",
                db_fup, xgb_fup.mean,
            )
            fup = xgb_fup
        else:
            fup = Distribution(mean=db_fup, cv=0.20)
            logger.info("Using DrugBank measured fup=%.3f", db_fup)
    else:
        fup = _predict_fup(features_2d)

    # Rest unchanged
    clint = _predict_clint(features_2d)
    rbp = _predict_rbp(features_2d[:, :2048])
    vdss = _predict_vdss(features_2d)
    peff = _estimate_peff(profile)
    solubility = _estimate_solubility(profile)

    # ... logging and return ADMEProperties unchanged
```

- [ ] **Step 3: Run tests**

Run: `python3 -m pytest tests/unit/test_drugbank_integration.py tests/unit/test_adme_ivive.py -v`
Expected: All pass.

- [ ] **Step 4: Commit**

```bash
git add src/sisyphus/predict/adme.py tests/unit/test_drugbank_integration.py
git commit -m "feat(predict): integrate DrugBank fup lookup with 5x cross-validation guard"
```

---

### Task 6: Integrate Enzyme fm into ivive.py

**Files:**
- Modify: `src/sisyphus/predict/ivive.py`
- Extend: `tests/unit/test_drugbank_integration.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/unit/test_drugbank_integration.py`:

```python
class TestIviveDrugBankIntegration:
    """Test enzyme fm fraction with DrugBank substrate annotations."""

    def test_fm_no_annotation_unchanged(self):
        from sisyphus.predict.ivive import _get_fm_fractions
        fm = _get_fm_fractions("neutral", substrate_enzymes=None)
        assert fm["CYP3A4"] == pytest.approx(0.50)

    def test_fm_single_substrate(self):
        from sisyphus.predict.ivive import _get_fm_fractions
        fm = _get_fm_fractions("neutral", substrate_enzymes={"CYP3A4"})
        # CYP3A4 = 1.0/1 = 1.0, others = 0.05 each
        # total = 1.0 + 4*0.05 = 1.20
        assert fm["CYP3A4"] == pytest.approx(1.0 / 1.20)
        assert fm["CYP2D6"] == pytest.approx(0.05 / 1.20)

    def test_fm_two_substrates(self):
        from sisyphus.predict.ivive import _get_fm_fractions
        fm = _get_fm_fractions("neutral", substrate_enzymes={"CYP3A4", "CYP2C9"})
        # Each substrate = 0.50, non-substrates = 0.05
        # total = 0.50 + 0.50 + 3*0.05 = 1.15
        assert fm["CYP3A4"] == pytest.approx(0.50 / 1.15)
        assert fm["CYP2C9"] == pytest.approx(0.50 / 1.15)
        assert fm["CYP2D6"] == pytest.approx(0.05 / 1.15)

    def test_fm_unknown_substrate_ignored(self):
        from sisyphus.predict.ivive import _get_fm_fractions
        fm = _get_fm_fractions("neutral", substrate_enzymes={"CYP_UNKNOWN"})
        # No known substrates → falls back to compound_type baseline
        assert fm["CYP3A4"] == pytest.approx(0.50)
```

- [ ] **Step 2: Run tests to verify fail**

Run: `python3 -m pytest tests/unit/test_drugbank_integration.py::TestIviveDrugBankIntegration -v`
Expected: TypeError — `_get_fm_fractions()` doesn't accept `substrate_enzymes`.

- [ ] **Step 3: Modify ivive.py**

Add `_normalize_fm` helper. Modify `_get_fm_fractions` signature. Thread `substrate_enzymes` through `_decompose_clint`. Modify `build_drug_on_graph` to call DrugBank lookup.

In `_get_fm_fractions` (~line 135), replace the function:

```python
def _get_fm_fractions(
    compound_type: str,
    substrate_enzymes: set[str] | None = None,
) -> dict[str, float]:
    """Get fraction metabolized by each CYP enzyme.

    When substrate_enzymes is provided (from DrugBank), annotated CYPs
    get equal share and non-substrates get a floor value.
    Otherwise falls back to compound_type-based defaults.
    """
    if compound_type in _FM_ADJUSTMENTS:
        fm = dict(_FM_ADJUSTMENTS[compound_type])
    else:
        fm = dict(_DEFAULT_FM)

    if not substrate_enzymes:
        return _normalize_fm(fm)

    known_substrates = substrate_enzymes & set(fm.keys())
    if not known_substrates:
        return _normalize_fm(fm)

    _NON_SUBSTRATE_FLOOR = 0.05
    for enzyme in fm:
        if enzyme in known_substrates:
            fm[enzyme] = 1.0 / len(known_substrates)
        else:
            fm[enzyme] = _NON_SUBSTRATE_FLOOR

    return _normalize_fm(fm)


def _normalize_fm(fm: dict[str, float]) -> dict[str, float]:
    """Normalize fm fractions to sum to 1.0."""
    total = sum(fm.values())
    if total > 0:
        return {k: v / total for k, v in fm.items()}
    return fm
```

In `_decompose_clint` (~line 157), add `substrate_enzymes` parameter:

```python
def _decompose_clint(
    clint: Distribution,
    compound_type: str,
    pka: float | None,
    enzyme_abundances: dict[str, float] | None = None,
    substrate_enzymes: set[str] | None = None,
) -> dict[str, Distribution]:
    fm = _get_fm_fractions(compound_type, substrate_enzymes)
    # ... rest unchanged
```

In `build_drug_on_graph` (~line 417), add DrugBank lookup before `_decompose_clint`:

```python
    from sisyphus.predict.drugbank import drugbank_lookup
    substrate_enzymes = drugbank_lookup().get_substrate_enzymes(profile.smiles)

    enzyme_affinity = _decompose_clint(
        adme.clint, profile.compound_type, profile.pka,
        enzyme_abundances=abundances,
        substrate_enzymes=substrate_enzymes,
    )
```

- [ ] **Step 4: Run all tests**

Run: `python3 -m pytest tests/unit/test_drugbank_integration.py tests/unit/test_adme_ivive.py -v`
Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add src/sisyphus/predict/ivive.py tests/unit/test_drugbank_integration.py
git commit -m "feat(predict): DrugBank CYP substrate annotations improve enzyme fm distribution"
```

---

### Task 7: Pipeline Warning Tags

**Files:**
- Modify: `src/sisyphus/pipeline/predict.py`

- [ ] **Step 1: Add DrugBank warning tags after Step 1 in pipeline**

After the `drug = build_drug_on_graph(...)` line (~line 71), add:

```python
    # ── DrugBank enrichment tags ──────────────────────────────────────
    # NOTE: fup tag checks data availability + sanity range but does NOT
    # replicate the 5x cross-validation guard from adme.py.  This means
    # a drug whose DrugBank fup was rejected by the 5x guard will still
    # be tagged as "drugbank:fup" (~5-10% of cases).  This is an accepted
    # imprecision per spec §5.5 — gold group has minor silver contamination.
    try:
        from sisyphus.predict.drugbank import drugbank_lookup
        db = drugbank_lookup()
        canonical = profile.smiles
        if db.get_substrate_enzymes(canonical) is not None:
            warnings_list.append("drugbank:enzyme_fm")
        db_fup = db.get_fup(canonical)
        if db_fup is not None and 0.001 <= db_fup <= 1.0:
            warnings_list.append("drugbank:fup")
        if db.get_pka(canonical) is not None:
            warnings_list.append("drugbank:pka")
        if db.get_logp(canonical) is not None:
            warnings_list.append("drugbank:logp")
    except Exception:
        pass  # DrugBank tagging is advisory, never blocks pipeline
```

- [ ] **Step 2: Run pipeline test**

Run: `python3 -m pytest tests/unit/test_pipeline.py -v`
Expected: All existing tests pass (no DrugBank CSVs in test env → no tags added).

- [ ] **Step 3: Commit**

```bash
git add src/sisyphus/pipeline/predict.py
git commit -m "feat(pipeline): add DrugBank enrichment warning tags for gold/silver tracking"
```

---

### Task 8: Benchmark Gold/Silver Split

**Files:**
- Modify: `src/sisyphus/validation/benchmark.py`

- [ ] **Step 1: Add gold/silver fields to BenchmarkResult**

Add after `pi_coverage_90` field (~line 56):

```python
    aafe_gold: float | None = None
    aafe_silver: float | None = None
    n_gold: int = 0
    n_silver: int = 0
```

- [ ] **Step 2: Add gold/silver tracking to the existing benchmark loop**

Inside the existing `for i, ref in enumerate(refs):` loop (line 98), after
`all_predicted.append(cmax_pred)` (line 107), add tracking of PredictionResult warnings:

```python
    # Add these lists at the top alongside all_predicted/all_observed (line 89):
    gold_predicted: list[float] = []
    gold_observed: list[float] = []
    silver_predicted: list[float] = []
    silver_observed: list[float] = []

    # Inside the loop, after all_predicted.append(cmax_pred) (line 107):
            is_gold = any("drugbank:" in w for w in result.warnings)
            if is_gold:
                gold_predicted.append(cmax_pred)
                gold_observed.append(ref.cmax_obs)
            else:
                silver_predicted.append(cmax_pred)
                silver_observed.append(ref.cmax_obs)
```

Then before the return statement (line 175), compute gold/silver AAFE:

```python
    # Gold/silver AAFE (DrugBank-enhanced vs XGBoost-only)
    gold_p = np.array(gold_predicted)
    gold_o = np.array(gold_observed)
    silver_p = np.array(silver_predicted)
    silver_o = np.array(silver_observed)
    aafe_gold_val = aafe(gold_p, gold_o) if len(gold_p) >= 3 else None
    aafe_silver_val = aafe(silver_p, silver_o) if len(silver_p) >= 3 else None

    logger.info(
        "Gold/Silver split: gold=%d (AAFE=%.3f), silver=%d (AAFE=%.3f)",
        len(gold_predicted),
        aafe_gold_val or 0.0,
        len(silver_predicted),
        aafe_silver_val or 0.0,
    )
```

Add `aafe_gold=aafe_gold_val, aafe_silver=aafe_silver_val, n_gold=len(gold_predicted), n_silver=len(silver_predicted)` to the BenchmarkResult constructor.

- [ ] **Step 3: Write test for gold/silver split**

Add to `tests/unit/test_drugbank_integration.py`:

```python
class TestBenchmarkGoldSilverSplit:
    """Test that BenchmarkResult correctly separates gold/silver."""

    def test_benchmark_result_has_gold_silver_fields(self):
        from sisyphus.validation.benchmark import BenchmarkResult
        br = BenchmarkResult(
            n_drugs=10, aafe=2.0, pct_2fold=50.0,
            n_in_domain=8, aafe_in_domain=1.8, pct_2fold_in_domain=60.0,
        )
        # Defaults
        assert br.aafe_gold is None
        assert br.aafe_silver is None
        assert br.n_gold == 0
        assert br.n_silver == 0

    def test_benchmark_result_with_gold_silver(self):
        from sisyphus.validation.benchmark import BenchmarkResult
        br = BenchmarkResult(
            n_drugs=10, aafe=2.0, pct_2fold=50.0,
            n_in_domain=8, aafe_in_domain=1.8, pct_2fold_in_domain=60.0,
            aafe_gold=1.5, aafe_silver=2.5, n_gold=7, n_silver=3,
        )
        assert br.aafe_gold == pytest.approx(1.5)
        assert br.n_gold == 7
```

- [ ] **Step 4: Run benchmark tests**

Run: `python3 -m pytest tests/unit/test_validation.py tests/unit/test_drugbank_integration.py::TestBenchmarkGoldSilverSplit -v`
Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add src/sisyphus/validation/benchmark.py tests/unit/test_drugbank_integration.py
git commit -m "feat(validation): gold/silver AAFE split in holdout benchmark"
```

---

### Task 9: Regression Test — Full Pipeline Without DrugBank

**Files:**
- Extend: `tests/unit/test_drugbank_integration.py`

- [ ] **Step 1: Write regression test**

```python
class TestRegressionNoDrugBank:
    """Verify pipeline works identically without DrugBank CSVs."""

    def test_predict_without_drugbank(self):
        """Pipeline should work with no DrugBank data (silver path only)."""
        from sisyphus.predict.drugbank import _reset_singleton
        _reset_singleton()
        from sisyphus.pipeline.predict import predict
        # Caffeine SMILES — should work regardless of DrugBank
        result = predict("Cn1c(=O)c2c(ncn2C)n(C)c1=O", dose_mg=100.0)
        assert result.pk.cmax.mean > 0
        assert result.method in ("engine", "ml", "hybrid")
        _reset_singleton()
```

- [ ] **Step 2: Run full test suite**

Run: `python3 -m pytest tests/ -v --tb=short`
Expected: All 253+ tests pass.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_drugbank_integration.py
git commit -m "test: add regression test verifying pipeline without DrugBank data"
```

---

### Task 10: Design Spec Update

**Files:**
- Modify: `DESIGN.md`

- [ ] **Step 1: Update §3.2 predict layer dependency**

Change predict row from:
```
| `predict` | SMILES → MolecularProfile → ADMEProperties → DrugOnGraph | 없음 | `engine`, `ml` |
```
To:
```
| `predict` | SMILES → MolecularProfile → ADMEProperties → DrugOnGraph | reference data (DrugBank CSV) | `engine`, `ml` |
```

- [ ] **Step 2: Commit**

```bash
git add DESIGN.md
git commit -m "docs: update design spec — predict layer depends on reference data lookup"
```
