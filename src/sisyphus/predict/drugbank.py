"""DrugBank data lookup for predict layer enrichment.

Loads extracted DrugBank CSV files and provides O(1) lookup by canonical
SMILES (primary) or InChIKey connectivity layer (fallback).

All lookups return None when data is unavailable — the predict layer
falls back to XGBoost/heuristic predictions. No exceptions are raised
for missing files or malformed data.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# CYP name normalization: DrugBank full names → Sisyphus YAML tags
_CYP_NORMALIZATION: dict[str, str] = {
    "Cytochrome P450 3A4": "CYP3A4",
    "Cytochrome P450 2D6": "CYP2D6",
    "Cytochrome P450 1A2": "CYP1A2",
    "Cytochrome P450 2C9": "CYP2C9",
    "Cytochrome P450 2E1": "CYP2E1",
    "Cytochrome P450 3A5": "CYP3A4",   # same gene family, merge
    "Cytochrome P450 2C19": "CYP2C9",  # same 2C subfamily
    "Cytochrome P450 2C8": "CYP2C9",   # same 2C subfamily
    # CYP2B6 intentionally absent — no Sisyphus equivalent
}

# UGT name normalization: DrugBank full names → Sisyphus YAML tags
# Only isoforms present in reference_man.yaml (liver or gut_wall)
_UGT_NORMALIZATION: dict[str, str] = {
    "UDP-glucuronosyltransferase 2B7": "UGT2B7",
    "UDP-glucuronosyltransferase 1A1": "UGT1A1",
    "UDP-glucuronosyltransferase 1A4": "UGT1A4",
    "UDP-glucuronosyltransferase 1A9": "UGT1A9",
    # UGTs not in YAML (no abundance → engine ignores them):
    # UGT1A3, UGT1A6, UGT1A8, UGT1A10, UGT2B4, UGT2B10, UGT2B15
}

_DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "drugbank"


@dataclass
class DrugBankConfig:
    """Feature flags for individual enrichment toggle (ablation support)."""
    enable_enzyme_fm: bool = True
    enable_fup: bool = True
    enable_pka: bool = True
    enable_logp: bool = True


class DrugBankLookup:
    """Lazy-loaded DrugBank data lookup.

    Args:
        data_dir: Directory containing extracted DrugBank CSVs.
        config: Feature flags for ablation.
    """

    def __init__(self, data_dir: Path = _DEFAULT_DATA_DIR, config: DrugBankConfig | None = None):
        self._data_dir = data_dir
        self._config = config or DrugBankConfig()
        self._loaded = False
        self._smiles_to_id: dict[str, str] = {}
        self._inchikey14_to_id: dict[str, str] = {}
        self._enzyme_substrates: dict[str, set[str]] = {}  # dbid → {CYP tags}
        self._ugt_substrates: dict[str, set[str]] = {}  # dbid → {UGT tags}
        self._fup: dict[str, float] = {}  # dbid → fup
        self._pka: dict[str, tuple[float, float]] = {}  # dbid → (acidic, basic)
        self._logp: dict[str, float] = {}  # dbid → logP

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not (self._data_dir / "drugs.csv").exists():
            # Public-clone state — DrugBank not present. Silent until queried,
            # then this single message announces the fallback at first use.
            logger.info(
                "DrugBank not present at %s — running in public-clone deterministic "
                "state. fup/pKa/logP enrichments will not apply. See AGENTS.md "
                "§\"Artifact gates\" for the headline AAFE footprint (~+2.7%% when "
                "absent vs present).",
                self._data_dir,
            )
            return
        self._load_drugs()
        self._load_enzymes()
        self._load_pk_data()
        self._load_experimental()
        logger.info(
            "DrugBank loaded from %s — enriching %d SMILES, %d InChIKey, %d enzyme, "
            "%d fup, %d pka, %d logp. Predictions will differ from public-clone "
            "state; see AGENTS.md §\"Artifact gates\".",
            self._data_dir, len(self._smiles_to_id), len(self._inchikey14_to_id),
            len(self._enzyme_substrates), len(self._fup), len(self._pka), len(self._logp),
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
                    # CYP substrates
                    cyp_tag = _CYP_NORMALIZATION.get(enz_name)
                    if cyp_tag and dbid:
                        self._enzyme_substrates.setdefault(dbid, set()).add(cyp_tag)
                    # UGT substrates
                    ugt_tag = _UGT_NORMALIZATION.get(enz_name)
                    if ugt_tag and dbid:
                        self._ugt_substrates.setdefault(dbid, set()).add(ugt_tag)
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

    # -- Lookup methods --

    def lookup(self, canonical_smiles: str) -> str | None:
        """Resolve canonical SMILES to DrugBank ID. 2-tier: SMILES exact → InChIKey-14 fallback."""
        self._ensure_loaded()
        dbid = self._smiles_to_id.get(canonical_smiles)
        if dbid:
            return dbid
        # InChIKey fallback
        try:
            from rdkit import Chem
            from rdkit.Chem.inchi import InchiToInchiKey, MolToInchi
            mol = Chem.MolFromSmiles(canonical_smiles)
            if mol is not None:
                inchi = MolToInchi(mol)
                if inchi:
                    inchikey = InchiToInchiKey(inchi)
                    if inchikey and len(inchikey) >= 14:
                        dbid = self._inchikey14_to_id.get(inchikey[:14])
                        if dbid:
                            logger.debug(
                                "InChIKey fallback match: %s → %s", canonical_smiles[:30], dbid
                            )
                            return dbid
        except Exception:
            pass
        return None

    def get_substrate_enzymes(self, canonical_smiles: str) -> set[str] | None:
        """Return set of CYP tags for which this drug is a substrate, or None if unknown."""
        if not self._config.enable_enzyme_fm:
            return None
        dbid = self.lookup(canonical_smiles)
        if dbid is None:
            return None
        enzymes = self._enzyme_substrates.get(dbid)
        return set(enzymes) if enzymes else None

    def get_ugt_enzymes(self, canonical_smiles: str) -> set[str] | None:
        """Return set of UGT tags for which this drug is a substrate, or None if unknown.

        Only returns UGT isoforms that have abundance entries in reference_man.yaml
        (UGT2B7, UGT1A1, UGT1A4, UGT1A9). Other isoforms are filtered out because
        the engine cannot compute clearance without a matching node abundance.
        """
        if not self._config.enable_enzyme_fm:
            return None
        dbid = self.lookup(canonical_smiles)
        if dbid is None:
            return None
        ugts = self._ugt_substrates.get(dbid)
        return set(ugts) if ugts else None

    def get_fup(self, canonical_smiles: str) -> float | None:
        """Return fraction unbound in plasma from DrugBank, or None if unknown."""
        if not self._config.enable_fup:
            return None
        dbid = self.lookup(canonical_smiles)
        if dbid is None:
            return None
        return self._fup.get(dbid)

    def get_pka(self, canonical_smiles: str) -> tuple[float, float] | None:
        """Return (acidic_pKa, basic_pKa) tuple, or None if unknown."""
        if not self._config.enable_pka:
            return None
        dbid = self.lookup(canonical_smiles)
        if dbid is None:
            return None
        return self._pka.get(dbid)

    def get_logp(self, canonical_smiles: str) -> float | None:
        """Return experimental logP, or None if unknown."""
        if not self._config.enable_logp:
            return None
        dbid = self.lookup(canonical_smiles)
        if dbid is None:
            return None
        return self._logp.get(dbid)


# Module-level singleton
_INSTANCE: DrugBankLookup | None = None


def drugbank_lookup(config: DrugBankConfig | None = None) -> DrugBankLookup:
    """Get module-level singleton. Creates on first call.

    Config is only used on first call.  Subsequent calls with a different
    config are ignored (call ``_reset_singleton()`` first to reconfigure).
    """
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = DrugBankLookup(config=config)
    elif config is not None:
        logger.warning("drugbank_lookup() singleton already initialized, config argument ignored")
    return _INSTANCE


def _reset_singleton() -> None:
    """Reset singleton for testing."""
    global _INSTANCE
    _INSTANCE = None
