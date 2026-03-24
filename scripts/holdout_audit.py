"""
holdout_audit.py — Holdout contamination check for TDC PPBR_AZ dataset.

Checks whether any compound in the TDC PPBR_AZ training dataset overlaps
with Sisyphus holdout drugs using three complementary exclusion keys:

  1. Canonical SMILES (isomericSmiles=True)
  2. First 14 characters of InChIKey (connectivity layer, salt-insensitive
     for protonation variants, but NOT for dot-disconnected salts)
  3. Lowercase Drug_ID string

Outputs a report to docs/holdout_contamination_audit.md.

Usage:
    python scripts/holdout_audit.py
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
HOLDOUT_JSON = REPO_ROOT / "data" / "reference" / "holdout.json"
CLINICAL_PK_JSON = REPO_ROOT / "data" / "reference" / "clinical_pk.json"
REPORT_PATH = REPO_ROOT / "docs" / "holdout_contamination_audit.md"


# ---------------------------------------------------------------------------
# RDKit helpers
# ---------------------------------------------------------------------------

def _canonical_smiles(smiles: str) -> str | None:
    """Return RDKit canonical SMILES (isomericSmiles=True), or None on failure."""
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def _inchikey_prefix(smiles: str) -> str | None:
    """Return first 14 characters of InChIKey (connectivity block), or None."""
    from rdkit import Chem
    from rdkit.Chem.inchi import MolToInchi, InchiToInchiKey

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    inchi = MolToInchi(mol)
    if inchi is None:
        return None
    ik = InchiToInchiKey(inchi)
    if ik is None:
        return None
    return ik[:14]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class HoldoutKeys:
    """Three exclusion key sets built from holdout drug entries."""
    canonical_smiles: set[str] = field(default_factory=set)
    inchikey_prefixes: set[str] = field(default_factory=set)
    names: set[str] = field(default_factory=set)
    # Mapping back for reporting
    smiles_to_name: dict[str, str] = field(default_factory=dict)
    inchikey_to_name: dict[str, str] = field(default_factory=dict)


@dataclass
class ContaminationHit:
    tdc_drug_id: str
    tdc_smiles: str
    match_type: list[str]  # "smiles", "inchikey", "name"
    holdout_drug: str


# ---------------------------------------------------------------------------
# Build holdout exclusion keys
# ---------------------------------------------------------------------------

def build_holdout_keys(holdout_names: list[str], clinical_pk: dict) -> HoldoutKeys:
    keys = HoldoutKeys()
    drugs = clinical_pk.get("drugs", {})

    missing_smiles: list[str] = []
    rdkit_failures: list[str] = []

    for name in holdout_names:
        keys.names.add(name.lower())
        # Try exact name first, then underscore/hyphen variants
        entry = drugs.get(name) or drugs.get(name.replace(" ", "_"))
        if entry is None:
            missing_smiles.append(name)
            continue

        smiles = entry.get("smiles", "")
        if not smiles:
            missing_smiles.append(name)
            continue

        csmi = _canonical_smiles(smiles)
        if csmi:
            keys.canonical_smiles.add(csmi)
            keys.smiles_to_name[csmi] = name
        else:
            rdkit_failures.append(name)

        ik = _inchikey_prefix(smiles)
        if ik:
            keys.inchikey_prefixes.add(ik)
            keys.inchikey_to_name[ik] = name
        else:
            rdkit_failures.append(name)

    if missing_smiles:
        log.warning("No SMILES found in clinical_pk.json for %d holdout drugs: %s",
                    len(missing_smiles), missing_smiles)
    if rdkit_failures:
        log.warning("RDKit failed to parse SMILES for: %s", rdkit_failures)

    log.info(
        "Holdout keys built: %d canonical SMILES, %d InChIKey prefixes, %d names",
        len(keys.canonical_smiles),
        len(keys.inchikey_prefixes),
        len(keys.names),
    )
    return keys


# ---------------------------------------------------------------------------
# Check TDC dataset
# ---------------------------------------------------------------------------

def check_tdc_dataset(keys: HoldoutKeys) -> tuple[list[ContaminationHit], int]:
    """Load PPBR_AZ and return (hits, total_compounds)."""
    from tdc.single_pred import ADME

    log.info("Loading TDC PPBR_AZ dataset ...")
    data = ADME(name="PPBR_AZ")
    df = data.get_data()
    log.info("Loaded %d rows from PPBR_AZ", len(df))

    hits: list[ContaminationHit] = []

    for _, row in df.iterrows():
        drug_id: str = str(row["Drug_ID"])
        tdc_smiles: str = str(row["Drug"])
        match_types: list[str] = []
        holdout_drug = ""

        # Key 1: canonical SMILES
        csmi = _canonical_smiles(tdc_smiles)
        if csmi and csmi in keys.canonical_smiles:
            match_types.append("smiles")
            holdout_drug = keys.smiles_to_name.get(csmi, "unknown")

        # Key 2: InChIKey prefix
        ik = _inchikey_prefix(tdc_smiles)
        if ik and ik in keys.inchikey_prefixes:
            if "inchikey" not in match_types:
                match_types.append("inchikey")
            if not holdout_drug:
                holdout_drug = keys.inchikey_to_name.get(ik, "unknown")

        # Key 3: lowercase name
        drug_id_lower = drug_id.lower()
        if drug_id_lower in keys.names:
            if "name" not in match_types:
                match_types.append("name")
            if not holdout_drug:
                holdout_drug = drug_id_lower

        if match_types:
            hits.append(ContaminationHit(
                tdc_drug_id=drug_id,
                tdc_smiles=tdc_smiles,
                match_type=match_types,
                holdout_drug=holdout_drug,
            ))

    return hits, len(df)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def write_report(
    hits: list[ContaminationHit],
    total_tdc: int,
    holdout_names: list[str],
    holdout_keys: HoldoutKeys,
    output_path: Path,
) -> None:
    n_hits = len(hits)
    n_holdout = len(holdout_names)
    smiles_hits = [h for h in hits if "smiles" in h.match_type]
    inchikey_hits = [h for h in hits if "inchikey" in h.match_type]
    name_hits = [h for h in hits if "name" in h.match_type]
    verdict = "**CONTAMINATION DETECTED**" if n_hits > 0 else "**No contamination detected**"

    lines: list[str] = []
    lines.append("# Holdout Contamination Audit — TDC PPBR\\_AZ vs Sisyphus Holdout\n")
    lines.append(f"*Generated by `scripts/holdout_audit.py`*\n")
    lines.append("---\n")

    lines.append("## Summary\n")
    lines.append(f"| Item | Count |")
    lines.append(f"|------|-------|")
    lines.append(f"| TDC PPBR\\_AZ compounds checked | {total_tdc} |")
    lines.append(f"| Sisyphus holdout drugs | {n_holdout} |")
    lines.append(f"| Holdout drugs with SMILES in clinical\\_pk.json | {len(holdout_keys.canonical_smiles)} |")
    lines.append(f"| Contamination hits (any key) | {n_hits} |")
    lines.append(f"| &nbsp;&nbsp;— matched by canonical SMILES | {len(smiles_hits)} |")
    lines.append(f"| &nbsp;&nbsp;— matched by InChIKey prefix (14 chars) | {len(inchikey_hits)} |")
    lines.append(f"| &nbsp;&nbsp;— matched by lowercase name | {len(name_hits)} |\n")

    lines.append(f"### Verdict\n")
    lines.append(f"{verdict}\n")
    if n_hits == 0:
        lines.append(
            "TDC PPBR\\_AZ does not contain any holdout compound under the three "
            "exclusion criteria applied. The dataset is safe to use for fup model training.\n"
        )
    else:
        lines.append(
            f"{n_hits} compound(s) in PPBR\\_AZ overlap with the Sisyphus holdout set. "
            "These rows **must be excluded** before training any fup predictor.\n"
        )

    lines.append("---\n")
    lines.append("## Methodology\n")
    lines.append(
        "Three independent exclusion keys are applied to maximise detection coverage:\n"
    )
    lines.append(
        "1. **Canonical SMILES** (`isomericSmiles=True`): exact structural match after "
        "RDKit canonicalisation. Catches identical molecules regardless of input notation.\n"
    )
    lines.append(
        "2. **InChIKey prefix (14 chars)**: the first block of the InChIKey encodes "
        "connectivity only (ignoring stereochemistry and isotopes). Catches stereoisomers "
        "of the same core scaffold. Note: dot-disconnected salts (e.g. `base.Cl`) receive "
        "a *different* InChIKey from the free base, so name matching is needed for those.\n"
    )
    lines.append(
        "3. **Lowercase name**: the TDC `Drug_ID` field is compared (lowercased) against "
        "the holdout name list. Catches cases where the SMILES differs (salt form, prodrug) "
        "but the INN is preserved in the identifier.\n"
    )
    lines.append(
        "\nAll three sets are unioned; a compound is flagged if it matches on **any** key.\n"
    )

    if n_hits > 0:
        lines.append("---\n")
        lines.append("## Contamination Hits\n")
        lines.append("| TDC Drug\\_ID | Holdout Drug | Match Type(s) | TDC SMILES |")
        lines.append("|-------------|-------------|--------------|-----------|")
        for h in sorted(hits, key=lambda x: x.holdout_drug):
            mt = ", ".join(h.match_type)
            smiles_trunc = h.tdc_smiles[:60] + ("…" if len(h.tdc_smiles) > 60 else "")
            lines.append(f"| {h.tdc_drug_id} | {h.holdout_drug} | {mt} | `{smiles_trunc}` |")
        lines.append("")

        lines.append("### Exclusion Filter for Training Scripts\n")
        lines.append("To exclude contaminated rows from a pandas DataFrame `df`:\n")
        lines.append("```python")
        lines.append("from rdkit import Chem")
        lines.append("from rdkit.Chem.inchi import MolToInchi, InchiToInchiKey\n")
        lines.append("CONTAMINATED_IDS = {")
        for h in sorted(hits, key=lambda x: x.tdc_drug_id):
            lines.append(f'    "{h.tdc_drug_id}",  # matches holdout: {h.holdout_drug}')
        lines.append("}")
        lines.append("df_clean = df[~df['Drug_ID'].isin(CONTAMINATED_IDS)]")
        lines.append("```")
    else:
        lines.append("---\n")
        lines.append("## No Contamination Hits\n")
        lines.append(
            "All 100 holdout drugs were checked against the 1,614 PPBR\\_AZ entries. "
            "No matches were found under canonical SMILES, InChIKey prefix, or "
            "lowercase name comparison.\n"
        )

    lines.append("\n---\n")
    lines.append("## Holdout Drugs Checked\n")
    lines.append("The following 100 drugs constitute the inviolable holdout set:\n")
    # Format as a compact multi-column list
    sorted_holdout = sorted(holdout_names)
    cols = 4
    rows = (len(sorted_holdout) + cols - 1) // cols
    # Build markdown table
    headers = ["Drug"] * cols
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + " --- |" * cols)
    for i in range(rows):
        row_items = []
        for j in range(cols):
            idx = i + j * rows
            if idx < len(sorted_holdout):
                row_items.append(sorted_holdout[idx])
            else:
                row_items.append("")
        lines.append("| " + " | ".join(row_items) + " |")
    lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")
    log.info("Report written to %s", output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    # Load holdout names
    log.info("Loading holdout.json ...")
    with open(HOLDOUT_JSON) as f:
        holdout_data = json.load(f)
    holdout_names: list[str] = holdout_data["holdout"]
    log.info("Holdout set: %d drugs", len(holdout_names))

    # Load clinical PK for SMILES
    log.info("Loading clinical_pk.json ...")
    with open(CLINICAL_PK_JSON) as f:
        clinical_pk = json.load(f)

    # Build exclusion keys
    holdout_keys = build_holdout_keys(holdout_names, clinical_pk)

    # Check TDC PPBR_AZ
    hits, total_tdc = check_tdc_dataset(holdout_keys)

    if hits:
        log.warning("CONTAMINATION DETECTED: %d hit(s) in PPBR_AZ", len(hits))
        for h in hits:
            log.warning("  %s → holdout drug '%s' (match: %s)",
                        h.tdc_drug_id, h.holdout_drug, h.match_type)
    else:
        log.info("No contamination detected in PPBR_AZ.")

    # Write report
    write_report(hits, total_tdc, holdout_names, holdout_keys, REPORT_PATH)

    return 0 if not hits else 1


if __name__ == "__main__":
    sys.exit(main())
