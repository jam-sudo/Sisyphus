"""Extract Sisyphus-relevant data from DrugBank full database XML.

Streaming parser — handles the 1.8GB XML without loading into memory.

Outputs (to data/drugbank/):
    drugs.csv                    — master list: identity + calculated properties
    experimental_properties.csv  — measured values (logP, solubility, caco2, pKa)
    enzyme_annotations.csv       — drug → CYP/UGT enzyme → substrate/inhibitor/inducer
    transporter_annotations.csv  — drug → transporter → substrate/inhibitor/inducer
    pk_data.csv                  — PK text fields + parsed numeric values
    extraction_summary.json      — counts and quality metrics
"""

from __future__ import annotations

import csv
import json
import logging
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

NS = "{http://www.drugbank.ca}"
INPUT_FILE = Path(__file__).resolve().parent.parent / "full database.xml"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "drugbank"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# PK text parsers
# ---------------------------------------------------------------------------

def parse_half_life_h(text: str) -> float | None:
    """Extract half-life in hours from free text.

    Handles: "4-5 hours", "~62.8 hours", "19 h", "25 min",
    "3.5h", "13.2 ± 3.2 days", "terminal half-life of 1.3 hours".
    Returns midpoint for ranges.
    """
    if not text:
        return None
    t = text.lower().replace("&plusmn;", "±").replace("\r", " ").replace("\n", " ")

    # Try range pattern first: "4-5 hours", "7.5 to 9 hours"
    m = re.search(
        r"(?:half[- ]?life|t½|t1/2)?\s*(?:of\s+)?(?:approximately\s+|about\s+|~)?"
        r"(\d+\.?\d*)\s*(?:[-–to]+)\s*(\d+\.?\d*)\s*(hours?|hrs?|h\b|min(?:utes?)?|days?)",
        t,
    )
    if m:
        lo, hi = float(m.group(1)), float(m.group(2))
        mid = (lo + hi) / 2
        return _to_hours(mid, m.group(3))

    # Single value: "62.8 hours", "25 min", "~19 h"
    m = re.search(
        r"(?:half[- ]?life|t½|t1/2)?\s*(?:of\s+)?(?:approximately\s+|about\s+|~)?"
        r"(\d+\.?\d*)\s*(?:±\s*\d+\.?\d*)?\s*(hours?|hrs?|h\b|min(?:utes?)?|days?)",
        t,
    )
    if m:
        return _to_hours(float(m.group(1)), m.group(2))

    # Bare number + hours at start: "4-5 hours" without prefix
    m = re.search(r"^[*\s]*~?(\d+\.?\d*)\s*(?:[-–to]+\s*(\d+\.?\d*))?\s*(hours?|hrs?|h\b|min|days?)", t)
    if m:
        val = float(m.group(1))
        if m.group(2):
            val = (val + float(m.group(2))) / 2
        return _to_hours(val, m.group(3))

    return None


def _to_hours(val: float, unit: str) -> float:
    u = unit.lower().strip()
    if u.startswith("min"):
        return val / 60
    elif u.startswith("day"):
        return val * 24
    return val  # hours


def parse_protein_binding_fup(text: str) -> float | None:
    """Extract fraction unbound in plasma from protein binding text.

    "90% bound" → 0.10, "27.3%" → fup = 0.727 (if ambiguous, assume % bound).
    Handles: "86%", "90-94%", "approximately 95%".
    """
    if not text:
        return None
    t = text.lower().replace("\r", " ").replace("\n", " ")

    # Explicit "unbound" or "free fraction"
    m = re.search(r"(?:unbound|free\s+fraction)\s*(?:of\s+)?(?:approximately\s+|about\s+|~)?(\d+\.?\d*)\s*%", t)
    if m:
        return float(m.group(1)) / 100

    # Range: "90-94% bound"
    m = re.search(r"(\d+\.?\d*)\s*[-–to]+\s*(\d+\.?\d*)\s*%", t)
    if m:
        lo, hi = float(m.group(1)), float(m.group(2))
        mid = (lo + hi) / 2
        if mid > 50:  # likely % bound
            return 1 - mid / 100
        return mid / 100  # likely % free — unusual

    # Single percentage: "86%", "27.3%"
    m = re.search(r"(?:approximately\s+|about\s+|~)?(\d+\.?\d*)\s*%", t)
    if m:
        pct = float(m.group(1))
        # Heuristic: > 10% is almost certainly % bound
        if pct > 10:
            return 1 - pct / 100
        # ≤ 10% could be either — assume % bound (conservative)
        return 1 - pct / 100

    return None


def parse_vd_L(text: str) -> tuple[float | None, str | None]:
    """Extract volume of distribution. Returns (value_L, unit_note).

    Handles: "0.2L/kg", "1.16 L/kg", "200 mL/kg", "70 L".
    Converts mL/kg to L/kg. Returns raw L/kg or L.
    """
    if not text:
        return None, None
    t = text.lower().replace("\r", " ").replace("\n", " ")

    # Range: "4-8 L/kg"
    m = re.search(r"(\d+\.?\d*)\s*[-–to]+\s*(\d+\.?\d*)\s*(l|ml|litre|liter)s?\s*/\s*kg", t)
    if m:
        lo, hi = float(m.group(1)), float(m.group(2))
        mid = (lo + hi) / 2
        if m.group(3).startswith("ml"):
            mid /= 1000
        return mid, "L/kg"

    # Single L/kg: "0.2 L/kg", "200 mL/kg", "44.1 ± 13.6 mL/kg"
    m = re.search(r"(\d+\.?\d*)\s*(?:±\s*\d+\.?\d*)?\s*(l|ml|litre|liter)s?\s*/\s*kg", t)
    if m:
        val = float(m.group(1))
        if m.group(2).startswith("ml"):
            val /= 1000
        return val, "L/kg"

    # Absolute L: "70 L", "12.2 L", "44.1 ± 13.6 L"
    m = re.search(r"(\d+\.?\d*)\s*(?:±\s*\d+\.?\d*)?\s*(l|litre|liter)s?\b(?!\s*/)", t)
    if m:
        return float(m.group(1)), "L"

    return None, None


def parse_clearance_L_per_h(text: str) -> tuple[float | None, str | None]:
    """Extract clearance. Returns (value, unit_note).

    Handles: "3.4 mL/min/kg", "7.2 mL/h/kg", "200 mL/min", "0.38 L×h/kg".
    Converts all to L/h (assuming 70 kg when per-kg).
    """
    if not text:
        return None, None
    t = text.lower().replace("\r", " ").replace("\n", " ").replace("×", "*").replace("x", "*")

    # mL/min/kg
    m = re.search(r"(\d+\.?\d*)\s*(?:±\s*\d+\.?\d*)?\s*ml\s*/\s*min\s*/\s*kg", t)
    if m:
        val = float(m.group(1))
        # mL/min/kg → L/h: × 70 (kg) × 60 (min→h) / 1000 (mL→L) = × 4.2
        return val * 4.2, "L/h (from mL/min/kg)"

    # mL/h/kg
    m = re.search(r"(\d+\.?\d*)\s*(?:±\s*\d+\.?\d*)?\s*ml\s*/\s*h(?:our)?s?\s*/\s*kg", t)
    if m:
        val = float(m.group(1))
        return val * 70 / 1000, "L/h (from mL/h/kg)"

    # L*h/kg or L/h/kg (Sisyphus convention)
    m = re.search(r"(\d+\.?\d*)\s*(?:±\s*\d+\.?\d*)?\s*l\s*[*/]\s*h(?:our)?s?\s*/?\s*kg", t)
    if m:
        val = float(m.group(1))
        return val * 70, "L/h (from L/h/kg)"

    # mL/min (absolute)
    m = re.search(r"(\d+\.?\d*)\s*(?:±\s*\d+\.?\d*)?\s*ml\s*/\s*min\b", t)
    if m:
        val = float(m.group(1))
        return val * 60 / 1000, "L/h (from mL/min)"

    # L/h (absolute)
    m = re.search(r"(\d+\.?\d*)\s*(?:±\s*\d+\.?\d*)?\s*l\s*/\s*h(?:our)?", t)
    if m:
        return float(m.group(1)), "L/h"

    return None, None


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------

def extract() -> None:
    logger.info("Starting DrugBank extraction from %s", INPUT_FILE)

    # Open output CSVs
    f_drugs = open(OUT_DIR / "drugs.csv", "w", newline="", encoding="utf-8")
    f_exp = open(OUT_DIR / "experimental_properties.csv", "w", newline="", encoding="utf-8")
    f_enz = open(OUT_DIR / "enzyme_annotations.csv", "w", newline="", encoding="utf-8")
    f_tr = open(OUT_DIR / "transporter_annotations.csv", "w", newline="", encoding="utf-8")
    f_pk = open(OUT_DIR / "pk_data.csv", "w", newline="", encoding="utf-8")

    w_drugs = csv.writer(f_drugs)
    w_drugs.writerow([
        "drugbank_id", "name", "cas", "smiles", "inchikey", "mw", "logp_calc",
        "pka_acidic", "pka_basic", "psa", "hba", "hbd", "rotatable_bonds",
        "state", "groups", "n_ddi",
    ])

    w_exp = csv.writer(f_exp)
    w_exp.writerow(["drugbank_id", "drug_name", "property", "value"])

    w_enz = csv.writer(f_enz)
    w_enz.writerow(["drugbank_id", "drug_name", "enzyme_name", "uniprot_id", "actions"])

    w_tr = csv.writer(f_tr)
    w_tr.writerow(["drugbank_id", "drug_name", "transporter_name", "uniprot_id", "actions"])

    w_pk = csv.writer(f_pk)
    w_pk.writerow([
        "drugbank_id", "drug_name", "field", "raw_text",
        "parsed_value", "parsed_unit",
    ])

    # Counters
    stats = {
        "total_sm": 0,
        "has_smiles": 0,
        "has_enzymes": 0,
        "has_transporters": 0,
        "pk_parsed": {"half_life": 0, "fup": 0, "vd": 0, "clearance": 0},
        "pk_total": {"half_life": 0, "fup": 0, "vd": 0, "clearance": 0},
        "experimental_properties": 0,
    }

    for event, elem in ET.iterparse(str(INPUT_FILE), events=["end"]):
        if elem.tag != NS + "drug" or elem.attrib.get("type") != "small molecule":
            continue

        stats["total_sm"] += 1
        if stats["total_sm"] % 2000 == 0:
            logger.info("Processed %d small molecules...", stats["total_sm"])

        # -- Identity --
        dbid_el = elem.find(NS + "drugbank-id[@primary='true']")
        if dbid_el is None:
            dbid_el = elem.find(NS + "drugbank-id")
        dbid = dbid_el.text if dbid_el is not None else ""
        name = _text(elem, "name")
        cas = _text(elem, "cas-number")
        state = _text(elem, "state")

        groups_el = elem.find(NS + "groups")
        groups = ",".join(g.text for g in groups_el if g.text) if groups_el is not None else ""

        # -- Calculated properties --
        calc_props = {}
        calc_el = elem.find(NS + "calculated-properties")
        if calc_el is not None:
            for prop in calc_el:
                kind = _text(prop, "kind")
                val = _text(prop, "value")
                if kind and val:
                    calc_props[kind] = val

        smiles = calc_props.get("SMILES", "")
        inchikey = calc_props.get("InChIKey", "")
        mw = calc_props.get("Molecular Weight", "")
        logp_calc = calc_props.get("logP", "")
        pka_acidic = calc_props.get("pKa (strongest acidic)", "")
        pka_basic = calc_props.get("pKa (strongest basic)", "")
        psa = calc_props.get("Polar Surface Area (PSA)", "")
        hba = calc_props.get("H Bond Acceptor Count", "")
        hbd = calc_props.get("H Bond Donor Count", "")
        rotatable = calc_props.get("Rotatable Bond Count", "")

        if smiles:
            stats["has_smiles"] += 1

        # -- DDI count --
        ddi_el = elem.find(NS + "drug-interactions")
        n_ddi = len(list(ddi_el)) if ddi_el is not None else 0

        w_drugs.writerow([
            dbid, name, cas, smiles, inchikey, mw, logp_calc,
            pka_acidic, pka_basic, psa, hba, hbd, rotatable,
            state, groups, n_ddi,
        ])

        # -- Experimental properties --
        exp_el = elem.find(NS + "experimental-properties")
        if exp_el is not None:
            for prop in exp_el:
                kind = _text(prop, "kind")
                val = _text(prop, "value")
                if kind and val:
                    w_exp.writerow([dbid, name, kind, val])
                    stats["experimental_properties"] += 1

        # -- Enzyme annotations --
        enz_el = elem.find(NS + "enzymes")
        if enz_el is not None:
            wrote_any = False
            for enz in enz_el:
                enz_name = _text(enz, "name")
                # UniProt ID from polypeptide
                uniprot = ""
                poly_el = enz.find(NS + "polypeptide")
                if poly_el is not None:
                    uniprot = poly_el.attrib.get("id", "")
                actions_el = enz.find(NS + "actions")
                actions = ",".join(a.text for a in actions_el if a.text) if actions_el is not None else ""
                if enz_name:
                    w_enz.writerow([dbid, name, enz_name, uniprot, actions])
                    wrote_any = True
            if wrote_any:
                stats["has_enzymes"] += 1

        # -- Transporter annotations --
        tr_el = elem.find(NS + "transporters")
        if tr_el is not None:
            wrote_any = False
            for tr in tr_el:
                tr_name = _text(tr, "name")
                uniprot = ""
                poly_el = tr.find(NS + "polypeptide")
                if poly_el is not None:
                    uniprot = poly_el.attrib.get("id", "")
                actions_el = tr.find(NS + "actions")
                actions = ",".join(a.text for a in actions_el if a.text) if actions_el is not None else ""
                if tr_name:
                    w_tr.writerow([dbid, name, tr_name, uniprot, actions])
                    wrote_any = True
            if wrote_any:
                stats["has_transporters"] += 1

        # -- PK text fields + parsing --
        pk_fields = {
            "half_life": "half-life",
            "protein_binding": "protein-binding",
            "volume_of_distribution": "volume-of-distribution",
            "clearance": "clearance",
            "absorption": "absorption",
        }

        for key, xml_tag in pk_fields.items():
            el = elem.find(NS + xml_tag)
            raw = el.text.strip() if el is not None and el.text else ""
            if not raw:
                continue

            parsed_val = None
            parsed_unit = None

            if key == "half_life":
                stats["pk_total"]["half_life"] += 1
                parsed_val = parse_half_life_h(raw)
                if parsed_val is not None:
                    parsed_unit = "h"
                    stats["pk_parsed"]["half_life"] += 1

            elif key == "protein_binding":
                stats["pk_total"]["fup"] += 1
                parsed_val = parse_protein_binding_fup(raw)
                if parsed_val is not None:
                    parsed_unit = "fup"
                    stats["pk_parsed"]["fup"] += 1

            elif key == "volume_of_distribution":
                stats["pk_total"]["vd"] += 1
                parsed_val, parsed_unit = parse_vd_L(raw)
                if parsed_val is not None:
                    stats["pk_parsed"]["vd"] += 1

            elif key == "clearance":
                stats["pk_total"]["clearance"] += 1
                parsed_val, parsed_unit = parse_clearance_L_per_h(raw)
                if parsed_val is not None:
                    stats["pk_parsed"]["clearance"] += 1

            # Absorption: save raw only (too unstructured for numeric parsing)

            w_pk.writerow([
                dbid, name, key,
                raw[:2000],  # cap at 2000 chars
                f"{parsed_val:.6g}" if parsed_val is not None else "",
                parsed_unit or "",
            ])

        elem.clear()

    # Close files
    for f in [f_drugs, f_exp, f_enz, f_tr, f_pk]:
        f.close()

    # Write summary
    summary = {
        "source": "DrugBank 5.1 (exported 2026-03-05)",
        "small_molecules_total": stats["total_sm"],
        "has_smiles": stats["has_smiles"],
        "has_enzyme_annotations": stats["has_enzymes"],
        "has_transporter_annotations": stats["has_transporters"],
        "experimental_property_rows": stats["experimental_properties"],
        "pk_parsing": {
            "half_life": {
                "total_with_text": stats["pk_total"]["half_life"],
                "parsed_numeric": stats["pk_parsed"]["half_life"],
                "parse_rate": f"{100 * stats['pk_parsed']['half_life'] / max(stats['pk_total']['half_life'], 1):.1f}%",
            },
            "fup": {
                "total_with_text": stats["pk_total"]["fup"],
                "parsed_numeric": stats["pk_parsed"]["fup"],
                "parse_rate": f"{100 * stats['pk_parsed']['fup'] / max(stats['pk_total']['fup'], 1):.1f}%",
            },
            "vd": {
                "total_with_text": stats["pk_total"]["vd"],
                "parsed_numeric": stats["pk_parsed"]["vd"],
                "parse_rate": f"{100 * stats['pk_parsed']['vd'] / max(stats['pk_total']['vd'], 1):.1f}%",
            },
            "clearance": {
                "total_with_text": stats["pk_total"]["clearance"],
                "parsed_numeric": stats["pk_parsed"]["clearance"],
                "parse_rate": f"{100 * stats['pk_parsed']['clearance'] / max(stats['pk_total']['clearance'], 1):.1f}%",
            },
        },
        "output_files": [str(p) for p in sorted(OUT_DIR.glob("*.csv"))],
    }

    with open(OUT_DIR / "extraction_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Extraction complete!")
    logger.info("  Small molecules: %d", stats["total_sm"])
    logger.info("  With SMILES: %d", stats["has_smiles"])
    logger.info("  With enzymes: %d", stats["has_enzymes"])
    logger.info("  With transporters: %d", stats["has_transporters"])
    logger.info("  PK half-life parsed: %d / %d", stats["pk_parsed"]["half_life"], stats["pk_total"]["half_life"])
    logger.info("  PK fup parsed: %d / %d", stats["pk_parsed"]["fup"], stats["pk_total"]["fup"])
    logger.info("  PK Vd parsed: %d / %d", stats["pk_parsed"]["vd"], stats["pk_total"]["vd"])
    logger.info("  PK CL parsed: %d / %d", stats["pk_parsed"]["clearance"], stats["pk_total"]["clearance"])
    print(json.dumps(summary, indent=2))


def _text(elem, tag: str) -> str:
    """Get text of a child element, or empty string."""
    child = elem.find(NS + tag)
    return child.text.strip() if child is not None and child.text else ""


if __name__ == "__main__":
    extract()
