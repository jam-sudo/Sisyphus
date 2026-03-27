#!/usr/bin/env python3
"""
Holdout expansion script — extracts clinical PK data from OSP repos and FDA labels.

Source 1: Open-Systems-Pharmacology GitHub repos (observed C(t) profiles)
Source 2: FDA labels via DailyMed / PubChem

Produces:
  data/reference/osp_observed.json
  data/reference/fda_clin_pharm.json
  data/reference/clinical_pk_v2.json (integrated)
"""

import csv
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
import urllib.error
import urllib.parse
from dataclasses import asdict, dataclass, field
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DATA_REF = ROOT / "data" / "reference"
DATA_DRUGBANK = ROOT / "data" / "drugbank"

# ─── Name → SMILES / InChIKey infrastructure ───────────────────────────────

def load_drugbank_lookup() -> dict[str, dict]:
    """Load DrugBank drugs.csv into name→{smiles, inchikey_14, mw} lookup."""
    lookup = {}
    path = DATA_DRUGBANK / "drugs.csv"
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"].strip().lower()
            smiles = row.get("canonical_smiles", row.get("smiles", "")).strip()
            ik14 = row.get("inchikey_14", "").strip()
            if not ik14 and row.get("inchikey", ""):
                ik14 = row["inchikey"].split("-")[0]
            mw = row.get("mw", "")
            if smiles and ik14:
                lookup[name] = {
                    "smiles": smiles,
                    "inchikey_14": ik14,
                    "mw": float(mw) if mw else None,
                }
    log.info(f"DrugBank lookup: {len(lookup)} drugs with SMILES+InChIKey")
    return lookup


def pubchem_name_to_smiles(name: str) -> dict | None:
    """PubChem PUG REST: name → CID → canonical SMILES + InChIKey."""
    try:
        # Step 1: name → CID
        url = (
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
            f"{urllib.parse.quote(name)}/cids/JSON"
        )
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        cid = data["IdentifierList"]["CID"][0]

        # Step 2: CID → properties
        url2 = (
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/"
            f"property/CanonicalSMILES,InChIKey,MolecularWeight/JSON"
        )
        with urllib.request.urlopen(url2, timeout=10) as resp:
            props = json.loads(resp.read())
        p = props["PropertyTable"]["Properties"][0]
        ik = p.get("InChIKey", "")
        return {
            "smiles": p.get("CanonicalSMILES", ""),
            "inchikey_14": ik.split("-")[0] if ik else "",
            "mw": p.get("MolecularWeight"),
        }
    except Exception as e:
        log.debug(f"PubChem lookup failed for '{name}': {e}")
        return None


def resolve_smiles(name: str, drugbank: dict) -> dict | None:
    """Resolve drug name to SMILES+InChIKey. DrugBank first, PubChem fallback."""
    key = name.strip().lower()
    # Direct match
    if key in drugbank:
        return drugbank[key]
    # Common aliases
    aliases = {
        "paracetamol": "acetaminophen",
        "s-mephenytoin": "mephenytoin",
        "mefenamic acid": "mefenamic acid",
        "nicotine": "nicotine",
        "cotinine": "cotinine",
    }
    if key in aliases and aliases[key] in drugbank:
        return drugbank[aliases[key]]
    # Partial match (e.g. "quinidine" in "quinidine-ddgi")
    for db_name, info in drugbank.items():
        if key in db_name or db_name in key:
            return info
    # PubChem fallback
    result = pubchem_name_to_smiles(name)
    if result and result["smiles"]:
        return result
    return None


# ─── Existing data loaders ──────────────────────────────────────────────────

def load_existing_holdout() -> tuple[set[str], set[str], dict]:
    """Load holdout.json and clinical_pk.json. Returns (holdout_names, train_names, clinical_drugs)."""
    with open(DATA_REF / "holdout.json") as f:
        h = json.load(f)
    with open(DATA_REF / "clinical_pk.json") as f:
        c = json.load(f)
    return set(h["holdout"]), set(h["train"]), c["drugs"]


def load_mmpk_drugs() -> set[str]:
    """Load MMPK training drug names (lowercase)."""
    drugs = set()
    path = ROOT / "data" / "training" / "mmpk_expanded_full.csv"
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            drugs.add(row["name"].strip().lower())
    return drugs


def load_existing_inchikeys(clinical_drugs: dict) -> set[str]:
    """Get InChIKey-14 set from existing clinical_pk drugs that have SMILES."""
    drugbank = load_drugbank_lookup()
    iks = set()
    for name, drug in clinical_drugs.items():
        smiles = drug.get("smiles", "")
        if smiles:
            info = resolve_smiles(name, drugbank)
            if info and info["inchikey_14"]:
                iks.add(info["inchikey_14"])
    return iks


# ─── Source 1: OSP extraction ───────────────────────────────────────────────

# All OSP *-Model repos (excluding biologics, animal models, non-drug)
OSP_EXCLUDE = {
    "7E3-Model", "CDA1-Model", "MEDI524-Model", "MEDI524YTE-Model",
    "Tefibazumab-Model", "dAb2-Model", "Idarucizumab-Model",
    "Birds-PBK-Model", "Glucose-Insulin-Model", "Incretins-Model",
    "Thyroid-Hormones-PB-QSP-Model", "Creatinine-Model",
    "Inulin-Model", "N1-Methylnicotinamide-NMN-Model",
    "Dabigatran-Reversal-Model", "Nicotine-Cotinine-Model",
}


def list_osp_model_repos() -> list[str]:
    """Get all *-Model repo names from OSP GitHub org."""
    repos = []
    for page in range(1, 10):
        url = (
            f"https://api.github.com/orgs/Open-Systems-Pharmacology/repos"
            f"?per_page=100&page={page}"
        )
        try:
            with urllib.request.urlopen(url, timeout=15) as resp:
                data = json.loads(resp.read())
        except Exception as e:
            log.warning(f"GitHub API page {page}: {e}")
            break
        if not data:
            break
        for r in data:
            name = r["name"]
            if name.endswith("-Model") and name not in OSP_EXCLUDE:
                repos.append(name)
    log.info(f"OSP model repos found: {len(repos)}")
    return sorted(repos)


@dataclass
class OSPObservation:
    drug_name: str
    smiles: str
    inchikey_14: str
    dose_mg: float
    cmax_obs: float
    tmax_obs: float | None
    source: str
    study: str
    route: str = "oral"
    formulation: str = "IR"
    population: str = "healthy"
    fed_state: str = "fasted"
    cmax_extraction: str = "profile_max"
    quality_tier: str = "medium"
    n_subjects: int | None = None


def extract_osp_repo(repo_name: str, clone_dir: Path, drugbank: dict) -> list[OSPObservation]:
    """Clone an OSP repo and extract qualifying observed data."""
    drug_raw = repo_name.replace("-Model", "").replace("-", " ")

    repo_dir = clone_dir / repo_name
    if not repo_dir.exists():
        log.info(f"Cloning {repo_name}...")
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1",
                 f"https://github.com/Open-Systems-Pharmacology/{repo_name}.git",
                 str(repo_dir)],
                capture_output=True, timeout=60, check=True,
            )
        except Exception as e:
            log.warning(f"Failed to clone {repo_name}: {e}")
            return []

    # Find the main JSON file
    json_file = repo_dir / f"{repo_name}.json"
    if not json_file.exists():
        # Try alternative names
        candidates = list(repo_dir.glob("*.json"))
        candidates = [c for c in candidates if "evaluation" not in c.name.lower()]
        if candidates:
            json_file = candidates[0]
        else:
            log.warning(f"No model JSON in {repo_name}")
            return []

    try:
        with open(json_file) as f:
            model = json.load(f)
    except Exception as e:
        log.warning(f"Failed to parse {json_file}: {e}")
        return []

    observed_data = model.get("ObservedData", [])
    if not observed_data:
        log.debug(f"{repo_name}: no ObservedData")
        return []

    # Resolve drug SMILES
    info = resolve_smiles(drug_raw, drugbank)
    if not info:
        log.warning(f"No SMILES for {drug_raw}")
        return []

    results = []
    for od in observed_data:
        props = {p["Name"]: p["Value"] for p in od.get("ExtendedProperties", [])}

        # Filter: human, oral, plasma, fasted, single dose
        if props.get("Species", "").lower() != "human":
            continue
        route = props.get("Route", "").upper()
        if route != "PO":
            continue
        food = props.get("Food state", "").lower()
        if food != "fasted":
            continue
        compartment = props.get("Compartment", "").lower()
        if "plasma" not in compartment:
            continue

        # Parse dose
        dose_str = props.get("Dose", "")
        dose_match = re.search(r"([\d.]+)\s*mg", dose_str, re.IGNORECASE)
        if not dose_match:
            continue
        dose_mg = float(dose_match.group(1))

        # Get concentration data
        columns = od.get("Columns", [])
        base_grid = od.get("BaseGrid", {})
        if not columns or not base_grid:
            continue

        times = base_grid.get("Values", [])
        concs = columns[0].get("Values", [])
        unit = columns[0].get("Unit", "mg/l").lower()

        if not concs or not times:
            continue

        # Unit conversion to mg/L
        if "µg/l" in unit or "ug/l" in unit:
            concs = [c / 1000 for c in concs]
        elif "ng/ml" in unit or "ng/l" in unit:
            # ng/mL = µg/L = mg/L / 1000
            concs = [c / 1000 for c in concs]
        elif "mg/l" in unit:
            pass  # already correct
        elif "µg/ml" in unit or "ug/ml" in unit:
            pass  # µg/mL = mg/L
        else:
            log.debug(f"Unknown unit '{unit}' in {repo_name}, skipping")
            continue

        cmax = max(concs)
        tmax_idx = concs.index(cmax)
        tmax = times[tmax_idx] if tmax_idx < len(times) else None

        # Sanity check: oral Cmax typically 0.001-100 mg/L
        if cmax < 0.0001 or cmax > 500:
            log.debug(f"Cmax {cmax} out of range for {drug_raw}, skipping")
            continue

        study_id = props.get("Study Id", od.get("Name", "unknown"))
        n_subjects = props.get("N")

        results.append(OSPObservation(
            drug_name=drug_raw.lower(),
            smiles=info["smiles"],
            inchikey_14=info["inchikey_14"],
            dose_mg=dose_mg,
            cmax_obs=round(cmax, 6),
            tmax_obs=round(tmax, 3) if tmax is not None else None,
            source=f"OSP/{repo_name}/{json_file.name}",
            study=str(study_id),
            n_subjects=int(n_subjects) if n_subjects else None,
        ))

    return results


def extract_all_osp(drugbank: dict) -> list[OSPObservation]:
    """Extract from all OSP repos."""
    repos = list_osp_model_repos()
    clone_dir = Path(tempfile.mkdtemp(prefix="osp_"))
    log.info(f"Clone directory: {clone_dir}")

    all_obs = []
    for repo in repos:
        obs = extract_osp_repo(repo, clone_dir, drugbank)
        if obs:
            log.info(f"  {repo}: {len(obs)} qualifying observations")
            all_obs.extend(obs)
        time.sleep(0.5)  # Be nice to GitHub

    log.info(f"Total OSP observations: {len(all_obs)}")
    return all_obs


def select_best_osp(observations: list[OSPObservation]) -> dict[str, OSPObservation]:
    """For each drug, select the best single observation.
    Prefer: largest N, then median Cmax among similar-dose studies.
    """
    by_drug: dict[str, list[OSPObservation]] = {}
    for obs in observations:
        by_drug.setdefault(obs.drug_name, []).append(obs)

    best = {}
    for drug, obs_list in by_drug.items():
        # Group by dose
        dose_groups: dict[float, list[OSPObservation]] = {}
        for o in obs_list:
            dose_groups.setdefault(o.dose_mg, []).append(o)

        # Pick the dose with most studies
        best_dose = max(dose_groups, key=lambda d: len(dose_groups[d]))
        candidates = dose_groups[best_dose]

        # Take median Cmax
        cmaxes = sorted(c.cmax_obs for c in candidates)
        median_cmax = cmaxes[len(cmaxes) // 2]

        # Pick the observation closest to median
        best_obs = min(candidates, key=lambda o: abs(o.cmax_obs - median_cmax))
        best[drug] = best_obs

    return best


# ─── Source 2: FDA label extraction ─────────────────────────────────────────

@dataclass
class FDAObservation:
    drug_name: str
    smiles: str
    inchikey_14: str
    dose_mg: float
    cmax_obs: float
    cmax_unit_original: str
    cmax_conversion_factor: float
    source: str
    route: str = "oral"
    formulation: str = "IR"
    population: str = "healthy"
    fed_state: str = "fasted"
    n_subjects: int | None = None
    quality_tier: str = "gold"


def fetch_dailymed_pk(drug_name: str) -> str | None:
    """Fetch pharmacokinetics text from DailyMed SPL for a drug."""
    try:
        # Search for the drug
        search_url = (
            f"https://dailymed.nlm.nih.gov/dailymed/services/v2/spls.json"
            f"?drug_name={urllib.parse.quote(drug_name)}&page=1&pagesize=1"
        )
        with urllib.request.urlopen(search_url, timeout=15) as resp:
            search = json.loads(resp.read())

        results = search.get("data", [])
        if not results:
            return None

        set_id = results[0].get("setid", "")
        if not set_id:
            return None

        # Get the SPL XML
        spl_url = f"https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/{set_id}.json"
        with urllib.request.urlopen(spl_url, timeout=15) as resp:
            spl_data = json.loads(resp.read())

        # Extract text — look for clinical pharmacology section
        # The JSON response has various sections
        # Try to get the full text and search for PK data
        full_text = json.dumps(spl_data)

        return full_text

    except Exception as e:
        log.debug(f"DailyMed fetch failed for '{drug_name}': {e}")
        return None


def parse_cmax_from_text(text: str) -> list[dict]:
    """Extract Cmax values from pharmacokinetics text.
    Returns list of {cmax_value, cmax_unit, dose_mg, context}.
    """
    results = []

    # Patterns for Cmax extraction
    # "Cmax of X ng/mL" or "Cmax was X ng/mL" or "Cmax = X ng/mL"
    # "mean Cmax of X ± Y ng/mL"
    # "peak plasma concentration (Cmax) of X ng/mL"
    patterns = [
        # "Cmax of/was/= X (± Y) unit"
        r"[Cc]max\s*(?:of|was|is|=|:)\s*(?:approximately\s+)?(\d+[\d,.]*)\s*(?:±\s*[\d,.]+\s*)?(ng/mL|µg/mL|mg/L|µg/L|ng/L|mcg/mL)",
        # "mean Cmax of X unit"
        r"[Mm]ean\s+[Cc]max\s*(?:of|was|is|=|:)\s*(?:approximately\s+)?(\d+[\d,.]*)\s*(?:±\s*[\d,.]+\s*)?(ng/mL|µg/mL|mg/L|µg/L|ng/L|mcg/mL)",
        # "peak.*concentration.*of X unit"
        r"peak\s+(?:plasma\s+)?concentration[^.]*?(\d+[\d,.]*)\s*(?:±\s*[\d,.]+\s*)?(ng/mL|µg/mL|mg/L|µg/L|ng/L|mcg/mL)",
        # "Cmax (unit)" in table-like format followed by value
        r"[Cc]max\s*\((ng/mL|µg/mL|mg/L|µg/L|mcg/mL)\)\s*[:\s]*(\d+[\d,.]*)",
    ]

    for pat in patterns:
        for m in re.finditer(pat, text):
            groups = m.groups()
            if len(groups) == 2:
                val_str, unit = groups
                # Handle pattern 4 where unit comes first
                try:
                    val = float(val_str.replace(",", ""))
                except ValueError:
                    try:
                        val = float(unit.replace(",", ""))
                        unit = val_str
                    except ValueError:
                        continue
            else:
                continue

            results.append({
                "cmax_value": val,
                "cmax_unit": unit,
                "context": text[max(0, m.start() - 100): m.end() + 100],
            })

    return results


def convert_cmax_to_mg_L(value: float, unit: str) -> tuple[float, float]:
    """Convert Cmax to mg/L. Returns (value_mg_L, conversion_factor)."""
    unit_lower = unit.lower().replace(" ", "")
    if unit_lower in ("ng/ml",):
        return value * 1e-6, 1e-6
    elif unit_lower in ("µg/ml", "ug/ml", "mcg/ml"):
        return value, 1.0  # µg/mL = mg/L
    elif unit_lower in ("mg/l",):
        return value, 1.0
    elif unit_lower in ("µg/l", "ug/l", "mcg/l"):
        return value / 1000, 0.001
    elif unit_lower in ("ng/l",):
        return value * 1e-6 / 1000, 1e-9  # This seems too small; ng/L is very dilute
    else:
        raise ValueError(f"Unknown unit: {unit}")


def extract_fda_for_drug(
    drug_name: str, drugbank: dict, target_dose: float | None = None
) -> FDAObservation | None:
    """Try to extract FDA label PK data for a single drug."""
    info = resolve_smiles(drug_name, drugbank)
    if not info:
        log.debug(f"No SMILES for {drug_name}")
        return None

    text = fetch_dailymed_pk(drug_name)
    if not text:
        log.debug(f"No DailyMed data for {drug_name}")
        return None

    cmax_hits = parse_cmax_from_text(text)
    if not cmax_hits:
        log.debug(f"No Cmax found in label for {drug_name}")
        return None

    # Filter: prefer fasted, single dose, healthy
    # Exclude: multiple doses, patients, fed, ER/MR
    exclusion_patterns = [
        r"multiple\s+dose", r"steady\s+state", r"patients?\s+with",
        r"extended.?release", r"modified.?release", r"sustained.?release",
        r"with\s+food", r"high.?fat\s+meal", r"fed\s+state",
    ]

    valid_hits = []
    for hit in cmax_hits:
        ctx = hit["context"].lower()
        excluded = any(re.search(p, ctx) for p in exclusion_patterns)
        if not excluded:
            valid_hits.append(hit)

    if not valid_hits:
        valid_hits = cmax_hits  # Fall back to all hits

    # Convert and pick the first valid one
    for hit in valid_hits:
        try:
            cmax_mg_L, conv_factor = convert_cmax_to_mg_L(
                hit["cmax_value"], hit["cmax_unit"]
            )
        except ValueError:
            continue

        # Sanity check
        if cmax_mg_L < 0.0001 or cmax_mg_L > 500:
            log.debug(f"Cmax {cmax_mg_L} mg/L out of range for {drug_name}")
            continue

        return FDAObservation(
            drug_name=drug_name.lower(),
            smiles=info["smiles"],
            inchikey_14=info["inchikey_14"],
            dose_mg=target_dose or 0,  # Will need manual fill if unknown
            cmax_obs=round(cmax_mg_L, 6),
            cmax_unit_original=hit["cmax_unit"],
            cmax_conversion_factor=conv_factor,
            source="FDA label via DailyMed",
        )

    return None


# ─── PubChem PK extraction (fallback for FDA) ──────────────────────────────

def pubchem_get_pk_text(drug_name: str) -> str | None:
    """Get pharmacokinetics annotation from PubChem."""
    try:
        url = (
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
            f"{urllib.parse.quote(drug_name)}/cids/JSON"
        )
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read())
        cid = data["IdentifierList"]["CID"][0]

        # Get annotations
        url2 = (
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/"
            f"JSON?heading=Pharmacokinetics"
        )
        with urllib.request.urlopen(url2, timeout=15) as resp:
            return resp.read().decode()
    except Exception as e:
        log.debug(f"PubChem PK fetch failed for {drug_name}: {e}")
        return None


# ─── Integration ────────────────────────────────────────────────────────────

def integrate_sources(
    osp_best: dict[str, OSPObservation],
    fda_obs: dict[str, FDAObservation],
    existing_drugs: dict,
    holdout_names: set[str],
    train_names: set[str],
    mmpk_drugs: set[str],
    drugbank: dict,
) -> dict:
    """Integrate all sources into clinical_pk_v2.json."""
    # Start with existing drugs
    new_drugs = dict(existing_drugs)

    # Track what we add
    stats = {
        "osp_found": len(osp_best),
        "fda_found": len(fda_obs),
        "osp_added": 0,
        "fda_added": 0,
        "flags": [],
    }

    # Collect existing InChIKeys — only for drugs that already have Cmax
    existing_iks_with_cmax = set()
    for name, drug in existing_drugs.items():
        if drug.get("pk_params", {}).get("cmax_mg_L"):
            info = resolve_smiles(name, drugbank)
            if info and info["inchikey_14"]:
                existing_iks_with_cmax.add(info["inchikey_14"])

    added_iks = set()

    # Helper to add a drug
    def add_drug(name, cmax, dose, smiles, ik14, source, tier, src_label):
        nonlocal stats
        name_lower = name.lower().strip()

        # Already in clinical_pk with Cmax?
        if name_lower in new_drugs and new_drugs[name_lower].get("pk_params", {}).get("cmax_mg_L"):
            existing_cmax = new_drugs[name_lower]["pk_params"]["cmax_mg_L"]
            ratio = cmax / existing_cmax if existing_cmax > 0 else float("inf")
            if ratio < 0.8 or ratio > 1.2:
                stats["flags"].append({
                    "drug": name_lower,
                    "existing_cmax": existing_cmax,
                    "new_cmax": cmax,
                    "ratio": round(ratio, 3),
                    "source": src_label,
                })
            return  # Don't overwrite existing

        # Drug exists but WITHOUT Cmax → update it
        if name_lower in new_drugs:
            existing = new_drugs[name_lower]
            existing.setdefault("pk_params", {})["cmax_mg_L"] = cmax
            if dose:
                existing["dose_mg"] = dose
            if smiles:
                existing["smiles"] = smiles
            existing["source"] = f"{existing.get('source', '')} + {source}"
            stats[f"{src_label}_added"] += 1
            return

        # InChIKey dedup (only for genuinely new drugs)
        if ik14 in added_iks:
            return
        # Check if InChIKey matches an existing drug that already has Cmax
        if ik14 in existing_iks_with_cmax:
            return

        new_drugs[name_lower] = {
            "name": name_lower,
            "tier": tier,
            "source": source,
            "dose_mg": dose,
            "route": "oral",
            "pk_params": {"cmax_mg_L": cmax},
            "smiles": smiles,
        }
        added_iks.add(ik14)
        stats[f"{src_label}_added"] += 1

    # Add FDA data first (higher quality)
    for name, obs in fda_obs.items():
        add_drug(name, obs.cmax_obs, obs.dose_mg, obs.smiles,
                 obs.inchikey_14, obs.source, "gold", "fda")

    # Add OSP data
    for name, obs in osp_best.items():
        add_drug(name, obs.cmax_obs, obs.dose_mg, obs.smiles,
                 obs.inchikey_14, obs.source, "silver", "osp")

    return new_drugs, stats


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("Holdout Expansion Pipeline")
    log.info("=" * 60)

    # Load infrastructure
    drugbank = load_drugbank_lookup()
    holdout_names, train_names, existing_drugs = load_existing_holdout()
    mmpk_drugs = load_mmpk_drugs()

    # Count existing holdout drugs with Cmax
    existing_holdout_with_cmax = set()
    for name in holdout_names:
        if name in existing_drugs:
            pk = existing_drugs[name].get("pk_params", {})
            if pk.get("cmax_mg_L"):
                existing_holdout_with_cmax.add(name)
    log.info(f"Current holdout with Cmax: {len(existing_holdout_with_cmax)}/100")

    # Missing holdout drugs (priority targets for FDA extraction)
    missing_holdout = holdout_names - existing_holdout_with_cmax
    log.info(f"Missing holdout drugs (priority): {len(missing_holdout)}")

    # ── Source 1: OSP ──
    log.info("\n--- Source 1: OSP Observed Data ---")
    all_osp = extract_all_osp(drugbank)

    # Save raw OSP data
    osp_raw = [asdict(o) for o in all_osp]
    with open(DATA_REF / "osp_observed_raw.json", "w") as f:
        json.dump(osp_raw, f, indent=2)
    log.info(f"Saved {len(osp_raw)} raw OSP observations")

    # Select best per drug
    osp_best = select_best_osp(all_osp)
    log.info(f"OSP unique drugs: {len(osp_best)}")

    osp_out = [asdict(v) for v in osp_best.values()]
    with open(DATA_REF / "osp_observed.json", "w") as f:
        json.dump(osp_out, f, indent=2)

    # ── Source 2: FDA labels ──
    log.info("\n--- Source 2: FDA Label Extraction ---")

    # Priority: missing holdout drugs
    fda_obs: dict[str, FDAObservation] = {}

    # Try DailyMed for all missing holdout drugs + OSP drugs
    all_targets = list(missing_holdout) + [
        name for name in osp_best if name.lower() not in holdout_names
    ]

    for drug_name in sorted(set(all_targets)):
        obs = extract_fda_for_drug(drug_name, drugbank)
        if obs:
            fda_obs[drug_name.lower()] = obs
            log.info(f"  FDA: {drug_name} → Cmax={obs.cmax_obs} mg/L")
        time.sleep(0.3)  # Rate limit

    fda_out = [asdict(v) for v in fda_obs.values()]
    with open(DATA_REF / "fda_clin_pharm.json", "w") as f:
        json.dump(fda_out, f, indent=2)
    log.info(f"FDA extractions: {len(fda_obs)}")

    # ── Integration ──
    log.info("\n--- Integration ---")
    new_drugs, stats = integrate_sources(
        osp_best, fda_obs, existing_drugs,
        holdout_names, train_names, mmpk_drugs, drugbank,
    )

    # Count new holdout drugs with Cmax
    new_holdout_with_cmax = set()
    for name in holdout_names:
        if name in new_drugs:
            pk = new_drugs[name].get("pk_params", {})
            if pk.get("cmax_mg_L"):
                new_holdout_with_cmax.add(name)

    # Build clinical_pk_v2
    # Update holdout.json if we have new drugs not in train/holdout split
    new_holdout_drugs_added = set(new_drugs.keys()) - set(existing_drugs.keys())
    new_not_in_split = new_holdout_drugs_added - holdout_names - train_names

    result = {
        "metadata": {
            "created": "2026-03-26",
            "n_drugs": len(new_drugs),
            "sources": {
                "original": len(existing_drugs),
                "osp_added": stats["osp_added"],
                "fda_added": stats["fda_added"],
            },
        },
        "drugs": dict(sorted(new_drugs.items())),
    }

    with open(DATA_REF / "clinical_pk_v2.json", "w") as f:
        json.dump(result, f, indent=2)

    # Report
    log.info("\n" + "=" * 60)
    log.info("REPORT")
    log.info("=" * 60)
    log.info(f"| Source   | Found | Added |")
    log.info(f"|---------|-------|-------|")
    log.info(f"| OSP     | {stats['osp_found']:5} | {stats['osp_added']:5} |")
    log.info(f"| FDA     | {stats['fda_found']:5} | {stats['fda_added']:5} |")
    log.info(f"| Total   |       | {stats['osp_added'] + stats['fda_added']:5} |")
    log.info(f"")
    log.info(f"Holdout drugs with Cmax: {len(existing_holdout_with_cmax)} → {len(new_holdout_with_cmax)}")
    log.info(f"Holdout STILL missing: {sorted(holdout_names - new_holdout_with_cmax)}")
    log.info(f"New drugs not in holdout/train split: {len(new_not_in_split)}")
    if new_not_in_split:
        log.info(f"  → These need to be added to holdout: {sorted(new_not_in_split)}")
    log.info(f"Total drugs in clinical_pk_v2: {len(new_drugs)}")

    if stats["flags"]:
        log.info(f"\n⚠ FLAGS ({len(stats['flags'])} discrepancies):")
        for flag in stats["flags"]:
            log.info(f"  {flag['drug']}: existing={flag['existing_cmax']}, "
                     f"new={flag['new_cmax']}, ratio={flag['ratio']}")

    return stats


if __name__ == "__main__":
    main()
