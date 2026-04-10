"""Phase 1: Batch docking via NVIDIA NIM DiffDock API.

Docks drugs against all 5 CYP structures using DiffDock.
Much faster than Vina (~1.5s/call) and supports all CYPs.

Usage:
    python scripts/batch_docking_diffdock.py [--max-drugs N] [--cyp CYP3A4|all]
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import requests
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
STRUCTURES_DIR = ROOT / "data" / "docking" / "structures"
CACHE_DIR = ROOT / "data" / "docking" / "cache_diffdock"

API_URL = "https://health.api.nvidia.com/v1/biology/mit/diffdock"
API_KEY = os.environ.get("NVIDIA_API_KEY", "")

# Rate limiting
MIN_INTERVAL_S = 0.5  # Min seconds between API calls


def load_env():
    """Load .env file if API key not in environment."""
    global API_KEY
    if API_KEY:
        return
    env_path = ROOT / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("NVIDIA_API_KEY="):
                    API_KEY = line.split("=", 1)[1].strip()
                    break
    if not API_KEY:
        logger.error("NVIDIA_API_KEY not found. Set env var or add to .env")
        sys.exit(1)


def inchikey_14(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return hashlib.md5(smiles.encode()).hexdigest()[:14]
    return Chem.inchi.MolToInchiKey(mol)[:14]


def smiles_to_sdf(smiles: str) -> str | None:
    """Convert SMILES to SDF mol block."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.maxAttempts = 50
        result = AllChem.EmbedMolecule(mol, params)
        if result != 0:
            params.useRandomCoords = True
            result = AllChem.EmbedMolecule(mol, params)
            if result != 0:
                return None
        AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
        mol = Chem.RemoveHs(mol)
        return Chem.MolToMolBlock(mol)
    except Exception:
        return None


def load_protein_pdb(cyp: str) -> str:
    """Load minimal PDB for a CYP (ATOM + HETATM only)."""
    pdb_path = STRUCTURES_DIR / f"{cyp}.pdb"
    lines = []
    with open(pdb_path) as f:
        for line in f:
            rec = line[:6].strip()
            if rec in ("ATOM", "HETATM", "TER", "END"):
                lines.append(line)
    return "".join(lines)


def call_diffdock(protein_pdb: str, ligand_sdf: str, num_poses: int = 1) -> dict | None:
    """Call DiffDock NIM API. Returns response dict or None."""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    payload = {
        "protein": protein_pdb,
        "ligand": ligand_sdf,
        "ligand_file_type": "sdf",
        "num_poses": num_poses,
        "time_divisions": 20,
        "steps": 18,
        "save_trajectory": False,
        "is_staged": False,
    }
    try:
        resp = requests.post(API_URL, headers=headers, json=payload, timeout=120)
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 429:
            logger.warning("Rate limited, waiting 10s...")
            time.sleep(10)
            resp = requests.post(API_URL, headers=headers, json=payload, timeout=120)
            if resp.status_code == 200:
                return resp.json()
        logger.debug(f"API error {resp.status_code}: {resp.text[:200]}")
        return None
    except Exception as e:
        logger.debug(f"API call failed: {e}")
        return None


def extract_features_from_pose(pose_sdf: str, confidence: float, heme_fe: dict) -> dict:
    """Extract features from a DiffDock docked pose."""
    fe = np.array([heme_fe["x"], heme_fe["y"], heme_fe["z"]])

    docked_mol = Chem.MolFromMolBlock(pose_sdf)
    if docked_mol is None:
        # Try with newline prefix
        docked_mol = Chem.MolFromMolBlock("\n" + pose_sdf)

    if docked_mol is None:
        return {
            "docking_score": confidence,
            "heme_fe_distance_min": float("nan"),
            "heme_fe_distance_atom_idx": -1,
            "heme_fe_distance_atom_symbol": "?",
            "n_heavy_atoms": 0,
            "pose_centroid_fe_dist": float("nan"),
        }

    conf = docked_mol.GetConformer()
    n_atoms = docked_mol.GetNumAtoms()
    coords = np.array([list(conf.GetAtomPosition(i)) for i in range(n_atoms)])

    # Heavy atoms only
    heavy = []
    for i in range(n_atoms):
        atom = docked_mol.GetAtomWithIdx(i)
        if atom.GetAtomicNum() > 1:
            dist = float(np.linalg.norm(coords[i] - fe))
            heavy.append((i, atom.GetSymbol(), dist, coords[i]))

    if not heavy:
        return {
            "docking_score": confidence,
            "heme_fe_distance_min": float("nan"),
            "heme_fe_distance_atom_idx": -1,
            "heme_fe_distance_atom_symbol": "?",
            "n_heavy_atoms": 0,
            "pose_centroid_fe_dist": float("nan"),
        }

    heavy.sort(key=lambda x: x[2])
    centroid = np.mean([h[3] for h in heavy], axis=0)
    centroid_dist = float(np.linalg.norm(centroid - fe))

    return {
        "docking_score": confidence,
        "heme_fe_distance_min": round(heavy[0][2], 3),
        "heme_fe_distance_atom_idx": heavy[0][0],
        "heme_fe_distance_atom_symbol": heavy[0][1],
        "n_heavy_atoms": len(heavy),
        "pose_centroid_fe_dist": round(centroid_dist, 3),
    }


def load_drug_list() -> list[dict]:
    """Load all drugs for docking."""
    drugs = []
    seen_smiles = set()

    tdc_path = ROOT / "data" / "clearance_hepatocyte_az.tab"
    if tdc_path.exists():
        with open(tdc_path) as f:
            f.readline()
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 3:
                    continue
                chembl_id = parts[0].strip('"')
                smiles = parts[1].strip('"')
                if smiles not in seen_smiles:
                    drugs.append({"id": chembl_id, "smiles": smiles, "source": "tdc_hep_az"})
                    seen_smiles.add(smiles)

    cpk_path = ROOT / "data" / "reference" / "clinical_pk.json"
    holdout_path = ROOT / "data" / "reference" / "holdout.json"
    if cpk_path.exists() and holdout_path.exists():
        with open(holdout_path) as f:
            holdout_names = set(json.load(f)["holdout"])
        with open(cpk_path) as f:
            cpk = json.load(f)
        for name, data in cpk["drugs"].items():
            if name in holdout_names and "smiles" in data:
                smiles = data["smiles"]
                if smiles not in seen_smiles:
                    drugs.append({"id": name, "smiles": smiles, "source": "holdout"})
                    seen_smiles.add(smiles)

    return drugs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cyp", default="all", help="CYP name or 'all'")
    parser.add_argument("--max-drugs", type=int, default=None)
    parser.add_argument("--num-poses", type=int, default=1)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    load_env()

    meta_path = STRUCTURES_DIR / "cyp_metadata.json"
    with open(meta_path) as f:
        metadata = json.load(f)

    cyps = list(metadata.keys()) if args.cyp == "all" else [args.cyp]
    drugs = load_drug_list()
    if args.max_drugs:
        drugs = drugs[:args.max_drugs]

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Pre-load protein PDBs
    protein_pdbs = {}
    for cyp in cyps:
        protein_pdbs[cyp] = load_protein_pdb(cyp)
        logger.info(f"Loaded {cyp} PDB: {len(protein_pdbs[cyp])} chars")

    # Pre-generate SDF for all drugs
    logger.info(f"Generating SDF for {len(drugs)} drugs...")
    drug_sdfs = {}
    sdf_fail = 0
    for drug in drugs:
        sdf = smiles_to_sdf(drug["smiles"])
        if sdf:
            drug_sdfs[drug["smiles"]] = sdf
        else:
            sdf_fail += 1
    logger.info(f"SDF generated: {len(drug_sdfs)} success, {sdf_fail} failed")

    for cyp in cyps:
        logger.info(f"\n=== DiffDock: {len(drugs)} drugs × {cyp} ===")
        protein_pdb = protein_pdbs[cyp]
        heme_fe = metadata[cyp]["heme_fe"]

        success = 0
        failure = 0
        cached = 0
        t0 = time.time()
        last_call_time = 0

        for i, drug in enumerate(drugs):
            key14 = inchikey_14(drug["smiles"])
            cache_path = CACHE_DIR / f"{key14}_{cyp}.json"

            if cache_path.exists():
                cached += 1
                continue

            sdf = drug_sdfs.get(drug["smiles"])
            if sdf is None:
                failure += 1
                continue

            # Rate limiting
            elapsed_since_last = time.time() - last_call_time
            if elapsed_since_last < MIN_INTERVAL_S:
                time.sleep(MIN_INTERVAL_S - elapsed_since_last)

            result = call_diffdock(protein_pdb, sdf, num_poses=args.num_poses)
            last_call_time = time.time()

            if result is None:
                failure += 1
                if (i + 1) % 50 == 0:
                    logger.info(f"[{cyp}] {i+1}/{len(drugs)} s={success} c={cached} f={failure}")
                continue

            # Extract features from top pose
            pose_sdf = result["ligand_positions"][0]
            confidence = result.get("position_confidence", [0.0])[0]
            features = extract_features_from_pose(pose_sdf, confidence, heme_fe)

            entry = {
                "smiles": drug["smiles"],
                "inchikey_14": key14,
                "cyp": cyp,
                "tool": "diffdock_nim",
                **features,
            }

            with open(cache_path, "w") as f:
                json.dump(entry, f, indent=2)

            success += 1

            if (i + 1) % 50 == 0:
                elapsed = time.time() - t0
                rate = (success + failure) / elapsed if elapsed > 0 else 0
                remaining = len(drugs) - i - 1 - cached
                eta = remaining / rate / 60 if rate > 0 else float("inf")
                logger.info(
                    f"[{cyp}] {i+1}/{len(drugs)} "
                    f"s={success} c={cached} f={failure} "
                    f"({rate:.1f}/s, ETA {eta:.0f}min)"
                )

        elapsed = time.time() - t0
        total_new = success + failure
        fail_rate = failure / total_new if total_new > 0 else 0

        logger.info(
            f"\n[{cyp}] DONE: {success} success, {cached} cached, {failure} fail "
            f"({fail_rate:.1%} fail rate), {elapsed:.0f}s"
        )

        if fail_rate > 0.30:
            logger.error(f"GATE FAIL: {cyp} failure rate {fail_rate:.1%} > 30%")


if __name__ == "__main__":
    main()
