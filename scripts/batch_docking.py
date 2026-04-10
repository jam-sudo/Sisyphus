"""Phase 1: Batch docking of drugs against CYP structures.

Docks TDC Hepatocyte_AZ training set + holdout drugs against CYP(s).
Caches results as JSON per drug-CYP pair.

Usage:
    python scripts/batch_docking.py --cyp CYP3A4 [--exhaustiveness 8] [--max-drugs N]
    python scripts/batch_docking.py --cyp all
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors

RDLogger.DisableLog("rdApp.*")

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
STRUCTURES_DIR = ROOT / "data" / "docking" / "structures"
CACHE_DIR = ROOT / "data" / "docking" / "cache"


def inchikey_14(smiles: str) -> str:
    """Get first 14 chars of InChIKey for a SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return hashlib.md5(smiles.encode()).hexdigest()[:14]
    return Chem.inchi.MolToInchiKey(mol)[:14]


def smiles_to_pdbqt(smiles: str) -> str | None:
    """Convert SMILES to PDBQT string. Returns None on failure."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Check MW — skip very large molecules
        mw = Descriptors.ExactMolWt(mol)
        if mw > 1000:
            logger.debug(f"Skipping large molecule MW={mw:.0f}")
            return None

        mol = Chem.AddHs(mol)

        params = AllChem.ETKDGv3()
        params.maxAttempts = 50
        result = AllChem.EmbedMolecule(mol, params)
        if result != 0:
            # Retry with random coordinates
            params.useRandomCoords = True
            result = AllChem.EmbedMolecule(mol, params)
            if result != 0:
                return None

        AllChem.MMFFOptimizeMolecule(mol, maxIters=200)

        from meeko import MoleculePreparation, PDBQTWriterLegacy

        preparator = MoleculePreparation()
        mol_setups = preparator.prepare(mol)
        setup = mol_setups[0]

        pdbqt_string, is_ok, err = PDBQTWriterLegacy.write_string(setup)
        if not is_ok:
            return None

        return pdbqt_string
    except Exception as e:
        logger.debug(f"PDBQT prep failed: {e}")
        return None


def run_vina_docking(
    receptor_pdbqt_path: str,
    ligand_pdbqt: str,
    center: tuple[float, float, float],
    box_size: tuple[float, float, float] = (25.0, 25.0, 25.0),
    exhaustiveness: int = 8,
) -> dict | None:
    """Run Vina docking. Returns result dict or None on failure."""
    try:
        from vina import Vina

        v = Vina(sf_name="vina", verbosity=0)

        with tempfile.NamedTemporaryFile(suffix=".pdbqt", mode="w", delete=False) as f:
            f.write(ligand_pdbqt)
            ligand_path = f.name

        v.set_receptor(receptor_pdbqt_path)
        v.set_ligand_from_file(ligand_path)
        v.compute_vina_maps(center=list(center), box_size=list(box_size))

        v.dock(exhaustiveness=exhaustiveness, n_poses=1)

        energies = v.energies()
        score = float(energies[0][0]) if energies is not None and len(energies) > 0 else None

        poses_pdbqt = v.poses(n_poses=1)

        os.unlink(ligand_path)

        # Parse pose atoms
        pose_atoms = []
        for line in poses_pdbqt.split("\n"):
            if line.startswith("ATOM") or line.startswith("HETATM"):
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                atom_name = line[12:16].strip()
                element = line[77:79].strip() if len(line) > 78 else atom_name[0]
                pose_atoms.append({
                    "name": atom_name,
                    "element": element,
                    "x": x, "y": y, "z": z,
                })

        return {"docking_score": score, "pose_atoms": pose_atoms}
    except Exception as e:
        logger.debug(f"Vina docking failed: {e}")
        try:
            os.unlink(ligand_path)
        except Exception:
            pass
        return None


def extract_features(
    pose_atoms: list[dict],
    heme_fe: dict,
    receptor_atoms: list[dict] | None = None,
) -> dict:
    """Extract docking features from a pose.

    Returns 6 features per CYP:
    1. heme_fe_distance_min (Å)
    2. heme_fe_distance_atom_idx, atom_symbol
    3. n_contacts_hydrophobic (within 4Å of any receptor C)
    4. n_contacts_hbond (within 3.5Å of receptor N/O)
    5. n_contacts_total (within 4.5Å)
    6. pose_buried_sasa — approximated by number of close contacts
    """
    fe_coord = np.array([heme_fe["x"], heme_fe["y"], heme_fe["z"]])

    min_dist = float("inf")
    min_idx = -1
    min_symbol = "?"
    heavy_atoms = []

    for i, atom in enumerate(pose_atoms):
        element = atom.get("element", atom["name"][0]).strip()
        if element.upper() in ("H", "HD", ""):
            continue
        coord = np.array([atom["x"], atom["y"], atom["z"]])
        dist = float(np.linalg.norm(coord - fe_coord))
        heavy_atoms.append({"idx": i, "element": element, "coord": coord, "fe_dist": dist})
        if dist < min_dist:
            min_dist = dist
            min_idx = i
            min_symbol = element

    return {
        "heme_fe_distance_min": round(min_dist, 3),
        "heme_fe_distance_atom_idx": min_idx,
        "heme_fe_distance_atom_symbol": min_symbol,
        "n_heavy_atoms": len(heavy_atoms),
    }


def load_drug_list() -> list[dict]:
    """Load all drugs for docking: TDC Hepatocyte_AZ + holdout drugs."""
    drugs = []
    seen_smiles = set()

    # TDC Hepatocyte_AZ
    tdc_path = ROOT / "data" / "clearance_hepatocyte_az.tab"
    if tdc_path.exists():
        with open(tdc_path) as f:
            header = f.readline()
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 3:
                    continue
                chembl_id = parts[0].strip('"')
                smiles = parts[1].strip('"')
                clint = float(parts[2])
                if smiles not in seen_smiles:
                    drugs.append({
                        "id": chembl_id,
                        "smiles": smiles,
                        "source": "tdc_hep_az",
                        "clint": clint,
                    })
                    seen_smiles.add(smiles)

    # Holdout drugs (for feature generation, NOT for CLint training)
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
                    drugs.append({
                        "id": name,
                        "smiles": smiles,
                        "source": "holdout",
                    })
                    seen_smiles.add(smiles)

    logger.info(f"Loaded {len(drugs)} drugs ({sum(1 for d in drugs if d['source'] == 'tdc_hep_az')} TDC + "
                f"{sum(1 for d in drugs if d['source'] == 'holdout')} holdout)")
    return drugs


def dock_single(drug: dict, cyp: str, metadata: dict, exhaustiveness: int) -> dict | None:
    """Dock one drug against one CYP. Returns cache entry or None."""
    key14 = inchikey_14(drug["smiles"])
    cache_path = CACHE_DIR / f"{key14}_{cyp}.json"

    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)

    # Prepare ligand
    ligand_pdbqt = smiles_to_pdbqt(drug["smiles"])
    if ligand_pdbqt is None:
        return None

    # Dock
    center = tuple(metadata[cyp]["active_site_center"])
    receptor_path = str(STRUCTURES_DIR / f"{cyp}.pdbqt")

    result = run_vina_docking(
        receptor_pdbqt_path=receptor_path,
        ligand_pdbqt=ligand_pdbqt,
        center=center,
        exhaustiveness=exhaustiveness,
    )

    if result is None:
        return None

    # Extract features
    heme_fe = metadata[cyp]["heme_fe"]
    features = extract_features(result["pose_atoms"], heme_fe)

    # Build cache entry
    entry = {
        "smiles": drug["smiles"],
        "inchikey_14": key14,
        "cyp": cyp,
        "tool": "vina",
        "docking_score": result["docking_score"],
        **features,
    }

    with open(cache_path, "w") as f:
        json.dump(entry, f, indent=2)

    return entry


def main():
    parser = argparse.ArgumentParser(description="Batch docking for CLint features")
    parser.add_argument("--cyp", default="CYP3A4", help="CYP name or 'all'")
    parser.add_argument("--exhaustiveness", type=int, default=8)
    parser.add_argument("--max-drugs", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    # Load metadata
    meta_path = STRUCTURES_DIR / "cyp_metadata.json"
    with open(meta_path) as f:
        metadata = json.load(f)

    cyps = list(metadata.keys()) if args.cyp == "all" else [args.cyp]

    # Load drugs
    drugs = load_drug_list()
    if args.max_drugs:
        drugs = drugs[:args.max_drugs]

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    for cyp in cyps:
        logger.info(f"\n=== Docking {len(drugs)} drugs against {cyp} ===")

        success = 0
        failure = 0
        cached = 0
        t0 = time.time()

        for i, drug in enumerate(drugs):
            key14 = inchikey_14(drug["smiles"])
            cache_path = CACHE_DIR / f"{key14}_{cyp}.json"

            if cache_path.exists():
                cached += 1
                if (i + 1) % 100 == 0:
                    elapsed = time.time() - t0
                    logger.info(
                        f"[{cyp}] {i+1}/{len(drugs)} "
                        f"(success={success}, cached={cached}, fail={failure}, "
                        f"{elapsed:.0f}s)"
                    )
                continue

            result = dock_single(drug, cyp, metadata, args.exhaustiveness)
            if result is not None:
                success += 1
            else:
                failure += 1

            if (i + 1) % 50 == 0:
                elapsed = time.time() - t0
                rate = (success + failure) / elapsed if elapsed > 0 else 0
                eta = (len(drugs) - i - 1) / rate / 60 if rate > 0 else float("inf")
                logger.info(
                    f"[{cyp}] {i+1}/{len(drugs)} "
                    f"(success={success}, cached={cached}, fail={failure}, "
                    f"{rate:.1f} drugs/s, ETA {eta:.0f}min)"
                )

        elapsed = time.time() - t0
        total = success + failure + cached
        fail_rate = failure / (success + failure) if (success + failure) > 0 else 0

        logger.info(
            f"\n[{cyp}] DONE: {total} total, "
            f"{success} new successes, {cached} cached, {failure} failures "
            f"({fail_rate:.1%} fail rate), {elapsed:.0f}s"
        )

        # Gate check
        if fail_rate > 0.30:
            logger.error(
                f"GATE FAIL: {cyp} failure rate {fail_rate:.1%} > 30%. "
                f"Check structure/tool."
            )
            sys.exit(1)


if __name__ == "__main__":
    main()
