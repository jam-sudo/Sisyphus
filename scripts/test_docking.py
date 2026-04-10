"""Phase 0C: Test docking — midazolam × CYP3A4.

Gate: heme Fe-ligand minimum distance 3-6Å = pass, >10Å = fail.
"""

import json
import logging
import tempfile
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

logger = logging.getLogger(__name__)

STRUCTURES_DIR = Path(__file__).parent.parent / "data" / "docking" / "structures"

# Midazolam SMILES
MIDAZOLAM_SMILES = "C1CN=C(C2=CC=C(F)C=C2F)C2=CC3=CC=C(Cl)N=C3N12"


def smiles_to_pdbqt(smiles: str) -> str:
    """Convert SMILES to PDBQT string via RDKit 3D embedding + meeko."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    mol = Chem.AddHs(mol)
    result = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    if result != 0:
        # Retry with random coords
        result = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        if result != 0:
            raise ValueError(f"Could not embed molecule: {smiles}")

    AllChem.MMFFOptimizeMolecule(mol, maxIters=200)

    from meeko import MoleculePreparation, PDBQTWriterLegacy

    preparator = MoleculePreparation()
    mol_setups = preparator.prepare(mol)
    setup = mol_setups[0]

    pdbqt_string, is_ok, err = PDBQTWriterLegacy.write_string(setup)
    if not is_ok:
        raise ValueError(f"PDBQT conversion failed: {err}")

    return pdbqt_string


def run_vina_docking(
    receptor_pdbqt: str,
    ligand_pdbqt: str,
    center: tuple[float, float, float],
    box_size: tuple[float, float, float] = (25.0, 25.0, 25.0),
    exhaustiveness: int = 8,
    n_poses: int = 1,
) -> dict:
    """Run Vina docking and return results."""
    from vina import Vina

    v = Vina(sf_name="vina", verbosity=0)

    # Write receptor and ligand to temp files
    with tempfile.NamedTemporaryFile(suffix=".pdbqt", mode="w", delete=False) as f:
        f.write(ligand_pdbqt)
        ligand_path = f.name

    v.set_receptor(str(receptor_pdbqt))
    v.set_ligand_from_file(ligand_path)
    v.compute_vina_maps(center=list(center), box_size=list(box_size))

    v.dock(exhaustiveness=exhaustiveness, n_poses=n_poses)

    energies = v.energies()
    score = float(energies[0][0]) if energies is not None and len(energies) > 0 else None

    # Get docked pose coordinates
    poses_pdbqt = v.poses(n_poses=1)

    # Parse pose coordinates
    pose_coords = []
    pose_atoms = []
    for line in poses_pdbqt.split("\n"):
        if line.startswith("ATOM") or line.startswith("HETATM"):
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            atom_name = line[12:16].strip()
            element = line[77:79].strip() if len(line) > 78 else atom_name[0]
            pose_coords.append([x, y, z])
            pose_atoms.append({"name": atom_name, "element": element, "x": x, "y": y, "z": z})

    import os
    os.unlink(ligand_path)

    return {
        "docking_score": score,
        "pose_coords": np.array(pose_coords) if pose_coords else None,
        "pose_atoms": pose_atoms,
        "poses_pdbqt": poses_pdbqt,
    }


def compute_heme_fe_distance(
    pose_atoms: list[dict], heme_fe: dict
) -> dict:
    """Compute minimum distance between any ligand heavy atom and heme Fe."""
    fe_coord = np.array([heme_fe["x"], heme_fe["y"], heme_fe["z"]])

    min_dist = float("inf")
    min_atom = None
    heavy_atom_dists = []

    for atom in pose_atoms:
        # Skip hydrogens
        element = atom.get("element", atom["name"][0])
        if element.upper() in ("H", "HD"):
            continue

        coord = np.array([atom["x"], atom["y"], atom["z"]])
        dist = float(np.linalg.norm(coord - fe_coord))
        heavy_atom_dists.append(dist)

        if dist < min_dist:
            min_dist = dist
            min_atom = atom

    return {
        "heme_fe_distance_min": round(min_dist, 2),
        "heme_fe_distance_atom_name": min_atom["name"] if min_atom else None,
        "heme_fe_distance_atom_element": min_atom.get("element", "?") if min_atom else None,
        "all_heavy_atom_distances": [round(d, 2) for d in sorted(heavy_atom_dists)],
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Load CYP3A4 metadata
    meta_path = STRUCTURES_DIR / "cyp_metadata.json"
    with open(meta_path) as f:
        metadata = json.load(f)

    cyp3a4 = metadata["CYP3A4"]
    heme_fe = cyp3a4["heme_fe"]
    receptor_pdbqt = STRUCTURES_DIR / "CYP3A4.pdbqt"

    print(f"CYP3A4 heme Fe: ({heme_fe['x']:.1f}, {heme_fe['y']:.1f}, {heme_fe['z']:.1f})")
    print(f"Receptor: {receptor_pdbqt}")

    # Prepare midazolam
    print(f"\nPreparing midazolam: {MIDAZOLAM_SMILES}")
    ligand_pdbqt = smiles_to_pdbqt(MIDAZOLAM_SMILES)
    print(f"Ligand PDBQT: {len(ligand_pdbqt)} chars")

    # Docking box centered above heme Fe (substrate access channel)
    # CYP active site is ~6Å above heme plane, opposite to axial Cys ligand
    # For CYP3A4 1TQN: substrate direction from Cys442-Fe axis
    center = (-10.1, -24.5, -12.2)  # Active site center above heme
    print(f"\nDocking box center (above heme): {center}")
    print(f"Fe to center distance: {np.sqrt(sum((a-b)**2 for a,b in zip(center, (heme_fe['x'],heme_fe['y'],heme_fe['z'])))):.1f} Å")
    print("Running Vina docking (exhaustiveness=32)...")

    result = run_vina_docking(
        receptor_pdbqt=str(receptor_pdbqt),
        ligand_pdbqt=ligand_pdbqt,
        center=center,
        box_size=(25.0, 25.0, 25.0),
        exhaustiveness=32,
        n_poses=1,
    )

    print(f"\nDocking score: {result['docking_score']:.2f} kcal/mol")

    # Compute heme Fe distance
    fe_dist = compute_heme_fe_distance(result["pose_atoms"], heme_fe)
    min_dist = fe_dist["heme_fe_distance_min"]

    print(f"Heme Fe-ligand min distance: {min_dist:.2f} Å")
    print(f"Closest atom: {fe_dist['heme_fe_distance_atom_name']} "
          f"({fe_dist['heme_fe_distance_atom_element']})")
    print(f"All heavy atom distances (sorted): "
          f"{fe_dist['all_heavy_atom_distances'][:5]}...")

    # Gate check
    print("\n=== GATE CHECK ===")
    if 3.0 <= min_dist <= 6.0:
        print(f"PASS: heme Fe distance {min_dist:.2f} Å is in expected range [3-6 Å]")
        print("Midazolam is docked near heme iron — consistent with CYP3A4 substrate binding")
    elif min_dist <= 10.0:
        print(f"MARGINAL: heme Fe distance {min_dist:.2f} Å is slightly outside ideal range")
        print("Still reasonable — ligand is in the active site vicinity")
    else:
        print(f"FAIL: heme Fe distance {min_dist:.2f} Å > 10 Å")
        print("Ligand docked far from active site — check structure/box settings")

    return result, fe_dist


if __name__ == "__main__":
    main()
