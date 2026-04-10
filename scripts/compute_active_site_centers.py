"""Compute active site centers for all 5 CYP structures.

Active site center = heme Fe + 6Å in substrate access direction
(opposite to axial Cys ligand).
"""

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

STRUCTURES_DIR = Path(__file__).parent.parent / "data" / "docking" / "structures"

# Axial cysteine residue numbers for each CYP
# These are the conserved Cys that coordinates the heme Fe
CYP_AXIAL_CYS = {
    "CYP3A4": 442,   # 1TQN
    "CYP2D6": 443,   # 4WNT
    "CYP2C9": 435,   # 1OG5
    "CYP2C19": 435,  # 4GQS
    "CYP1A2": 458,   # 2HI4
}


def find_axial_cys_sg(pdb_path: Path, cys_resnum: int) -> np.ndarray | None:
    """Find SG atom of the axial cysteine."""
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            atom_name = line[12:16].strip()
            res_name = line[17:20].strip()
            res_seq = int(line[22:26])
            if res_name == "CYS" and atom_name == "SG" and res_seq == cys_resnum:
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                return np.array([x, y, z])
    return None


def compute_active_site_center(
    fe_coord: np.ndarray, cys_sg_coord: np.ndarray, offset_angstrom: float = 6.0
) -> np.ndarray:
    """Compute active site center = Fe + offset in substrate direction.

    Substrate direction is opposite to Cys-Fe axis.
    """
    cys_to_fe = fe_coord - cys_sg_coord
    substrate_dir = -cys_to_fe / np.linalg.norm(cys_to_fe)
    return fe_coord + substrate_dir * offset_angstrom


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    meta_path = STRUCTURES_DIR / "cyp_metadata.json"
    with open(meta_path) as f:
        metadata = json.load(f)

    for cyp_name, meta in metadata.items():
        heme_fe = meta["heme_fe"]
        if heme_fe is None:
            print(f"{cyp_name}: SKIP — no heme Fe")
            continue

        fe = np.array([heme_fe["x"], heme_fe["y"], heme_fe["z"]])
        pdb_id = meta["pdb_id"]
        raw_pdb = STRUCTURES_DIR / f"{pdb_id}_raw.pdb"

        cys_resnum = CYP_AXIAL_CYS[cyp_name]
        cys_sg = find_axial_cys_sg(raw_pdb, cys_resnum)

        if cys_sg is None:
            print(f"{cyp_name}: WARNING — Cys{cys_resnum} SG not found, using Fe as center")
            center = fe.tolist()
        else:
            center = compute_active_site_center(fe, cys_sg).tolist()
            fe_cys_dist = np.linalg.norm(fe - cys_sg)
            print(
                f"{cyp_name}: Fe=({fe[0]:.1f},{fe[1]:.1f},{fe[2]:.1f}), "
                f"Cys{cys_resnum} SG=({cys_sg[0]:.1f},{cys_sg[1]:.1f},{cys_sg[2]:.1f}), "
                f"Fe-SG={fe_cys_dist:.2f}Å, "
                f"center=({center[0]:.1f},{center[1]:.1f},{center[2]:.1f})"
            )

        meta["active_site_center"] = [round(c, 2) for c in center]

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nMetadata updated with active_site_center for all CYPs.")


if __name__ == "__main__":
    main()
