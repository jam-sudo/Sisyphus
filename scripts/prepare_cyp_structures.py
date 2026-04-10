"""Download and prepare CYP crystal structures for docking.

Downloads 5 CYP structures from PDB, removes co-crystallized ligands,
identifies heme Fe coordinates, and saves clean structures for Vina docking.
"""

import json
import logging
import urllib.request
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# CYP PDB IDs — well-established crystal structures
CYP_STRUCTURES = {
    "CYP3A4": "1TQN",
    "CYP2D6": "4WNT",
    "CYP2C9": "1OG5",
    "CYP2C19": "4GQS",
    "CYP1A2": "2HI4",
}

# Known co-crystallized ligand residue names to remove
# These are non-protein, non-heme heteroatoms
KEEP_HETATM = {"HEM", "HEC", "HOH"}  # Keep heme (HEM/HEC variants), optionally water

STRUCTURES_DIR = Path(__file__).parent.parent / "data" / "docking" / "structures"


def download_pdb(pdb_id: str, output_path: Path) -> str:
    """Download PDB file from RCSB."""
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    logger.info(f"Downloading {pdb_id} from {url}")
    urllib.request.urlretrieve(url, output_path)
    return str(output_path)


def find_heme_fe(lines: list[str]) -> dict | None:
    """Find heme iron (FE) coordinates from HETATM records."""
    for line in lines:
        if not line.startswith("HETATM"):
            continue
        atom_name = line[12:16].strip()
        res_name = line[17:20].strip()
        if res_name in ("HEM", "HEC") and atom_name == "FE":
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            chain = line[21].strip()
            res_seq = int(line[22:26])
            return {
                "atom_name": atom_name,
                "res_name": res_name,
                "chain": chain,
                "res_seq": res_seq,
                "x": x,
                "y": y,
                "z": z,
            }
    return None


def clean_pdb(input_path: Path, output_path: Path, cyp_name: str) -> dict:
    """Remove co-crystallized ligands, keep protein + heme.

    Returns metadata dict with heme Fe coordinates and removed ligands.
    """
    with open(input_path) as f:
        lines = f.readlines()

    # Find heme Fe first
    heme_fe = find_heme_fe(lines)
    if heme_fe is None:
        logger.warning(f"{cyp_name}: No heme FE found in structure")

    clean_lines = []
    removed_ligands = set()

    for line in lines:
        record = line[:6].strip()

        if record == "HETATM":
            res_name = line[17:20].strip()
            if res_name in KEEP_HETATM:
                clean_lines.append(line)
            else:
                removed_ligands.add(res_name)
                continue
        elif record in ("ATOM", "TER", "END", "MODEL", "ENDMDL"):
            clean_lines.append(line)
        elif record in ("CONECT",):
            # Keep CONECT records for heme
            atom_serial = int(line[6:11])
            clean_lines.append(line)
        elif record in ("REMARK", "HEADER", "TITLE", "CRYST1", "SCALE1",
                        "SCALE2", "SCALE3", "ORIGX1", "ORIGX2", "ORIGX3"):
            clean_lines.append(line)

    with open(output_path, "w") as f:
        f.writelines(clean_lines)

    metadata = {
        "cyp": cyp_name,
        "pdb_id": CYP_STRUCTURES[cyp_name],
        "heme_fe": heme_fe,
        "removed_ligands": sorted(removed_ligands),
        "n_atoms_clean": sum(1 for l in clean_lines if l[:6].strip() in ("ATOM", "HETATM")),
    }

    logger.info(
        f"{cyp_name} ({CYP_STRUCTURES[cyp_name]}): "
        f"removed {removed_ligands}, "
        f"heme Fe at ({heme_fe['x']:.1f}, {heme_fe['y']:.1f}, {heme_fe['z']:.1f})"
        if heme_fe else f"{cyp_name}: no heme Fe found"
    )

    return metadata


def prepare_receptor_pdbqt(clean_pdb_path: Path, pdbqt_path: Path) -> None:
    """Convert clean PDB to PDBQT for Vina.

    Keeps protein ATOM + heme organic atoms (C/N/O). Excludes FE (unsupported by Vina).
    Heme Fe coordinates stored in metadata for post-docking distance calculation.
    """
    # Valid AD types for Vina
    VALID_AD_ELEMENTS = {"C", "N", "O", "S", "H", "P", "F", "CL", "BR", "I"}

    lines = []
    with open(clean_pdb_path) as f:
        for line in f:
            record = line[:6].strip()
            if record in ("ATOM", "HETATM"):
                atom_name = line[12:16].strip()
                res_name = line[17:20].strip()
                element = line[76:78].strip() if len(line) > 76 else atom_name[0]

                # Skip metals (FE, ZN, etc.) — Vina doesn't support them
                if element.upper() in ("FE", "ZN", "MG", "MN", "CU", "CO", "NI"):
                    continue

                # Skip water
                if res_name == "HOH":
                    continue

                ad_type = _element_to_ad_type(element, atom_name, line)

                pdbqt_line = line[:54]
                pdbqt_line += line[54:60] if len(line) > 60 else "  1.00"
                pdbqt_line += line[60:66] if len(line) > 66 else "  0.00"
                pdbqt_line += f"    {0.000:+6.3f} {ad_type:<2s}\n"
                lines.append(pdbqt_line)
            elif record in ("TER", "END"):
                lines.append(line)

    with open(pdbqt_path, "w") as f:
        f.writelines(lines)


def _element_to_ad_type(element: str, atom_name: str, line: str) -> str:
    """Map element to AutoDock atom type."""
    res_name = line[17:20].strip()
    type_map = {
        "C": "C",
        "N": "NA" if "HIS" in res_name or "N" in atom_name else "N",
        "O": "OA",
        "S": "SA",
        "H": "HD",
        "FE": "FE",
        "ZN": "ZN",
        "MG": "MG",
        "MN": "MN",
        "CA": "CA" if res_name != "CA" else "C",
    }
    return type_map.get(element.upper(), element.upper())


def prepare_all():
    """Download and prepare all CYP structures."""
    STRUCTURES_DIR.mkdir(parents=True, exist_ok=True)

    all_metadata = {}

    for cyp_name, pdb_id in CYP_STRUCTURES.items():
        raw_path = STRUCTURES_DIR / f"{pdb_id}_raw.pdb"
        clean_path = STRUCTURES_DIR / f"{cyp_name}.pdb"
        pdbqt_path = STRUCTURES_DIR / f"{cyp_name}.pdbqt"

        # Download
        if not raw_path.exists():
            download_pdb(pdb_id, raw_path)
        else:
            logger.info(f"{pdb_id} already downloaded")

        # Clean
        metadata = clean_pdb(raw_path, clean_path, cyp_name)
        all_metadata[cyp_name] = metadata

        # Convert to PDBQT
        prepare_receptor_pdbqt(clean_path, pdbqt_path)
        logger.info(f"{cyp_name}: PDBQT written to {pdbqt_path}")

    # Save metadata
    meta_path = STRUCTURES_DIR / "cyp_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(all_metadata, f, indent=2)

    return all_metadata


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    metadata = prepare_all()

    print("\n=== CYP Structure Summary ===")
    for cyp, meta in metadata.items():
        fe = meta["heme_fe"]
        if fe:
            print(
                f"{cyp} ({meta['pdb_id']}): "
                f"heme Fe=({fe['x']:.1f}, {fe['y']:.1f}, {fe['z']:.1f}), "
                f"removed={meta['removed_ligands']}, "
                f"atoms={meta['n_atoms_clean']}"
            )
        else:
            print(f"{cyp} ({meta['pdb_id']}): NO HEME FE FOUND — check structure")
