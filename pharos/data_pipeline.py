"""Pharos Data Pipeline: SMILES → PyG molecular graphs + endpoint matrix.

Merges MMPK (Cmax, AUC, t½) + TDC (fup, CLint, Peff) datasets.
InChIKey dedup, holdout 107 exclusion, scaffold split.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent

# ═══════════════════════════════════════════════════════════════════════════
# RDKit → PyG graph conversion
# ═══════════════════════════════════════════════════════════════════════════

# Atom feature specification (78-dim)
ATOM_FEATURES = {
    "atomic_num": list(range(1, 119)),  # 118 elements → one-hot not feasible, use idx
    "degree": [0, 1, 2, 3, 4, 5],
    "formal_charge": [-2, -1, 0, 1, 2],
    "num_hs": [0, 1, 2, 3, 4],
    "hybridization": [0, 1, 2, 3, 4],  # SP, SP2, SP3, SP3D, SP3D2
}


def _one_hot(val, allowed):
    """One-hot encode a value against allowed list, with 'other' bin."""
    vec = [0] * (len(allowed) + 1)
    if val in allowed:
        vec[allowed.index(val)] = 1
    else:
        vec[-1] = 1
    return vec


def smiles_to_graph(smiles: str) -> Data | None:
    """Convert SMILES to PyG Data object with atom/bond features."""
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Atom features
    atom_feats = []
    for atom in mol.GetAtoms():
        feat = []
        feat += _one_hot(atom.GetAtomicNum(), [6, 7, 8, 9, 15, 16, 17, 35, 53])  # 10
        feat += _one_hot(atom.GetDegree(), [0, 1, 2, 3, 4, 5])  # 7
        feat += _one_hot(atom.GetFormalCharge(), [-2, -1, 0, 1, 2])  # 6
        feat += _one_hot(atom.GetNumExplicitHs(), [0, 1, 2, 3, 4])  # 6
        feat += _one_hot(int(atom.GetHybridization()), [1, 2, 3, 4, 5])  # 6
        feat.append(1.0 if atom.GetIsAromatic() else 0.0)  # 1
        feat.append(1.0 if atom.IsInRing() else 0.0)  # 1
        feat.append(atom.GetMass() / 200.0)  # 1 (normalized)
        atom_feats.append(feat)

    if len(atom_feats) == 0:
        return None

    x = torch.tensor(atom_feats, dtype=torch.float)
    node_dim = x.shape[1]  # should be 38

    # Edge index (undirected)
    edge_indices = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_indices.append([i, j])
        edge_indices.append([j, i])

    if len(edge_indices) == 0:
        # Single atom molecule
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index)


def smiles_to_morgan(smiles: str, nbits: int = 2048, radius: int = 2) -> np.ndarray | None:
    """Convert SMILES to Morgan fingerprint vector."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    arr = np.zeros(nbits, dtype=np.float32)
    for bit in fp.GetOnBits():
        arr[bit] = 1.0
    return arr


# ═══════════════════════════════════════════════════════════════════════════
# Data loading + merging
# ═══════════════════════════════════════════════════════════════════════════

def _canonical_smiles(smiles: str) -> str | None:
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol, isomericSmiles=True) if mol else None


def _inchikey(smiles: str) -> str | None:
    from rdkit import Chem
    from rdkit.Chem.inchi import MolToInchi, InchiToInchiKey
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None
    inchi = MolToInchi(mol)
    if not inchi:
        return None
    ik = InchiToInchiKey(inchi)
    return ik[:14] if ik else None


def load_holdout_keys() -> set[str]:
    """Load holdout InChIKey-14 set for exclusion."""
    with open(ROOT / "data" / "reference" / "holdout.json") as f:
        hd = json.load(f)
    with open(ROOT / "data" / "reference" / "clinical_pk.json") as f:
        cp = json.load(f)

    all_names = hd.get("holdout", []) + hd.get("train", [])
    drugs = cp.get("drugs", {})
    keys = set()
    for name in all_names:
        entry = drugs.get(name) or drugs.get(name.replace(" ", "_"))
        if not entry:
            continue
        smi = entry.get("smiles", "")
        if smi:
            ik = _inchikey(smi)
            if ik:
                keys.add(ik)
    return keys


def load_all_data() -> dict:
    """Load and merge all data sources. Returns dict of DataFrames."""
    holdout_ik = load_holdout_keys()
    log.info("Holdout InChIKeys: %d", len(holdout_ik))

    # --- MMPK: Cmax, AUC, t½ (oral) ---
    mmpk = pd.read_csv(ROOT / "data" / "training" / "mmpk_expanded_full.csv")
    # Use only non-holdout entries
    mmpk = mmpk[~mmpk["in_holdout"]].reset_index(drop=True)
    # Per-drug: take the entry with median Cmax (most representative dose)
    mmpk_dedup = mmpk.groupby("canon_smiles").agg(
        dose_mg=("dose_mg", "median"),
        cmax=("cmax_mg_L", "median"),
        auc_raw=("auc_ng_h_ml", "median"),  # ng·h/mL
        thalf=("thalf_h", "median"),
    ).reset_index()
    mmpk_dedup["auc"] = mmpk_dedup["auc_raw"] / 1000.0  # ng·h/mL → µg·h/mL = mg·h/L
    mmpk_dedup["ik"] = mmpk_dedup["canon_smiles"].apply(_inchikey)
    mmpk_dedup = mmpk_dedup.dropna(subset=["ik"])
    mmpk_dedup = mmpk_dedup[~mmpk_dedup["ik"].isin(holdout_ik)]
    log.info("MMPK (non-holdout): %d drugs", len(mmpk_dedup))

    # --- TDC: fup (PPBR_AZ) ---
    ppbr = pd.read_csv(ROOT / "data" / "ppbr_az.tab", sep="\t")
    # Filter for human data only
    if "Species" in ppbr.columns:
        ppbr = ppbr[ppbr["Species"] == "Homo sapiens"].reset_index(drop=True)
    ppbr = ppbr[["Drug_ID", "Drug", "Y"]]
    ppbr.columns = ["ID", "SMILES", "Y"]
    ppbr["canon"] = ppbr["SMILES"].apply(_canonical_smiles)
    ppbr = ppbr.dropna(subset=["canon"])
    ppbr_dedup = ppbr.groupby("canon").agg(fup=("Y", "mean")).reset_index()
    ppbr_dedup["fup"] = ppbr_dedup["fup"] / 100.0  # PPBR is % → fraction
    ppbr_dedup["fup"] = ppbr_dedup["fup"].clip(0.001, 1.0)
    ppbr_dedup["ik"] = ppbr_dedup["canon"].apply(_inchikey)
    ppbr_dedup = ppbr_dedup.dropna(subset=["ik"])
    ppbr_dedup = ppbr_dedup[~ppbr_dedup["ik"].isin(holdout_ik)]
    log.info("TDC PPBR (fup): %d drugs", len(ppbr_dedup))

    # --- TDC: CLint (Hepatocyte_AZ) ---
    clint = pd.read_csv(ROOT / "data" / "training" / "clearance_hepatocyte_az.tab", sep="\t")
    clint.columns = ["ID", "SMILES", "Y"]
    clint["canon"] = clint["SMILES"].apply(_canonical_smiles)
    clint = clint.dropna(subset=["canon"])
    clint_dedup = clint.groupby("canon").agg(clint=("Y", "mean")).reset_index()
    clint_dedup["ik"] = clint_dedup["canon"].apply(_inchikey)
    clint_dedup = clint_dedup.dropna(subset=["ik"])
    clint_dedup = clint_dedup[~clint_dedup["ik"].isin(holdout_ik)]
    log.info("TDC CLint: %d drugs", len(clint_dedup))

    # --- TDC: Peff (Caco2_Wang) ---
    caco2 = pd.read_csv(ROOT / "data" / "caco2_wang.tab", sep="\t")
    caco2.columns = ["ID", "SMILES", "Y"]
    caco2["canon"] = caco2["SMILES"].apply(_canonical_smiles)
    caco2 = caco2.dropna(subset=["canon"])
    caco2_dedup = caco2.groupby("canon").agg(peff=("Y", "mean")).reset_index()
    caco2_dedup["ik"] = caco2_dedup["canon"].apply(_inchikey)
    caco2_dedup = caco2_dedup.dropna(subset=["ik"])
    caco2_dedup = caco2_dedup[~caco2_dedup["ik"].isin(holdout_ik)]
    log.info("TDC Caco2 (Peff): %d drugs", len(caco2_dedup))

    # --- Merge all by InChIKey-14 ---
    # Start with union of all compounds
    all_iks = set()
    all_smiles = {}  # ik → canonical SMILES

    for df, smi_col in [
        (mmpk_dedup, "canon_smiles"),
        (ppbr_dedup, "canon"),
        (clint_dedup, "canon"),
        (caco2_dedup, "canon"),
    ]:
        for _, row in df.iterrows():
            ik = row["ik"]
            all_iks.add(ik)
            if ik not in all_smiles:
                all_smiles[ik] = row[smi_col]

    log.info("Total unique compounds (InChIKey): %d", len(all_iks))

    # Build merged dataframe using dict lookup (drop_duplicates on ik first)
    mmpk_dict = mmpk_dedup.drop_duplicates(subset="ik").set_index("ik").to_dict("index")
    ppbr_dict = ppbr_dedup.drop_duplicates(subset="ik").set_index("ik").to_dict("index")
    clint_dict = clint_dedup.drop_duplicates(subset="ik").set_index("ik").to_dict("index")
    caco2_dict = caco2_dedup.drop_duplicates(subset="ik").set_index("ik").to_dict("index")

    records = []
    for ik in all_iks:
        rec = {"ik": ik, "smiles": all_smiles[ik]}

        # MMPK endpoints
        if ik in mmpk_dict:
            row = mmpk_dict[ik]
            rec["dose_mg"] = float(row["dose_mg"])
            rec["cmax"] = float(row["cmax"])
            rec["auc"] = float(row["auc"]) if pd.notna(row.get("auc")) else None
            rec["thalf"] = float(row["thalf"]) if pd.notna(row.get("thalf")) else None
        else:
            rec["dose_mg"] = None
            rec["cmax"] = None
            rec["auc"] = None
            rec["thalf"] = None

        # Auxiliary
        rec["fup"] = float(ppbr_dict[ik]["fup"]) if ik in ppbr_dict else None
        rec["clint"] = float(clint_dict[ik]["clint"]) if ik in clint_dict else None
        rec["peff"] = float(caco2_dict[ik]["peff"]) if ik in caco2_dict else None

        records.append(rec)

    merged = pd.DataFrame(records)

    # Report overlap
    has_cmax = merged["cmax"].notna().sum()
    has_auc = merged["auc"].notna().sum()
    has_thalf = merged["thalf"].notna().sum()
    has_fup = merged["fup"].notna().sum()
    has_clint = merged["clint"].notna().sum()
    has_peff = merged["peff"].notna().sum()

    log.info("Merged dataset: %d compounds", len(merged))
    log.info("  Cmax: %d  AUC: %d  t½: %d", has_cmax, has_auc, has_thalf)
    log.info("  fup: %d  CLint: %d  Peff: %d", has_fup, has_clint, has_peff)

    # Overlap: compounds with both Cmax AND at least one auxiliary
    cmax_set = set(merged[merged["cmax"].notna()]["ik"])
    aux_set = set(merged[(merged["fup"].notna()) | (merged["clint"].notna()) | (merged["peff"].notna())]["ik"])
    overlap = len(cmax_set & aux_set)
    log.info("  Cmax ∩ auxiliary: %d (%.1f%% of Cmax)", overlap, overlap / max(has_cmax, 1) * 100)

    return merged


# ═══════════════════════════════════════════════════════════════════════════
# Scaffold split
# ═══════════════════════════════════════════════════════════════════════════

def scaffold_split(smiles_list: list[str], val_frac: float = 0.2, seed: int = 42):
    """Scaffold-based train/val split. Returns (train_idx, val_idx)."""
    from rdkit import Chem
    from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles

    scaffold_to_idx: dict[str, list[int]] = {}
    for i, smi in enumerate(smiles_list):
        try:
            mol = Chem.MolFromSmiles(smi)
            sc = MurckoScaffoldSmiles(mol=mol, includeChirality=False) if mol else ""
        except Exception:
            sc = ""
        scaffold_to_idx.setdefault(sc, []).append(i)

    rng = np.random.default_rng(seed)
    scaffolds = list(scaffold_to_idx.keys())
    rng.shuffle(scaffolds)

    val_target = int(len(smiles_list) * val_frac)
    train_idx, val_idx = [], []
    val_count = 0

    for sc in scaffolds:
        indices = scaffold_to_idx[sc]
        if val_count < val_target:
            val_idx.extend(indices)
            val_count += len(indices)
        else:
            train_idx.extend(indices)

    return train_idx, val_idx


# ═══════════════════════════════════════════════════════════════════════════
# Dataset class
# ═══════════════════════════════════════════════════════════════════════════

class PharosDataset:
    """Preprocessed dataset with graphs, fingerprints, and targets."""

    def __init__(self, merged_df: pd.DataFrame, default_dose: float = 100.0,
                 default_bw: float = 70.0):
        self.records = []
        n_fail = 0

        for _, row in merged_df.iterrows():
            smi = row["smiles"]
            graph = smiles_to_graph(smi)
            if graph is None:
                n_fail += 1
                continue

            fp = smiles_to_morgan(smi)
            if fp is None:
                n_fail += 1
                continue

            dose = float(row["dose_mg"]) if pd.notna(row.get("dose_mg")) else default_dose
            bw = default_bw

            targets = {}
            masks = {}
            for key in ["cmax", "auc", "thalf", "fup", "clint", "peff"]:
                val = row.get(key)
                if pd.notna(val) and val is not None and float(val) > 0:
                    targets[key] = float(val)
                    masks[key] = True
                else:
                    targets[key] = 0.0
                    masks[key] = False

            self.records.append({
                "smiles": smi,
                "graph": graph,
                "fp": torch.tensor(fp, dtype=torch.float),
                "dose": dose,
                "bw": bw,
                "targets": targets,
                "masks": masks,
            })

        log.info("PharosDataset: %d compounds loaded (%d failed)", len(self.records), n_fail)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]
