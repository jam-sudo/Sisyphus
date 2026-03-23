"""Scaffold and temporal splits for train/holdout partitioning.

The holdout split is generated once (Murcko scaffold stratified,
seed=42) and frozen.  This module provides the splitting logic
for reproducibility and auditing.
"""

from __future__ import annotations


def scaffold_split(
    smiles_list: list[str],
    holdout_fraction: float = 0.25,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    """Murcko scaffold-stratified split.

    Groups molecules by Murcko scaffold, then assigns entire
    scaffold groups to train or holdout to avoid data leakage.

    Args:
        smiles_list: List of SMILES strings.
        holdout_fraction: Target fraction for holdout.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_indices, holdout_indices).
    """
    raise NotImplementedError
