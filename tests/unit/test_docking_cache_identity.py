"""A docking pose for one stereoisomer must not be reused for another."""

import json

import numpy as np
import pytest
from rdkit import Chem

from scripts import train_docking_surrogate
from scripts.benchmark_clint_docking import get_docking_features
from scripts.train_clint_docking import load_docking_features
from sisyphus.validation.docking_cache import cache_path, load_matching_cache


def test_legacy_cache_requires_exact_isomer_and_full_key_separates_new_entries(tmp_path):
    unspecified = "CC(F)Cl"
    stereo = "C[C@H](F)Cl"
    opposite = "C[C@@H](F)Cl"
    cyp = "CYP3A4"
    legacy_key = Chem.MolToInchiKey(Chem.MolFromSmiles(unspecified))[:14]
    legacy = tmp_path / f"{legacy_key}_{cyp}.json"
    legacy.write_text(json.dumps({"smiles": unspecified, "docking_score": 1.0}))

    assert load_matching_cache(tmp_path, unspecified, cyp)["docking_score"] == 1.0
    assert load_matching_cache(tmp_path, stereo, cyp) is None
    assert np.isnan(get_docking_features(stereo, [cyp], tmp_path)).all()
    assert np.isnan(load_docking_features(stereo, tmp_path, [cyp])[f"{cyp}_docking_score"])

    specific = cache_path(tmp_path, stereo, cyp)
    assert specific != cache_path(tmp_path, opposite, cyp)
    specific.write_text(json.dumps({"smiles": stereo, "docking_score": 2.0}))
    assert load_matching_cache(tmp_path, stereo, cyp)["docking_score"] == 2.0
    assert get_docking_features(stereo, [cyp], tmp_path)[0] == 2.0
    assert load_docking_features(stereo, tmp_path, [cyp])[f"{cyp}_docking_score"] == 2.0
    assert load_matching_cache(tmp_path, opposite, cyp) is None
    assert load_matching_cache(tmp_path, "", cyp) is None
    with pytest.raises(ValueError, match="Invalid docking SMILES"):
        cache_path(tmp_path, "", cyp)


def test_surrogate_does_not_join_cross_isomer_cyp_poses(tmp_path, monkeypatch):
    unspecified = "CC(F)Cl"
    stereo = "C[C@H](F)Cl"
    first, second = "CYP3A4", "CYP2D6"
    legacy_key = Chem.MolToInchiKey(Chem.MolFromSmiles(unspecified))[:14]
    (tmp_path / f"{legacy_key}_{first}.json").write_text(
        json.dumps({"smiles": unspecified, "docking_score": 1.0})
    )
    for cyp in (first, second):
        cache_path(tmp_path, stereo, cyp).write_text(json.dumps({
            "smiles": stereo,
            "docking_score": 2.0,
            "heme_fe_distance_min": 3.0,
            "pose_centroid_fe_dist": 4.0,
            "n_heavy_atoms": 5,
        }))
    monkeypatch.setattr(train_docking_surrogate, "CACHE_DIFFDOCK", tmp_path)
    features, targets, smiles, _ = train_docking_surrogate.collect_training_data({
        "cyps": [first, second], "tool": "diffdock",
    })
    assert features.shape == (1, 2057)
    assert targets.shape == (1, 8)
    assert smiles == [stereo]


def test_same_inchikey_tautomers_have_distinct_paths(tmp_path):
    keto, enol = "O=C1NC=CC=C1", "OC1=NC=CC=C1"
    assert Chem.MolToInchiKey(Chem.MolFromSmiles(keto)) == Chem.MolToInchiKey(
        Chem.MolFromSmiles(enol)
    )
    assert cache_path(tmp_path, keto, "CYP3A4") != cache_path(tmp_path, enol, "CYP3A4")
