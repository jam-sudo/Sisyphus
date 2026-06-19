"""Unit tests for the hepatic_fu_correction registry loader (B-11)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from sisyphus.predict.hepatic_fu_correction import lookup_hepatic_fu_correction

_CLOPIDOGREL_STEREO = "COC(=O)[C@H](c1ccccc1Cl)N1CCc2sccc2C1"
_CLOPIDOGREL_NONSTEREO = "COC(=O)C(C1=CC=CC=C1Cl)N2CCC3=C(C2)C=CS3"
_MORPHINE = "CN1CCC23C4C1CC5=C2C(=C(C=C5)O)OC3C(C=C4)O"


def _write_registry(tmp_path: Path, overrides: list[dict]) -> Path:
    p = tmp_path / "hepatic_fu_correction.json"
    p.write_text(json.dumps({"overrides": overrides}))
    return p


def _valid_entry(**overrides) -> dict:
    base = {
        "drug": "clopidogrel",
        "smiles": _CLOPIDOGREL_STEREO,
        "inchikey": "GKTWGGQPFAXNFI-HNNXBMFYSA-N",
        "fu_correction_liver": {"mean": 8.5, "cv": 0.5},
        "disposition": "literature_applied",
        "literature": ["Watanabe 2009 DMD 37:1471 Table 1"],
        "notes": "test fixture",
        "n_candidates_reviewed": 3,
        "source_dbs_searched": ["PubMed"],
    }
    base.update(overrides)
    return base


def test_default_returns_one_for_unregistered(tmp_path):
    reg = _write_registry(tmp_path, [])
    out = lookup_hepatic_fu_correction(_MORPHINE, registry_path=reg)
    assert out.mean == pytest.approx(1.0)
    assert out.cv == pytest.approx(0.0)


def test_inchikey_full_match(tmp_path):
    reg = _write_registry(tmp_path, [_valid_entry()])
    out = lookup_hepatic_fu_correction(_CLOPIDOGREL_STEREO, registry_path=reg)
    assert out.mean == pytest.approx(8.5)
    assert out.cv == pytest.approx(0.5)


def test_inchikey_connectivity_fallback_for_stereo_variant(tmp_path):
    """Non-stereo query SMILES matches stereospecific registry via connectivity block."""
    reg = _write_registry(tmp_path, [_valid_entry()])
    out = lookup_hepatic_fu_correction(_CLOPIDOGREL_NONSTEREO, registry_path=reg)
    assert out.mean == pytest.approx(8.5)


def test_loader_rejects_value_below_one(tmp_path):
    """Anti-fudge guard: fu_correction_liver < 1.0 is not allowed (invariant #8)."""
    bad = _valid_entry(fu_correction_liver={"mean": 0.7, "cv": 0.1})
    reg = _write_registry(tmp_path, [bad])
    with pytest.raises(ValueError, match=r"fu_correction_liver.*>= 1\.0"):
        lookup_hepatic_fu_correction(_CLOPIDOGREL_STEREO, registry_path=reg)


def test_loader_rejects_missing_disposition(tmp_path):
    bad = _valid_entry()
    del bad["disposition"]
    reg = _write_registry(tmp_path, [bad])
    with pytest.raises(ValueError, match="disposition"):
        lookup_hepatic_fu_correction(_CLOPIDOGREL_STEREO, registry_path=reg)


def test_loader_rejects_unknown_disposition(tmp_path):
    bad = _valid_entry(disposition="fudge_applied")
    reg = _write_registry(tmp_path, [bad])
    with pytest.raises(ValueError, match="disposition"):
        lookup_hepatic_fu_correction(_CLOPIDOGREL_STEREO, registry_path=reg)


def test_connectivity_collision_returns_default(tmp_path):
    """If two registry entries share the same InChIKey connectivity block but
    different stereo blocks, the connectivity fallback must NOT silently pick
    one — it must return the default (no scaling) for safety."""
    # Construct two distinct full InChIKeys that share the connectivity block.
    # Real InChIKeys are 14char-10char-1char (e.g., GKTWGGQPFAXNFI-HNNXBMFYSA-N).
    entry_a = _valid_entry(
        drug="alpha",
        inchikey="GKTWGGQPFAXNFI-HNNXBMFYSA-N",
        fu_correction_liver={"mean": 5.0, "cv": 0.3},
    )
    entry_b = _valid_entry(
        drug="beta",
        inchikey="GKTWGGQPFAXNFI-DFRTYUOPQX-N",  # different stereo block
        fu_correction_liver={"mean": 9.0, "cv": 0.4},
    )
    reg = _write_registry(tmp_path, [entry_a, entry_b])

    # Verify the loader stores two entries under the same connectivity block.
    from sisyphus.predict.hepatic_fu_correction import _load_uncached
    full_index, conn_index = _load_uncached(reg)
    assert len(full_index) == 2
    assert "GKTWGGQPFAXNFI" in conn_index
    assert len(conn_index["GKTWGGQPFAXNFI"]) == 2
