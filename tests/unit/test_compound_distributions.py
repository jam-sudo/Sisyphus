"""Curated compound YAML distribution compatibility."""

from pathlib import Path

import yaml

from sisyphus.compounds import load_compound

_ROOT = Path(__file__).resolve().parents[2]


def test_curated_compound_scalar_and_distribution(tmp_path):
    source = yaml.safe_load((_ROOT / "data/compounds/caffeine.yaml").read_text())
    source["fup"] = {"mean": 0.7, "cv": 0.2}
    source["kp_overrides"]["liver"] = {"mean": 2.0, "cv": 0.1}
    path = tmp_path / "drug.yaml"
    path.write_text(yaml.safe_dump(source))
    drug = load_compound(path)
    assert (drug.fup.mean, drug.fup.cv) == (0.7, 0.2)
    assert (drug.kp_overrides["liver"].mean, drug.kp_overrides["liver"].cv) == (2.0, 0.1)
    assert drug.rbp.cv == 0.0
