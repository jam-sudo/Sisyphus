from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from sisyphus.core import Distribution, DrugOnGraph
from sisyphus.engine.compiler import ResolvedParams
from sisyphus.graph.builder import build_from_yaml


def _drug(enzyme_km=None):
    return DrugOnGraph(
        name="t", smiles="C", dose_mg=100.0, route="oral",
        administration_node="stomach_lumen", mw=300.0, pka=None, compound_type="neutral",
        fup=Distribution(0.1, 0.0), rbp=Distribution(1.0, 0.0),
        kp_method="rodgers_rowland", kp_overrides={},
        peff=Distribution(5.0, 0.0), solubility=Distribution(1000.0, 0.0),
        enzyme_affinity={"CYP2D6": Distribution(0.5, 0.0)},
        renal_clearance=Distribution(0.0, 0.0),
        enzyme_km=enzyme_km or {},
    )


def test_enzyme_km_defaults_empty():
    assert _drug().enzyme_km == {}


def test_enzyme_km_rejects_nonpositive():
    with pytest.raises(ValueError):
        _drug({"CYP2D6": Distribution(0.0, 0.0)})


def test_realize_means_preserves_enzyme_km():
    d = _drug({"CYP2D6": Distribution(2.5, 0.0)}).realize_means()
    assert d.enzyme_km["CYP2D6"].mean == pytest.approx(2.5)


def test_sample_preserves_enzyme_km():
    d = _drug({"CYP2D6": Distribution(2.5, 0.0)}).sample(np.random.default_rng(0))
    assert d.enzyme_km["CYP2D6"].mean == pytest.approx(2.5)


def test_accessors_present_and_absent():
    g = build_from_yaml(Path("data/physiology/reference_man.yaml")).realize_means()
    rp = ResolvedParams(g, _drug({"CYP2D6": Distribution(2.5, 0.0)}).realize_means())
    assert rp.drug_enzyme_km("CYP2D6") == pytest.approx(2.5)
    assert math.isinf(rp.drug_enzyme_km("ABSENT"))
    assert rp.drug_has_enzyme_km() is True
    rp_empty = ResolvedParams(g, _drug().realize_means())
    assert rp_empty.drug_has_enzyme_km() is False
    assert math.isinf(rp_empty.drug_enzyme_km("CYP2D6"))
