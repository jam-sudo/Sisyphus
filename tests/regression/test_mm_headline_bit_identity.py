"""v2.2a is headline-isolated: a drug with empty enzyme_km takes the verbatim linear flux
path, so its engine output is bit-identical to pre-v2.2a. Pins one drug's Cmax to the value
recorded on the linear-path code (Task 3 Step 1)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

import sisyphus.engine.flux  # noqa: F401
from sisyphus.core import Distribution, DrugOnGraph
from sisyphus.engine.compiler import ODECompiler, ResolvedParams
from sisyphus.engine.solver import solve
from sisyphus.graph.builder import build_from_yaml

_PINNED_CMAX = 1.8165541833005436  # <-- paste the Step-1 value (full repr precision)


def test_empty_enzyme_km_is_bit_identical():
    g = build_from_yaml(Path("data/physiology/reference_man.yaml")).realize_means()
    d = DrugOnGraph(
        name="caf", smiles="C", dose_mg=100.0, route="oral",
        administration_node="stomach_lumen", mw=194.0, pka=None, compound_type="base",
        fup=Distribution(0.65, 0), rbp=Distribution(1.0, 0), kp_method="rodgers_rowland",
        kp_overrides={}, peff=Distribution(5.0, 0), solubility=Distribution(1000.0, 0),
        enzyme_affinity={"CYP1A2": Distribution(1.0e-3, 0)},
        renal_clearance=Distribution(0.0, 0),
    ).realize_means()
    assert d.enzyme_km == {}  # empty ⇒ linear path
    c = ODECompiler().compile(g)
    p = ResolvedParams(g, d)
    y0 = np.zeros(c.n_states)
    y0[c.state_index["stomach_lumen"]] = d.dose_mg
    r = solve(c, p, y0, t_span=(0.0, 200.0))
    assert float(r.concentrations["venous_blood"].max()) == _PINNED_CMAX
