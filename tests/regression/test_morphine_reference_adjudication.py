"""Pin the source-adjudicated oral morphine development reference."""

import json
from pathlib import Path

import pytest

from scripts.evaluate_tdm_ci_coverage import DRUGS as CI_DRUGS
from scripts.run_tdm_benchmark import DRUGS as TDM_DRUGS
from scripts.verify_tdm_ci_floor import CASES as FLOOR_CASES
from sisyphus.validation.reference import load_reference

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "data/reference/clinical_pk.json"


def test_morphine_reference_is_fasted_single_dose_oral_ir():
    drug = json.loads(REFERENCE.read_text())["drugs"]["morphine"]
    assert "PMC9705466" in drug["source"]
    assert "Sevredol" in drug["source"]
    assert "ct_curve" not in drug
    assert drug["pk_params"]["cmax_mg_L"] == pytest.approx(28.5 / 1000)
    assert drug["dose_mg"] == pytest.approx(30 * 0.75)

    reference = next(row for row in load_reference(REFERENCE) if row.name == "morphine")
    assert reference.in_holdout
    assert reference.route == "oral"
    assert reference.cmax_obs == pytest.approx(0.0285)

    for case in (TDM_DRUGS["morphine"], CI_DRUGS["morphine"]):
        assert case["dose_mg"] == reference.dose_mg
        assert case["cmax_obs"] == reference.cmax_obs
    assert [(dose, cmax) for name, _, dose, cmax, *_ in FLOOR_CASES if name == "morphine"] == [
        (reference.dose_mg, reference.cmax_obs)
    ]
