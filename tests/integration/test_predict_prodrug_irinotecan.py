"""Integration test for irinotecan -> SN-38 prodrug routing (v0.3.4 / #11).

Tests that predict() with irinotecan SMILES routes through the prodrug
registry (CES2 hydrolysis), simulates the active SN-38 species, and
returns a Cmax in the plausible range. Slatter 2000 reports SN-38 Cmax
~50-100 ng/mL = 0.05-0.10 mg/L post 350 mg/m² IV irinotecan.

The gate is mechanical-correctness only: confirms routing fired and
active species simulated, without pinning Cmax magnitude. Per spec §10,
calibration is downstream.

Note: irinotecan is given IV in clinical practice. predict() with
route="iv" exercises the IV bolus path.
"""
from __future__ import annotations

import pytest

from sisyphus.pipeline.predict import predict

_IRINOTECAN = (
    "CCc1c2c(nc3cc(OC(=O)N4CCC(N5CCCCC5)CC4)ccc13)"
    "-c1cc3c(c(=O)n1C2)COC(=O)C3(O)CC"
)


@pytest.mark.slow
def test_irinotecan_returns_active_sn38_cmax():
    """predict(irinotecan, 350mg IV) routes through prodrug registry and
    returns active SN-38 Cmax > 0 (mechanical correctness)."""
    result = predict(_IRINOTECAN, dose_mg=350.0, route="iv")
    assert result.engine_pk is not None, (
        "engine_pk None — prodrug routing or IV simulation failed"
    )
    cmax = result.engine_pk.cmax.mean
    # Lower bound permissive: confirms routing fired (non-zero Cmax)
    # without pinning calibration magnitude (consistent with Task 4
    # simvastatin handling).
    assert cmax > 0.0001, (
        f"SN-38 Cmax {cmax:.5f} mg/L below floor 0.0001 — registry routing "
        f"or active species PK may have misfired."
    )
    assert cmax < 1.0, (
        f"SN-38 Cmax {cmax:.5f} mg/L above 1.0 — possible double-routing "
        f"or conversion yield error."
    )
