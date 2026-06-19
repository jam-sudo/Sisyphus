"""Gate tests for the multi-species engine-convection spike (B1.x). Pins the OBSERVED YES
outcome: the axial machinery composes with the active-metabolite species (spatially-resolved,
convected, mass-conserving), and the Damkohler map is monotone (convection matters at low Da,
local/post-processor-correct at high Da). Stack-independent assertions only."""
import importlib.util
import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


def _load():
    spec = importlib.util.spec_from_file_location(
        "ms_spike", _ROOT / "scripts" / "spike_multispecies_convection.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


spike = _load()
_R = spike.run()  # engine solves are slow; compute once, all tests read from this


def test_builds_and_solves():
    # The axial multi-species composition compiled + solved (the spike's core YES).
    assert _R["verdict"] == "YES"


def test_mass_conserved():
    assert _R["mass_balance_error"] < 1e-3            # engine global invariant
    formed, in_system = _R["formed"], _R["chain_end"] + _R["sink_end"]
    assert abs(formed - in_system) / max(formed, 1e-30) < 1e-2   # species-level balance


def test_low_da_shows_downstream_shift():
    # Da << 1: convection sweeps the metabolite toward the outlet (center-of-mass moves).
    assert _R["shift_low_da"] > 0.5


def test_high_da_agrees_with_local():
    # Da >> 1: reaction-dominated, profile stays local -> shift collapses toward 0,
    # validating the local-only post-processor as the high-Da limit.
    assert abs(_R["shift_high_da"]) < _R["shift_low_da"]
    assert abs(_R["shift_high_da"]) < 1.0


def test_da_map_monotone_decreasing():
    shifts = _R["da_shift_curve"]                     # ordered by increasing Da
    assert all(shifts[i] >= shifts[i + 1] - 1e-6 for i in range(len(shifts) - 1))


def test_headline_isolation_unchanged():
    p = _ROOT / "data" / "training" / "4track_holdout_predictions.json"
    d = json.loads(p.read_text())
    assert abs(d["overall"]["meta"]["aafe"] - 2.731) < 5e-3
