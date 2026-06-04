# tests/integration/test_engine_validation.py
"""Integration test: validate full pipeline against reference ODE Cmax.

Runs 4 validation drugs through the complete pipeline:
    YAML -> BodyGraph -> compile -> solve -> Cmax
and compares against reference deterministic ODE Cmax within +/-5%.

caffeine/warfarin are LOW-extraction and pinned to Omega's predecessor values.
midazolam/propranolol are HIGH-extraction: the FLUX-1 fix (2026-06-03) corrected
a flow-limitation double-count that Omega shared, so their targets are now
FLUX-1-corrected Sisyphus snapshots, not independent Omega values (see
OMEGA_TARGETS comment). The test thus mixes 2 cross-engine parity checks with 2
self-consistency snapshots; all 4 still serve as drift regressions.
"""

from pathlib import Path

import numpy as np
import pytest

import sisyphus.engine.flux  # noqa: F401 -- register flux specs
from sisyphus.compounds import load_compound
from sisyphus.engine.compiler import ODECompiler, ResolvedParams
from sisyphus.engine.solver import solve
from sisyphus.graph.builder import build_from_yaml
from sisyphus.pk.endpoints import compute_endpoints

# Parity targets — drift-regression snapshots. caffeine (RBP=1.0) is the only
# RBP-invariant case and stays at Omega's deterministic ODE value. The other
# three carry RBP != 1 in their curated configs (midazolam 0.66, warfarin 0.58,
# propranolol 0.81), so two correctness fixes legitimately moved them off Omega:
#   FLUX-1 (2026-06-03): flow-limitation double-count (E capped at 0.5).
#   RBP-2  (2026-06-04): blood:plasma concentration-basis — convective flow now
#     carries whole-blood A/V (no RBP at blood pools) and the metabolic/renal
#     sink drives off unbound PLASMA, so the realized hepatic extraction is the
#     canonical fu_b form E = fup*CLint/(Q*RBP + fup*CLint), fu_b = fup/RBP.
#     For RBP < 1 this raises first-pass extraction -> lower Cmax (verified
#     direction; caffeine RBP=1.0 is a bit-identical no-op, the surgical witness).
# Targets below are the post-RBP-2 macOS-stack snapshots; the +/-5% gate absorbs
# the documented macOS<->CI stack drift (caffeine/warfarin goldens already
# carried ~1.3%/1.1% such drift pre-RBP-2). A CI regen would tighten them.
# Specs: 2026-06-03-flux1-extraction-double-count-design.md,
#        2026-06-04-rbp-concentration-basis-design.md.
OMEGA_TARGETS = {
    "midazolam": {"cmax": 0.002800, "tmax": 1.5},   # RBP-2: was 0.005909 (FLUX-1)
    "caffeine": {"cmax": 1.7139, "tmax": 1.0},       # RBP=1.0 -> unchanged
    "warfarin": {"cmax": 0.343133, "tmax": 3.0},     # RBP-2: was 0.4922
    "propranolol": {"cmax": 0.059875, "tmax": 1.5},  # RBP-2: was 0.082528 (FLUX-1)
}


def run_drug(drug_name: str):
    """Full pipeline: YAML -> BodyGraph -> compile -> solve -> PKEndpoints."""
    graph = build_from_yaml(Path("data/physiology/reference_man.yaml"))
    drug = load_compound(Path(f"data/compounds/{drug_name}.yaml"))

    compiler = ODECompiler()
    compiled = compiler.compile(graph)

    # Deterministic mean-only realization (Hardening 2026-05-01).
    # Replaced graph.sample(rng=42) with realize_means() so this test is
    # truly deterministic — adding a new Distribution to physiology YAML
    # does not shift realized values for unrelated drugs.
    realized_graph = graph.realize_means()
    realized_drug = drug.realize_means()
    params = ResolvedParams(realized_graph, realized_drug)

    # Initial conditions: dose in administration node
    y0 = np.zeros(compiled.n_states)
    admin_idx = compiled.state_index[drug.administration_node]
    y0[admin_idx] = drug.dose_mg

    # Solve for 24 hours with fine time resolution
    t_eval = np.linspace(0, 24, 2000)
    result = solve(compiled, params, y0, t_span=(0, 24), t_eval=t_eval)
    pk = compute_endpoints(result, observation_node="venous_blood")
    return pk, result


_CMAX_DRUGS = [
    "midazolam",
    "caffeine",
    "warfarin",
    "propranolol",  # was xfail (~16% drift) — passes post-Hardening (RNG artifact resolved)
]


class TestEngineValidation:
    @pytest.mark.parametrize("drug", _CMAX_DRUGS)
    def test_cmax_within_5pct(self, drug):
        pk, _ = run_drug(drug)
        target = OMEGA_TARGETS[drug]["cmax"]
        actual = pk.cmax.mean
        rel_error = abs(actual - target) / target
        print(f"\n{drug}: Cmax={actual:.6f}, target={target:.6f}, error={rel_error:.1%}")
        assert rel_error < 0.05, (
            f"{drug}: Cmax={actual:.6f}, target={target:.6f}, error={rel_error:.1%}"
        )

    @pytest.mark.parametrize("drug", OMEGA_TARGETS.keys())
    def test_mass_balance(self, drug):
        _, result = run_drug(drug)
        assert result.mass_balance_error < 1e-6, (
            f"{drug}: mass balance error = {result.mass_balance_error:.2e}"
        )

    @pytest.mark.parametrize("drug", OMEGA_TARGETS.keys())
    def test_solver_success(self, drug):
        _, result = run_drug(drug)
        assert result.solver_success, f"{drug}: solver failed"
