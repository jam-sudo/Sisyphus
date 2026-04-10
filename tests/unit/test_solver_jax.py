"""Tests for JAX/Diffrax solver — Phase 0 of UDE roadmap.

Validates:
1. Numerical equivalence to SciPy LSODA on the wrapped path.
2. Autograd support through Diffrax Kvaerno5 on pure-JAX path.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

# Skip entire module if jax/diffrax not installed (optional dependency).
pytest.importorskip("jax")
pytest.importorskip("diffrax")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from sisyphus.engine.solver_jax import solve as solve_jax  # noqa: E402
from sisyphus.engine.solver_jax import solve_jax_pure  # noqa: E402


# ---------------------------------------------------------------------------
# Phase 0b: pure-JAX RHS + gradient verification on 2-compartment model
# ---------------------------------------------------------------------------


def _two_compartment_rhs(t, y, args):
    """2-compartment oral absorption: A_gut → A_central → sink."""
    ka, CL, V = args
    A_gut, A_cent = y
    return jnp.array([-ka * A_gut, ka * A_gut - (CL / V) * A_cent])


def _analytical_cmax(dose: float, ka: float, CL: float, V: float) -> float:
    """Closed-form Cmax for 1st-order oral absorption, 1-compartment elim."""
    k_el = CL / V
    tmax = float(np.log(ka / k_el) / (ka - k_el))
    return float(dose / V * ka / (ka - k_el) * (np.exp(-k_el * tmax) - np.exp(-ka * tmax)))


def _cmax_fn(args, dose: float):
    """Compute Cmax via Diffrax forward solve."""
    y0 = jnp.array([dose, 0.0])
    t_eval = jnp.linspace(0.01, 24.0, 241)
    _, ys, _ = solve_jax_pure(
        _two_compartment_rhs, y0, (0.0, 24.0), args=args, t_eval=t_eval,
        allow_jvp=True, stiff=False,
    )
    C_cent = ys[:, 1] / args[2]
    return jnp.max(C_cent)


class TestJAXSolverForward:
    """Forward solve accuracy against analytical solution."""

    def test_two_compartment_matches_analytical(self):
        dose, ka, CL, V = 10.0, 1.5, 5.0, 50.0
        cmax_analytical = _analytical_cmax(dose, ka, CL, V)
        cmax_diffrax = float(_cmax_fn((ka, CL, V), dose))
        rel_err = abs(cmax_diffrax - cmax_analytical) / cmax_analytical
        assert rel_err < 1e-3, f"rel_err={rel_err:.3e}, expected < 1e-3"

    def test_high_clearance_drug(self):
        """High CL drug: short t½, low Cmax."""
        dose, ka, CL, V = 100.0, 2.0, 100.0, 70.0
        cmax_an = _analytical_cmax(dose, ka, CL, V)
        cmax_dx = float(_cmax_fn((ka, CL, V), dose))
        assert abs(cmax_dx - cmax_an) / cmax_an < 1e-3

    def test_low_clearance_drug(self):
        """Low CL drug: long t½, plateau behavior."""
        dose, ka, CL, V = 50.0, 0.5, 2.0, 100.0
        cmax_an = _analytical_cmax(dose, ka, CL, V)
        cmax_dx = float(_cmax_fn((ka, CL, V), dose))
        assert abs(cmax_dx - cmax_an) / cmax_an < 1e-3


class TestJAXSolverGradients:
    """Verify jax.grad through Diffrax matches finite differences."""

    @staticmethod
    def _fd(f, x, eps=1e-5):
        return (f(x + eps) - f(x - eps)) / (2 * eps)

    def test_grad_clearance(self):
        dose, ka, CL, V = 10.0, 1.5, 5.0, 50.0
        g = jax.grad(lambda cl: _cmax_fn((ka, cl, V), dose))(CL)
        fd = self._fd(lambda cl: float(_cmax_fn((ka, float(cl), V), dose)), CL)
        assert abs(float(g) - fd) / abs(fd) < 1e-3, f"grad={float(g):.6f} fd={fd:.6f}"

    def test_grad_ka(self):
        dose, ka, CL, V = 10.0, 1.5, 5.0, 50.0
        g = jax.grad(lambda k: _cmax_fn((k, CL, V), dose))(ka)
        fd = self._fd(lambda k: float(_cmax_fn((float(k), CL, V), dose)), ka)
        assert abs(float(g) - fd) / abs(fd) < 1e-3

    def test_grad_volume(self):
        dose, ka, CL, V = 10.0, 1.5, 5.0, 50.0
        g = jax.grad(lambda v: _cmax_fn((ka, CL, v), dose))(V)
        fd = self._fd(lambda v: float(_cmax_fn((ka, CL, float(v)), dose)), V)
        assert abs(float(g) - fd) / abs(fd) < 1e-3


# ---------------------------------------------------------------------------
# Phase 0a: numerical equivalence between SciPy LSODA and Diffrax (wrapped)
# ---------------------------------------------------------------------------


class TestJAXSolverMatchesSciPy:
    """Validate solve_jax wrapped path matches existing SciPy-based solver."""

    @pytest.fixture(scope="class")
    def setup(self):
        import sisyphus.engine.flux  # noqa: F401 -- register flux specs
        from sisyphus.engine.compiler import ODECompiler
        from sisyphus.graph.builder import build_from_yaml

        physiology = Path(__file__).resolve().parent.parent.parent / "data" / "physiology" / "reference_man.yaml"
        graph = build_from_yaml(physiology)
        compiler = ODECompiler()
        compiled = compiler.compile(graph)
        return graph, compiled

    def _run_both(self, setup, smiles, dose, route="oral"):
        from sisyphus.engine.compiler import ResolvedParams
        from sisyphus.engine.solver import solve as solve_scipy
        from sisyphus.predict.adme import predict_adme
        from sisyphus.predict.chemistry import compute_profile
        from sisyphus.predict.ivive import build_drug_on_graph

        graph, compiled = setup
        profile = compute_profile(smiles)
        adme = predict_adme(profile)
        drug = build_drug_on_graph(profile, adme, dose, route)
        rng = np.random.default_rng(42)
        params = ResolvedParams(graph.sample(rng), drug.sample(rng))
        y0 = np.zeros(compiled.n_states)
        y0[compiled.state_index[drug.administration_node]] = dose
        sol_s = solve_scipy(compiled, params, y0, t_span=(0, 24))
        sol_j = solve_jax(compiled, params, y0, t_span=(0, 24))
        return sol_s, sol_j

    @pytest.mark.slow
    def test_midazolam_venous_blood(self, setup):
        smi = "Cc1ncc2n1-c1ccc(Cl)cc1C(c1ccccc1F)=NC2"
        sol_s, sol_j = self._run_both(setup, smi, 15.0)
        c_s = sol_s.concentrations["venous_blood"]
        c_j = sol_j.concentrations["venous_blood"]
        rel_err = np.abs(c_s - c_j).max() / max(c_s.max(), 1e-10)
        assert rel_err < 1e-3, f"venous_blood rel_err={rel_err:.3e}"
        assert sol_j.solver_success

    @pytest.mark.slow
    def test_warfarin_mass_balance(self, setup):
        smi = "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O"
        _, sol_j = self._run_both(setup, smi, 10.0)
        # Mass balance must be tight (same threshold as SciPy path)
        assert sol_j.mass_balance_error < 1e-4, f"MBE={sol_j.mass_balance_error:.3e}"
        assert sol_j.solver_success
