"""Tests for ParacellularAbsorptionFluxSpec — Renkin pore-sieving absorption.

The paracellular pathway is a second, parallel absorption route (lumen ->
gut_wall) representing pore-restricted, charge-selective permeation through
intestinal tight junctions.  It is the dominant absorption route for small
hydrophilic drugs (cimetidine, metformin, atenolol class) that the single
transcellular ``AbsorptionFluxSpec`` cannot represent.

All constants are external-physiology anchored (Avdeef 2010 PMID 20069445;
Renkin 1954 PMID 13211998; Linnankoski 2010 PMID 19827099; atenolol Peff
Dahlgren 2016 PMID 27504798; charge factor Adson 1994 / Tam-Velicky-Dryfe).
NONE were fit to holdout Cmax (Invariants #5/#8).
"""

import numpy as np
import pytest

from sisyphus.core import Distribution, DrugOnGraph
from sisyphus.engine.compiler import ResolvedParams
from sisyphus.engine.flux import (
    FLUX_REGISTRY,
    ParacellularAbsorptionFluxSpec,
    _hydrodynamic_radius_angstrom,
    _paracellular_charge_factor,
    _renkin_sieving,
)
from sisyphus.graph.body import BodyGraph
from sisyphus.graph.types import Node, ParacellularAbsorptionEdge

# Literature-anchored physiology (matches reference_man.yaml global_params).
_R_PORE = 5.6  # Angstrom (Linnankoski 2010 / Sugano 2002 / Kataoka 2011)
_P_SCALE = 62.8  # x10^-4 cm/s (atenolol-anchored with cation E=1.4)
_E_CATION = 1.4  # base/cation enhancement (Tam-Velicky-Dryfe / Adson 1994)
_E_ANION = 0.7  # acid/anion hindrance


def _make_drug(**overrides) -> DrugOnGraph:
    defaults = dict(
        name="test",
        smiles="C",
        dose_mg=100.0,
        route="oral",
        administration_node="lumen",
        mw=250.0,
        pka=None,
        compound_type="neutral",
        fup=Distribution(0.5),
        rbp=Distribution(1.0),
        kp_method="provided",
        kp_overrides={},
        peff=Distribution(1.0),
        solubility=Distribution(10.0),
        enzyme_affinity={},
        renal_clearance=Distribution(0.0),
        particle_radius_um=25.0,
        ps_overrides={},
    )
    defaults.update(overrides)
    return DrugOnGraph(**defaults)


def _graph(globals_on: bool = True, lumen="lumen", gut="gut") -> BodyGraph:
    g = BodyGraph()
    g.add_node(Node(name=lumen, node_type="lumen", volume=Distribution(0.25)))
    g.add_node(Node(name=gut, node_type="organ", volume=Distribution(1.0)))
    g.add_edge(
        ParacellularAbsorptionEdge(source=lumen, target=gut, ka_fraction=Distribution(1.0))
    )
    if globals_on:
        g.global_params["paracellular_pore_radius_angstrom"] = Distribution(_R_PORE)
        g.global_params["paracellular_permeability_scale"] = Distribution(_P_SCALE)
        g.global_params["paracellular_charge_factor_cation"] = Distribution(_E_CATION)
        g.global_params["paracellular_charge_factor_anion"] = Distribution(_E_ANION)
    return g


def _apply(drug: DrugOnGraph, globals_on: bool = True, lumen="lumen", gut="gut") -> np.ndarray:
    g = _graph(globals_on=globals_on, lumen=lumen, gut=gut)
    params = ResolvedParams(g, drug)
    state_index = {gut: 0, lumen: 1}
    spec = ParacellularAbsorptionFluxSpec.from_edge(0, g.edges[0], state_index)
    y = np.array([0.0, 100.0])
    dydt = np.zeros(2)
    spec.apply(0.0, y, dydt, params)
    return dydt  # [gut_gain, lumen_loss]


# ---------------------------------------------------------------------------
# Physics helpers
# ---------------------------------------------------------------------------


class TestRenkinSieving:
    def test_small_lambda_near_one(self):
        assert _renkin_sieving(0.01) == pytest.approx(1.0, abs=0.05)

    def test_monotonic_decreasing(self):
        vals = [_renkin_sieving(lam) for lam in (0.2, 0.4, 0.6, 0.8)]
        assert all(a > b for a, b in zip(vals, vals[1:]))

    def test_zero_at_or_above_pore(self):
        assert _renkin_sieving(1.0) == 0.0
        assert _renkin_sieving(1.2) == 0.0

    def test_never_negative(self):
        assert all(_renkin_sieving(lam) >= 0.0 for lam in np.linspace(0.01, 0.999, 200))


class TestHydrodynamicRadius:
    @pytest.mark.parametrize(
        "mw,expected_angstrom",
        [(151, 3.43), (252, 4.09), (266, 4.17), (455, 5.14), (559, 5.59)],
    )
    def test_matches_avdeef_worked_values(self, mw, expected_angstrom):
        # Avdeef 2010 refined Stokes-Einstein; values pre-derived independently.
        assert _hydrodynamic_radius_angstrom(mw) == pytest.approx(expected_angstrom, abs=0.05)

    def test_monotonic_with_mw(self):
        rs = [_hydrodynamic_radius_angstrom(mw) for mw in (100, 250, 450, 600)]
        assert all(a < b for a, b in zip(rs, rs[1:]))


class TestChargeFactor:
    def test_cation_enhanced(self):
        assert _paracellular_charge_factor("base", _E_CATION, _E_ANION) == _E_CATION

    def test_anion_hindered(self):
        assert _paracellular_charge_factor("acid", _E_CATION, _E_ANION) == _E_ANION

    def test_neutral_and_zwitterion_unity(self):
        assert _paracellular_charge_factor("neutral", _E_CATION, _E_ANION) == 1.0
        assert _paracellular_charge_factor("zwitterion", _E_CATION, _E_ANION) == 1.0


# ---------------------------------------------------------------------------
# Flux behavior
# ---------------------------------------------------------------------------


class TestParacellularFlux:
    def test_registered(self):
        assert "paracellular_absorption" in FLUX_REGISTRY
        assert FLUX_REGISTRY["paracellular_absorption"] is ParacellularAbsorptionFluxSpec

    def test_small_hydrophilic_is_absorbed(self):
        """Small drug (MW 252, cimetidine-like) gets a meaningful paracellular rate."""
        drug = _make_drug(mw=252.0, compound_type="base", particle_radius_um=25.0)
        dydt = _apply(drug)
        # peff_para = 62.8 * F_renkin(4.09/5.6) * 1.4 ;  ka = 2.88*peff_para/25
        assert dydt[0] > 0.0  # gut gains
        assert dydt[1] < 0.0  # lumen loses
        # bounded, physically-sized rate (h^-1 * 100 mg), not absurd
        assert 0.01 < dydt[0] < 50.0

    def test_mass_conservation(self):
        drug = _make_drug(mw=252.0, compound_type="base")
        dydt = _apply(drug)
        assert dydt[0] == pytest.approx(-dydt[1], abs=1e-12)

    def test_large_molecule_sieved_to_zero(self):
        """MW giving lambda >= 1 (r_HYD(650) ~ 5.95 A > pore 5.6 A) -> F=0 -> no flux."""
        drug = _make_drug(mw=650.0, compound_type="acid")
        dydt = _apply(drug)
        assert dydt[0] == 0.0 and dydt[1] == 0.0

    def test_near_pore_cutoff_is_negligible(self):
        """A molecule right at the pore edge (atorvastatin 559, lambda~0.998)
        gets a negligible — not necessarily exactly zero — paracellular rate."""
        big = _apply(_make_drug(mw=559.0, compound_type="acid"))
        small = _apply(_make_drug(mw=171.0, compound_type="neutral"))
        assert big[0] < 0.01 * small[0]

    def test_lipophilic_control_negligible_vs_small_target(self):
        """A large drug's paracellular rate is far smaller than a small drug's
        (the size gate auto-protects high-MW controls)."""
        small = _apply(_make_drug(mw=171.0, compound_type="neutral"))  # metronidazole
        big = _apply(_make_drug(mw=455.0, compound_type="base"))  # verapamil
        assert big[0] < 0.10 * small[0]

    def test_anion_hindered_vs_cation(self):
        """Same size: an acid (anion) absorbs less paracellularly than a base (cation)."""
        base = _apply(_make_drug(mw=200.0, compound_type="base"))
        acid = _apply(_make_drug(mw=200.0, compound_type="acid"))
        neutral = _apply(_make_drug(mw=200.0, compound_type="neutral"))
        assert base[0] > neutral[0] > acid[0]
        assert acid[0] == pytest.approx(base[0] * _E_ANION / _E_CATION, rel=1e-9)

    def test_independent_of_transcellular_peff(self):
        """Paracellular permeation does not read the drug's transcellular peff:
        a zero-peff drug still absorbs paracellularly."""
        d0 = _apply(_make_drug(mw=200.0, compound_type="base", peff=Distribution(0.0)))
        d1 = _apply(_make_drug(mw=200.0, compound_type="base", peff=Distribution(5.0)))
        assert d0[0] > 0.0
        assert d0[0] == pytest.approx(d1[0], rel=1e-12)

    def test_zero_ka_fraction_no_flux(self):
        g = _graph()
        # rebuild edge with ka_fraction 0
        g2 = BodyGraph()
        g2.add_node(Node(name="lumen", node_type="lumen", volume=Distribution(0.25)))
        g2.add_node(Node(name="gut", node_type="organ", volume=Distribution(1.0)))
        g2.add_edge(
            ParacellularAbsorptionEdge(source="lumen", target="gut", ka_fraction=Distribution(0.0))
        )
        for k, v in g.global_params.items():
            g2.global_params[k] = v
        params = ResolvedParams(g2, _make_drug(mw=200.0, compound_type="base"))
        spec = ParacellularAbsorptionFluxSpec.from_edge(0, g2.edges[0], {"gut": 0, "lumen": 1})
        dydt = np.zeros(2)
        spec.apply(0.0, np.array([0.0, 100.0]), dydt, params)
        assert dydt[0] == 0.0 and dydt[1] == 0.0

    def test_no_paracellular_when_globals_absent(self):
        """Graceful degradation: if pore physiology is not configured, the flux
        is inert (a graph without the global params behaves as before)."""
        drug = _make_drug(mw=200.0, compound_type="base")
        dydt = _apply(drug, globals_on=False)
        assert dydt[0] == 0.0 and dydt[1] == 0.0

    def test_identity_blind(self):
        """Invariant #1: renaming nodes must not change the numerical flux."""
        drug = _make_drug(mw=252.0, compound_type="base")
        a = _apply(drug, lumen="lumen", gut="gut")
        b = _apply(drug, lumen="xq7", gut="zz9")
        assert a[0] == pytest.approx(b[0], rel=1e-12)
