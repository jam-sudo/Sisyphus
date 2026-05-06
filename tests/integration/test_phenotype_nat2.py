"""Integration test for NAT2 phenotype propagation (issue #10).

Tests:
- parse_phenotype_spec("NAT2:PM") accepted
- apply_phenotype_to_graph(graph, {"NAT2": "PM"}) warns nothing
- predict(isoniazid, phenotypes={"NAT2": "PM"}) > predict(isoniazid).cmax * 1.3
- predict(metoprolol, phenotypes={"NAT2": "PM"}) ~= predict(metoprolol).cmax (no NAT2 affinity → silent zero)
"""
from __future__ import annotations

import logging

import pytest

from sisyphus.pipeline.predict import predict


_ISONIAZID = "NNC(=O)c1ccncc1"
_METOPROLOL = "COCCc1ccc(OCC(O)CNC(C)C)cc1"


def test_parse_nat2_pm():
    from sisyphus.predict.phenotype import parse_phenotype_spec
    out = parse_phenotype_spec("NAT2:PM")
    assert out == {"NAT2": "PM"}


def test_apply_nat2_pm_no_warning(caplog):
    from sisyphus.predict.phenotype import apply_phenotype_to_graph
    from sisyphus.graph.builder import build_from_yaml
    import pathlib

    caplog.set_level(logging.WARNING)
    g = build_from_yaml(pathlib.Path("data/physiology/reference_man.yaml"))
    _ = apply_phenotype_to_graph(g, {"NAT2": "PM"})
    warnings_about_nat2 = [r for r in caplog.records if "NAT2" in r.getMessage() and "not found" in r.getMessage()]
    assert not warnings_about_nat2, (
        f"apply_phenotype_to_graph emitted 'tag not found' for NAT2: {warnings_about_nat2}"
    )


@pytest.mark.slow
def test_isoniazid_nat2_pm_propagates():
    """NAT2:PM should drop isoniazid clearance, raising Cmax > 1.3× EM.

    Ellard 1976: slow-acetylator t1/2 ~3h vs rapid ~1h (AUC ratio 3-4×).
    Cmax effect smaller than AUC due to absorption-time saturation;
    gate at 1.3× is conservative.
    """
    em = predict(_ISONIAZID, dose_mg=300.0, phenotypes={"NAT2": "EM"})
    pm = predict(_ISONIAZID, dose_mg=300.0, phenotypes={"NAT2": "PM"})
    assert em.engine_pk is not None and pm.engine_pk is not None
    ratio = pm.engine_pk.cmax.mean / em.engine_pk.cmax.mean
    assert ratio > 1.3, (
        f"NAT2:PM/EM Cmax ratio {ratio:.3f} ≤ 1.3 — phenotype propagation gate."
    )


@pytest.mark.slow
def test_metoprolol_nat2_pm_silent_zero():
    """Non-NAT2 substrate must be invariant under NAT2 phenotype scaling.

    Metoprolol has no NAT2 affinity (registry miss → fm has no NAT2),
    so even though apply_phenotype_to_graph scales liver.NAT2 abundance
    by 0.10, the engine multiplies by zero affinity → silent zero
    (graph-blind invariant).
    """
    base = predict(_METOPROLOL, dose_mg=100.0)
    pm = predict(_METOPROLOL, dose_mg=100.0, phenotypes={"NAT2": "PM"})
    assert base.engine_pk is not None and pm.engine_pk is not None
    rel_err = abs(pm.engine_pk.cmax.mean - base.engine_pk.cmax.mean) / base.engine_pk.cmax.mean
    assert rel_err < 1e-6, (
        f"Metoprolol Cmax shifted under NAT2:PM (rel_err {rel_err:.2e}); "
        f"silent-zero invariant violated."
    )
