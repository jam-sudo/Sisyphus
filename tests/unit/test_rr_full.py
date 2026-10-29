"""Published Rodgers–Rowland tissue Kp examples and input sensitivity."""

import pytest

from sisyphus.core import Distribution
from sisyphus.predict.rr_full import _RR_ROWS, compute_rr_kp


@pytest.mark.parametrize(
    ("drug", "logp", "pka", "kind", "fup", "rbp", "tissue", "published_kp"),
    [
        ("griseofulvin", 2.0, None, "neutral", .16, .776, "brain", .8),
        ("griseofulvin", 2.0, None, "neutral", .16, .776, "gut_wall", .9),
        ("itraconazole", 4.0, 4.57, "base", .002, .665, "liver", .5),
        ("posaconazole", 4.0, 4.67, "base", .02, .668, "brain", 7.9),
        ("posaconazole", 4.0, 4.67, "base", .02, .668, "gut_wall", 8.3),
    ],
)
def test_named_drug_examples_match_published_kp(
    drug, logp, pka, kind, fup, rbp, tissue, published_kp
):
    # Coutinho et al., Pharm Res 2024, supplementary Tables S3–S6, logP grid.
    # https://doi.org/10.1007/s11095-024-03703-4 (Kp rounded to 0.1).
    kp = compute_rr_kp(logp, pka, kind, _RR_ROWS[tissue], fup, rbp, .45)
    assert kp == pytest.approx(published_kp, abs=.051), drug


def test_acebutolol_blood_cell_affinity_uses_rbp_and_hematocrit():
    # Acebutolol input row: Harvard thesis Table 4, drawn from Rodgers 2005.
    tissue = _RR_ROWS["lung"]
    low = compute_rr_kp(1.87, 9.7, "base", tissue, .79, 1.0, .45)
    measured = compute_rr_kp(1.87, 9.7, "base", tissue, .79, 1.09, .45)
    lower_h = compute_rr_kp(1.87, 9.7, "base", tissue, .79, 1.09, .40)
    assert measured > low
    assert lower_h > measured


def test_zwitterion_uses_both_pka_values():
    tissue = _RR_ROWS["kidney"]
    a = compute_rr_kp(1.0, 9.5, "zwitterion", tissue, .1, 1.0, .45, 4.0)
    b = compute_rr_kp(1.0, 9.5, "zwitterion", tissue, .1, 1.0, .45, 6.0)
    assert a != b


def test_etodolac_acid_uses_albumin_and_ionization():
    # Etodolac logP, pKa and fup: Harvard thesis Table 4 (Rodgers 2006 inputs).
    tissue = _RR_ROWS["liver"]
    etodolac = compute_rr_kp(3.6, 4.7, "acid", tissue, .0108, 1.0, .45)
    changed_pka = compute_rr_kp(3.6, 6.0, "acid", tissue, .0108, 1.0, .45)
    assert etodolac == pytest.approx(.0910, abs=.0001)
    assert etodolac != changed_pka


def test_compositions_cover_all_modeled_tissues_as_distributions():
    from sisyphus.predict.ivive import _TISSUE_COMPOSITIONS

    assert set(_RR_ROWS) == set(_TISSUE_COMPOSITIONS)
    for composition in _RR_ROWS.values():
        assert all(isinstance(value, Distribution) for value in vars(composition).values())


def test_graph_hematocrit_reaches_blood_cell_affinity(tmp_path):
    from dataclasses import replace

    from sisyphus.pipeline.context import prepare_simulation_context
    from sisyphus.predict.adme import predict_adme
    from sisyphus.predict.chemistry import compute_profile
    from sisyphus.resources import get_resource_config

    smiles = "O(CCN(C)C)C(c1ccccc1)c1ccccc1"  # diphenhydramine, basic
    profile = compute_profile(smiles)
    adme = replace(predict_adme(profile), rbp=Distribution(1.1))
    standard = prepare_simulation_context(smiles, 10, profile=profile, adme=adme)
    physiology = get_resource_config().data("physiology", "reference_man.yaml").read_text()
    altered = tmp_path / "lower_hct.yaml"
    altered.write_text(physiology.replace(
        "hematocrit: {mean: 0.45, cv: 0.0}", "hematocrit: {mean: 0.40, cv: 0.0}"
    ))
    lower_hct = prepare_simulation_context(
        smiles, 10, profile=profile, adme=adme, base_yaml=altered
    )
    assert lower_hct.drug.kp_overrides["lung"].mean > standard.drug.kp_overrides["lung"].mean
