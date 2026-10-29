"""Measured inputs and provenance flags on the public prediction path."""

import json
import sys

import pytest

from sisyphus.core import Distribution
from sisyphus.pipeline.context import prepare_simulation_context
from sisyphus.pipeline.predict import predict
from sisyphus.predict.adme import MeasuredADMEInput
from sisyphus.predict.chemistry import compute_profile
from sisyphus.resources import get_resource_config

CAFFEINE = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"


def test_unknown_kp_method_rejected():
    with pytest.raises(ValueError, match="Unknown kp_method"):
        predict(CAFFEINE, 100, kp_method="typo")
    with pytest.raises(ValueError, match="requires measured_adme.kp_overrides"):
        predict(CAFFEINE, 100, kp_method="provided")


def test_measured_kp_reaches_drug_and_changes_engine():
    base = prepare_simulation_context(CAFFEINE, 100)
    kp = {key: Distribution(value.mean * 2, value.cv)
          for key, value in base.drug.kp_overrides.items()}
    measured = MeasuredADMEInput(kp_overrides=kp)
    context = prepare_simulation_context(CAFFEINE, 100, kp_method="provided",
                                         measured_adme=measured)
    assert context.drug.kp_method == "provided"
    assert context.drug.kp_overrides == kp
    supplied = predict(CAFFEINE, 100, kp_method="provided", measured_adme=measured)
    assert supplied.engine_pk.cmax.mean != predict(CAFFEINE, 100).engine_pk.cmax.mean


def test_measured_enzyme_km_reaches_existing_saturation_path():
    km = {"CYP3A4": Distribution(0.000001, 0.2)}
    measured = MeasuredADMEInput(enzyme_km=km)
    context = prepare_simulation_context(CAFFEINE, 100, measured_adme=measured)
    assert context.drug.enzyme_km == km
    assert predict(CAFFEINE, 100, measured_adme=measured).engine_pk.cmax.mean != (
        predict(CAFFEINE, 100).engine_pk.cmax.mean
    )


def test_invalid_measured_maps_rejected():
    with pytest.raises(ValueError, match="kp_overrides"):
        MeasuredADMEInput(kp_overrides={"liver": Distribution(-1, 0.2)})
    with pytest.raises(ValueError, match="enzyme_km"):
        MeasuredADMEInput(enzyme_km={"CYP1A2": 1.0})
    with pytest.raises(ValueError, match="CV >= 0.10"):
        MeasuredADMEInput(enzyme_km={"CYP3A4": Distribution(1.0)})
    with pytest.raises(ValueError, match="Unknown Kp tissue"):
        prepare_simulation_context(
            CAFFEINE, 100,
            measured_adme=MeasuredADMEInput(kp_overrides={"typo": Distribution(2, 0.2)}),
        )
    with pytest.raises(ValueError, match="requires all tissues"):
        prepare_simulation_context(
            CAFFEINE, 100, kp_method="provided",
            measured_adme=MeasuredADMEInput(kp_overrides={"liver": Distribution(2, 0.2)}),
        )


def test_unannotated_phenotype_warns_about_default_fm():
    result = predict(CAFFEINE, 100, phenotypes={"CYP2D6": "PM"})
    assert any("fm:compound-type default" in warning for warning in result.warnings)


def test_active_species_phenotype_warns_about_aggregate_clearance():
    registry = json.loads(get_resource_config().data(
        "sbi", "prodrug_activation_registry.json").read_text())
    smiles = next(smiles for smiles, entry in registry.items() if entry["name"] == "irinotecan")
    result = predict(smiles, 350, route="iv", phenotypes={"UGT1A1": "PM"})
    assert any("active metabolite uses aggregate clearance" in w for w in result.warnings)


def test_berezhkovskiy_method_is_available():
    result = predict(CAFFEINE, 100, kp_method="berezhkovskiy")
    assert result.pk.cmax.mean > 0


def test_phenotype_help_lists_only_graph_supported_cyp_tags(monkeypatch, capsys):
    from sisyphus import cli
    from sisyphus.predict import phenotype

    monkeypatch.setattr(sys, "argv", ["sisyphus", "tdm", "--help"])
    with pytest.raises(SystemExit):
        cli.main()
    help_text = capsys.readouterr().out
    for tag in ("2D6", "2C9", "1A2", "3A4", "2E1"):
        assert tag in help_text
    for tag in ("2C19", "3A5", "2B6"):
        assert tag not in help_text
        assert f"CYP{tag}" not in phenotype.__doc__


def test_nonester_registry_prodrugs_are_flagged():
    registry = json.loads(get_resource_config().data(
        "sbi", "prodrug_activation_registry.json").read_text())
    for name in ("BH4", "R406"):
        smiles = next(smiles for smiles, entry in registry.items() if entry["name"] == name)
        assert "PRODRUG" in compute_profile(smiles).ad_flags
