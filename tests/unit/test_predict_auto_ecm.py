"""Unit tests for OATP1B1 ECM auto-loading on pipeline.predict.predict.

Closes issue #9. Previously, ``predict()`` did not activate the engine's
ECM transporter machinery for known OATP1B1 substrates — callers had to
construct the graph + drug + ECM kwargs manually. This caused
``apply_phenotype_to_graph(graph, {"SLCO1B1": "PM"})`` to be a no-op
through ``predict()`` even though the underlying engine path supported it.

These tests verify the wiring contract: SMILES of a registered substrate
must trigger auto-loading of both transporter kinetics and hepatic ECM
parameters; a non-substrate SMILES must leave them None.
"""

from __future__ import annotations

from unittest.mock import patch

from sisyphus.predict.transporter_db import find_oatp1b1_substrate_name

_PRAVASTATIN_CANONICAL = (
    "CC[C@@H](C)C(=O)O[C@@H]1C[C@@H](O)C=C2[C@@H]"
    "(CC[C@@H](O)C[C@@H](O)CC(=O)O)[C@H](C)CC[C@@H]21"
)
_PRAVASTATIN_VARIANT = (
    "CC[C@@H](C)C(=O)O[C@@H]1C[C@H](C=C2[C@@H]1CC[C@H]"
    "([C@@H]2CC[C@H](C[C@H](CC(=O)O)O)O)C)O"
)
_CAFFEINE = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"
_MORPHINE = "CN1CCC23C4C1CC5=C2C(=C(C=C5)O)OC3C(C=C4)O"


class TestFindOatp1b1Substrate:
    def test_pravastatin_canonical_returns_name(self):
        assert find_oatp1b1_substrate_name(_PRAVASTATIN_CANONICAL) == "pravastatin"

    def test_pravastatin_variant_smiles_returns_name(self):
        """SMILES variants of the same molecule must resolve via InChIKey."""
        assert find_oatp1b1_substrate_name(_PRAVASTATIN_VARIANT) == "pravastatin"

    def test_stereo_stripped_smiles_resolves_via_connectivity_block(self):
        """Reference datasets (e.g. clinical_pk.json) sometimes carry
        SMILES without stereochemistry annotations. The lookup must
        resolve such variants via the InChIKey connectivity block (first
        14 chars).
        """
        # Atorvastatin without stereo annotations (PubChem canonical
        # without @ markers).
        atorvastatin_no_stereo = (
            "CC(C)C1=C(C(=O)NC2=CC=CC=C2)C(=C(N1CCC(O)CC(O)CC(=O)O)"
            "C3=CC=C(F)C=C3)C4=CC=CC=C4"
        )
        assert find_oatp1b1_substrate_name(atorvastatin_no_stereo) == "atorvastatin"

    def test_caffeine_not_a_substrate(self):
        assert find_oatp1b1_substrate_name(_CAFFEINE) is None

    def test_morphine_not_a_substrate(self):
        assert find_oatp1b1_substrate_name(_MORPHINE) is None

    def test_invalid_smiles_returns_none(self):
        assert find_oatp1b1_substrate_name("not a smiles") is None

    def test_empty_smiles_returns_none(self):
        assert find_oatp1b1_substrate_name("") is None

    def test_all_seven_registered_substrates_resolvable(self):
        """The 7 substrate entries (pravastatin, rosuvastatin, atorvastatin,
        pitavastatin, fluvastatin, glimepiride, valsartan) all have InChIKey
        fields and resolve.
        """
        from sisyphus.predict.transporter_db import _load_oatp1b1_table
        for name, entry in _load_oatp1b1_table().items():
            assert "inchikey" in entry, f"{name} missing inchikey field"
            assert "smiles" in entry, f"{name} missing smiles field"
            resolved = find_oatp1b1_substrate_name(entry["smiles"])
            assert resolved == name, f"round-trip failed for {name}: {resolved}"


class TestPredictAutoEcmWiring:
    def test_non_substrate_does_not_trigger_auto_load(self):
        """Caffeine is not an OATP1B1 substrate. predict() must build the
        drug without transporter_kinetics or hepatic_ecm_params.
        """
        from sisyphus.predict import ivive

        real_build = ivive.build_drug_on_graph
        captured: list[dict] = []

        def spy(*args, **kwargs):
            captured.append({
                "transporter_kinetics": kwargs.get("transporter_kinetics"),
                "hepatic_ecm_params": kwargs.get("hepatic_ecm_params"),
            })
            return real_build(*args, **kwargs)

        from sisyphus.pipeline.predict import predict
        with patch("sisyphus.predict.ivive.build_drug_on_graph", side_effect=spy):
            predict(_CAFFEINE, dose_mg=100.0)
        assert captured, "expected build_drug_on_graph to be invoked"
        assert all(c["transporter_kinetics"] is None for c in captured)
        assert all(c["hepatic_ecm_params"] is None for c in captured)

    def test_pravastatin_triggers_auto_load(self):
        """Pravastatin is in both oatp1b1.json and hepatic_ecm.json, so
        predict() must auto-load both registries and forward to every
        build_drug_on_graph call.
        """
        from sisyphus.predict import ivive

        real_build = ivive.build_drug_on_graph
        captured: list[dict] = []

        def spy(*args, **kwargs):
            captured.append({
                "transporter_kinetics": kwargs.get("transporter_kinetics"),
                "hepatic_ecm_params": kwargs.get("hepatic_ecm_params"),
            })
            return real_build(*args, **kwargs)

        from sisyphus.pipeline.predict import predict
        with patch("sisyphus.predict.ivive.build_drug_on_graph", side_effect=spy):
            result = predict(_PRAVASTATIN_VARIANT, dose_mg=40.0)
        assert captured, "expected build_drug_on_graph to be invoked"
        assert all(c["transporter_kinetics"] is not None for c in captured), (
            "expected every build_drug_on_graph call to receive auto-loaded kinetics"
        )
        assert all(c["hepatic_ecm_params"] is not None for c in captured), (
            "expected every build_drug_on_graph call to receive auto-loaded ECM params"
        )
        assert "oatp1b1:auto_ecm:pravastatin" in result.warnings, (
            f"expected auto_ecm warning tag, got {result.warnings}"
        )

    def test_pravastatin_auto_ecm_lifts_cmax_vs_caffeine_baseline(self):
        """Sanity: with ECM auto-loaded, pravastatin Cmax through predict()
        must be in the same magnitude family as the integration test
        (~0.04 mg/L for 40 mg PO) rather than the pre-#9 value (~0.01 mg/L
        with no hepatic clearance path because metabolic_fraction=0 zeroed
        CYP and ECM was not active).

        This is a smoke gate, not a tight calibration assertion — the exact
        value depends on graph realization details that we don't lock here.
        """
        from sisyphus.pipeline.predict import predict
        result = predict(_PRAVASTATIN_VARIANT, dose_mg=40.0)
        assert result.pk.cmax.mean > 0.020, (
            f"expected pravastatin Cmax > 0.02 mg/L with auto-ECM, got {result.pk.cmax.mean:.4f}"
        )
        assert result.pk.cmax.mean < 0.200, (
            f"expected pravastatin Cmax < 0.2 mg/L (sanity upper bound), "
            f"got {result.pk.cmax.mean:.4f}"
        )
