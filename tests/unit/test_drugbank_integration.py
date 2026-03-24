"""Integration tests for DrugBank enrichment in predict layer."""
import pytest


class TestChemistryDrugBankIntegration:
    def test_pka_classify_acid(self):
        from sisyphus.predict.chemistry import _classify_from_pka
        pka, ct = _classify_from_pka(4.5, 2.0)
        assert ct == "acid"
        assert pka == pytest.approx(4.5)

    def test_pka_classify_base(self):
        from sisyphus.predict.chemistry import _classify_from_pka
        pka, ct = _classify_from_pka(12.0, 9.0)
        assert ct == "base"
        assert pka == pytest.approx(9.0)

    def test_pka_classify_zwitterion(self):
        from sisyphus.predict.chemistry import _classify_from_pka
        pka, ct = _classify_from_pka(4.0, 9.5)
        assert ct == "zwitterion"
        assert pka == pytest.approx(9.5)

    def test_pka_classify_neutral(self):
        from sisyphus.predict.chemistry import _classify_from_pka
        pka, ct = _classify_from_pka(11.0, 3.0)
        assert ct == "neutral"
        assert pka is None

    def test_pka_uncertainty_zone_classified_neutral(self):
        from sisyphus.predict.chemistry import _classify_from_pka
        _, ct = _classify_from_pka(7.2, 7.5)
        assert ct == "neutral"
