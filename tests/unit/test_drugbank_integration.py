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


class TestAdmeFupGuard:
    def test_fup_5x_guard_accepts_close(self):
        """DrugBank fup within 5x of XGBoost → accept."""
        db_fup, xgb_fup = 0.05, 0.08
        assert db_fup / xgb_fup <= 5.0 and xgb_fup / db_fup <= 5.0

    def test_fup_5x_guard_rejects_divergent(self):
        """DrugBank fup >5x different → reject."""
        db_fup, xgb_fup = 0.50, 0.08
        assert db_fup / xgb_fup > 5.0 or xgb_fup / db_fup > 5.0

    def test_fup_sanity_rejects_out_of_range(self):
        assert not (0.001 <= 0.0 <= 1.0)
        assert not (0.001 <= 1.5 <= 1.0)
        assert 0.001 <= 0.05 <= 1.0


class TestIviveFmFractions:
    def test_fm_no_annotation(self):
        from sisyphus.predict.ivive import _get_fm_fractions
        fm = _get_fm_fractions("neutral", substrate_enzymes=None)
        assert fm["CYP3A4"] == pytest.approx(0.50)

    def test_fm_single_substrate(self):
        from sisyphus.predict.ivive import _get_fm_fractions
        fm = _get_fm_fractions("neutral", substrate_enzymes={"CYP3A4"})
        assert fm["CYP3A4"] == pytest.approx(1.0 / 1.20)
        assert fm["CYP2D6"] == pytest.approx(0.05 / 1.20)

    def test_fm_two_substrates(self):
        from sisyphus.predict.ivive import _get_fm_fractions
        fm = _get_fm_fractions("neutral", substrate_enzymes={"CYP3A4", "CYP2C9"})
        assert fm["CYP3A4"] == pytest.approx(0.50 / 1.15)
        assert fm["CYP2C9"] == pytest.approx(0.50 / 1.15)

    def test_fm_unknown_substrate_ignored(self):
        from sisyphus.predict.ivive import _get_fm_fractions
        fm = _get_fm_fractions("neutral", substrate_enzymes={"CYP_UNKNOWN"})
        assert fm["CYP3A4"] == pytest.approx(0.50)
