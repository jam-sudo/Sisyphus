"""Tests for DrugBank lookup module."""
import pytest
from sisyphus.predict.drugbank import DrugBankConfig, DrugBankLookup


class TestDrugBankConfig:
    def test_default_all_enabled(self):
        cfg = DrugBankConfig()
        assert cfg.enable_enzyme_fm is True
        assert cfg.enable_fup is True
        assert cfg.enable_pka is True
        assert cfg.enable_logp is True

    def test_individual_disable(self):
        cfg = DrugBankConfig(enable_fup=False)
        assert cfg.enable_fup is False
        assert cfg.enable_logp is True


class TestDrugBankLookupNoData:
    def test_lookup_returns_none(self, tmp_path):
        lookup = DrugBankLookup(data_dir=tmp_path)
        assert lookup.lookup("CCO") is None

    def test_get_substrate_enzymes_returns_none(self, tmp_path):
        lookup = DrugBankLookup(data_dir=tmp_path)
        assert lookup.get_substrate_enzymes("CCO") is None

    def test_get_fup_returns_none(self, tmp_path):
        lookup = DrugBankLookup(data_dir=tmp_path)
        assert lookup.get_fup("CCO") is None

    def test_get_pka_returns_none(self, tmp_path):
        lookup = DrugBankLookup(data_dir=tmp_path)
        assert lookup.get_pka("CCO") is None

    def test_get_logp_returns_none(self, tmp_path):
        lookup = DrugBankLookup(data_dir=tmp_path)
        assert lookup.get_logp("CCO") is None


class TestDrugBankLookupWithData:
    @pytest.fixture
    def data_dir(self, tmp_path):
        (tmp_path / "drugs.csv").write_text(
            "drugbank_id,name,cas,smiles,inchikey,mw,logp_calc,pka_acidic,pka_basic,"
            "psa,hba,hbd,rotatable_bonds,state,groups,n_ddi,canonical_smiles,inchikey_14\n"
            'DB99901,TestDrug,,,TESTINCHIKEY1234-REST,180,2.5,4.2,9.1,50,3,1,2,solid,approved,0,'
            'c1ccc(O)cc1,TESTINCHIKEY1234\n'
            'DB99902,TestDrug2,,,OTHERINCHIKEY567-REST,200,3.0,10.5,2.0,60,4,2,3,solid,approved,0,'
            'CC(=O)O,OTHERINCHIKEY567\n'
        )
        (tmp_path / "enzyme_annotations.csv").write_text(
            "drugbank_id,drug_name,enzyme_name,uniprot_id,actions\n"
            "DB99901,TestDrug,Cytochrome P450 3A4,P08684,substrate\n"
            "DB99901,TestDrug,Cytochrome P450 2D6,P10635,substrate,inhibitor\n"
            "DB99901,TestDrug,Cytochrome P450 3A5,P20815,substrate\n"
        )
        (tmp_path / "pk_data.csv").write_text(
            "drugbank_id,drug_name,field,raw_text,parsed_value,parsed_unit\n"
            "DB99901,TestDrug,protein_binding,95% bound,0.05,fup\n"
        )
        (tmp_path / "experimental_properties.csv").write_text(
            "drugbank_id,drug_name,property,value\n"
            "DB99901,TestDrug,logP,2.8\n"
        )
        return tmp_path

    def test_lookup_by_canonical_smiles(self, data_dir):
        lookup = DrugBankLookup(data_dir=data_dir)
        assert lookup.lookup("c1ccc(O)cc1") == "DB99901"

    def test_lookup_inchikey_index_populated(self, data_dir):
        lookup = DrugBankLookup(data_dir=data_dir)
        lookup._ensure_loaded()
        assert "TESTINCHIKEY1234" in lookup._inchikey14_to_id

    def test_get_substrate_enzymes_with_cyp_normalization(self, data_dir):
        lookup = DrugBankLookup(data_dir=data_dir)
        enzymes = lookup.get_substrate_enzymes("c1ccc(O)cc1")
        assert enzymes is not None
        # CYP3A4 direct + CYP3A5→CYP3A4 (merged), CYP2D6 direct
        assert enzymes == {"CYP3A4", "CYP2D6"}

    def test_get_fup(self, data_dir):
        lookup = DrugBankLookup(data_dir=data_dir)
        assert lookup.get_fup("c1ccc(O)cc1") == pytest.approx(0.05)

    def test_get_pka(self, data_dir):
        lookup = DrugBankLookup(data_dir=data_dir)
        pka = lookup.get_pka("c1ccc(O)cc1")
        assert pka == pytest.approx((4.2, 9.1))

    def test_get_logp(self, data_dir):
        lookup = DrugBankLookup(data_dir=data_dir)
        assert lookup.get_logp("c1ccc(O)cc1") == pytest.approx(2.8)

    def test_feature_flag_disables_lookup(self, data_dir):
        cfg = DrugBankConfig(enable_fup=False, enable_logp=False)
        lookup = DrugBankLookup(data_dir=data_dir, config=cfg)
        assert lookup.get_fup("c1ccc(O)cc1") is None
        assert lookup.get_logp("c1ccc(O)cc1") is None
        assert lookup.get_pka("c1ccc(O)cc1") is not None
        assert lookup.get_substrate_enzymes("c1ccc(O)cc1") is not None

    def test_miss_returns_none(self, data_dir):
        lookup = DrugBankLookup(data_dir=data_dir)
        assert lookup.get_fup("CCCNOTINDB") is None
