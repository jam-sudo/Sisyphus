"""Scored development labels must be observed parent-drug arms."""

import json
from pathlib import Path

import pytest
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from sisyphus.validation.reference import load_reference

ROOT = Path(__file__).resolve().parents[2]


def test_adjudicated_holdout_arms_match_scored_cache():
    data = json.loads((ROOT / "data/reference/clinical_pk.json").read_text())
    assert data["metadata"]["n_with_cmax"] == sum(
        "cmax_mg_L" in drug.get("pk_params", {}) for drug in data["drugs"].values()
    )
    assert {name for name, drug in data["drugs"].items() if "ct_curve" in drug} == {"simvastatin"}
    assert data["drugs"]["codeine"]["tier"] == "gold"
    refs = {row.name: row for row in load_reference() if row.in_holdout}
    cache = json.loads((ROOT / "data/training/4track_holdout_predictions.json").read_text())
    assert cache["n_holdout"] == data["metadata"]["holdout_with_cmax"] == len(refs) == 73
    assert {row["name"] for row in cache["drugs"]} == set(refs)
    for name in ("cimetidine", "mefenamic acid"):
        assert data["drugs"][name]["tier"] == "unverified"
        assert not data["drugs"][name]["pk_params"]
        assert name not in refs
    assert refs["probenecid"].cmax_obs == pytest.approx(35.3)
    assert "mean-profile maximum" in data["drugs"]["alprazolam"]["source"]
    assert "mean-profile maximum" in data["drugs"]["triazolam"]["source"]
    assert refs["apixaban"].cmax_obs == pytest.approx(0.126)
    assert "Wang et al." in data["drugs"]["apixaban"]["source"]
    assert "thalf_h" not in data["drugs"]["apixaban"]["pk_params"]
    assert refs["famotidine"].cmax_obs == pytest.approx(0.073)
    assert "auc_mg_h_L" not in data["drugs"]["famotidine"]["pk_params"]
    assert refs["sildenafil"].cmax_obs == pytest.approx(0.271)
    assert data["drugs"]["sildenafil"]["pk_params"]["thalf_h"] == pytest.approx(2.96)
    assert "bioavailability_pct" not in data["drugs"]["sildenafil"]["pk_params"]
    assert data["drugs"]["penicillamine"]["tier"] == "unverified"
    assert not data["drugs"]["penicillamine"]["pk_params"]
    assert "penicillamine" not in refs
    assert refs["lenacapavir"].cmax_obs == pytest.approx(0.0234)
    assert "normal-hepatic-function matched controls" in data["drugs"]["lenacapavir"]["source"]
    assert refs["lorlatinib"].cmax_obs == pytest.approx(0.5013)
    assert "Hibma et al." in data["drugs"]["lorlatinib"]["source"]
    assert refs["vonoprazan"].cmax_obs == pytest.approx(0.025)
    assert "Clin Transl Gastroenterol" in data["drugs"]["vonoprazan"]["source"]
    for row in cache["drugs"]:
        assert row["obs"] == refs[row["name"]].cmax_obs
        assert not any(word in data["drugs"][row["name"]]["source"].lower()
                       for word in ("estimated", "simulated"))

    for name in (
        "abiraterone", "atovaquone", "clozapine", "darolutamide",
        "darunavir ethanolate", "glasdegib", "itraconazole", "leflunomide",
        "pomalidomide", "ranolazine", "sirolimus", "sonidegib", "tamsulosin",
        "vilazodone",
    ):
        assert data["drugs"][name]["tier"] == "unverified"
        assert not data["drugs"][name]["pk_params"]
        assert name not in refs
    for name in (
        "adefovir dipivoxil", "fesoterodine", "molnupiravir", "prasugrel",
        "tenofovir disoproxil", "valacyclovir", "valganciclovir",
    ):
        assert data["drugs"][name]["tier"] == "unverified"
        assert not data["drugs"][name]["pk_params"]
        assert name not in refs
    assert refs["clopidogrel"].dose_mg == 300.0
    assert refs["clopidogrel"].cmax_obs == 0.0145
    assert refs["codeine"].dose_mg == pytest.approx(30 * 2 * 299.3642 / 750.85, rel=1e-6)
    assert refs["codeine"].cmax_obs == 0.0714
    assert refs["amantadine"].dose_mg == pytest.approx(100 * 151.25 / 187.71, rel=1e-6)
    assert refs["amantadine"].cmax_obs == 0.22
    assert refs["fluvoxamine"].dose_mg == pytest.approx(50 * 318.33 / 434.4, rel=1e-6)
    assert refs["fluvoxamine"].cmax_obs == 0.017
    assert refs["hydroxyzine"].dose_mg == pytest.approx(25 * 374.9 / 447.83, rel=1e-6)
    assert refs["hydroxyzine"].cmax_obs == 0.03
    assert refs["carbinoxamine"].dose_mg == pytest.approx(8 * 290.79 / 406.86, rel=1e-6)
    assert refs["carbinoxamine"].cmax_obs == 0.024
    assert refs["montelukast"].dose_mg == 10.0
    assert refs["montelukast"].cmax_obs == 0.35
    assert refs["pravastatin"].dose_mg == 19.01
    assert refs["pravastatin"].cmax_obs == 0.0265
    assert refs["quizartinib"].dose_mg == 26.5
    assert refs["quizartinib"].cmax_obs == 0.102
    assert refs["ranitidine"].dose_mg == 150.0
    assert refs["ranitidine"].cmax_obs == 0.4506
    assert refs["selegiline"].dose_mg == pytest.approx(10 * 187.28 / 223.74, rel=1e-6)
    assert refs["selegiline"].cmax_obs == 0.003093
    assert "ulipristal acetate" not in refs  # historical split key is retained
    assert data["drugs"]["ulipristal"]["name"] == "ulipristal acetate"
    assert refs["ulipristal"].dose_mg == 30.0
    assert refs["ulipristal"].cmax_obs == 0.176
    ester = Chem.MolFromSmiles(refs["ulipristal"].smiles)
    assert rdMolDescriptors.CalcMolFormula(ester) == "C30H37NO4"
    assert Chem.MolToInchiKey(ester) == "OOLLAFOLCSJHRE-ZHAKMVSLSA-N"
    prior = Chem.MolFromSmiles(data["drugs"]["ulipristal"]["prior_reference_smiles"])
    assert rdMolDescriptors.CalcMolFormula(prior) == "C28H35NO3"
    curated_rows = json.loads((ROOT / "data/reference/curated_pk_data.json").read_text())
    assert next(row for row in curated_rows if row["drug_name"] == "ulipristal")["cmax_mg_L"] is None
    fda_rows = json.loads((ROOT / "data/reference/fda_extraction_results.json").read_text())
    for name in ("penicillamine", "lenacapavir"):
        row = next(row for row in fda_rows if row["drug_name"] == name)
        assert row["status"] == "unverified"
        assert row["cmax_mg_L"] is None
    assert next(row for row in fda_rows if row["drug_name"] == "hydroxyzine")["status"] != "extracted"
    assert next(row for row in fda_rows if row["drug_name"] == "selegiline")["status"] != "extracted"
    assert next(row for row in fda_rows if row["drug_name"] == "ulipristal")["status"] != "extracted"
    assert refs["trazodone"].dose_mg == pytest.approx(100 * 371.864 / 408.33, rel=1e-6)
    assert refs["trazodone"].cmax_obs == 1.5469
    assert "bioavailability_pct" not in data["drugs"]["fluvoxamine"]["pk_params"]
    assert refs["levocetirizine"].dose_mg == pytest.approx(5 * 388.9 / 461.8, rel=1e-6)
    assert refs["levocetirizine"].cmax_obs == 0.27
    assert refs["methylphenidate"].dose_mg == pytest.approx(20 * 233.31 / 269.77, rel=1e-6)
    assert refs["methylphenidate"].cmax_obs == 0.0091
    assert refs["paroxetine"].dose_mg == 25.0
    assert refs["paroxetine"].cmax_obs == 0.0055
    assert refs["nilotinib"].dose_mg == 200.0
    assert refs["nilotinib"].cmax_obs == 0.615
    assert refs["norethindrone"].dose_mg == 0.35
    assert refs["norethindrone"].cmax_obs == 0.004817
    assert refs["carbamazepine"].dose_mg == 200.0
    assert refs["carbamazepine"].cmax_obs == 1.9
    assert refs["zonisamide"].dose_mg == 300.0
    assert refs["zonisamide"].cmax_obs == 3.42
    assert refs["quinine"].dose_mg == 538.0
    assert refs["quinine"].cmax_obs == 3.2
    assert refs["quinine"].auc_obs is None
    assert refs["oxybutynin"].dose_mg == 5.0
    assert refs["oxybutynin"].cmax_obs == 0.0082
    assert refs["oxybutynin"].auc_obs is None
    assert refs["dasatinib"].dose_mg == 100.0
    assert refs["dasatinib"].cmax_obs == 0.2246
    assert refs["dasatinib"].auc_obs is None
    assert refs["cetirizine"].dose_mg == pytest.approx(10 * 388.89 / 461.82, rel=1e-6)
    assert refs["cetirizine"].cmax_obs == pytest.approx(0.266)
    assert refs["cetirizine"].auc_obs == pytest.approx(2.526)
    assert refs["febuxostat"].dose_mg == 40.0
    assert refs["febuxostat"].cmax_obs == pytest.approx(1.82)
    assert refs["febuxostat"].auc_obs == pytest.approx(4.61)
    assert "ct_curve" not in data["drugs"]["febuxostat"]
    assert refs["clomipramine"].dose_mg == pytest.approx(50 * 314.852 / 351.31, rel=1e-6)
    for name, dose, cmax in (
        ("alosetron", 1, 0.005),
        ("azacitidine", 300, 0.145),
        ("tamoxifen", 20, 0.04),
    ):
        assert refs[name].dose_mg == dose
        assert refs[name].cmax_obs == cmax
    for name in ("alosetron", "azacitidine", "clomipramine", "tamoxifen"):
        assert "ct_curve" not in data["drugs"][name]
        assert "thalf_h" not in data["drugs"][name]["pk_params"]
    assert data["drugs"]["clonidine"]["tier"] == "unverified"
    assert data["drugs"]["clonidine"]["dose_mg"] == 0.087
    assert not data["drugs"]["clonidine"]["pk_params"]
    assert "clonidine" not in refs
    assert refs["pindolol"].cmax_obs == 0.0331
    assert "thalf_h" not in data["drugs"]["pindolol"]["pk_params"]
    assert refs["sumatriptan"].dose_mg == 25.0
    assert refs["sumatriptan"].cmax_obs == 0.018
    assert refs["bexagliflozin"].dose_mg == 20.0
    assert refs["bexagliflozin"].cmax_obs == 0.134
    for name in ("clonidine", "pindolol", "sumatriptan", "bexagliflozin"):
        assert "ct_curve" not in data["drugs"][name]
    assert data["drugs"]["indomethacin"]["tier"] == "gold"
    assert refs["indomethacin"].dose_mg == 50.0
    assert refs["indomethacin"].cmax_obs == 3.107
    assert refs["ketoconazole"].cmax_obs == 4.22
    assert refs["levofloxacin"].dose_mg == 500.0
    assert refs["levofloxacin"].cmax_obs == 5.1
    assert refs["metronidazole"].cmax_obs == 13.0
    for name in ("indomethacin", "ketoconazole", "levofloxacin", "metronidazole"):
        assert "ct_curve" not in data["drugs"][name]
    for name in ("lopinavir", "pilocarpine", "temozolomide", "venlafaxine"):
        assert data["drugs"][name]["tier"] == "unverified"
        assert not data["drugs"][name]["pk_params"]
        assert "ct_curve" not in data["drugs"][name]
        assert name not in refs
    for name in ("isosorbide mononitrate", "losartan"):
        assert data["drugs"][name]["tier"] == "unverified"
        assert not data["drugs"][name]["pk_params"]
        assert "ct_curve" not in data["drugs"][name]
        assert name not in refs
    assert refs["ciprofloxacin"].cmax_obs == 2.4
    assert refs["moxifloxacin"].cmax_obs == 3.1
    assert refs["diclofenac"].dose_mg == pytest.approx(25 * 296.1 / 318.1)
    assert refs["diclofenac"].cmax_obs == 1.0
    assert refs["zolpidem"].dose_mg == 10.0
    for name in ("azithromycin", "ciprofloxacin", "colchicine", "diclofenac", "moxifloxacin", "zolpidem"):
        assert "ct_curve" not in data["drugs"][name]
    assert refs["budesonide"].auc_obs == pytest.approx(0.01413)
    assert data["drugs"]["budesonide"]["pk_params"].get("thalf_h") is None
    assert refs["dalfampridine"].cmax_obs == pytest.approx(0.0427)
    assert "thalf_h" not in data["drugs"]["dalfampridine"]["pk_params"]
    assert "bioavailability_pct" not in data["drugs"]["dalfampridine"]["pk_params"]
    assert refs["dapagliflozin"].dose_mg == 10.0
    assert refs["dapagliflozin"].cmax_obs == pytest.approx(0.136)
    assert "Table 1" in data["drugs"]["dapagliflozin"]["source"]
    assert next(row for row in curated_rows if row["drug_name"] == "dapagliflozin")["cmax_mg_L"] is None
    assert next(row for row in fda_rows if row["drug_name"] == "dapagliflozin")["status"] != "extracted"
    assert refs["etodolac"].cmax_obs == pytest.approx(14.0)
    assert "bioavailability_pct" not in data["drugs"]["etodolac"]["pk_params"]
    assert refs["ramelteon"].auc_obs == pytest.approx(0.0187)
    assert data["drugs"]["rivaroxaban"]["pk_params"]["thalf_h"] == pytest.approx(7.57)
    assert "bioavailability_pct" not in data["drugs"]["rivaroxaban"]["pk_params"]
    for name in ("budesonide", "dalfampridine", "dapagliflozin", "etodolac", "ramelteon", "rivaroxaban"):
        assert "ct_curve" not in data["drugs"][name]
    assert refs["cabozantinib"].dose_mg == 140.0
    assert refs["cabozantinib"].cmax_obs == pytest.approx(0.554)
    assert refs["erythromycin"].cmax_obs == pytest.approx(1.014211)
    assert refs["ruxolitinib"].dose_mg == 25.0
    assert refs["ruxolitinib"].cmax_obs == pytest.approx(0.4627)
    assert refs["ruxolitinib"].auc_obs == pytest.approx(1.630)
    osp = {row["drug_name"]: row for row in json.loads((ROOT / "data/reference/osp_observed.json").read_text())}
    assert not {"cabozantinib", "felodipine", "itraconazole", "ruxolitinib", "verapamil"} & osp.keys()
    assert osp["erythromycin"]["study"] == "DiSanto 1981"
    assert data["drugs"]["acamprosate"]["dose_mg"] == 600.0
    assert "acamprosate" not in refs
    assert data["drugs"]["acamprosate"]["tier"] == "unverified"
    assert next(row for row in curated_rows if row["drug_name"] == "acamprosate")["cmax_mg_L"] is None
    assert data["drugs"]["alvimopan"]["tier"] == "unverified"
    assert not data["drugs"]["alvimopan"]["pk_params"]
    assert "alvimopan" not in refs
    assert next(row for row in curated_rows if row["drug_name"] == "alvimopan")["cmax_mg_L"] is None
    assert next(row for row in fda_rows if row["drug_name"] == "alvimopan")["cmax_mg_L"] is None
    assert refs["donepezil"].dose_mg == pytest.approx(4.56)
    assert refs["donepezil"].cmax_obs == pytest.approx(0.0077)
    assert "oral suspension" in data["drugs"]["fruquintinib"]["source"]
    assert refs["ketorolac"].dose_mg == pytest.approx(10 * 255.27 / 376.41, rel=1e-6)
    assert refs["ketorolac"].cmax_obs == pytest.approx(0.87)
    assert next(row for row in fda_rows if row["drug_name"] == "ketorolac")["cmax_mg_L"] == pytest.approx(0.87)
    assert refs["brincidofovir"].cmax_obs == pytest.approx(0.251)
    assert "CMX001-120" in data["drugs"]["brincidofovir"]["source"]
    assert refs["lamivudine"].dose_mg == 300.0
    assert refs["lamivudine"].cmax_obs == pytest.approx(2.6)
    assert "Table 7" in data["drugs"]["lamivudine"]["source"]
    assert refs["mercaptopurine"].cmax_obs == pytest.approx(0.069)
    assert "tablet" in data["drugs"]["mercaptopurine"]["source"]
    assert next(row for row in fda_rows if row["drug_name"] == "mercaptopurine")["source"].startswith("DailyMed PURIXAN oral suspension")
    assert data["drugs"]["progesterone"]["tier"] == "unverified"
    assert not data["drugs"]["progesterone"]["pk_params"]
    assert "progesterone" not in refs
    assert next(row for row in curated_rows if row["drug_name"] == "progesterone")["cmax_mg_L"] is None
    assert next(row for row in fda_rows if row["drug_name"] == "progesterone")["status"] == "unverified"
    assert refs["rifabutin"].dose_mg == 300.0
    assert refs["rifabutin"].cmax_obs == pytest.approx(0.375)
    assert next(row for row in fda_rows if row["drug_name"] == "rifabutin")["cmax_value_original"] == 375
    assert refs["phenytoin"].dose_mg == pytest.approx(300 * 252.27 / 274.25, rel=1e-6)
    assert refs["phenytoin"].cmax_obs == pytest.approx(2.32)
    assert refs["phenytoin"].auc_obs == pytest.approx(108.99)
    assert "ct_curve" not in data["drugs"]["phenytoin"]
