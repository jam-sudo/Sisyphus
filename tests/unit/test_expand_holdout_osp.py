"""OSP imports must not treat co-modeled drugs or repeated dosing as target observations."""

import json

from scripts.expand_holdout import OSPObservation, extract_osp_repo, select_best_osp


def test_extract_osp_repo_requires_parent_and_single_dose(tmp_path):
    repo = tmp_path / "Cabozantinib-Model"
    repo.mkdir()

    def observation(molecule, administrations, peak):
        props = {
            "Molecule": molecule,
            "Species": "Human",
            "Route": "PO",
            "Food state": "Fasted",
            "Compartment": "Plasma",
            "Dose": "140 mg",
            "Times of Administration [h]": administrations,
            "Formulation": "tablet",
            "Data type": "aggregated",
        }
        return {
            "ExtendedProperties": [{"Name": key, "Value": value} for key, value in props.items()],
            "BaseGrid": {"Values": [1, 2]},
            "Columns": [{"Values": [peak / 2, peak], "Unit": "mg/l"}],
        }

    data = {
        "ObservedData": [
            observation("Rifampicin", 0, 9.8),
            observation("Cabozantinib", "0-24-48", 1.0),
            observation("Cabozantinib", 0, 0.554),
        ]
    }
    (repo / "Cabozantinib-Model.json").write_text(json.dumps(data))
    drugbank = {"cabozantinib": {"smiles": "C", "inchikey_14": "TEST"}}

    result = extract_osp_repo("Cabozantinib-Model", tmp_path, drugbank)

    assert len(result) == 1
    assert result[0].cmax_obs == 0.554
    assert result[0].formulation == "tablet"
    assert result[0].data_type == "aggregated"


def test_select_best_osp_excludes_individual_curve():
    common = dict(drug_name="example", smiles="C", inchikey_14="TEST", dose_mg=400,
                  tmax_obs=1, source="OSP", study="study")
    observations = [
        OSPObservation(cmax_obs=2.3, n_subjects=1, data_type="individual", **common),
        OSPObservation(cmax_obs=1.8, n_subjects=12, data_type="aggregated", **common),
    ]
    assert select_best_osp(observations)["example"].cmax_obs == 1.8
