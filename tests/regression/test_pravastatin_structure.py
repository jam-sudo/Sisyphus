"""Keep pravastatin's model input aligned with its documented parent structure."""

import json
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import inchi, rdMolDescriptors

from sisyphus.predict.cyp_clearance_overrides import lookup_metabolic_fraction
from sisyphus.predict.transporter_db import find_oatp1b1_substrate_name

ROOT = Path(__file__).resolve().parents[2]
KEY = "TUZYXOIXSAXUGO-PZAWKZKUSA-N"  # PubChem CID 54687


def test_pravastatin_structure_and_routing():
    clinical = json.loads((ROOT / "data/reference/clinical_pk.json").read_text())
    oatp = json.loads((ROOT / "data/transporters/oatp1b1.json").read_text())
    overrides = json.loads((ROOT / "data/transporters/cyp_clearance_overrides.json").read_text())
    entries = [
        clinical["drugs"]["pravastatin"],
        oatp["drugs"]["pravastatin"],
        next(row for row in overrides["overrides"] if row["drug"] == "pravastatin"),
    ]
    for entry in entries:
        molecule = Chem.MolFromSmiles(entry["smiles"])
        assert molecule is not None
        assert inchi.MolToInchiKey(molecule) == KEY
        assert rdMolDescriptors.CalcMolFormula(molecule) == "C23H36O7"
        if "inchikey" in entry:
            assert entry["inchikey"] == KEY

    correct = entries[0]["smiles"]
    assert find_oatp1b1_substrate_name(correct) == "pravastatin"
    assert lookup_metabolic_fraction(correct) == 0.0
    old_wrong = (
        "CC[C@@H](C)C(=O)O[C@@H]1C[C@@H](O)C=C2[C@@H]"
        "(CC[C@@H](O)C[C@@H](O)CC(=O)O)[C@H](C)CC[C@@H]21"
    )
    assert find_oatp1b1_substrate_name(old_wrong) is None
    assert lookup_metabolic_fraction(old_wrong) == 1.0
