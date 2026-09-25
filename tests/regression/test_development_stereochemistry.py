"""Pin source-adjudicated stereochemistry in the development reference set.

Keys come from PubChem name records checked through 2026-09-25. The four
exceptions are mixtures or FDA-unspecified E/Z structures, so choosing a
single PubChem stereoisomer would misrepresent their reference formulations.
"""

import json
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import inchi

ROOT = Path(__file__).resolve().parents[2]
EXPECTED = {
    "atorvastatin": "XUKUURHRXDUEBC-KAYWLYCHSA-N",
    "azacitidine": "NMUSYJAQQFHJEW-KVTDHHQDSA-N",
    "azithromycin": "MQTOSJVFKKJCRP-BICOPXKESA-N",
    "bexagliflozin": "BTCRKOKVYTVOLU-SJSRKZJXSA-N",
    "clarithromycin": "AGOYDEPGAOXOCK-KCBOHYOISA-N",
    "clopidogrel": "GKTWGGQPFAXNFI-HNNXBMFYSA-N",
    "colchicine": "IAKHMKGGTNLKSZ-INIZCTEOSA-N",
    "darolutamide": "BLIJXOOIHRSQRB-PXYINDEMSA-N",
    "entacapone": "JRURYQJSLYLRLN-BJMVGYQFSA-N",
    "fesoterodine": "DCCSDBARQIPTGU-HSZRJFAPSA-N",
    "fluvoxamine": "CJOFXWAVKWHTFT-XSFVSMFZSA-N",
    "glasdegib": "SFNSLLSYNZWZQG-VQIMIIECSA-N",
    "isosorbide mononitrate": "YWXYYJSYQOXTPL-SLPGGIOYSA-N",
    "isotretinoin": "SHGAZHPCJJPHSC-XFYACQKRSA-N",
    "levocetirizine": "ZKLPARSLTMPFCP-OAQYLSRUSA-N",
    "molnupiravir": "HTNPEHXGEKVIHG-QCNRFFRDSA-N",
    "morphine": "BQJCRHHNABKAKU-KBQPJGBKSA-N",
    "moxifloxacin": "FABPRXSRWADJSP-MEDUHNTESA-N",
    "naproxen oral": "CMWTZPSULFXXJA-VIFPVBQESA-N",
    "norethindrone": "VIKNJXKGJWUCNN-XGXHKTLJSA-N",
    "sonidegib": "VZZJRYRQSPEMTK-CALCHBBNSA-N",
    "tamoxifen": "NKANXQFJJICGDU-QPLCGJKRSA-N",
    "tamsulosin": "DRHKJLXJIQTDTD-OAHLLOKOSA-N",
    "tenofovir disoproxil": "JFVZFKDSXNQEJW-CQSZACIVSA-N",
    "valacyclovir": "HDOVUKNUBWVHOX-QMMMGPOBSA-N",
    "valganciclovir": "WPVFJKSGQUFQAP-GKAPJAKFSA-N",
}
MIXTURE_EXCEPTIONS = {"itraconazole", "ketoconazole", "ranitidine", "tramadol"}


def test_development_structures_and_registries_share_stereo_identity():
    clinical = json.loads((ROOT / "data/reference/clinical_pk.json").read_text())["drugs"]
    for name, key in EXPECTED.items():
        molecule = Chem.MolFromSmiles(clinical[name]["smiles"])
        assert molecule is not None
        assert inchi.MolToInchiKey(molecule) == key, name

    assert len(EXPECTED) == 26
    for name in MIXTURE_EXCEPTIONS:
        assert not any(marker in clinical[name]["smiles"] for marker in ("@", "/", "\\"))

    registries = {
        "data/transporters/hepatic_fu_correction.json": (
            "overrides", {"azacitidine", "clopidogrel", "fesoterodine", "fluvoxamine", "morphine"}
        ),
        "data/enzymes/ugt1a9_substrates.json": ("substrates", {"bexagliflozin", "glasdegib"}),
        "data/enzymes/ugt2b7_substrates.json": ("substrates", {"morphine"}),
    }
    for path, (field, expected_names) in registries.items():
        rows = json.loads((ROOT / path).read_text())[field]
        assert {row["drug"] for row in rows if row["drug"] in EXPECTED} == expected_names
        for row in rows:
            if row["drug"] in EXPECTED:
                assert inchi.MolToInchiKey(Chem.MolFromSmiles(row["smiles"])) == row["inchikey"]
                assert row["inchikey"] == EXPECTED[row["drug"]]
