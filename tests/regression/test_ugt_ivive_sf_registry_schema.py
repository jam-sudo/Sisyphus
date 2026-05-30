"""B-14 registry integrity: enforces the anti-fudge + correct-basis invariants."""
from __future__ import annotations

import json
import pathlib

_PATH = pathlib.Path("data/enzymes/ugt_ivive_sf.json")
_VALID_BASIS = {"hepatocyte", "hepatocyte_scaled"}
_VALID_DISP = {"literature_applied", "ceiling_accepted", "not_applicable", "default_1.0"}


def test_ugt_ivive_sf_schema():
    data = json.loads(_PATH.read_text())
    subs = data["substrates"]
    assert subs, "registry must list the seed substrates"
    seen = set()
    for e in subs:
        drug = e["drug"]
        ikey = e["inchikey"]
        assert ikey not in seen, f"duplicate inchikey {ikey}"
        seen.add(ikey)
        assert "<" not in ikey, f"{drug}: placeholder InChIKey not replaced"
        sf = e["ivive_sf"]
        assert isinstance(sf, dict) and sf, f"{drug}: ivive_sf must be a non-empty map"
        assert all(k.startswith("UGT") for k in sf), f"{drug}: SF keys must be UGT tags"
        disp = e["disposition"]
        assert disp in _VALID_DISP, f"{drug}: bad disposition {disp}"
        if disp in {"default_1.0", "not_applicable", "ceiling_accepted"}:
            assert all(v == 1.0 for v in sf.values()), f"{drug}: {disp} must be all 1.0"
        if disp == "literature_applied":
            assert any(v != 1.0 for v in sf.values()), f"{drug}: literature_applied all 1.0"
            assert e["basis"] in _VALID_BASIS, f"{drug}: basis must be in {_VALID_BASIS}"
            lits = e.get("literature", [])
            assert lits and all(
                lit.get("verified") and lit.get("pmid_or_doi") for lit in lits
            ), f"{drug}: literature_applied needs a verified PMID/DOI"
            if any(v > 5 for v in sf.values()):
                assert len(lits) >= 2, f"{drug}: ivive_sf>5 needs a second source"
        if drug in {"morphine", "codeine"}:
            assert "hepatic_fraction_of_deficit" in e, f"{drug}: must record partition"
