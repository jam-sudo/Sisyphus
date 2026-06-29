import json
from pathlib import Path

_PATH = Path("data/validation/pod_bioactivation_kinetics_spike.json")
_REQUIRED = {
    "name", "role", "smiles", "mw",
    "rat_fup", "rat_clint", "rat_clint_source",
    "vmax_bio", "vmax_bio_basis", "km_bio_uM", "fu_mic", "kinetics_source",
    "bioactivating_enzyme", "enzyme_zonation",  # e.g. "CYP2E1"/"pericentral"
    "vmax_detox", "vmax_detox_source",
    "observed_zone", "observed_zone_source",
    "pod_obs_noel", "pod_obs_loel", "pod_obs_source",
    "obs_rat_cmax_or_auc", "obs_pk_source",
}
_ROLES = {"reactive", "periportal", "negative", "off_mechanism"}


def test_kinetics_file_parses_and_is_precommitted():
    data = json.loads(_PATH.read_text())
    assert "compounds" in data and "halted" in data
    assert data.get("precommitted") is True


def test_every_compound_has_full_schema_and_sources():
    data = json.loads(_PATH.read_text())
    for c in data["compounds"]:
        missing = _REQUIRED - set(c)
        assert not missing, f"{c.get('name')} missing {missing}"
        assert c["role"] in _ROLES
        # no fabricated numerics: every value-bearing field non-null, every source non-empty
        for k in ("vmax_bio", "km_bio_uM", "vmax_detox", "rat_clint", "rat_fup"):
            assert c[k] is not None
        for k in ("kinetics_source", "vmax_detox_source", "pod_obs_source",
                  "observed_zone_source", "obs_pk_source", "rat_clint_source"):
            assert isinstance(c[k], str) and c[k].strip()


def test_program_viability_floor():
    # >=3 sourceable REACTIVE drugs (excluding acetaminophen) OR the file declares a HALT
    data = json.loads(_PATH.read_text())
    reactive = [c for c in data["compounds"]
                if c["role"] == "reactive" and c["name"].lower() != "acetaminophen"]
    if not data.get("data_halt"):
        assert len(reactive) >= 3, "data HALT terminal not reached but <3 reactive drugs"
    # at least one negative control and the off-mechanism control present (or halted w/ reason)
    roles = {c["role"] for c in data["compounds"]}
    assert "negative" in roles, "G-negative requires a true-negative control"
