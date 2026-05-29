"""B-13 regression: gut_wall UGT abundance is the corrected, literature-grounded set.

Guards against re-introducing the confabulated B-13 values (DE-39). The gut_wall node must:
  - carry UGT2B7 at a literature-defensible abundance (Al-Majdoub 2021 CPT 109:1136 /
    Couto 2020 DMD 48:245: ~0.60 pmol/mg total-mucosal x 6000 mg = 3.6e3 pmol), NOT the
    confabulated 9.0e4 ("15 pmol/mg", cite "Bhatt 2019 DMD 47:498" — an unrelated maraviroc
    DDI paper, PMID 30862625);
  - NOT carry UGT1A9 — it is not expressed in human small intestine (Oda 2012 DMD 40:1620
    isoform-specific antibody finds it in kidney+liver only; UGT1A10 is the intestine-specific
    1A isoform; absent from Couto 2020 >5000-protein global proteomics). The confabulated
    1.2e4 entry cited a non-existent paper ("Akabane 2012 DMD 40:1310"; NCBI esearch count=0).

The drug-level UGT1A9 affinity (B-02 registry) still acts at the liver node — only the gut
abundance is omitted.
"""

from __future__ import annotations

from pathlib import Path

import yaml

_YAML = Path("data/physiology/reference_man.yaml")


def _gut_enzymes() -> dict:
    doc = yaml.safe_load(_YAML.read_text())
    gut = next(n for n in doc["nodes"] if n["name"] == "gut_wall")
    return gut["enzymes"]


def _mean(entry) -> float:
    return float(entry["mean"]) if isinstance(entry, dict) else float(entry)


def test_gut_wall_has_ugt2b7_literature_value() -> None:
    enz = _gut_enzymes()
    assert "UGT2B7" in enz, "gut_wall missing UGT2B7 abundance (B-13)"
    mean = _mean(enz["UGT2B7"])
    # Literature-defensible band around the curated 3.6e3 (0.60 pmol/mg total-mucosal x 6000).
    # Upper bound rejects the confabulated 9.0e4 ("15 pmol/mg", DE-39); the band still admits
    # the published intestinal spread (~0.6-3.6 pmol/mg total-protein -> ~3.6e3-2.2e4 pmol).
    assert 1.0e3 <= mean <= 1.5e4, (
        f"gut UGT2B7 abundance {mean:.3g} pmol outside the literature-defensible band "
        "[1e3, 1.5e4]; the confabulated 9.0e4 (15 pmol/mg, DE-39) must not return"
    )


def test_gut_wall_omits_ugt1a9() -> None:
    # UGT1A9 is not expressed in human small intestine (DE-39). The gut_wall node must not
    # carry a UGT1A9 abundance; drug-level UGT1A9 affinity still acts at the liver node.
    assert "UGT1A9" not in _gut_enzymes(), (
        "gut_wall must NOT contain UGT1A9 — not expressed in human intestine (Oda 2012); "
        "the confabulated 1.2e4 entry (DE-39) must not return"
    )
