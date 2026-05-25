"""B-03.x literature-IVIVE disposition pin.

Verifies clopidogrel CES1/CYP affinities flipped from B-03 placeholders
(0.030 each, ceiling_accepted) to literature-IVIVE values
(Subash 2025 PMC12673578 rCES1 Vmax/Km + Boberg 2017 abundance;
Kazui 2010 85/15 fate split partition). Prevents silent regression back
to placeholders.

Closed 2026-05-25 doctrine completion sprint Phase B (SUCCESS).
"""
from __future__ import annotations

import json
from pathlib import Path

REGISTRY = Path("data/sbi/prodrug_activation_registry.json")
CLOPIDOGREL_SMILES = "COC(=O)[C@H](c1ccccc1Cl)N1CCc2sccc2C1"


def test_clopidogrel_disposition_literature_applied():
    """Clopidogrel B-03.x: disposition_state must be literature_applied."""
    entry = json.loads(REGISTRY.read_text())[CLOPIDOGREL_SMILES]
    assert entry["v3_metadata"]["disposition_state"] == "literature_applied", (
        f"clopidogrel disposition_state regressed to "
        f"{entry['v3_metadata']['disposition_state']!r} from literature_applied "
        f"(B-03.x SUCCESS 2026-05-25). If intentional, update this test."
    )
    assert entry["affinity_source"] == "literature_ivive", (
        f"clopidogrel affinity_source regressed to {entry['affinity_source']!r} "
        f"from literature_ivive (B-03.x SUCCESS 2026-05-25)."
    )


def test_clopidogrel_ces1_affinity_not_placeholder():
    """CES1 affinity must differ from the B-03 placeholder 0.030."""
    entry = json.loads(REGISTRY.read_text())[CLOPIDOGREL_SMILES]
    ces1_mean = entry["enzyme_affinity_for_conversion"]["CES1"]["mean"]
    assert ces1_mean != 0.03, (
        "CES1 affinity is the B-03 placeholder 0.030 — B-03.x literature-IVIVE "
        "values not applied. Expected ~0.0586 per Subash 2025 PMC12673578 + "
        "Boberg 2017 PMC5267516 IVIVE chain."
    )
    # Unit-sanity window per spec §B.2 (sprint 2026-05-24 design doc)
    assert 0.003 <= ces1_mean <= 0.30, (
        f"CES1 affinity {ces1_mean} outside spec §B.2 sanity window [0.003, 0.30]; "
        f"likely unit conversion regression."
    )


def test_clopidogrel_85_15_fate_split():
    """CES1 vs CYP_total flux ratio must be 85/15 ± 5pp.

    Per Kazui 2010 DMD 38:92-99 and clopidogrel literature consensus: ~85% of
    parent goes to inactive carboxylate via CES1 (yield=0); ~15% to active
    thiol metabolite R-130964 via CYP3A4 + CYP2C19 (mapped to CYP2C9 surrogate,
    both with yield=1).
    """
    entry = json.loads(REGISTRY.read_text())[CLOPIDOGREL_SMILES]
    aff = entry["enzyme_affinity_for_conversion"]
    # Liver-total abundances (per data/physiology/reference_man.yaml)
    abund = {"CES1": 8.0e7, "CYP3A4": 9.25e6, "CYP2C9": 6.48e6}
    fluxes = {tag: aff[tag]["mean"] * abund[tag] for tag in ("CES1", "CYP3A4", "CYP2C9")}
    total = sum(fluxes.values())
    ces1_frac = fluxes["CES1"] / total
    assert 0.80 <= ces1_frac <= 0.90, (
        f"CES1 flux fraction {ces1_frac:.3f} outside [0.80, 0.90] (Kazui 2010 "
        f"85% inactive target ±5pp). CYP3A4+CYP2C9 may be miscalibrated."
    )
