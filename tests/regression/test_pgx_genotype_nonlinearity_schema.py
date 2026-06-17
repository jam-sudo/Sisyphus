"""Schema guard for the (HALTED-crossed-grid) genotype-nonlinearity evidence dataset.

The crossed dose×genotype grid does not exist in citable form (see the dataset's
halt_reason). This guard locks the HONEST shape: the HALT is recorded, the one citable
clinical saturation signature (propafenone EM dose-response, Siddoway 1987) is present and
well-formed for the P1 check, and the non-saturation context fold is labelled as such.
Spec: 2026-06-16-pgx-genotype-nonlinearity-two-arm-design.md §4.1, §7.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
_DATA = ROOT / "data" / "validation" / "pgx_genotype_nonlinearity_folds.json"


def _load():
    return json.loads(_DATA.read_text())


def test_dataset_exists_and_records_the_halt():
    d = _load()
    assert d["meta"]["status"] == "CROSSED_GRID_HALTED"
    assert d["meta"]["halt_reason"]  # the honest-negative reason is recorded
    assert "available_evidence" in d and d["available_evidence"]


def test_propafenone_em_p1_evidence_is_wellformed():
    """The one citable clinical saturation signature must carry doses, a supra-proportional
    observed beta, and a source — it feeds the engine P1 check."""
    ev = [e for e in _load()["available_evidence"]
          if e["drug"] == "propafenone" and e["tier"] == "p1_em_dose_response"]
    assert len(ev) == 1
    obs = ev[0]["observed"]
    assert len(obs["doses_mg_per_day"]) >= 2
    assert len(obs["relative_steadystate_conc"]) == len(obs["doses_mg_per_day"])
    assert obs["beta_obs_loglog"] > 1.0  # supra-proportional (saturation)
    assert "PMID 2434237" in obs["source"]  # Siddoway 1987


def test_context_fold_is_labelled_not_saturation_specific():
    """The 300 mg PM/EM fold is recorded as CONTEXT, explicitly not a saturation signal
    (it exceeds 1/(1-fm), the linear-null ceiling)."""
    ev = [e for e in _load()["available_evidence"]
          if e.get("tier") == "context_single_dose_fold"]
    assert len(ev) == 1
    assert "NOT" in ev[0]["note"] and "saturation" in ev[0]["note"].lower()
    assert ev[0]["observed"]["auc_fold_pm_over_em"] > 5.0  # > 1/(1-0.8) linear-null ceiling


def test_phenytoin_arm_is_halted_not_fabricated():
    ev = [e for e in _load()["available_evidence"]
          if e["drug"] == "phenytoin"]
    assert len(ev) == 1 and ev[0]["tier"] == "HALTED"
    # no fabricated dose×Css rows
    assert "observed" not in ev[0] or "doses" not in str(ev[0].get("observed", ""))
    assert ev[0]["halt"]


def test_no_invented_pm_gene_activity_fraction():
    """The forbidden cherry-pick: no *3/*3 activity number was invented anywhere."""
    # the dataset must not assert a phenytoin pm_gene_activity value (it was not found)
    pheny = [e for e in _load()["available_evidence"] if e["drug"] == "phenytoin"][0]
    assert "pm_gene_activity" not in pheny
