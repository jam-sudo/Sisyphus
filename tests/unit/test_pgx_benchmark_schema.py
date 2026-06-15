# tests/unit/test_pgx_benchmark_schema.py
"""Guard the PGx benchmark: no circular fm, no excluded compounds in the
quantitative set, required fields present. Engine-free (no heavy imports)."""
from __future__ import annotations

import json
import pathlib

BENCH = pathlib.Path(__file__).resolve().parents[2] / "data/validation/pgx_genotype_folds.json"
_REQUIRED = {
    "drug", "gene", "phenotype", "fm_invitro", "fm_source_type",
    "obs_auc_fold_pm", "obs_auc_fold_ci", "is_prodrug", "is_nonlinear",
    "quantitative", "citation_fm", "citation_fold",
}
_ALLOWED_GENES = {"CYP2D6", "CYP2C19", "CYP2C9"}


def _pairs():
    return json.loads(BENCH.read_text())["pairs"]


def test_required_fields_present():
    for p in _pairs():
        missing = _REQUIRED - set(p)
        assert not missing, f"{p.get('drug')}: missing {missing}"


def test_no_circular_fm():
    for p in _pairs():
        assert p["fm_source_type"] == "in_vitro_phenotyping", (
            f"{p['drug']}: circular fm source {p['fm_source_type']!r}"
        )


def test_quantitative_set_is_clean():
    for p in _pairs():
        if p["quantitative"]:
            assert not p["is_prodrug"], f"{p['drug']}: prodrug in quantitative set"
            assert "extreme_fold" not in p.get("flags", []), p["drug"]


def test_genes_and_min_count():
    pairs = _pairs()
    assert all(p["gene"] in _ALLOWED_GENES for p in pairs)
    quant = [p for p in pairs if p["quantitative"]]
    assert len(quant) >= 6, f"feasibility gate: only {len(quant)} quantitative pairs"
    by_gene = {g: sum(p["gene"] == g for p in quant) for g in _ALLOWED_GENES}
    assert all(n >= 2 for n in by_gene.values()), by_gene
