# tests/integration/test_pgx_engine_fold.py
"""Engine genotype response must match the analytical oral-AUC fold (production-
path correctness oracle for v2). See spec sec 4.1 / 7."""
from __future__ import annotations

import importlib.util
import pathlib

import pytest

from sisyphus.validation.pgx_metrics import analytical_fold

_HARNESS = pathlib.Path(__file__).resolve().parents[2] / "scripts/validate_pgx_genotype_folds.py"
_spec = importlib.util.spec_from_file_location("pgx_harness", _HARNESS)
pgx_harness = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pgx_harness)


@pytest.mark.parametrize("gene", ["CYP2D6", "CYP2C19", "CYP2C9"])
@pytest.mark.parametrize("fm", [0.7, 0.9])
def test_engine_pm_fold_matches_analytical(gene, fm):
    engine = pgx_harness.engine_auc_fold(gene_tag=gene, fm=fm, activity_variant=0.0)
    expected = analytical_fold(fm=fm, activity=0.0)  # PM -> 1/(1-fm)
    rel = abs(engine - expected) / expected
    assert rel < 0.02, f"{gene} fm={fm}: engine {engine:.3f} vs analytical {expected:.3f} (rel {rel:.3%})"
