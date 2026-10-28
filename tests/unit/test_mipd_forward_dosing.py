"""Forward dose search and loading-dose checks on known posterior particles."""
import numpy as np
import pytest

from sisyphus.mipd.core import Posterior, PosteriorPK
from sisyphus.mipd.dosing import Constraint, DoseTarget, forward_dose_search, recommend_dose
from sisyphus.regimen.types import DosingRegimen

ATENOLOL = "CC(C)NCC(O)COc1ccc(CC(N)=O)cc1"


def _posterior(f=1.0, ka=None):
    n = 100
    return PosteriorPK(
        f=Posterior(np.full(n, f)), cmax=Posterior(np.ones(n)),
        auc=Posterior(np.ones(n)), n_eff=float(n),
        total_clearance_l_h=Posterior(np.full(n, 5.0)),
        volume_l=Posterior(np.full(n, 25.0)),
        absorption_rate_per_h=Posterior(np.full(n, ka)) if ka is not None else None,
    )


@pytest.mark.parametrize(
    "constraint, expected_dose",
    [
        (Constraint("trough", low=0.7, high=0.9), 200.0),
        (Constraint("cmax", low=4.0, high=6.0), 100.0),
        (Constraint("auc24", low=70.0, high=90.0), 200.0),
    ],
)
def test_forward_search_attains_each_target_type(constraint, expected_dose):
    regimen = DosingRegimen.iv_infusion(100, 0, 12, 5)
    rec = forward_dose_search(
        _posterior(), regimen, DoseTarget((constraint,)), (100.0, 200.0), (12.0,),
        loading_target_mg_l=4.0,
    )
    assert rec.dose_mg == expected_dose
    assert rec.attainment_prob == 1.0
    assert rec.candidates[1].cmax_median == pytest.approx(2 * rec.candidates[0].cmax_median)
    assert rec.loading_dose.point == pytest.approx(100.0)
    assert rec.loading_dose.ci90 == (100.0, 100.0)


def test_oral_loading_dose_divides_by_bioavailability():
    regimen = DosingRegimen.oral_repeated(100, 12, 5)
    rec = forward_dose_search(
        _posterior(f=0.5, ka=1.0), regimen,
        DoseTarget((Constraint("cmax", low=0.1),)), (100.0,), (12.0,),
        loading_target_mg_l=4.0,
    )
    assert rec.loading_dose.point == pytest.approx(200.0)


def test_recommend_dose_routes_explicit_candidates_through_forward_search():
    regimen = DosingRegimen.iv_infusion(50, 0.5, 8, 5)
    rec = recommend_dose(
        ATENOLOL, regimen, (), DoseTarget((Constraint("auc24", low=1.0),)),
        candidate_doses_mg=(25.0, 50.0), candidate_intervals=(8.0,),
        loading_target_mg_l=1.0, n_samples=100,
    )
    assert len(rec.candidates) == 2
    assert rec.loading_dose is not None
    assert rec.loading_dose.ci90[0] > 0
