"""Public-worktree 3-fold prodrug gate; unsupported cases remain strict xfail.

2026-10-29 deterministic Cmax and fold errors (mg/L, n_mc_samples=0):
  sepiapterin parent       13.897 / 0.0024 = 5790x
  remdesivir parent         5.894 / 4.38   = 1.35x
  tebipenem active          0.369 / 4.01   = 10.86x under
  fostamatinib active       0.0483 / 0.61  = 12.63x under

Gut-wall CES2/ALPI and gut/liver/kidney SPR activation are already wired.
Missing substrate-specific kinetics and presystemic disposition are described
in docs/research/prodrug-v4-mechanism-audit.md. Affinities are not fitted to
clinical Cmax.
"""
from __future__ import annotations

import pytest

from sisyphus.graph.builder import augment_for_active_species
from sisyphus.graph.presets import reference_man
from sisyphus.graph.types import ProdrugActivationEdge
from sisyphus.pipeline.predict import predict
from sisyphus.predict.registry import lookup_active_metabolite
from tests._artifact_helpers import skip_if_local_artifacts
from tests.unit.test_prodrug_v2_drug import _minimal_drug

_CLINICAL_CMAX = {
    "sepiapterin":       0.0024,
    "remdesivir":        4.38,
    "tebipenem_pivoxil": 4.01,
    "fostamatinib":      0.61,
}

_SMILES = {
    "sepiapterin":       "C[C@H](O)C(=O)C1=Nc2c(nc(N)[nH]c2=O)NC1",
    "remdesivir":        "CCC(CC)COC(=O)[C@H](C)N[P@](=O)(OC[C@H]1O[C@@](C#N)(c2ccc3c(N)ncnn23)[C@H](O)[C@@H]1O)Oc1ccccc1",  # noqa: E501
    "tebipenem_pivoxil": "C[C@@H](O)[C@H]1C(=O)N2C(C(=O)OCOC(=O)C(C)(C)C)=C(SC3CN(C4=NCCS4)C3)[C@H](C)[C@H]12",  # noqa: E501
    "fostamatinib":      "COc1cc(Nc2ncc(F)c(Nc3ccc4c(n3)N(COP(=O)(O)O)C(=O)C(C)(C)O4)n2)cc(OC)c1OC",
}

_DOSE_ROUTE = {
    "sepiapterin":       (4200.0, "oral"),
    "remdesivir":        (200.0, "iv"),
    "tebipenem_pivoxil": (300.0, "oral"),
    "fostamatinib":      (75.0, "oral"),
}


_KNOWN_FAILURES = {
    "sepiapterin":
        "v4 audit: 5790x parent overprediction. Gut/liver/kidney SPR sites "
        "exist, but human tissue abundance and presystemic conversion kinetics "
        "for sepiapterin are unmeasured; no Cmax-based fit is allowed.",
    # Remdesivir passes the current public-worktree gate.
    "tebipenem_pivoxil":
        "v4 audit: 10.86x active underprediction. Gut-wall CES2 site exists, "
        "but substrate-specific human CES2 Vmax/Km and gastric/intestinal "
        "hydrolysis kinetics were not located in public sources.",
    "fostamatinib":
        "v4 audit: 12.63x active underprediction. Gut-wall ALPI site exists, "
        "but brush-border ALPI/fostamatinib Vmax/Km were not located in "
        "public sources.",
}


def test_strict_xfail_activation_sites_are_wired():
    expected = {
        "sepiapterin": {("gut_wall", "SPR"), ("liver", "SPR"), ("kidney", "SPR")},
        "tebipenem_pivoxil": {("gut_wall", "CES2"), ("liver", "CES2")},
        "fostamatinib": {("gut_wall", "ALPI")},
    }
    for name, sites in expected.items():
        active, observation_species, affinities, _ = lookup_active_metabolite(_SMILES[name])
        graph = reference_man()
        drug = _minimal_drug(
            active_metabolite=active,
            observation_species=observation_species,
            enzyme_affinity_for_conversion=affinities,
        )
        augment_for_active_species(graph, drug)
        actual = {
            (edge.source, next(iter(edge.enzyme_tags)))
            for edge in graph.edges
            if isinstance(edge, ProdrugActivationEdge)
        }
        assert actual == sites


@skip_if_local_artifacts
@pytest.mark.parametrize(
    "drug_name",
    [
        pytest.param(
            name,
            marks=(
                pytest.mark.xfail(reason=_KNOWN_FAILURES[name], strict=True)
                if name in _KNOWN_FAILURES
                else ()
            ),
        )
        for name in _CLINICAL_CMAX
    ],
)
def test_prodrug_3fold(drug_name):
    smiles = _SMILES[drug_name]
    dose, route = _DOSE_ROUTE[drug_name]
    result = predict(smiles, dose_mg=dose, route=route, n_mc_samples=0)
    pred = result.pk.cmax.mean
    obs = _CLINICAL_CMAX[drug_name]
    fold_error = max(pred / obs, obs / pred)
    assert fold_error < 3.0, (
        f"{drug_name}: pred={pred:.4g} mg/L, obs={obs:.4g} mg/L, "
        f"fold_error={fold_error:.2f}x (target < 3.0x)"
    )
