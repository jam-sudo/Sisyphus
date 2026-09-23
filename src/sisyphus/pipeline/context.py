"""Shared construction of a mechanistic simulation context.

All entry points that solve the PBPK graph must use this builder so disposition
routing, phenotype application, active-species augmentation, axial expansion,
and the hepatic-fu contract cannot silently diverge.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from sisyphus.core import DrugOnGraph
from sisyphus.engine.compiler import CompiledODE, ResolvedParams
from sisyphus.graph.body import BodyGraph
from sisyphus.predict.phenotype import PhenotypeApplicationReport

if TYPE_CHECKING:
    from sisyphus.predict.adme import ADMEProperties
    from sisyphus.predict.chemistry import MolecularProfile


@dataclass(frozen=True)
class SimulationContext:
    """Fully prepared inputs for deterministic or regimen simulation."""

    profile: MolecularProfile
    adme: ADMEProperties
    drug: DrugOnGraph
    graph: BodyGraph
    compiled: CompiledODE
    params: ResolvedParams
    phenotype_report: PhenotypeApplicationReport
    auto_oatp: bool
    non_cyp_tags: tuple[str, ...]


def prepare_simulation_context(
    smiles: str,
    dose_mg: float,
    route: str = "oral",
    *,
    phenotypes: dict[str, str] | None = None,
    phenotype_scale_overrides: dict[str, float] | None = None,
    kp_method: str = "rodgers_rowland",
    profile: MolecularProfile | None = None,
    adme: ADMEProperties | None = None,
    base_yaml: Path | None = None,
) -> SimulationContext:
    """Build one canonical simulation context for every public entry point."""

    import sisyphus.engine.flux  # noqa: F401 -- register flux specifications
    from sisyphus.engine.compiler import ODECompiler
    from sisyphus.engine.contracts import assert_fu_correction_honored
    from sisyphus.graph.axial import expand_axial
    from sisyphus.graph.builder import augment_for_active_species, build_from_yaml
    from sisyphus.predict.adme import predict_adme
    from sisyphus.predict.chemistry import compute_profile
    from sisyphus.predict.ivive import build_drug_on_graph, detect_disposition
    from sisyphus.predict.phenotype import apply_phenotype_to_graph
    from sisyphus.resources import get_resource_config

    profile = profile or compute_profile(smiles)
    adme = adme or predict_adme(profile)

    oatp_kinetics, ecm_params, non_cyp_fractions = detect_disposition(profile)
    drug = build_drug_on_graph(
        profile,
        adme,
        dose_mg,
        route,
        kp_method=kp_method,
        transporter_kinetics=oatp_kinetics,
        hepatic_ecm_params=ecm_params,
        non_cyp_fractions=non_cyp_fractions,
    )

    physiology = base_yaml or get_resource_config().data(
        "physiology", "reference_man.yaml"
    )
    graph = build_from_yaml(physiology)
    liver_enzymes_pre = None
    if "liver" in graph.nodes and graph.nodes["liver"].enzymes:
        liver_enzymes_pre = {
            tag: dist.mean for tag, dist in graph.nodes["liver"].enzymes.items()
        }

    if phenotypes:
        graph, phenotype_report = apply_phenotype_to_graph(
            graph,
            phenotypes,
            phenotype_scale_overrides=phenotype_scale_overrides,
            return_report=True,
        )
    else:
        phenotype_report = PhenotypeApplicationReport((), (), ())

    if liver_enzymes_pre is not None:
        drug = build_drug_on_graph(
            profile,
            adme,
            dose_mg,
            route,
            liver_enzymes=liver_enzymes_pre,
            kp_method=kp_method,
            transporter_kinetics=oatp_kinetics,
            hepatic_ecm_params=ecm_params,
            non_cyp_fractions=non_cyp_fractions,
        )

    graph = augment_for_active_species(graph, drug)
    graph = expand_axial(graph)
    assert_fu_correction_honored(graph, drug.fu_correction_liver.mean)
    compiled = ODECompiler().compile(graph)
    params = ResolvedParams(graph.realize_means(), drug.realize_means())

    return SimulationContext(
        profile=profile,
        adme=adme,
        drug=drug,
        graph=graph,
        compiled=compiled,
        params=params,
        phenotype_report=phenotype_report,
        auto_oatp=oatp_kinetics is not None and ecm_params is not None,
        non_cyp_tags=tuple(sorted(non_cyp_fractions)),
    )
