"""Compound loader -- reads drug YAML files into DrugOnGraph instances.

Each compound YAML file specifies physicochemical properties, binding
parameters, enzyme affinities, and tissue partition coefficients.
Bare float values are wrapped in deterministic ``Distribution(value, cv=0.0)``
for Phase 1 (no parameter uncertainty on drug properties).
"""

from __future__ import annotations

from pathlib import Path

import yaml

from sisyphus.core import Distribution, DrugOnGraph


def load_compound(path: Path) -> DrugOnGraph:
    """Read a compound YAML file and return a DrugOnGraph.

    All numeric drug properties that the engine treats as uncertain
    are wrapped in ``Distribution(value, cv=0.0)`` (deterministic for
    Phase 1).

    Args:
        path: Path to the compound YAML file.

    Returns:
        A fully constructed DrugOnGraph.
    """
    with open(path) as f:
        data = yaml.safe_load(f)

    # Simple scalar fields.
    name = data["name"]
    smiles = data["smiles"]
    mw = float(data["mw"])
    dose_mg = float(data["dose_mg"])
    route = data["route"]
    administration_node = data["administration_node"]
    compound_type = data["compound_type"]
    particle_radius_um = float(data.get("particle_radius_um", 25.0))

    # pKa: float or None (if null/missing in YAML).
    raw_pka = data.get("pka")
    pka: float | None = float(raw_pka) if raw_pka is not None else None

    # Deterministic distributions for Phase 1.
    fup = Distribution(mean=float(data["fup"]), cv=0.0)
    rbp = Distribution(mean=float(data["rbp"]), cv=0.0)
    peff = Distribution(mean=float(data["peff"]), cv=0.0)
    solubility = Distribution(mean=float(data["solubility"]), cv=0.0)
    renal_clearance = Distribution(mean=float(data["renal_clearance"]), cv=0.0)

    # Kp method and overrides.
    kp_method = data.get("kp_method", "provided")
    kp_overrides: dict[str, Distribution] = {}
    for tissue, kp_val in data.get("kp_overrides", {}).items():
        kp_overrides[tissue] = Distribution(mean=float(kp_val), cv=0.0)

    # Enzyme affinity.
    enzyme_affinity: dict[str, Distribution] = {}
    for enzyme, affinity_val in data.get("enzyme_affinity", {}).items():
        enzyme_affinity[enzyme] = Distribution(mean=float(affinity_val), cv=0.0)

    # PS overrides.
    ps_overrides: dict[str, Distribution] = {}
    for tissue, ps_val in data.get("ps_overrides", {}).items():
        ps_overrides[tissue] = Distribution(mean=float(ps_val), cv=0.0)

    return DrugOnGraph(
        name=name,
        smiles=smiles,
        mw=mw,
        dose_mg=dose_mg,
        route=route,
        administration_node=administration_node,
        compound_type=compound_type,
        particle_radius_um=particle_radius_um,
        pka=pka,
        fup=fup,
        rbp=rbp,
        peff=peff,
        solubility=solubility,
        renal_clearance=renal_clearance,
        kp_method=kp_method,
        kp_overrides=kp_overrides,
        enzyme_affinity=enzyme_affinity,
        ps_overrides=ps_overrides,
    )
