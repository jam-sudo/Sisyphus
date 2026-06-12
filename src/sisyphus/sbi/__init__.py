"""Amortized Simulation-Based Inference for PBPK.

This module implements Phase 2 of the breakthrough path: train a neural density
estimator to amortize the posterior p(ADME_params | observed_Cmax, drug, dose)
by using the Sisyphus engine as a simulator.

Unlike IBIS/EnKF TDM which runs MCMC per patient (seconds), an amortized
posterior returns an answer in milliseconds after an offline training phase.

Architecture:
    prior  →  simulator(engine)  →  (theta, x) pairs  →  NPE  →  amortized posterior

Current scope (POC):
    - Single-drug amortizer (per-drug training)
    - 3D theta: (log10_clint_scale, fup, log10_peff_scale)
    - 1D observation: log10(Cmax)
    - Simulator: full scipy engine (not surrogate — OOD bug)

Imports are LAZY (PEP 562 ``__getattr__``): ``torch`` is an optional extra
(pyproject ``[project.optional-dependencies]``) and the multi-drug / simulator
members need it. The physiology generator and priors are torch-free and must
remain importable without torch (e.g. by ``mipd`` weight/age covariates), so the
torch-dependent submodules are NOT imported at package-import time — each public
name is resolved from its submodule only when first accessed.
"""
from __future__ import annotations

import importlib
from typing import Any

# Public name -> defining submodule. Torch-free: physiology_generator, priors.
# Torch-dependent (imported only on access): multi_drug, simulator.
_EXPORTS = {
    "ENZYME_PARAMS": "sisyphus.sbi.physiology_generator",
    "enzyme_factor": "sisyphus.sbi.physiology_generator",
    "generate_physiology": "sisyphus.sbi.physiology_generator",
    "build_box_prior": "sisyphus.sbi.priors",
    "DRUG_FEATURE_NAMES": "sisyphus.sbi.multi_drug",
    "N_DRUG_FEATURES": "sisyphus.sbi.multi_drug",
    "DrugSpec": "sisyphus.sbi.multi_drug",
    "HierarchicalMultiDrugSimulator": "sisyphus.sbi.multi_drug",
    "MultiDrugSimulator": "sisyphus.sbi.multi_drug",
    "PopulationSpec": "sisyphus.sbi.multi_drug",
    "extract_drug_features": "sisyphus.sbi.multi_drug",
    "load_drug_specs_from_json": "sisyphus.sbi.multi_drug",
    "load_populations": "sisyphus.sbi.multi_drug",
    "pack_observation": "sisyphus.sbi.multi_drug",
    "pack_observation_hierarchical": "sisyphus.sbi.multi_drug",
    "population_onehot": "sisyphus.sbi.multi_drug",
    "stack_training_pairs": "sisyphus.sbi.multi_drug",
    "stack_training_pairs_hierarchical": "sisyphus.sbi.multi_drug",
    "EngineSimulator": "sisyphus.sbi.simulator",
    "apply_theta_to_drug": "sisyphus.sbi.simulator",
}


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(module), name)


def __dir__() -> list[str]:
    return sorted(_EXPORTS)


__all__ = [
    "DRUG_FEATURE_NAMES",
    "DrugSpec",
    "ENZYME_PARAMS",
    "EngineSimulator",
    "enzyme_factor",
    "generate_physiology",
    "HierarchicalMultiDrugSimulator",
    "MultiDrugSimulator",
    "N_DRUG_FEATURES",
    "PopulationSpec",
    "apply_theta_to_drug",
    "build_box_prior",
    "extract_drug_features",
    "load_drug_specs_from_json",
    "load_populations",
    "pack_observation",
    "pack_observation_hierarchical",
    "population_onehot",
    "stack_training_pairs",
    "stack_training_pairs_hierarchical",
]
