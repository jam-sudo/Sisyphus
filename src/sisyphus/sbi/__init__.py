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
"""

from sisyphus.sbi.multi_drug import (
    DRUG_FEATURE_NAMES,
    N_DRUG_FEATURES,
    DrugSpec,
    HierarchicalMultiDrugSimulator,
    MultiDrugSimulator,
    PopulationSpec,
    extract_drug_features,
    load_drug_specs_from_json,
    load_populations,
    pack_observation,
    pack_observation_hierarchical,
    population_onehot,
    stack_training_pairs,
    stack_training_pairs_hierarchical,
)
from sisyphus.sbi.physiology_generator import (
    ENZYME_PARAMS,
    enzyme_factor,
    generate_physiology,
)
from sisyphus.sbi.priors import build_box_prior
from sisyphus.sbi.simulator import EngineSimulator, apply_theta_to_drug

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
