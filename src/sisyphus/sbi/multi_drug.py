"""Multi-drug conditional SBI utilities (Phase 2.0, Track A).

Generalizes the single-drug POC (`sisyphus.sbi.simulator.EngineSimulator`)
to the conditional setting where one amortizer serves all drugs:

    p(theta | log10_cmax_obs, drug_features)

Here ``drug_features`` is a fixed 12-D summary of the nominal ADME profile
(MW, logP, compound_type one-hot, nominal CLint magnitude, fup, ...), and
``theta`` layout (Phase 2.0.5):

    theta[0] = log10(CLint_actual / CLint_nominal)
    theta[1] = logit(fup)   (sigmoid inverse recovers physical fup)
    theta[2] = log10(Peff_actual / Peff_nominal)

The drug features enter training as the context half of the observation,
the density estimator learns ``p(theta | x)`` where ``x`` is
``concat([log10(cmax), drug_features])``.

The feature layout intentionally matches the neural surrogate's 12D layout
so downstream code (e.g. Track D1 surrogate fix) can share helpers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterable

import numpy as np

from sisyphus.sbi.simulator import EngineSimulator, apply_theta_to_drug

if TYPE_CHECKING:
    from sisyphus.core import DrugOnGraph
    from sisyphus.graph.body import BodyGraph

logger = logging.getLogger(__name__)


DRUG_FEATURE_NAMES: tuple[str, ...] = (
    "log10_clint_hepatic",
    "fup",
    "logp",
    "pka_or_7",
    "is_neutral",
    "is_acid",
    "is_base",
    "mw_scaled",
    "log10_dose",
    "log10_peff",
    "log10_sol",
    "log10_renal_cl",
)
N_DRUG_FEATURES = len(DRUG_FEATURE_NAMES)


def extract_drug_features(
    sim_or_drug=None,
    logp: float | None = None,
    *,
    drug: "DrugOnGraph | None" = None,
    graph: "BodyGraph | None" = None,
) -> np.ndarray:
    """Compute a 12-D feature vector from a drug's nominal ADME profile.

    Two calling conventions:

    1. ``extract_drug_features(sim)`` — legacy form, where ``sim`` is an
       :class:`EngineSimulator`. Pulls ``sim.nominal_drug`` and ``sim.graph``.
    2. ``extract_drug_features(drug=drug, graph=graph)`` — preferred form
       used by :func:`sisyphus.regimen.tdm.sbi_update` so no simulator
       round-trip is needed.

    The layout matches ``sisyphus.engine.surrogate.FEATURE_NAMES`` but with
    ``log10_clint`` derived from the *liver* node's enzyme abundance ×
    affinity product (hepatic-clearance proxy, not the buggy
    sum-over-all-nodes used by ``params_to_features_single``).

    ``logp`` is not carried on ``DrugOnGraph``; pass it explicitly (e.g.
    from ``compute_profile(smiles).logp``) so the default ``2.0`` sentinel
    is not silently substituted.

    Returns a length-12 float64 array.
    """
    if drug is None and graph is None:
        if sim_or_drug is None:
            raise ValueError(
                "extract_drug_features requires either a simulator "
                "positional arg or drug=/graph= keyword args"
            )
        sim = sim_or_drug  # legacy EngineSimulator form
        drug_obj = sim.nominal_drug
        graph_obj = sim.graph
    else:
        if drug is None or graph is None:
            raise ValueError("drug= and graph= must both be provided together")
        drug_obj = drug
        graph_obj = graph

    # Hepatic CLint = Σ(enzyme_abundance_liver × enzyme_affinity) on liver node.
    clint_hepatic = 0.0
    if "liver" in graph_obj.nodes and graph_obj.nodes["liver"].enzymes:
        for tag, abundance in graph_obj.nodes["liver"].enzymes.items():
            aff = drug_obj.enzyme_affinity.get(tag)
            if aff is None:
                continue
            clint_hepatic += float(abundance.mean) * float(aff.mean)
    clint_hepatic = max(clint_hepatic, 1e-6)

    fup = float(drug_obj.fup.mean)
    peff = float(drug_obj.peff.mean)
    sol = float(drug_obj.solubility.mean) if drug_obj.solubility is not None else 1.0
    renal_cl = float(drug_obj.renal_clearance.mean) if drug_obj.renal_clearance is not None else 1e-6
    dose = float(drug_obj.dose_mg)
    mw = float(drug_obj.mw)
    logp_val = float(logp) if logp is not None else 2.0
    pka = getattr(drug_obj, "pka", None)
    ctype = getattr(drug_obj, "compound_type", "neutral")

    feats = np.array(
        [
            np.log10(clint_hepatic),
            fup,
            logp_val,
            float(pka) if pka is not None else 7.0,
            1.0 if ctype == "neutral" else 0.0,
            1.0 if ctype == "acid" else 0.0,
            1.0 if ctype == "base" else 0.0,
            mw / 500.0,
            np.log10(max(dose, 1e-6)),
            np.log10(max(peff, 1e-6)),
            np.log10(max(sol, 1e-10)),
            np.log10(max(renal_cl, 1e-10)),
        ],
        dtype=np.float64,
    )
    return feats


@dataclass
class DrugSpec:
    """Minimal descriptor for loading a drug into a MultiDrugSimulator."""

    name: str
    smiles: str
    dose_mg: float
    route: str = "oral"


@dataclass
class MultiDrugSimulator:
    """Hold a population of per-drug EngineSimulators keyed by name.

    Build via ``MultiDrugSimulator.from_specs`` which constructs each
    simulator eagerly. Use ``simulate_for_drug(name, theta, seed)`` to
    run a single sample, or ``simulate_batch_per_drug(...)`` to sweep
    many thetas on one drug efficiently.
    """

    simulators: dict[str, EngineSimulator]
    features: dict[str, np.ndarray]

    @classmethod
    def from_specs(
        cls,
        specs: Iterable[DrugSpec],
        obs_sigma_log10: float = 0.0414,
    ) -> "MultiDrugSimulator":
        from sisyphus.predict.chemistry import compute_profile

        sims: dict[str, EngineSimulator] = {}
        feats: dict[str, np.ndarray] = {}
        for spec in specs:
            logger.info("building simulator for %s", spec.name)
            sim = EngineSimulator.for_drug(
                smiles=spec.smiles,
                dose_mg=spec.dose_mg,
                route=spec.route,
                obs_sigma_log10=obs_sigma_log10,
            )
            profile = compute_profile(spec.smiles)
            sims[spec.name] = sim
            feats[spec.name] = extract_drug_features(sim, logp=float(profile.logp))
        return cls(simulators=sims, features=feats)

    @classmethod
    def for_single(cls, spec: DrugSpec, obs_sigma_log10: float = 0.0414) -> "MultiDrugSimulator":
        return cls.from_specs([spec], obs_sigma_log10=obs_sigma_log10)

    @property
    def drug_names(self) -> list[str]:
        return list(self.simulators.keys())

    def get_features(self, name: str) -> np.ndarray:
        return self.features[name].copy()

    def simulate_single(self, name: str, theta: np.ndarray, seed: int) -> float:
        return self.simulators[name].simulate_single(theta, seed)

    def simulate_batch_per_drug(
        self,
        name: str,
        thetas: np.ndarray,
        seed: int = 0,
        progress: Callable[[int], None] | None = None,
    ) -> np.ndarray:
        return self.simulators[name].simulate_batch(thetas, seed=seed, progress=progress)


def stack_training_pairs(
    theta_per_drug: dict[str, np.ndarray],
    logcmax_per_drug: dict[str, np.ndarray],
    features_per_drug: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Concatenate per-drug arrays into the flat ``(theta, x, context)``
    tuple the trainer consumes.

    Rows with non-finite ``log10(cmax)`` are dropped, matching the
    single-drug POC convention.

    Returns:
        theta_all: (N, d_theta) float64
        x_all:     (N, 1) float64  — log10(cmax)
        feat_all:  (N, d_feat) float64 — drug features
        names:     list of length N, drug name per row
    """
    names_sorted = sorted(theta_per_drug.keys())
    theta_parts: list[np.ndarray] = []
    x_parts: list[np.ndarray] = []
    feat_parts: list[np.ndarray] = []
    row_names: list[str] = []
    for name in names_sorted:
        theta_i = np.asarray(theta_per_drug[name], dtype=np.float64)
        x_i = np.asarray(logcmax_per_drug[name], dtype=np.float64)
        if x_i.ndim == 1:
            x_i = x_i[:, None]
        finite_mask = np.isfinite(x_i[:, 0])
        if finite_mask.sum() == 0:
            logger.warning("drug %s has 0 finite sims, skipping", name)
            continue
        theta_parts.append(theta_i[finite_mask])
        x_parts.append(x_i[finite_mask])
        feats = features_per_drug[name].astype(np.float64)
        feat_parts.append(np.tile(feats[None, :], (finite_mask.sum(), 1)))
        row_names.extend([name] * int(finite_mask.sum()))

    if not theta_parts:
        raise RuntimeError("No finite simulations across any drug")

    theta_all = np.concatenate(theta_parts, axis=0)
    x_all = np.concatenate(x_parts, axis=0)
    feat_all = np.concatenate(feat_parts, axis=0)
    return theta_all, x_all, feat_all, row_names


def pack_observation(log10_cmax: np.ndarray, drug_features: np.ndarray) -> np.ndarray:
    """Concatenate observation and drug features into a single x vector.

    The conditional NPE treats the full concatenation as the observation
    so the density estimator can condition on both.

    log10_cmax: (N, 1) or (N,)
    drug_features: (N, d_feat) or (d_feat,)  — broadcast if needed
    Returns: (N, 1 + d_feat)
    """
    lc = np.asarray(log10_cmax, dtype=np.float64)
    if lc.ndim == 1:
        lc = lc[:, None]
    df = np.asarray(drug_features, dtype=np.float64)
    if df.ndim == 1:
        df = np.tile(df[None, :], (lc.shape[0], 1))
    if df.shape[0] != lc.shape[0]:
        raise ValueError(
            f"shape mismatch: log10_cmax={lc.shape}, drug_features={df.shape}"
        )
    return np.concatenate([lc, df], axis=1)


def load_drug_specs_from_json(path: Path) -> list[DrugSpec]:
    """Load a list of DrugSpec from a train/validation set JSON file."""
    import json

    with open(path) as f:
        data = json.load(f)
    specs: list[DrugSpec] = []
    for d in data["drugs"]:
        specs.append(
            DrugSpec(
                name=d["name"],
                smiles=d["smiles"],
                dose_mg=float(d["dose_mg"]),
                route=d.get("route", "oral"),
            )
        )
    return specs
