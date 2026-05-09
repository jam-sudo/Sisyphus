"""Multi-drug conditional SBI utilities (Phase 2.0, Track A + Track C1).

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

**Track C1 — Hierarchical populations (2026-04-12)**:

Extends the multi-drug conditional amortizer to also condition on
population class (adult vs pediatric). The observation vector becomes:

    x = [log10(cmax), drug_features (12D), pop_onehot (N_pop D)]

Drug features are always computed using the *adult reference* graph so they
describe intrinsic drug properties, not population-specific PK.  Population
differences (enzyme ontogeny, allometric volumes, cardiac output) are
captured by the one-hot population conditioning vector.

Key classes:
    - ``HierarchicalMultiDrugSimulator`` — holds simulators for all
      (population, drug) combinations.
    - ``load_populations()`` — reads ``data/sbi/populations.json``.
    - ``pack_observation_hierarchical()`` — appends pop one-hot to the
      existing 13D observation vector.
    - ``stack_training_pairs_hierarchical()`` — stacks per-drug per-pop data.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from sisyphus.sbi.simulator import EngineSimulator

if TYPE_CHECKING:
    from sisyphus.core import DrugOnGraph
    from sisyphus.graph.body import BodyGraph

logger = logging.getLogger(__name__)

_POPULATIONS_JSON = Path("data/sbi/populations.json")
_ADULT_PHYSIOLOGY = Path("data/physiology/reference_man.yaml")


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


# ---------------------------------------------------------------------------
# Population registry (Track C1)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PopulationSpec:
    """Descriptor for a population loaded from populations.json."""

    name: str
    physiology_yaml: Path
    body_weight_kg: float
    description: str = ""


def load_populations(path: Path | None = None) -> dict[str, PopulationSpec]:
    """Load population definitions from ``data/sbi/populations.json``.

    Returns a dict mapping population name to ``PopulationSpec``.
    """
    if path is None:
        path = _POPULATIONS_JSON
    with open(path) as f:
        data = json.load(f)
    pops: dict[str, PopulationSpec] = {}
    for name, info in data["populations"].items():
        pops[name] = PopulationSpec(
            name=name,
            physiology_yaml=Path(info["physiology_yaml"]),
            body_weight_kg=float(info["body_weight_kg"]),
            description=str(info.get("description", "")),
        )
    return pops


def population_onehot(
    pop_name: str,
    population_names: list[str],
) -> np.ndarray:
    """Return a one-hot vector for the given population.

    Args:
        pop_name: name of the population (e.g. ``"adult"``).
        population_names: canonical ordered list of population names.

    Returns:
        float64 array of shape ``(len(population_names),)``.
    """
    vec = np.zeros(len(population_names), dtype=np.float64)
    idx = population_names.index(pop_name)
    vec[idx] = 1.0
    return vec


def extract_drug_features(
    sim_or_drug=None,
    logp: float | None = None,
    *,
    drug: DrugOnGraph | None = None,
    graph: BodyGraph | None = None,
) -> np.ndarray:
    """Compute a 12-D feature vector from a drug's nominal ADME profile.

    Two calling conventions:

    1. ``extract_drug_features(sim)`` — legacy form, where ``sim`` is an
       :class:`EngineSimulator`. Pulls ``sim.nominal_drug`` and ``sim.graph``.
    2. ``extract_drug_features(drug=drug, graph=graph)`` — preferred form
       used by :func:`sisyphus.regimen.tdm.sbi_update` so no simulator
       round-trip is needed.

    The layout matches ``sisyphus.ml.surrogate.FEATURE_NAMES`` but with
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
    renal_cl = float(drug_obj.renal_clearance.mean) if drug_obj.renal_clearance is not None else 1e-6  # noqa: E501
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
    ) -> MultiDrugSimulator:
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
    def for_single(cls, spec: DrugSpec, obs_sigma_log10: float = 0.0414) -> MultiDrugSimulator:
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


# ---------------------------------------------------------------------------
# Hierarchical multi-population simulator (Track C1)
# ---------------------------------------------------------------------------

@dataclass
class HierarchicalMultiDrugSimulator:
    """Hold per-(population, drug) EngineSimulators.

    Drug features are always computed against the adult reference graph so
    they remain population-independent.  The population difference is encoded
    via a one-hot vector appended to the observation.

    Build via ``HierarchicalMultiDrugSimulator.from_specs``.
    """

    # pop_name -> drug_name -> EngineSimulator
    simulators: dict[str, dict[str, EngineSimulator]]
    # drug_name -> 12D feature vector (population-independent, from adult graph)
    drug_features: dict[str, np.ndarray]
    # sorted population names — defines one-hot order
    population_names: list[str]
    # pop_name -> one-hot vector
    pop_onehot: dict[str, np.ndarray]

    @property
    def n_populations(self) -> int:
        return len(self.population_names)

    @classmethod
    def from_specs(
        cls,
        specs: Iterable[DrugSpec],
        populations: dict[str, PopulationSpec] | None = None,
        obs_sigma_log10: float = 0.0414,
    ) -> HierarchicalMultiDrugSimulator:
        """Build simulators for every (population, drug) pair.

        Drug features use the adult reference graph regardless of population.
        """
        from sisyphus.graph.builder import build_from_yaml
        from sisyphus.predict.chemistry import compute_profile

        if populations is None:
            populations = load_populations()
        pop_names = sorted(populations.keys())  # canonical order: ["adult", "pediatric_5y"]

        pop_oh: dict[str, np.ndarray] = {}
        for pn in pop_names:
            pop_oh[pn] = population_onehot(pn, pop_names)

        # Build the adult reference graph once for population-independent feature extraction
        adult_graph = build_from_yaml(_ADULT_PHYSIOLOGY)  # noqa: F841

        specs_list = list(specs)
        sims: dict[str, dict[str, EngineSimulator]] = {pn: {} for pn in pop_names}
        feats: dict[str, np.ndarray] = {}

        for spec in specs_list:
            logger.info("building hierarchical simulators for %s", spec.name)
            profile = compute_profile(spec.smiles)

            # Drug features from adult graph (population-independent)
            # Build a single adult simulator to extract features
            adult_sim = EngineSimulator.for_drug(
                smiles=spec.smiles,
                dose_mg=spec.dose_mg,
                route=spec.route,
                physiology_yaml=_ADULT_PHYSIOLOGY,
                obs_sigma_log10=obs_sigma_log10,
            )
            feats[spec.name] = extract_drug_features(
                adult_sim, logp=float(profile.logp)
            )
            sims["adult"][spec.name] = adult_sim

            # Build simulators for non-adult populations
            for pn in pop_names:
                if pn == "adult":
                    continue  # already built
                pop_spec = populations[pn]
                sim = EngineSimulator.for_drug(
                    smiles=spec.smiles,
                    dose_mg=spec.dose_mg,
                    route=spec.route,
                    physiology_yaml=pop_spec.physiology_yaml,
                    obs_sigma_log10=obs_sigma_log10,
                )
                sims[pn][spec.name] = sim

        return cls(
            simulators=sims,
            drug_features=feats,
            population_names=pop_names,
            pop_onehot=pop_oh,
        )

    def get_features(self, drug_name: str) -> np.ndarray:
        """Return a copy of the population-independent 12D drug features."""
        return self.drug_features[drug_name].copy()

    def get_pop_onehot(self, pop_name: str) -> np.ndarray:
        """Return a copy of the one-hot population vector."""
        return self.pop_onehot[pop_name].copy()

    @property
    def drug_names(self) -> list[str]:
        return sorted(self.drug_features.keys())

    def simulate_single(
        self,
        pop_name: str,
        drug_name: str,
        theta: np.ndarray,
        seed: int,
    ) -> float:
        """Simulate log10(Cmax) for one (population, drug, theta)."""
        return self.simulators[pop_name][drug_name].simulate_single(theta, seed)

    def simulate_batch_per_drug(
        self,
        pop_name: str,
        drug_name: str,
        thetas: np.ndarray,
        seed: int = 0,
        progress: Callable[[int], None] | None = None,
    ) -> np.ndarray:
        """Simulate a batch of thetas for one (population, drug)."""
        return self.simulators[pop_name][drug_name].simulate_batch(
            thetas, seed=seed, progress=progress
        )


def pack_observation_hierarchical(
    log10_cmax: np.ndarray,
    drug_features: np.ndarray,
    pop_onehot: np.ndarray,
) -> np.ndarray:
    """Concatenate observation, drug features, and population one-hot.

    The hierarchical conditional NPE conditions on:
        x = [log10(cmax) (1D), drug_features (12D), pop_onehot (N_pop D)]

    log10_cmax: (N, 1) or (N,)
    drug_features: (N, d_feat) or (d_feat,) — broadcast if needed
    pop_onehot: (N, n_pop) or (n_pop,) — broadcast if needed
    Returns: (N, 1 + d_feat + n_pop)
    """
    lc = np.asarray(log10_cmax, dtype=np.float64)
    if lc.ndim == 1:
        lc = lc[:, None]
    n = lc.shape[0]

    df = np.asarray(drug_features, dtype=np.float64)
    if df.ndim == 1:
        df = np.tile(df[None, :], (n, 1))
    if df.shape[0] != n:
        raise ValueError(
            f"shape mismatch: log10_cmax has {n} rows, drug_features has {df.shape[0]}"
        )

    ph = np.asarray(pop_onehot, dtype=np.float64)
    if ph.ndim == 1:
        ph = np.tile(ph[None, :], (n, 1))
    if ph.shape[0] != n:
        raise ValueError(
            f"shape mismatch: log10_cmax has {n} rows, pop_onehot has {ph.shape[0]}"
        )

    return np.concatenate([lc, df, ph], axis=1)


def stack_training_pairs_hierarchical(
    theta_per_pop_drug: dict[str, dict[str, np.ndarray]],
    logcmax_per_pop_drug: dict[str, dict[str, np.ndarray]],
    features_per_drug: dict[str, np.ndarray],
    pop_onehot_map: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    """Concatenate per-(population, drug) arrays into flat training data.

    Args:
        theta_per_pop_drug: pop_name -> drug_name -> (N, d_theta)
        logcmax_per_pop_drug: pop_name -> drug_name -> (N, 1) or (N,)
        features_per_drug: drug_name -> (d_feat,) — pop-independent
        pop_onehot_map: pop_name -> (n_pop,)

    Returns:
        theta_all:    (N_total, d_theta)
        x_all:        (N_total, 1)  — log10(cmax)
        feat_all:     (N_total, d_feat) — drug features
        pop_all:      (N_total, n_pop) — population one-hot
        drug_names:   list of length N_total
        pop_names:    list of length N_total
    """
    pop_sorted = sorted(theta_per_pop_drug.keys())
    theta_parts: list[np.ndarray] = []
    x_parts: list[np.ndarray] = []
    feat_parts: list[np.ndarray] = []
    pop_parts: list[np.ndarray] = []
    row_drugs: list[str] = []
    row_pops: list[str] = []

    for pop in pop_sorted:
        drug_dict = theta_per_pop_drug[pop]
        drug_sorted = sorted(drug_dict.keys())
        pop_vec = pop_onehot_map[pop]
        for drug in drug_sorted:
            theta_i = np.asarray(drug_dict[drug], dtype=np.float64)
            x_i = np.asarray(logcmax_per_pop_drug[pop][drug], dtype=np.float64)
            if x_i.ndim == 1:
                x_i = x_i[:, None]
            finite_mask = np.isfinite(x_i[:, 0])
            n_valid = int(finite_mask.sum())
            if n_valid == 0:
                logger.warning(
                    "pop=%s drug=%s has 0 finite sims, skipping", pop, drug
                )
                continue
            theta_parts.append(theta_i[finite_mask])
            x_parts.append(x_i[finite_mask])
            feat_parts.append(
                np.tile(features_per_drug[drug][None, :], (n_valid, 1))
            )
            pop_parts.append(np.tile(pop_vec[None, :], (n_valid, 1)))
            row_drugs.extend([drug] * n_valid)
            row_pops.extend([pop] * n_valid)

    if not theta_parts:
        raise RuntimeError("No finite simulations across any (population, drug)")

    return (
        np.concatenate(theta_parts, axis=0),
        np.concatenate(x_parts, axis=0),
        np.concatenate(feat_parts, axis=0),
        np.concatenate(pop_parts, axis=0),
        row_drugs,
        row_pops,
    )


# ---------------------------------------------------------------------------
# Continuous hierarchical (P4) — (body_weight, age) conditioning
# ---------------------------------------------------------------------------


def pack_observation_continuous(
    log10_cmax: np.ndarray,
    drug_features: np.ndarray,
    log_bw_norm: float | np.ndarray,
    log_age_norm: float | np.ndarray,
) -> np.ndarray:
    """Pack 15D observation for continuous hierarchical NPE.

    Layout: [log10_cmax(1), drug_features(12), log_bw_norm(1), log_age_norm(1)]
    """
    lc = np.asarray(log10_cmax, dtype=np.float64)
    if lc.ndim == 1:
        lc = lc[:, None]
    n = lc.shape[0]

    df = np.asarray(drug_features, dtype=np.float64)
    if df.ndim == 1:
        df = np.tile(df[None, :], (n, 1))

    bw = np.broadcast_to(np.asarray(log_bw_norm, dtype=np.float64), (n,))[:, None]
    ag = np.broadcast_to(np.asarray(log_age_norm, dtype=np.float64), (n,))[:, None]

    return np.concatenate([lc, df, bw, ag], axis=1)


def stack_training_pairs_continuous(
    theta_per_drug: dict[str, np.ndarray],
    logcmax_per_drug: dict[str, np.ndarray],
    features_per_drug: dict[str, np.ndarray],
    bw_per_drug: dict[str, np.ndarray],
    age_per_drug: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Stack per-drug arrays into flat training data for continuous hierarchical.

    Returns:
        theta_all:  (N, 3)
        x_all:      (N, 1) — log10(cmax)
        feat_all:   (N, 12)
        bw_all:     (N,) — body_weight_kg
        age_all:    (N,) — age_years
        names:      list[str] of length N
    """
    names_sorted = sorted(theta_per_drug.keys())
    theta_parts: list[np.ndarray] = []
    x_parts: list[np.ndarray] = []
    feat_parts: list[np.ndarray] = []
    bw_parts: list[np.ndarray] = []
    age_parts: list[np.ndarray] = []
    row_names: list[str] = []

    for name in names_sorted:
        theta_i = np.asarray(theta_per_drug[name], dtype=np.float64)
        x_i = np.asarray(logcmax_per_drug[name], dtype=np.float64)
        if x_i.ndim == 1:
            x_i = x_i[:, None]
        bw_i = np.asarray(bw_per_drug[name], dtype=np.float64)
        age_i = np.asarray(age_per_drug[name], dtype=np.float64)

        finite_mask = np.isfinite(x_i[:, 0])
        n_valid = int(finite_mask.sum())
        if n_valid == 0:
            continue

        theta_parts.append(theta_i[finite_mask])
        x_parts.append(x_i[finite_mask])
        feat_parts.append(np.tile(features_per_drug[name][None, :], (n_valid, 1)))
        bw_parts.append(bw_i[finite_mask])
        age_parts.append(age_i[finite_mask])
        row_names.extend([name] * n_valid)

    if not theta_parts:
        raise RuntimeError("No finite simulations across any drug")

    return (
        np.concatenate(theta_parts, axis=0),
        np.concatenate(x_parts, axis=0),
        np.concatenate(feat_parts, axis=0),
        np.concatenate(bw_parts, axis=0),
        np.concatenate(age_parts, axis=0),
        row_names,
    )
