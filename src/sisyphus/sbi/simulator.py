"""Engine-backed simulator for SBI.

Wraps the full scipy ODE engine as a simulator that maps
    theta = (log10_clint_shift, logit_fup, log10_peff_shift) → log10(Cmax)

Usage::

    sim = EngineSimulator.for_drug(smiles="...", dose_mg=30.0, route="oral")
    log10_cmax = sim.simulate_single(theta=np.array([0.0, 0.0, 0.0]), seed=0)
    batch = sim.simulate_batch(thetas, seed=42)  # (N, 1)

The simulator is deterministic for a given (theta, seed) pair so SBI
training is reproducible.  Distributional realization of the graph and
drug uses the seed; theta then *overrides* the sampled ADME values.

Phase 2.0.5: theta[1] is now logit(fup), not absolute fup. The
``apply_theta_to_drug`` function applies sigmoid to recover the physical
fup value.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from sisyphus.core import Distribution, DrugOnGraph
from sisyphus.engine.compiler import CompiledODE, ODECompiler, ResolvedParams
from sisyphus.engine.solver import solve
from sisyphus.graph.body import BodyGraph
from sisyphus.graph.builder import build_from_yaml

logger = logging.getLogger(__name__)

_DEFAULT_PHYSIOLOGY = Path("data/physiology/reference_man.yaml")
_DEFAULT_OBS_NODE = "venous_blood"
_DEFAULT_T_SPAN = (0.0, 48.0)
# Assay-scale Gaussian noise added to log10(Cmax) (σ = log10(1 + CV))
_DEFAULT_OBS_SIGMA_LOG10 = 0.0414  # ≈ 10% CV


def apply_theta_to_drug(
    drug: DrugOnGraph,
    theta: np.ndarray,
    nominal_peff: float,
) -> DrugOnGraph:
    """Return a new DrugOnGraph with ADME overridden by theta.

    theta[0] — log10(CLint_actual / CLint_nominal): multiply every
               enzyme_affinity.mean by 10**theta[0], keep CV.
    theta[1] — logit(fup): replace fup distribution mean by
               sigmoid(theta[1]), keep CV. Phase 2.0.5: the prior
               samples uniformly in logit space for better low-fup
               coverage.
    theta[2] — log10(Peff_actual / Peff_nominal): replace Peff mean by
               10**theta[2] * nominal_peff, keep CV.

    This mutates only ADME fields that actually live in the surrogate's
    12D feature space and that the engine uses for clearance / absorption
    / binding. Kp overrides, solubility, renal_cl remain at nominal.
    """
    from dataclasses import replace

    from sisyphus.sbi.priors import _sigmoid

    clint_scale = float(10 ** theta[0])
    fup_val = float(np.clip(_sigmoid(theta[1]), 1e-4, 1.0))
    peff_val = float(10 ** theta[2]) * nominal_peff

    new_enzyme_affinity = {
        tag: Distribution(mean=dist.mean * clint_scale, cv=dist.cv, dist_type=dist.dist_type)
        for tag, dist in drug.enzyme_affinity.items()
    }
    new_fup = Distribution(mean=fup_val, cv=drug.fup.cv, dist_type=drug.fup.dist_type)
    new_peff = Distribution(mean=peff_val, cv=drug.peff.cv, dist_type=drug.peff.dist_type)

    return replace(
        drug,
        fup=new_fup,
        peff=new_peff,
        enzyme_affinity=new_enzyme_affinity,
    )


@dataclass
class EngineSimulator:
    """Wrap the full scipy engine as an SBI simulator for one drug.

    Build once via ``for_drug`` and reuse across many theta evaluations.
    """

    graph: BodyGraph
    compiled: CompiledODE
    nominal_drug: DrugOnGraph
    nominal_peff: float
    obs_node: str = _DEFAULT_OBS_NODE
    t_span: tuple[float, float] = _DEFAULT_T_SPAN
    obs_sigma_log10: float = _DEFAULT_OBS_SIGMA_LOG10
    n_failures: int = field(default=0, init=False)

    @classmethod
    def for_drug(
        cls,
        smiles: str,
        dose_mg: float,
        route: str = "oral",
        physiology_yaml: Path = _DEFAULT_PHYSIOLOGY,
        obs_sigma_log10: float = _DEFAULT_OBS_SIGMA_LOG10,
    ) -> "EngineSimulator":
        """Construct a simulator for a specific drug from SMILES + dose."""
        import sisyphus.engine.flux  # noqa: F401 — register flux specs
        from sisyphus.predict.adme import predict_adme
        from sisyphus.predict.chemistry import compute_profile
        from sisyphus.predict.ivive import build_drug_on_graph

        graph = build_from_yaml(physiology_yaml)
        compiled = ODECompiler().compile(graph)
        profile = compute_profile(smiles)
        adme = predict_adme(profile)
        liver_enzymes = None
        if "liver" in graph.nodes and graph.nodes["liver"].enzymes:
            liver_enzymes = {
                tag: dist.mean for tag, dist in graph.nodes["liver"].enzymes.items()
            }
        drug = build_drug_on_graph(profile, adme, dose_mg, route, liver_enzymes=liver_enzymes)
        return cls(
            graph=graph,
            compiled=compiled,
            nominal_drug=drug,
            nominal_peff=float(adme.peff.mean),
            obs_sigma_log10=obs_sigma_log10,
        )

    def simulate_single(self, theta: np.ndarray, seed: int) -> float:
        """Simulate log10(Cmax) for one theta with Gaussian assay noise.

        Returns ``-np.inf`` if the solver fails or gives non-positive Cmax.
        """
        overridden = apply_theta_to_drug(self.nominal_drug, theta, self.nominal_peff)
        rng = np.random.default_rng(seed)
        rg = self.graph.sample(rng)
        rd = overridden.sample(rng)
        params = ResolvedParams(rg, rd)
        y0 = np.zeros(self.compiled.n_states)
        y0[self.compiled.state_index[overridden.administration_node]] = overridden.dose_mg
        try:
            result = solve(self.compiled, params, y0, t_span=self.t_span)
        except Exception as exc:  # defensive — ODE solver can raise
            logger.debug("Solver exception: %s", exc)
            self.n_failures += 1
            return float("-inf")
        if not result.solver_success or self.obs_node not in result.concentrations:
            self.n_failures += 1
            return float("-inf")
        cmax = float(result.concentrations[self.obs_node].max())
        if cmax <= 0 or not np.isfinite(cmax):
            self.n_failures += 1
            return float("-inf")
        noise = rng.normal(0.0, self.obs_sigma_log10)
        return float(np.log10(cmax) + noise)

    def simulate_batch(
        self,
        thetas: np.ndarray,
        seed: int = 0,
        progress: Callable[[int], None] | None = None,
    ) -> np.ndarray:
        """Simulate a batch of thetas sequentially.

        Args:
            thetas: shape (N, 3)
            seed: base RNG seed; each sample uses ``seed + i``.
            progress: optional callback invoked every 100 samples as
                      ``progress(n_done)``.

        Returns:
            log10(Cmax) array of shape (N, 1). Failed solves become NaN.
        """
        n = thetas.shape[0]
        out = np.full((n, 1), np.nan, dtype=np.float64)
        for i in range(n):
            val = self.simulate_single(thetas[i], seed + i)
            out[i, 0] = val if np.isfinite(val) else np.nan
            if progress is not None and (i + 1) % 100 == 0:
                progress(i + 1)
        return out

    # ---- sbi interface ----
    def torch_simulator(self) -> Callable[[torch.Tensor], torch.Tensor]:
        """Return a simulator function suitable for ``sbi`` inference.

        Accepts a (B, 3) torch tensor of thetas, returns (B, 1) tensor of
        log10(Cmax).  Failed solves come back as very negative sentinel
        values so the density estimator rejects them naturally.
        """
        FAIL_SENTINEL = -20.0  # far below any physical log10(Cmax)

        def _sim(theta_t: torch.Tensor) -> torch.Tensor:
            thetas = theta_t.detach().cpu().numpy().astype(np.float64)
            if thetas.ndim == 1:
                thetas = thetas[None, :]
            out = self.simulate_batch(thetas, seed=int(torch.randint(0, 2**31 - 1, (1,)).item()))
            out = np.where(np.isnan(out), FAIL_SENTINEL, out)
            return torch.as_tensor(out, dtype=torch.float32)

        return _sim
