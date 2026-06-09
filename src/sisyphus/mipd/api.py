"""Public MIPD API: SMILES + dose + sparse measured data -> posterior PK.

``predict_posterior`` wires the existing a-priori ``predict()`` to the SIR
inference core. It calls the engine twice (once a-priori for Cmax0/AUC0, once
with a measured-F probe to recover the engine's emergent F_engine), then runs
SIR over the bioavailability latent given the supplied observations. With no
observations it returns the a-priori engine prediction as a (prior) posterior,
so the SMILES-only path is unchanged.
"""
from __future__ import annotations

import re

import numpy as np

from sisyphus.mipd.amortizer import SIRAmortizer
from sisyphus.mipd.core import APrioriPK, PosteriorPK

_F_ENGINE_RE = re.compile(r"f_engine=([0-9.]+)")
_F_PROBE = 0.5  # arbitrary in-range F used only to recover F_engine via the routing


def _recover_f_engine(probe_result, cmax0: float) -> float:
    """Recover the engine's emergent F_engine from a measured-F probe call.

    Prefers the value the routing surfaces in its warning (clamp-proof); falls
    back to the exact linear arithmetic ``F_engine = Cmax0 * F_probe / Cmax_scaled``.
    """
    for w in probe_result.warnings or []:
        m = _F_ENGINE_RE.search(w)
        if m:
            return float(m.group(1))
    scaled = probe_result.engine_pk.cmax.mean
    if scaled <= 0:
        raise ValueError("measured-F probe produced non-positive Cmax; cannot recover F_engine")
    return cmax0 * _F_PROBE / scaled


def predict_posterior(
    smiles: str,
    dose_mg: float,
    observations=(),
    *,
    route: str = "oral",
    prior_cv: float = 1.0,
    n_samples: int = 20000,
    seed: int = 0,
    **predict_kwargs,
) -> PosteriorPK:
    """Posterior PK for ``smiles`` at ``dose_mg`` given ``observations``.

    Args:
        observations: sequence of MeasuredF / MeasuredCmax / MeasuredAUC. Empty
            (default) returns the a-priori engine prediction as a prior posterior.
        prior_cv: width of the F prior (wide by default — the engine F is the
            dominant structural error).
        seed: RNG seed for reproducible SIR.
        predict_kwargs: forwarded to ``pipeline.predict.predict`` (e.g. kp_method).
    """
    from sisyphus.pipeline.predict import predict
    from sisyphus.predict.adme import MeasuredADMEInput

    ap = predict(smiles, dose_mg, route=route, **predict_kwargs)
    if ap.engine_pk is None or ap.engine_pk.cmax is None:
        raise ValueError("engine produced no Cmax for this input; cannot build a posterior")
    cmax0 = ap.engine_pk.cmax.mean
    auc0 = ap.engine_pk.auc_0t.mean if ap.engine_pk.auc_0t is not None else 0.0

    probe = predict(
        smiles, dose_mg, route=route,
        measured_adme=MeasuredADMEInput(f_bioavail=_F_PROBE),
        **predict_kwargs,
    )
    f_engine = min(max(_recover_f_engine(probe, cmax0), 1e-4), 1.0)

    apriori = APrioriPK(cmax0=cmax0, auc0=auc0, f_engine=f_engine)
    rng = np.random.default_rng(seed)
    return SIRAmortizer(prior_cv=prior_cv, n_samples=n_samples).posterior(
        apriori, list(observations), rng=rng
    )
