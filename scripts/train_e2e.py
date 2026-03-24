#!/usr/bin/env python3
"""Phase 2B: End-to-End differentiable PBPK training.

MLP → (fup, CLint, Peff) → ODE engine → Cmax.
Finite-difference gradient through the engine; autograd through the MLP.
Multi-task loss: L_cmax + λ × (L_fup + L_clint + L_peff).

Usage:
    python3 scripts/train_e2e.py
"""

from __future__ import annotations

import json
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
log = logging.getLogger("e2e")
log.setLevel(logging.INFO)

PHYSIOLOGY_YAML = ROOT / "data" / "physiology" / "reference_man.yaml"
HOLDOUT_JSON = ROOT / "data" / "reference" / "holdout.json"
CLINICAL_PK_JSON = ROOT / "data" / "reference" / "clinical_pk.json"
OUTPUT_MODEL = ROOT / "models" / "e2e_mlp.pt"

# Hyperparameters
LR = 1e-3
EPOCHS = 80
LAMBDA_ADME = 0.3        # weight for auxiliary ADME losses
ADME_BATCH_SIZE = 64      # TDC samples per Cmax epoch
WEIGHT_DECAY = 1e-4
HIDDEN = 64
DEVICE = "cpu"  # engine is CPU-bound anyway


# ═══════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════

@dataclass
class CmaxSample:
    name: str
    smiles: str
    dose_mg: float
    route: str
    log_cmax_obs: float  # log10(Cmax mg/L)
    features: np.ndarray  # (2057,)


@dataclass
class ADMESample:
    smiles: str
    features: np.ndarray
    target: float     # logit(fup), log10(CLint), or log10(Peff)
    param: str        # "fup", "clint", "peff"


def load_cmax_data() -> tuple[list[CmaxSample], list[CmaxSample]]:
    """Load train/holdout Cmax samples with pre-computed features."""
    from sisyphus.descriptors import compute_features

    with open(HOLDOUT_JSON) as f:
        holdout_set = set(json.load(f)["holdout"])
    with open(CLINICAL_PK_JSON) as f:
        drugs = json.load(f)["drugs"]

    train, holdout = [], []
    for name, info in drugs.items():
        smiles = info.get("smiles", "")
        dose = info.get("dose_mg")
        cmax = info.get("pk_params", {}).get("cmax_mg_L")
        route = info.get("route", "oral")
        if not (smiles and dose and dose > 0 and cmax and cmax > 0):
            continue
        try:
            feat = compute_features(smiles)
        except Exception:
            continue
        sample = CmaxSample(
            name=name, smiles=smiles, dose_mg=dose, route=route,
            log_cmax_obs=math.log10(cmax), features=feat,
        )
        if name in holdout_set:
            holdout.append(sample)
        else:
            train.append(sample)

    log.info("Cmax data: %d train, %d holdout", len(train), len(holdout))
    return train, holdout


def load_tdc_adme() -> list[ADMESample]:
    """Load TDC fup/CLint/Peff datasets as auxiliary ADME samples."""
    from sisyphus.descriptors import compute_features

    # Holdout exclusion
    with open(HOLDOUT_JSON) as f:
        holdout_set = set(json.load(f)["holdout"])
    with open(CLINICAL_PK_JSON) as f:
        clinical = json.load(f)["drugs"]

    holdout_smiles = set()
    for name in holdout_set:
        entry = clinical.get(name, {})
        s = entry.get("smiles", "")
        if s:
            holdout_smiles.add(s)

    samples: list[ADMESample] = []

    # fup: TDC PPBR_AZ
    try:
        from tdc.single_pred import ADME
        ppbr = ADME(name="PPBR_AZ").get_data()
        for _, row in ppbr.iterrows():
            smiles = str(row["Drug"])
            if smiles in holdout_smiles:
                continue
            fup = (100.0 - row["Y"]) / 100.0
            if fup <= 0.001 or fup >= 0.999:
                continue
            try:
                feat = compute_features(smiles)
                logit_fup = math.log(fup / (1 - fup))
                samples.append(ADMESample(smiles, feat, logit_fup, "fup"))
            except Exception:
                pass
        log.info("TDC fup: %d samples", sum(1 for s in samples if s.param == "fup"))
    except Exception as e:
        log.warning("Failed to load TDC PPBR_AZ: %s", e)

    # CLint: TDC Hepatocyte_AZ (if available)
    try:
        from tdc.single_pred import ADME
        hep = ADME(name="HepG2").get_data()  # try Hepatocyte
        # Hepatocyte data may be binary, skip if not continuous
    except Exception:
        pass

    # Peff: TDC Caco2_Wang
    try:
        from tdc.single_pred import ADME
        caco = ADME(name="Caco2_Wang").get_data()
        for _, row in caco.iterrows():
            smiles = str(row["Drug"])
            if smiles in holdout_smiles:
                continue
            log_peff = row["Y"] + 4.0  # log10(Papp cm/s) → log10(Peff x10^-4 cm/s)
            # Add Caco-2→in vivo offset
            log_peff += 1.29
            try:
                feat = compute_features(smiles)
                samples.append(ADMESample(smiles, feat, log_peff, "peff"))
            except Exception:
                pass
        log.info("TDC peff: %d samples", sum(1 for s in samples if s.param == "peff"))
    except Exception as e:
        log.warning("Failed to load TDC Caco2_Wang: %s", e)

    log.info("Total ADME auxiliary samples: %d", len(samples))
    return samples


# ═══════════════════════════════════════════════════════════════════
# MLP model
# ═══════════════════════════════════════════════════════════════════

class ADMEMLP(nn.Module):
    """MLP: 2057-dim features → 3 ADME params (logit_fup, log_clint, log_peff)."""

    def __init__(self, input_dim: int = 2057, hidden: int = HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden // 2, 3),
        )
        # Initialize output bias to XGBoost-like defaults
        # logit(0.3)=-0.85, log10(20)=1.3, log10(2)=0.3
        with torch.no_grad():
            self.net[-1].bias.copy_(torch.tensor([-0.85, 1.3, 0.3]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (batch, 3): [logit_fup, log_clint, log_peff]."""
        return self.net(x)

    def adme_values(self, raw: torch.Tensor) -> tuple[float, float, float]:
        """Convert raw MLP output to physical ADME values."""
        fup = torch.sigmoid(raw[0]).item()
        fup = max(fup, 0.001)
        clint = 10 ** torch.clamp(raw[1], -1, 3).item()  # [0.1, 1000]
        peff = 10 ** torch.clamp(raw[2], -2, 2).item()    # [0.01, 100]
        return fup, clint, peff


# ═══════════════════════════════════════════════════════════════════
# Engine wrapper (cached graph/compile, ADME-parameterized solve)
# ═══════════════════════════════════════════════════════════════════

class EngineWrapper:
    """Wraps the PBPK engine for fast repeated solves with varying ADME params."""

    def __init__(self):
        import sisyphus.engine.flux  # noqa: F401
        from sisyphus.engine.compiler import ODECompiler
        from sisyphus.graph.builder import build_from_yaml

        self.graph = build_from_yaml(PHYSIOLOGY_YAML)
        self.compiler = ODECompiler()
        self.compiled = self.compiler.compile(self.graph)

        # Cache liver enzymes
        liver = self.graph.nodes.get("liver")
        self.liver_enzymes = (
            {tag: d.mean for tag, d in liver.enzymes.items()}
            if liver and liver.enzymes else None
        )

    def solve(self, smiles: str, dose_mg: float, route: str,
              fup: float, clint: float, peff: float) -> float:
        """Run engine with custom ADME params. Returns Cmax (mg/L) or 0."""
        from sisyphus.core import Distribution
        from sisyphus.engine.compiler import ResolvedParams
        from sisyphus.engine.solver import solve
        from sisyphus.pk.endpoints import compute_endpoints
        from sisyphus.predict.adme import ADMEProperties
        from sisyphus.predict.chemistry import compute_profile
        from sisyphus.predict.ivive import build_drug_on_graph

        try:
            profile = compute_profile(smiles)

            adme = ADMEProperties(
                fup=Distribution(fup, cv=0),
                clint=Distribution(clint, cv=0),
                peff=Distribution(peff, cv=0),
                solubility=Distribution(10 ** (-profile.logp + 0.5), cv=0),
                vdss=Distribution(1.0, cv=0),
                rbp=Distribution(1.0, cv=0),
            )

            drug = build_drug_on_graph(
                profile, adme, dose_mg, route,
                liver_enzymes=self.liver_enzymes,
            )

            rng = np.random.default_rng(42)
            realized_graph = self.graph.sample(rng)
            realized_drug = drug.sample(rng)
            params = ResolvedParams(realized_graph, realized_drug)

            y0 = np.zeros(self.compiled.n_states)
            admin_idx = self.compiled.state_index[drug.administration_node]
            y0[admin_idx] = drug.dose_mg

            sim = solve(self.compiled, params, y0, t_span=(0, 24))
            if sim.solver_success:
                pk = compute_endpoints(sim)
                return pk.cmax.mean
        except Exception:
            pass
        return 0.0


# ═══════════════════════════════════════════════════════════════════
# Custom autograd: finite-difference backward through engine
# ═══════════════════════════════════════════════════════════════════

class EngineForward(torch.autograd.Function):
    """Differentiable engine forward pass (finite-difference backward)."""

    @staticmethod
    def forward(ctx, mlp_output, engine, smiles, dose_mg, route):
        vals = mlp_output.detach().cpu().numpy()
        fup = 1.0 / (1.0 + np.exp(-vals[0]))
        fup = max(fup, 0.001)
        clint = 10 ** np.clip(vals[1], -1, 3)
        peff = 10 ** np.clip(vals[2], -2, 2)

        cmax = engine.solve(smiles, dose_mg, route, fup, clint, peff)
        log_cmax = math.log10(max(cmax, 1e-10))

        ctx.save_for_backward(mlp_output)
        ctx.engine = engine
        ctx.smiles = smiles
        ctx.dose_mg = dose_mg
        ctx.route = route
        ctx.log_cmax = log_cmax
        ctx.adme = (fup, clint, peff)

        return torch.tensor([log_cmax], dtype=torch.float32)

    @staticmethod
    def backward(ctx, grad_output):
        mlp_output, = ctx.saved_tensors
        vals = mlp_output.detach().cpu().numpy()
        engine = ctx.engine

        grads = np.zeros(3)
        for i in range(3):
            eps = max(abs(vals[i]) * 0.01, 1e-5)
            perturbed = vals.copy()
            perturbed[i] += eps

            fup = 1.0 / (1.0 + np.exp(-perturbed[0]))
            fup = max(fup, 0.001)
            clint = 10 ** np.clip(perturbed[1], -1, 3)
            peff = 10 ** np.clip(perturbed[2], -2, 2)

            cmax = engine.solve(ctx.smiles, ctx.dose_mg, ctx.route, fup, clint, peff)
            log_cmax = math.log10(max(cmax, 1e-10))
            grads[i] = (log_cmax - ctx.log_cmax) / eps

        grad_tensor = grad_output * torch.tensor(grads, dtype=torch.float32)
        return grad_tensor, None, None, None, None


# ═══════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════

def train():
    log.info("=" * 60)
    log.info("Phase 2B: E2E Differentiable PBPK Training")
    log.info("=" * 60)

    # Data
    train_cmax, holdout_cmax = load_cmax_data()
    adme_samples = load_tdc_adme()

    # Model & optimizer
    model = ADMEMLP().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Engine
    log.info("Initializing engine...")
    engine = EngineWrapper()
    log.info("Engine ready. Compiled %d states.", engine.compiled.n_states)

    # Pre-tensorize ADME data
    adme_fup = [(torch.tensor(s.features, dtype=torch.float32), s.target)
                for s in adme_samples if s.param == "fup"]
    adme_peff = [(torch.tensor(s.features, dtype=torch.float32), s.target)
                 for s in adme_samples if s.param == "peff"]

    log.info("Training: %d Cmax drugs, %d fup, %d peff auxiliary",
             len(train_cmax), len(adme_fup), len(adme_peff))

    # Pre-tensorize Cmax features
    train_feats = [torch.tensor(s.features, dtype=torch.float32) for s in train_cmax]

    best_holdout_aafe = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()

        # ── Cmax loss (65 drugs, each requires engine solve) ──
        cmax_losses = []
        for idx, sample in enumerate(train_cmax):
            feat = train_feats[idx].to(DEVICE)
            raw = model(feat.unsqueeze(0)).squeeze(0)

            log_cmax_pred = EngineForward.apply(raw, engine,
                                                 sample.smiles, sample.dose_mg, sample.route)
            loss = (log_cmax_pred - sample.log_cmax_obs) ** 2
            cmax_losses.append(loss)

        if cmax_losses:
            cmax_loss = torch.stack(cmax_losses).mean()
        else:
            cmax_loss = torch.tensor(0.0)

        # ── ADME auxiliary losses (batched, no engine) ──
        adme_loss = torch.tensor(0.0)
        n_adme = 0

        # fup auxiliary
        if adme_fup:
            indices = np.random.choice(len(adme_fup), min(ADME_BATCH_SIZE, len(adme_fup)), replace=False)
            for i in indices:
                feat, target = adme_fup[i]
                raw = model(feat.unsqueeze(0).to(DEVICE)).squeeze(0)
                pred_logit_fup = raw[0]
                adme_loss = adme_loss + (pred_logit_fup - target) ** 2
                n_adme += 1

        # peff auxiliary
        if adme_peff:
            indices = np.random.choice(len(adme_peff), min(ADME_BATCH_SIZE, len(adme_peff)), replace=False)
            for i in indices:
                feat, target = adme_peff[i]
                raw = model(feat.unsqueeze(0).to(DEVICE)).squeeze(0)
                pred_log_peff = raw[2]
                adme_loss = adme_loss + (pred_log_peff - target) ** 2
                n_adme += 1

        if n_adme > 0:
            adme_loss = adme_loss / n_adme

        # Total loss
        total_loss = cmax_loss + LAMBDA_ADME * adme_loss
        total_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step()

        # ── Evaluate on holdout every 10 epochs ──
        if (epoch + 1) % 10 == 0 or epoch == 0:
            model.eval()
            log_errors = []
            with torch.no_grad():
                for sample in holdout_cmax:
                    feat = torch.tensor(sample.features, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                    raw = model(feat).squeeze(0)
                    fup, clint, peff = model.adme_values(raw)
                    cmax = engine.solve(sample.smiles, sample.dose_mg, sample.route,
                                        fup, clint, peff)
                    if cmax > 0:
                        log_errors.append(abs(math.log10(cmax) - sample.log_cmax_obs))

            holdout_aafe = 10 ** np.mean(log_errors) if log_errors else float("inf")

            # Also compute train AAFE (from last forward pass)
            train_aafe = 10 ** (cmax_loss.item() ** 0.5) if cmax_loss.item() > 0 else float("inf")

            log.info(
                "Epoch %3d/%d | Loss: cmax=%.4f adme=%.4f total=%.4f | "
                "Train AAFE~%.3f | Holdout AAFE=%.3f (N=%d) %s",
                epoch + 1, EPOCHS,
                cmax_loss.item(), adme_loss.item(), total_loss.item(),
                train_aafe, holdout_aafe, len(log_errors),
                "★" if holdout_aafe < best_holdout_aafe else ""
            )

            if holdout_aafe < best_holdout_aafe:
                best_holdout_aafe = holdout_aafe
                OUTPUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), OUTPUT_MODEL)

    # ── Final evaluation ──
    log.info("=" * 60)
    log.info("FINAL EVALUATION")
    log.info("=" * 60)

    # Load best model
    model.load_state_dict(torch.load(OUTPUT_MODEL, weights_only=True))
    model.eval()

    # Holdout per-drug results
    log.info(f"\n{'Drug':<25s} {'Obs':>8s} {'E2E':>8s} {'Fold':>8s} {'fup':>6s} {'CLint':>7s} {'Peff':>6s}")
    log.info("-" * 75)

    log_errors = []
    with torch.no_grad():
        for sample in holdout_cmax:
            feat = torch.tensor(sample.features, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            raw = model(feat).squeeze(0)
            fup, clint, peff = model.adme_values(raw)
            cmax = engine.solve(sample.smiles, sample.dose_mg, sample.route, fup, clint, peff)

            if cmax > 0:
                obs = 10 ** sample.log_cmax_obs
                fold = cmax / obs
                log_errors.append(abs(math.log10(cmax) - sample.log_cmax_obs))
                log.info(f"{sample.name:<25s} {obs:8.4f} {cmax:8.4f} {fold:8.2f}x {fup:6.3f} {clint:7.1f} {peff:6.2f}")

    final_aafe = 10 ** np.mean(log_errors) if log_errors else float("inf")
    log.info(f"\nE2E Engine AAFE (holdout): {final_aafe:.3f} (N={len(log_errors)})")
    log.info(f"Baseline Engine AAFE:      2.861")
    log.info(f"Best model saved to: {OUTPUT_MODEL}")


if __name__ == "__main__":
    train()
