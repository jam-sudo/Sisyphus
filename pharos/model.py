"""Pharos: End-to-End PK Prediction with Mixture of Experts.

Architecture:
  SMILES → MPNN Encoder → z (256-dim)
  z → Gating → π (K expert weights)
  z → K Expert Heads → (CL, V, ka) per expert
  1-comp oral PK backbone → Cmax, AUC, t½ per expert
  Mixture: weighted sum → final predictions

Bypasses IVIVE chain entirely. No fup, CLint, Peff → CLh calculation.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_add_pool


# ═══════════════════════════════════════════════════════════════════════════
# Layer 1: Molecular Encoder (GCN-based MPNN)
# ═══════════════════════════════════════════════════════════════════════════
class MolecularEncoder(nn.Module):
    """Graph neural network encoder: molecular graph → latent vector z."""

    def __init__(self, node_dim: int = 78, hidden_dim: int = 256,
                 n_layers: int = 4, dropout: float = 0.2):
        super().__init__()
        self.node_embed = nn.Linear(node_dim, hidden_dim)
        self.convs = nn.ModuleList([
            GCNConv(hidden_dim, hidden_dim) for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch):
        h = self.node_embed(x)
        for conv, norm in zip(self.convs, self.norms):
            h_new = conv(h, edge_index)
            h_new = norm(h_new)
            h_new = F.relu(h_new)
            h_new = self.dropout(h_new)
            h = h + h_new  # residual connection
        z = global_mean_pool(h, batch)  # (batch_size, hidden_dim)
        return z


# ═══════════════════════════════════════════════════════════════════════════
# Morgan FP Encoder (baseline comparison)
# ═══════════════════════════════════════════════════════════════════════════
class MorganEncoder(nn.Module):
    """Morgan fingerprint encoder: FP vector → latent z."""

    def __init__(self, fp_dim: int = 2048, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(fp_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, fp):
        return self.net(fp)


# ═══════════════════════════════════════════════════════════════════════════
# Layer 2: PK Regime Gating (MoE)
# ═══════════════════════════════════════════════════════════════════════════
class RegimeGating(nn.Module):
    """Mixture of Experts gating network: z → expert weights π."""

    def __init__(self, z_dim: int = 256, n_experts: int = 3, dropout: float = 0.2):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(z_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_experts),
        )

    def forward(self, z):
        logits = self.gate(z)
        pi = F.softmax(logits, dim=-1)
        return pi


# ═══════════════════════════════════════════════════════════════════════════
# Layer 3: Expert PK Predictor (1-comp oral backbone)
# ═══════════════════════════════════════════════════════════════════════════
class PKExpert(nn.Module):
    """Single expert: z → (CL, V, ka) → 1-comp oral PK endpoints."""

    def __init__(self, z_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3),  # log_CL (mL/min/kg), log_V (L/kg), log_ka (/h)
        )

    def forward(self, z, dose, bw):
        """
        Args:
            z: (batch, z_dim) molecular embedding
            dose: (batch,) dose in mg
            bw: (batch,) body weight in kg

        Returns:
            cmax, auc, thalf: (batch,) PK endpoints
        """
        log_params = self.head(z)

        # PK parameters (per-kg, then scale by body weight)
        CL_ml_min_kg = torch.exp(log_params[:, 0])  # mL/min/kg
        V_L_kg = torch.exp(log_params[:, 1])          # L/kg
        ka = torch.exp(log_params[:, 2])               # /h

        CL_Lh = CL_ml_min_kg * bw * 60.0 / 1000.0  # mL/min/kg → L/h
        V = V_L_kg * bw                               # L/kg → L

        ke = CL_Lh / V  # /h

        # 1-comp oral: handle ka ≈ ke singularity
        diff = ka - ke
        safe = torch.abs(diff) > 1e-4

        # tmax = ln(ka/ke) / (ka - ke)
        tmax = torch.where(
            safe,
            torch.log(torch.clamp(ka / ke, min=1e-10)) / (diff + 1e-10),
            1.0 / (ka + 1e-10),  # L'Hôpital limit
        )
        tmax = torch.clamp(tmax, min=0.01, max=100.0)

        # Cmax = (Dose × F × ka) / (V × (ka - ke)) × (e^(-ke×tmax) - e^(-ka×tmax))
        # F is implicitly absorbed into CL (this is CL/F for oral)
        cmax = torch.where(
            safe,
            (dose * ka / (V * (diff + 1e-10))) *
            (torch.exp(-ke * tmax) - torch.exp(-ka * tmax)),
            dose / (V * math.e),  # limit when ka→ke
        )
        cmax = torch.clamp(cmax, min=1e-8)

        auc = dose / (CL_Lh + 1e-10)  # mg / (L/h) = mg·h/L
        thalf = 0.693 / (ke + 1e-10)   # h

        return cmax, auc, thalf


# ═══════════════════════════════════════════════════════════════════════════
# Layer 4: Full Pharos Model
# ═══════════════════════════════════════════════════════════════════════════
class PharosModel(nn.Module):
    """End-to-end Pharos: SMILES → PK endpoints via MoE."""

    def __init__(self, node_dim: int = 78, hidden_dim: int = 256,
                 n_layers: int = 4, n_experts: int = 3, dropout: float = 0.2,
                 use_morgan: bool = False, morgan_dim: int = 2048):
        super().__init__()
        self.use_morgan = use_morgan
        if use_morgan:
            self.encoder = MorganEncoder(morgan_dim, hidden_dim, dropout)
        else:
            self.encoder = MolecularEncoder(node_dim, hidden_dim, n_layers, dropout)
        self.gating = RegimeGating(hidden_dim, n_experts, dropout)
        self.experts = nn.ModuleList([
            PKExpert(hidden_dim, dropout) for _ in range(n_experts)
        ])
        self.n_experts = n_experts

        # Auxiliary task heads (encoder regularization)
        self.aux_fup = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1))
        self.aux_clint = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1))
        self.aux_peff = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, batch_data, dose, bw):
        # Encode
        if self.use_morgan:
            z = self.encoder(batch_data)  # batch_data = FP tensor
        else:
            z = self.encoder(batch_data.x, batch_data.edge_index, batch_data.batch)

        # Gating
        pi = self.gating(z)  # (batch, K)

        # Expert predictions
        cmax_experts = []
        auc_experts = []
        thalf_experts = []
        for expert in self.experts:
            cmax_k, auc_k, thalf_k = expert(z, dose, bw)
            cmax_experts.append(cmax_k)
            auc_experts.append(auc_k)
            thalf_experts.append(thalf_k)

        cmax_stack = torch.stack(cmax_experts, dim=1)   # (batch, K)
        auc_stack = torch.stack(auc_experts, dim=1)
        thalf_stack = torch.stack(thalf_experts, dim=1)

        # Mixture
        cmax_pred = (pi * cmax_stack).sum(dim=1)
        auc_pred = (pi * auc_stack).sum(dim=1)
        thalf_pred = (pi * thalf_stack).sum(dim=1)

        # Mixture variance (between-expert)
        cmax_var = (pi * (cmax_stack - cmax_pred.unsqueeze(1)) ** 2).sum(dim=1)

        # Auxiliary predictions
        aux_fup_pred = torch.sigmoid(self.aux_fup(z).squeeze(-1))  # (0,1)
        aux_clint_pred = torch.exp(self.aux_clint(z).squeeze(-1))  # log-space → positive
        aux_peff_pred = torch.exp(self.aux_peff(z).squeeze(-1))

        return {
            "cmax": cmax_pred,
            "auc": auc_pred,
            "thalf": thalf_pred,
            "cmax_var": cmax_var,
            "pi": pi,
            "expert_cmax": cmax_stack,
            "aux_fup": aux_fup_pred,
            "aux_clint": aux_clint_pred,
            "aux_peff": aux_peff_pred,
        }
