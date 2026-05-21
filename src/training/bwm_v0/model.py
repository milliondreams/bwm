"""v0 JEPA model: transformer encoder + EMA target + horizon-conditioned predictor.

Phase B replaces JEPABackbone's internals with a hybrid Mamba-3 + attention
stack (ADR-1). The I/O signature is preserved: (B, T, n_concepts) →
(B, d_model). So the rest of the training graph (predictor + loss + EMA)
stays the same.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
from torch import Tensor

from training.bwm_v0.concepts import N_CONCEPTS


class JEPABackbone(nn.Module):
    """Sequence encoder operating on per-quarter feature vectors.

    Input  : x (B, T, n_concepts), mask (B, T) with 1=real, 0=pad
    Output : (B, d_model) — sequence-level representation via masked mean-pool

    Two backbone variants share this outer shell:
      backbone_type = "transformer" (default, Stage 1.8 path) — uses
        nn.TransformerEncoder
      backbone_type = "hybrid" (Stage 1.9) — uses HybridBackbone, which
        interleaves SSMBlock and AttnBlock per ssm_ratio (see ADR-1)

    Both produce a (B, T, d_model) hidden-state tensor over the same I/O
    contract; the masked mean-pool + out_norm logic below is shared.
    """

    def __init__(
        self,
        n_concepts: int = N_CONCEPTS,
        d_model: int = 64,
        n_layers: int = 2,
        n_heads: int = 4,
        ff_dim: int = 256,
        dropout: float = 0.1,
        max_history_q: int = 32,
        backbone_type: str = "transformer",
        ssm_ratio: float = 0.75,
        d_state: int = 16,
        d_conv: int = 4,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.backbone_type = backbone_type
        self.input_proj = nn.Linear(n_concepts, d_model)
        self.pos_emb = nn.Embedding(max_history_q, d_model)

        if backbone_type == "transformer":
            # Stage 1.8 default — preserved bit-for-bit for back-compat.
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=ff_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        elif backbone_type == "hybrid":
            # Stage 1.9: lazy import to keep blocks.py from being a hard
            # dependency of every JEPABackbone instantiation.
            from training.bwm_v0.blocks import HybridBackbone

            self.encoder = HybridBackbone(
                d_model=d_model,
                n_layers=n_layers,
                n_heads=n_heads,
                ff_dim=ff_dim,
                dropout=dropout,
                ssm_ratio=ssm_ratio,
                d_state=d_state,
                d_conv=d_conv,
            )
        else:
            raise ValueError(
                f"backbone_type must be 'transformer' or 'hybrid'; got {backbone_type!r}"
            )

        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        # x: (B, T, n_concepts), mask: (B, T) with 1=real, 0=pad
        B, T, _ = x.shape
        h = self.input_proj(x)
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        h = h + self.pos_emb(positions)

        # Encoder dispatch. Both paths produce (B, T, d_model).
        if self.backbone_type == "transformer":
            # PyTorch expects True = ignore for src_key_padding_mask
            pad_mask = mask == 0
            any_real_per_sample = mask.sum(dim=1) > 0
            if not bool(any_real_per_sample.all()):
                safe_pad_mask = pad_mask.clone()
                all_pad = ~any_real_per_sample
                safe_pad_mask[all_pad, 0] = False
                pad_mask = safe_pad_mask
            h = self.encoder(h, src_key_padding_mask=pad_mask)
        else:
            # HybridBackbone's blocks each handle their own masking
            # (AttnBlock has the all-pad guard internally; SSMBlock zeros
            # B_bar at pad positions).
            h = self.encoder(h, mask)

        # Masked mean-pool over real positions.
        m = mask.float().unsqueeze(-1)  # (B, T, 1)
        denom = m.sum(dim=1).clamp(min=1.0)
        pooled = (h * m).sum(dim=1) / denom
        any_real = (mask.sum(dim=1) > 0).float().unsqueeze(-1)
        pooled = pooled * any_real
        return self.out_norm(pooled)


class JEPAPredictor(nn.Module):
    """Maps (s_t, horizon_idx) → ŝ_{t+h}.

    Conditioning: additive horizon embedding. Skip connection from s_t for
    stability when h is small (e.g., h=1, where ŝ ≈ s_t is a sensible
    initial guess).
    """

    def __init__(self, d_model: int = 64, n_horizons: int = 2, hidden: int = 128) -> None:
        super().__init__()
        self.h_emb = nn.Embedding(n_horizons, d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )

    def forward(self, s_t: Tensor, horizon_idx: Tensor) -> Tensor:
        # s_t: (B, d_model), horizon_idx: (B,) int64
        h = self.h_emb(horizon_idx)
        return s_t + self.mlp(s_t + h)


class JEPAModel(nn.Module):
    """Composite: online encoder + EMA-tracked target encoder + predictor.

    The target encoder is a deepcopy of the online encoder at init, with
    requires_grad=False on its parameters. It is updated via ema.ema_update
    outside the autograd graph, called once per training step after
    optimizer.step().
    """

    def __init__(
        self,
        n_concepts: int = N_CONCEPTS,
        d_model: int = 64,
        n_layers: int = 2,
        n_heads: int = 4,
        ff_dim: int = 256,
        dropout: float = 0.1,
        max_history_q: int = 32,
        n_horizons: int = 2,
        predictor_hidden: int = 128,
        backbone_type: str = "transformer",
        ssm_ratio: float = 0.75,
        d_state: int = 16,
        d_conv: int = 4,
    ) -> None:
        super().__init__()
        self.online = JEPABackbone(
            n_concepts=n_concepts,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            max_history_q=max_history_q,
            backbone_type=backbone_type,
            ssm_ratio=ssm_ratio,
            d_state=d_state,
            d_conv=d_conv,
        )
        # Target is a structurally-identical deepcopy at init time.
        self.target = copy.deepcopy(self.online)
        for p in self.target.parameters():
            p.requires_grad = False
        self.predictor = JEPAPredictor(
            d_model=d_model, n_horizons=n_horizons, hidden=predictor_hidden
        )

    @torch.no_grad()
    def target_encode(self, x: Tensor, mask: Tensor) -> Tensor:
        """Forward the target encoder under no_grad. Returns s_target.detach()."""
        self.target.eval()
        out = self.target(x, mask)
        self.target.train()
        return out
