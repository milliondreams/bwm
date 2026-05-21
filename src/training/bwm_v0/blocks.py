"""Backbone blocks for the v0 hybrid SSM + attention encoder (Stage 1.9).

Three components:

  - AttnBlock         standard pre-norm transformer encoder layer (refactor
                      from Stage 1.8's inline TransformerEncoderLayer body)
  - SSMBlock          Mamba-2-style selective state-space block; pure
                      PyTorch implementation for CPU compatibility. Phase B
                      swaps the inner scan for mamba-ssm's CUDA fused kernel.
  - HybridBackbone    composes N blocks chosen by ssm_ratio per the
                      pattern fixed in the Stage 1.9 plan

Each block preserves the (B, T, d_model) shape so they can be freely
interleaved. Padding is handled defensively: AttnBlock has an all-pad guard
against attention-softmax NaN; SSMBlock zeros B_bar at pad positions so
they contribute nothing to the recurrence state.

This file is the only new code Stage 1.9 introduces; everything else
(JEPABackbone, predictor, loss, EMA, training loop, dataset) is unchanged.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------- AttnBlock -------------------------------------------------------


class AttnBlock(nn.Module):
    """Pre-norm transformer encoder layer (multi-head self-attention + FFN).

    Equivalent in structure to nn.TransformerEncoderLayer(norm_first=True),
    but extracted as a standalone block so it can be interleaved with
    SSMBlock inside HybridBackbone.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        ff_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        # x: (B, T, d_model); mask: (B, T) with 1=real, 0=pad
        # All-pad guard: attention softmax over an entirely-masked row gives NaN.
        # If any sample is fully masked, temporarily unmask position 0 so attention
        # has something to attend to; the masked mean-pool in JEPABackbone zeros
        # the sample's output anyway.
        pad_mask = mask == 0
        any_real = (mask.sum(dim=1) > 0)
        if not bool(any_real.all()):
            pad_mask = pad_mask.clone()
            pad_mask[~any_real, 0] = False

        h = self.norm1(x)
        attn_out, _ = self.attn(
            h, h, h, key_padding_mask=pad_mask, need_weights=False
        )
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


# ---------- SSMBlock --------------------------------------------------------


class SSMBlock(nn.Module):
    """Mamba-2-style selective state-space block (pure PyTorch, CPU-compatible).

    Pattern (pre-norm, residual):
        h = norm(x)
        h_proj = in_proj(h)            -> (B, T, 2·d_inner): split into x_branch + gate
        x_branch = silu(causal_conv1d(x_branch))
        y = selective_scan(x_branch, mask)
        y = y * silu(gate)
        y = out_proj(y)
        return x + dropout(y)

    The selective scan follows the Mamba-2 recurrence:
        h_t = A_bar(dt_t, A) · h_{t-1} + B_bar(dt_t, B_t) · x_t
        y_t = C_t · h_t
    where (dt, B, C) are projected per-token from x_branch, and A is shared
    log-parameterized for stability. Real-valued state (Mamba-2; Phase B's
    Mamba-3 uses complex state).

    Phase B replaces `_selective_scan` with `mamba_ssm.selective_scan_fn`
    (fused CUDA kernel). The algorithm here is the naive Python equivalent;
    mathematically identical, different compute pattern.
    """

    def __init__(
        self,
        d_model: int,
        d_inner: int | None = None,
        d_state: int = 16,
        d_conv: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_inner if d_inner is not None else 2 * d_model
        self.d_state = d_state
        self.d_conv = d_conv

        self.norm = nn.LayerNorm(d_model)
        # in_proj: produces both x_branch and the gate
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner)
        # Causal 1D conv along the time axis. Implemented via groups=d_inner
        # to apply the same kernel per-channel (depthwise) — the standard
        # Mamba conv. Padding is (d_conv - 1) on the left only.
        # bias=False so that convolving zero (masked-pad) input produces zero
        # output, preserving pad isolation. Matches standard Mamba impls.
        self.conv = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=False,
        )
        # Selective-scan projections: dt, B, C are functions of x
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner)
        self.B_proj = nn.Linear(self.d_inner, d_state)
        self.C_proj = nn.Linear(self.d_inner, d_state)
        # A is shared across tokens, log-parameterized so A = -exp(A_log) is
        # strictly negative-real for stability.
        # Initialize A to negative integers 1..d_state, broadcast across d_inner.
        a_init = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0)
        a_init = a_init.expand(self.d_inner, d_state).contiguous()
        self.A_log = nn.Parameter(torch.log(a_init))
        # Output projection back to d_model
        self.out_proj = nn.Linear(self.d_inner, d_model)
        self.dropout = nn.Dropout(dropout)

    def _causal_conv(self, x: Tensor) -> Tensor:
        """x: (B, T, d_inner). Returns (B, T, d_inner) after a left-causal 1D conv."""
        # Conv1d expects (B, C, T); we have (B, T, C). Transpose, conv, transpose back.
        h = x.transpose(1, 2)  # (B, d_inner, T)
        h = self.conv(h)        # (B, d_inner, T + d_conv - 1)
        # Drop the rightmost (d_conv - 1) positions so output is causal and length-T.
        h = h[..., : x.size(1)]
        return h.transpose(1, 2)  # (B, T, d_inner)

    def _selective_scan(self, u: Tensor, mask: Tensor) -> Tensor:
        """Naive sequential Mamba-2 scan.

        Args:
            u:    (B, T, d_inner) — the input sequence
            mask: (B, T)          — 1=real, 0=pad; pad positions contribute
                                    no input to the state recurrence

        Returns:
            y: (B, T, d_inner)
        """
        B, T, D = u.shape
        d_state = self.d_state

        # Project per-token SSM parameters
        dt = F.softplus(self.dt_proj(u))               # (B, T, D), strictly positive
        B_ssm = self.B_proj(u)                          # (B, T, d_state)
        C = self.C_proj(u)                              # (B, T, d_state)
        A = -torch.exp(self.A_log)                      # (D, d_state), strictly negative-real

        # Discretize: A_bar = exp(dt * A), B_bar = dt * B
        # Broadcast shapes to (B, T, D, d_state)
        dt_A = dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)   # (B, T, D, d_state)
        A_bar = torch.exp(dt_A)
        B_bar = dt.unsqueeze(-1) * B_ssm.unsqueeze(-2)          # (B, T, D, d_state)

        # Pad isolation: zero B_bar at pad positions so they contribute
        # nothing to state. (A_bar at pad still multiplies whatever state
        # exists, which decays it harmlessly.)
        m = mask.float().unsqueeze(-1).unsqueeze(-1)             # (B, T, 1, 1)
        B_bar = B_bar * m

        # Sequential scan
        h = torch.zeros(B, D, d_state, device=u.device, dtype=u.dtype)
        outs = []
        for t in range(T):
            # h_t = A_bar_t * h_{t-1} + B_bar_t * x_t
            h = A_bar[:, t] * h + B_bar[:, t] * u[:, t].unsqueeze(-1)
            # y_t = C_t · h_t  (contract over d_state)
            y_t = (h * C[:, t].unsqueeze(1)).sum(dim=-1)         # (B, D)
            outs.append(y_t)
        return torch.stack(outs, dim=1)                          # (B, T, D)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        h = self.norm(x)
        h = self.in_proj(h)                            # (B, T, 2·d_inner)
        x_branch, gate = h.chunk(2, dim=-1)            # each (B, T, d_inner)
        # Pad isolation step 1: zero x_branch at pad positions BEFORE the conv,
        # so the conv kernel can never mix pad-position values into adjacent
        # real positions. (The SSM scan applies a second mask on B_bar.)
        m = mask.float().unsqueeze(-1)                 # (B, T, 1)
        x_branch = x_branch * m
        x_branch = self._causal_conv(x_branch)
        x_branch = F.silu(x_branch)
        y = self._selective_scan(x_branch, mask)
        y = y * F.silu(gate)
        y = self.out_proj(y)
        return x + self.dropout(y)


# ---------- HybridBackbone --------------------------------------------------


def _ssm_positions(n_layers: int, ssm_ratio: float) -> set[int]:
    """Position-picking: place ssm_ratio·n_layers SSM blocks first, then attn.

    For n_layers=4, ssm_ratio=0.75 → {0, 1, 2}  (3 SSM, then 1 attn at index 3).
    For n_layers=4, ssm_ratio=0.5  → {0, 2}     (interleave 1:1 — see below).

    NOTE on interleaving: the cleanest "first all SSM, then all attn" pattern
    works for ratios like 0.75 and 1.0, but for 0.5 / 0.25 it would cluster
    attention blocks together. For visual interleaving on 0.5 specifically,
    we override to an alternating pattern. This matches the most common
    Qwen3.5/Kimi visualization and is the natural reading of "1:1 interleave".
    Other ratios (0.25, 0.75, 0.875) still use the cluster pattern.
    """
    n_ssm = round(n_layers * ssm_ratio)
    if abs(ssm_ratio - 0.5) < 1e-9:
        # Alternate: SSM at even indices, attn at odd.
        return {i for i in range(n_layers) if i % 2 == 0}
    # Cluster: first n_ssm positions get SSM
    return set(range(n_ssm))


class HybridBackbone(nn.Module):
    """N blocks; each block is SSMBlock or AttnBlock per ssm_ratio.

    Block layout examples (n_layers=4):
      ssm_ratio=0.0   → [Attn, Attn, Attn, Attn]
      ssm_ratio=0.25  → [SSM, Attn, Attn, Attn]
      ssm_ratio=0.5   → [SSM, Attn, SSM, Attn]   (alternating override)
      ssm_ratio=0.75  → [SSM, SSM, SSM, Attn]
      ssm_ratio=1.0   → [SSM, SSM, SSM, SSM]
    """

    def __init__(
        self,
        d_model: int,
        n_layers: int = 4,
        n_heads: int = 4,
        ff_dim: int = 256,
        dropout: float = 0.1,
        ssm_ratio: float = 0.75,
        d_state: int = 16,
        d_conv: int = 4,
    ) -> None:
        super().__init__()
        if not 0.0 <= ssm_ratio <= 1.0:
            raise ValueError(f"ssm_ratio must be in [0, 1]; got {ssm_ratio}")
        self.n_layers = n_layers
        self.ssm_ratio = ssm_ratio

        ssm_set = _ssm_positions(n_layers, ssm_ratio)
        self.block_types: list[str] = [
            "ssm" if i in ssm_set else "attn" for i in range(n_layers)
        ]
        self.blocks = nn.ModuleList([
            SSMBlock(d_model=d_model, d_state=d_state, d_conv=d_conv, dropout=dropout)
            if t == "ssm"
            else AttnBlock(d_model=d_model, n_heads=n_heads, ff_dim=ff_dim, dropout=dropout)
            for t in self.block_types
        ])

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        # x: (B, T, d_model); mask: (B, T)
        for block in self.blocks:
            x = block(x, mask)
        return x

    @property
    def n_ssm(self) -> int:
        return sum(1 for t in self.block_types if t == "ssm")

    @property
    def n_attn(self) -> int:
        return sum(1 for t in self.block_types if t == "attn")
