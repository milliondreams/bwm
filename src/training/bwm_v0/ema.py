"""EMA target encoder update.

Spec § 6.3: EMA target update with τ = 0.996. Called once per training step
after optimizer.step(), in-place, outside the autograd graph.

The update rule is the standard:
    θ_target ← τ · θ_target + (1 - τ) · θ_online

Buffers (LayerNorm running stats, etc.) are copied directly rather than
EMA-averaged. The target and online encoders share architecture, so their
buffers are structurally identical; the difference between target.bn.mean
and online.bn.mean is dominated by the EMA in the param weights, not in
the buffers themselves.
"""
from __future__ import annotations

import torch
import torch.nn as nn


@torch.no_grad()
def ema_update(target: nn.Module, online: nn.Module, tau: float = 0.996) -> None:
    """In-place EMA: θ_target ← τ·θ_target + (1-τ)·θ_online."""
    for tp, op in zip(target.parameters(), online.parameters()):
        tp.mul_(tau).add_(op.detach(), alpha=1.0 - tau)
    # Direct buffer copy (see module docstring).
    for tb, ob in zip(target.buffers(), online.buffers()):
        tb.copy_(ob)
