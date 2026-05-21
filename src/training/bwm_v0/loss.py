"""JEPA loss with VICReg variance + covariance anti-collapse terms.

Spec § 6.3:
    L = ||ŝ_{t+h} − sg(s_{t+h})||² + λ_var·VICReg_var + λ_cov·VICReg_cov

VICReg variance term (per-dimension std ≥ γ) prevents the encoder from
collapsing all inputs to a constant vector. VICReg covariance term
decorrelates latent dimensions, preventing trivial collapse to a low-rank
subspace.

References: Bardes et al., "VICReg: Variance-Invariance-Covariance
Regularization for Self-Supervised Learning" (2022). Coefficients tuned
for the original VICReg paper at λ_var=25, λ_cov=1; we use λ_var=1.0,
λ_cov=0.05 because at v0's tiny latent dim (64) and tiny dataset (15
entities) the relative scales differ.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def vicreg_variance(z: Tensor, gamma: float = 1.0, eps: float = 1e-4) -> Tensor:
    """Per-dimension std should be ≥ gamma.

    z: (B, d). Returns mean over dimensions of max(0, gamma - std)².
    """
    # Unbiased=False because at small batch sizes the unbiased estimator is
    # noisy and can produce negatives under sqrt.
    std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)
    return F.relu(gamma - std).pow(2).mean()


def vicreg_covariance(z: Tensor) -> Tensor:
    """Sum of squared off-diagonals of the centered covariance, scaled by 1/d.

    z: (B, d). Returns a scalar.
    """
    B, d = z.shape
    if B < 2:
        # Covariance is undefined with a single sample; return zero so the
        # optimizer step is well-defined (no NaN propagation).
        return torch.zeros((), device=z.device, dtype=z.dtype)
    z_centered = z - z.mean(dim=0, keepdim=True)
    cov = (z_centered.transpose(0, 1) @ z_centered) / (B - 1)
    off_diag = cov - torch.diag(torch.diagonal(cov))
    return off_diag.pow(2).sum() / d


def jepa_loss(
    pred: Tensor,
    target: Tensor,
    online_s_t: Tensor,
    *,
    gamma: float = 1.0,
    lambda_var: float = 1.0,
    lambda_cov: float = 0.05,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Multi-horizon JEPA loss.

    Args:
        pred:        (B, d) predictor output ŝ_{t+h}
        target:      (B, d) target encoder output s_{t+h} (already detached by caller)
        online_s_t:  (B, d) online encoder output at time t (gradient flows through)
        gamma:       VICReg variance threshold
        lambda_var:  weight on variance term
        lambda_cov:  weight on covariance term

    Returns:
        (total_loss, components)  where components has detached scalars:
            "mse":   the JEPA MSE
            "var":   the VICReg variance term
            "cov":   the VICReg covariance term
            "total": the full loss
    """
    # `target` should be detached by the caller (model.target_encode returns
    # a no_grad tensor), but enforce defensively.
    mse = F.mse_loss(pred, target.detach())
    var = vicreg_variance(online_s_t, gamma=gamma)
    cov = vicreg_covariance(online_s_t)
    total = mse + lambda_var * var + lambda_cov * cov
    return total, {
        "mse": mse.detach(),
        "var": var.detach(),
        "cov": cov.detach(),
        "total": total.detach(),
    }
