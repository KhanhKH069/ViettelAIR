"""
Loss functions for 3DGS training.

Main loss:
    L = (1 - lambda_ssim) * L1(pred, gt) + lambda_ssim * (1 - SSIM(pred, gt))

Optional regularisers (activated via config):
    scale_regularization   — penalises elongated Gaussians (helps thin structures)
    opacity_regularization — L1 on opacities (encourages sparsity)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# SSIM (structural similarity) — differentiable, single-scale
# ---------------------------------------------------------------------------

def _gaussian_kernel_1d(size: int = 11, sigma: float = 1.5) -> Tensor:
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    gauss  = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    return gauss / gauss.sum()


def _gaussian_kernel_2d(size: int = 11, sigma: float = 1.5) -> Tensor:
    k1d = _gaussian_kernel_1d(size, sigma)
    k2d = k1d.outer(k1d)
    return k2d.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)


_SSIM_KERNEL: Tensor | None = None


def _get_ssim_kernel(device: torch.device) -> Tensor:
    global _SSIM_KERNEL
    if _SSIM_KERNEL is None or _SSIM_KERNEL.device != device:
        _SSIM_KERNEL = _gaussian_kernel_2d().to(device)
    return _SSIM_KERNEL


def ssim(pred: Tensor, gt: Tensor, kernel_size: int = 11) -> Tensor:
    """
    Compute SSIM between pred and gt tensors of shape (C, H, W) or (B, C, H, W).
    Returns a scalar tensor.

    Constants follow the original Wang et al. paper: C1=(0.01*255)^2, C2=(0.03*255)^2.
    Since images are normalised to [0,1], we use C1=0.01^2, C2=0.03^2.
    """
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
        gt   = gt.unsqueeze(0)

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # Process each channel independently using grouped convolution
    B, C, H, W = pred.shape
    kernel = _get_ssim_kernel(pred.device)  # (1, 1, k, k)
    kernel = kernel.repeat(C, 1, 1, 1)     # (C, 1, k, k)
    pad    = kernel_size // 2

    def _conv(x: Tensor) -> Tensor:
        return F.conv2d(x, kernel, padding=pad, groups=C)

    mu1    = _conv(pred)
    mu2    = _conv(gt)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu12   = mu1 * mu2

    sigma1_sq = _conv(pred * pred) - mu1_sq
    sigma2_sq = _conv(gt   * gt)   - mu2_sq
    sigma12   = _conv(pred * gt)   - mu12

    num = (2 * mu12   + C1) * (2 * sigma12   + C2)
    den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = num / den.clamp(min=1e-8)

    return ssim_map.mean()


# ---------------------------------------------------------------------------
# Main photometric loss
# ---------------------------------------------------------------------------

def photometric_loss(
    pred: Tensor,           # (C, H, W) or (B, C, H, W)
    gt:   Tensor,           # same shape
    lambda_ssim: float = 0.2,
) -> Tensor:
    """
    Combined L1 + SSIM loss (standard 3DGS training objective).

    L = (1 - λ) * L1 + λ * (1 - SSIM)
    """
    l1   = F.l1_loss(pred, gt)
    ssim_val = ssim(pred, gt)
    loss = (1.0 - lambda_ssim) * l1 + lambda_ssim * (1.0 - ssim_val)
    return loss


# ---------------------------------------------------------------------------
# Regularisers
# ---------------------------------------------------------------------------

def scale_regularization(scales: Tensor, max_ratio: float = 10.0) -> Tensor:
    """
    Penalise Gaussians whose largest / smallest scale ratio exceeds max_ratio.
    Encourages compact, roughly isotropic Gaussians — helps thin BTS structures
    avoid degenerate "needle" Gaussians that hurt geometry.

    Args:
        scales:    (N, 3) positive scale values (after exp)
        max_ratio: tolerated max/min ratio before penalty kicks in
    Returns:
        scalar loss (mean over Gaussians)
    """
    # Sort scales per Gaussian
    sorted_scales, _ = scales.sort(dim=-1)               # ascending
    ratio = sorted_scales[..., -1] / sorted_scales[..., 0].clamp(min=1e-8)
    penalty = F.relu(ratio - max_ratio)
    return penalty.mean()


def opacity_regularization(opacities: Tensor) -> Tensor:
    """
    L1 regularisation on opacity — encourages sparsity / removes floaters.
    Args:
        opacities: (N,) values in (0, 1) after sigmoid
    """
    return opacities.mean()
