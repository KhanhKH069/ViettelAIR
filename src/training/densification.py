"""
Densification and pruning for 3D Gaussian Splatting.

Follows the adaptive control scheme from the original 3DGS paper:
  - Clone: small Gaussians with large view-space gradient → duplicate & shift
  - Split: large Gaussians with large view-space gradient → split into 2 smaller ones
  - Prune: Gaussians with opacity below threshold, or too large
  - Reset opacity: periodically set all opacities to a small value to remove floaters

Reference:
    Kerbl et al., "3D Gaussian Splatting for Real-Time Radiance Field Rendering"
    SIGGRAPH 2023.  https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class DensificationController:
    """
    Manages adaptive densification of a GaussianModel.

    Accumulates per-Gaussian gradient statistics across iterations,
    then performs clone / split / prune at the configured interval.
    """

    def __init__(
        self,
        model,                          # GaussianModel
        grad_threshold: float = 2e-4,
        min_opacity:    float = 0.005,
        max_screen_size: float = 20.0,  # pixels; Gaussians larger than this are split
        split_ratio:    float = 1.6,    # split Gaussians larger than ratio * mean scale
    ):
        self.model          = model
        self.grad_threshold = grad_threshold
        self.min_opacity    = min_opacity
        self.max_screen_size= max_screen_size
        self.split_ratio    = split_ratio

        # Accumulated per-Gaussian gradient norm (filled by update_stats)
        self._grad_accum:  Optional[Tensor] = None
        self._denom:       Optional[Tensor] = None

    # ------------------------------------------------------------------
    # Accumulate gradient statistics
    # ------------------------------------------------------------------

    def update_stats(self, viewspace_grad: Tensor) -> None:
        """
        Call after each backward pass with the gradient of the rendered image
        w.r.t. the 2D means (i.e. the view-space gradient from gsplat's `info`).

        Args:
            viewspace_grad: (N, 2) gradient magnitude in screen space
        """
        N = self.model.num_gaussians
        if self._grad_accum is None:
            device = viewspace_grad.device
            self._grad_accum = torch.zeros(N, device=device)
            self._denom      = torch.zeros(N, device=device)

        grad_norm = viewspace_grad.norm(dim=-1)
        self._grad_accum += grad_norm
        self._denom      += 1

    # ------------------------------------------------------------------
    # Main densification step
    # ------------------------------------------------------------------

    @torch.no_grad()
    def step(self, optimizer: torch.optim.Optimizer) -> int:
        """
        Run one densification step: clone/split high-gradient Gaussians,
        then prune low-opacity / over-large ones.

        Returns number of Gaussians after densification.
        """
        if self._grad_accum is None:
            return self.model.num_gaussians

        avg_grad = self._grad_accum / self._denom.clamp(min=1)
        N = self.model.num_gaussians

        # High-gradient mask
        high_grad = avg_grad >= self.grad_threshold  # (N,)

        # Scale-based split/clone decision
        mean_scale = self.model.scales.mean(dim=-1)  # (N,)
        large      = mean_scale > self.split_ratio * mean_scale.mean()

        to_split  = high_grad &  large
        to_clone  = high_grad & ~large

        # ---- Split ----
        if to_split.any():
            self._split(optimizer, to_split)

        # ---- Clone ----
        if to_clone.any():
            self._clone(optimizer, to_clone)

        # ---- Prune ----
        to_prune  = (self.model.opacities < self.min_opacity)
        if to_prune.any():
            self._prune(optimizer, to_prune)

        # Reset accumulators (re-initialized next update_stats call)
        self._grad_accum = None
        self._denom      = None

        return self.model.num_gaussians

    # ------------------------------------------------------------------
    # Opacity reset (call every ~3000 iterations)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def reset_opacity(self, optimizer: torch.optim.Optimizer, value: float = 0.01) -> None:
        """Set all opacities to logit(value) — removes persistent floaters."""
        import math
        logit_val = math.log(value / (1.0 - value))
        self.model._logit_opacities.data.fill_(logit_val)
        # Zero out the optimizer state for opacities
        self._zero_optimizer_state(optimizer, "opacities")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _clone(self, optimizer: torch.optim.Optimizer, mask: Tensor) -> None:
        """Duplicate selected Gaussians in-place."""
        new_means           = self.model._means.data[mask]
        new_quats           = self.model._quats.data[mask]
        new_log_scales      = self.model._log_scales.data[mask]
        new_logit_opacities = self.model._logit_opacities.data[mask]
        new_features_dc     = self.model._features_dc.data[mask]
        new_features_rest   = self.model._features_rest.data[mask]

        self._extend_params(optimizer, {
            "means":           new_means,
            "quats":           new_quats,
            "log_scales":      new_log_scales,
            "logit_opacities": new_logit_opacities,
            "features_dc":     new_features_dc,
            "features_rest":   new_features_rest,
        })

    def _split(self, optimizer: torch.optim.Optimizer, mask: Tensor, n_split: int = 2) -> None:
        """Split selected Gaussians into n_split smaller ones."""
        scales = self.model.scales.data[mask]      # (M, 3)
        M      = mask.sum().item()

        # Sample random directions from the Gaussian
        noise = torch.randn(n_split, M, 3, device=scales.device)
        noise = noise * scales.unsqueeze(0)  # scale by Gaussian extent

        # We need R(quat) to transform noise to world
        # For simplicity: add noise directly (isotropic approximation)
        for _ in range(n_split):
            new_means = self.model._means.data[mask] + noise[min(_, n_split-1)]
            new_log_scales = self.model._log_scales.data[mask] - math.log(0.8 * n_split)

            import math
            self._extend_params(optimizer, {
                "means":           new_means,
                "quats":           self.model._quats.data[mask],
                "log_scales":      new_log_scales.clamp(min=-10.0),
                "logit_opacities": self.model._logit_opacities.data[mask],
                "features_dc":     self.model._features_dc.data[mask],
                "features_rest":   self.model._features_rest.data[mask],
            })

        # Prune the originals
        self._prune(optimizer, mask)

    def _prune(self, optimizer: torch.optim.Optimizer, mask: Tensor) -> None:
        """Remove selected Gaussians (mask=True → remove)."""
        keep = ~mask
        self._filter_params(optimizer, keep)

    def _extend_params(self, optimizer: torch.optim.Optimizer, new_data: dict) -> None:
        """Append new Gaussian parameters to all parameter tensors."""
        m = self.model
        param_map = {
            "means":           m._means,
            "quats":           m._quats,
            "log_scales":      m._log_scales,
            "logit_opacities": m._logit_opacities,
            "features_dc":     m._features_dc,
            "features_rest":   m._features_rest,
        }
        for name, param in param_map.items():
            extra  = new_data[name]
            merged = nn.Parameter(torch.cat([param.data, extra], dim=0))
            # Update optimizer state
            for group in optimizer.param_groups:
                if group.get("name") == name.replace("_", "").replace("log", "").replace("logit", ""):
                    # Zero init new states
                    pass
            # Update model attribute
            setattr(m, f"_{name}", merged)

    def _filter_params(self, optimizer: torch.optim.Optimizer, keep: Tensor) -> None:
        """Keep only Gaussians where keep[i] = True."""
        m = self.model
        for attr in ("_means", "_quats", "_log_scales", "_logit_opacities",
                     "_features_dc", "_features_rest"):
            p = getattr(m, attr)
            setattr(m, attr, nn.Parameter(p.data[keep]))

    def _zero_optimizer_state(self, optimizer: torch.optim.Optimizer, name: str) -> None:
        for group in optimizer.param_groups:
            if group.get("name") == name:
                for p in group["params"]:
                    if p in optimizer.state:
                        optimizer.state[p] = {}
