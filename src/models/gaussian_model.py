"""
3D Gaussian Splatting model — wraps the `gsplat` library.

Install: pip install gsplat
GitHub:  https://github.com/nerfstudio-project/gsplat

COLMAP/OpenCV camera convention used throughout:
  - Z points forward (into the scene)
  - Y points down
  - X points right

gsplat rasterization API:
    from gsplat import rasterization
    renders, alphas, info = rasterization(
        means,          # (N, 3) Gaussian centres in world space
        quats,          # (N, 4) rotation quaternions (w, x, y, z)
        scales,         # (N, 3) log-scale (exponentiated inside gsplat)
        opacities,      # (N,)   logit-opacity (sigmoid inside gsplat)
        colors,         # (N, K, 3) SH coefficients or (N, 3) simple RGB
        viewmats,       # (C, 4, 4) world-to-camera matrices
        Ks,             # (C, 3, 3) intrinsic matrices
        width, height,
        sh_degree=sh_degree,
    )
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# Number of SH coefficients per degree
# ---------------------------------------------------------------------------
def num_sh_bases(degree: int) -> int:
    """Total SH coefficients for degrees 0..degree inclusive."""
    return (degree + 1) ** 2


# ---------------------------------------------------------------------------
# Gaussian parameter initialisation helpers
# ---------------------------------------------------------------------------

def _inverse_sigmoid(x: float) -> float:
    return math.log(x / (1.0 - x))


def _init_scales_from_knn(means: Tensor, k: int = 3) -> Tensor:
    """
    Estimate initial Gaussian scale as the mean distance to the k-nearest
    neighbours in the SfM point cloud.  This gives a sensible starting size.
    """
    from torch import cdist
    dists = cdist(means, means)  # (N, N)
    # exclude self (diagonal = 0)
    topk = torch.topk(dists, k + 1, largest=False, dim=-1).values[:, 1:]  # (N, k)
    mean_dist = topk.mean(dim=-1, keepdim=True).clamp(min=1e-7)            # (N, 1)
    log_scale = torch.log(mean_dist).expand(-1, 3)                         # (N, 3)
    return log_scale


# ---------------------------------------------------------------------------
# GaussianModel
# ---------------------------------------------------------------------------

class GaussianModel(nn.Module):
    """
    Learnable 3D Gaussian Splatting scene representation.

    All parameters are stored as raw (unbounded) learnable tensors and
    mapped to their constrained counterparts during rendering:
        means:      raw ℝ³             → world-space Gaussian centres
        _quats:     raw ℝ⁴, normalized → unit quaternions (w,x,y,z)
        _log_scales:raw ℝ³             → exp(·) = scales > 0
        _logit_opacities: raw ℝ        → sigmoid(·) ∈ (0,1)
        _features_dc:  raw ℝ(N,1,3)   → degree-0 SH (base colour)
        _features_rest:raw ℝ(N,K-1,3) → higher-order SH
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config    = config
        self.sh_degree = config.get("sh_degree", 3)
        self._n_sh     = num_sh_bases(self.sh_degree)
        self._active_sh_degree = 0   # progressively activated during training

        # Will be populated by init_from_colmap_points()
        self._means:             Optional[nn.Parameter] = None
        self._quats:             Optional[nn.Parameter] = None
        self._log_scales:        Optional[nn.Parameter] = None
        self._logit_opacities:   Optional[nn.Parameter] = None
        self._features_dc:       Optional[nn.Parameter] = None
        self._features_rest:     Optional[nn.Parameter] = None

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def init_from_colmap_points(
        self,
        xyz: np.ndarray,        # (N, 3) float64
        colors: np.ndarray,     # (N, 3) float32  RGB in [0,1]
        device: str = "cuda",
    ) -> None:
        """
        Initialise Gaussians from SfM sparse point cloud.

        Each SfM point spawns one Gaussian with:
          - centre  = SfM point location
          - scale   = KNN-estimated neighbourhood size
          - opacity = 0.1 (opaque but not saturated)
          - colour  = SfM point colour (stored as degree-0 SH)
          - rotation= identity quaternion
        """
        means_np = xyz.astype(np.float32)
        N = means_np.shape[0]

        means_t = torch.tensor(means_np, device=device)

        # Rotation: identity quaternion (w=1, x=y=z=0)
        quats_t = torch.zeros(N, 4, device=device)
        quats_t[:, 0] = 1.0

        # Scale: KNN-estimated
        log_scales_t = _init_scales_from_knn(means_t).to(device)

        # Opacity: small initial value (logit of 0.1)
        logit_opacities_t = torch.full(
            (N,), _inverse_sigmoid(0.1), device=device
        )

        # Colour: RGB → degree-0 SH coefficient
        # SH to RGB: rgb = sh_coeff * C0 + 0.5   where C0 = 0.28209479
        C0 = 0.28209479177387814
        rgb_t = torch.tensor(colors, dtype=torch.float32, device=device)
        sh0 = (rgb_t - 0.5) / C0  # (N, 3)
        features_dc   = sh0.unsqueeze(1)                             # (N, 1, 3)
        features_rest = torch.zeros(N, self._n_sh - 1, 3, device=device)

        self._means           = nn.Parameter(means_t)
        self._quats           = nn.Parameter(quats_t)
        self._log_scales      = nn.Parameter(log_scales_t)
        self._logit_opacities = nn.Parameter(logit_opacities_t)
        self._features_dc     = nn.Parameter(features_dc)
        self._features_rest   = nn.Parameter(features_rest)

    # ------------------------------------------------------------------
    # Properties (constrained values)
    # ------------------------------------------------------------------

    @property
    def means(self) -> Tensor:
        return self._means  # type: ignore[return-value]

    @property
    def quats(self) -> Tensor:
        """Unit quaternions (w,x,y,z)."""
        q = self._quats  # type: ignore[return-value]
        return q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    @property
    def scales(self) -> Tensor:
        return torch.exp(self._log_scales)  # type: ignore[return-value]

    @property
    def opacities(self) -> Tensor:
        return torch.sigmoid(self._logit_opacities)  # type: ignore[return-value]

    @property
    def sh_features(self) -> Tensor:
        """All SH features (N, n_sh, 3), first band = DC."""
        return torch.cat([self._features_dc, self._features_rest], dim=1)  # type: ignore[return-value]

    @property
    def num_gaussians(self) -> int:
        return 0 if self._means is None else self._means.shape[0]

    # ------------------------------------------------------------------
    # Forward / rendering
    # ------------------------------------------------------------------

    def forward(
        self,
        c2w: Tensor,    # (4, 4) camera-to-world
        K:   Tensor,    # (3, 3) intrinsics
        width:  int,
        height: int,
    ) -> Dict[str, Tensor]:
        """
        Render one view via gsplat rasterization.

        Returns:
            rgb:   (H, W, 3) float32 in [0, 1]
            alpha: (H, W, 1) float32 in [0, 1]
        """
        from gsplat import rasterization

        device = self._means.device  # type: ignore[union-attr]

        # Convert c2w → w2c (viewmat) for gsplat
        viewmat = torch.inverse(c2w).unsqueeze(0)   # (1, 4, 4)
        Ks      = K.unsqueeze(0)                     # (1, 3, 3)

        renders, alphas, _info = rasterization(
            means     = self.means,
            quats     = self.quats,
            scales    = self.scales,
            opacities = self.opacities,
            colors    = self.sh_features,
            viewmats  = viewmat,
            Ks        = Ks,
            width     = width,
            height    = height,
            sh_degree = self._active_sh_degree,
            near_plane= 0.01,
            far_plane  = 1e9,
            render_mode= "RGB+D",
        )
        # renders: (1, H, W, 3),  alphas: (1, H, W, 1)
        return {
            "rgb":   renders[0, ..., :3].clamp(0.0, 1.0),
            "alpha": alphas[0],
        }

    # ------------------------------------------------------------------
    # Optimiser parameter groups
    # ------------------------------------------------------------------

    def get_optimizable_params(self, cfg: dict) -> List[Dict]:
        """
        Return parameter groups for torch.optim.Adam.
        Each group has a dedicated learning rate.
        """
        if self._means is None:
            raise RuntimeError("Call init_from_colmap_points first.")
        return [
            {"params": [self._means],           "lr": cfg.get("lr_position", 1.6e-4),  "name": "means"},
            {"params": [self._features_dc],     "lr": cfg.get("lr_feature",  2.5e-3),  "name": "features_dc"},
            {"params": [self._features_rest],   "lr": cfg.get("lr_feature",  2.5e-3) / 20.0, "name": "features_rest"},
            {"params": [self._logit_opacities], "lr": cfg.get("lr_opacity",  5.0e-2),  "name": "opacities"},
            {"params": [self._log_scales],      "lr": cfg.get("lr_scaling",  5.0e-3),  "name": "scales"},
            {"params": [self._quats],           "lr": cfg.get("lr_rotation", 1.0e-3),  "name": "quats"},
        ]

    # ------------------------------------------------------------------
    # SH degree scheduling
    # ------------------------------------------------------------------

    def oneup_sh_degree(self) -> None:
        """Increase active SH degree by one (call periodically during training)."""
        if self._active_sh_degree < self.sh_degree:
            self._active_sh_degree += 1

    # ------------------------------------------------------------------
    # Checkpoint save/load
    # ------------------------------------------------------------------

    def state_dict_gaussians(self) -> Dict[str, Tensor]:
        return {
            "means":           self._means.data,               # type: ignore
            "quats":           self._quats.data,               # type: ignore
            "log_scales":      self._log_scales.data,          # type: ignore
            "logit_opacities": self._logit_opacities.data,     # type: ignore
            "features_dc":     self._features_dc.data,         # type: ignore
            "features_rest":   self._features_rest.data,       # type: ignore
            "sh_degree":       torch.tensor(self.sh_degree),
            "active_sh_degree":torch.tensor(self._active_sh_degree),
        }

    def load_state_dict_gaussians(self, d: Dict[str, Tensor]) -> None:
        device = d["means"].device
        self.sh_degree         = int(d["sh_degree"].item())
        self._active_sh_degree = int(d["active_sh_degree"].item())
        self._n_sh = num_sh_bases(self.sh_degree)
        self._means           = nn.Parameter(d["means"].to(device))
        self._quats           = nn.Parameter(d["quats"].to(device))
        self._log_scales      = nn.Parameter(d["log_scales"].to(device))
        self._logit_opacities = nn.Parameter(d["logit_opacities"].to(device))
        self._features_dc     = nn.Parameter(d["features_dc"].to(device))
        self._features_rest   = nn.Parameter(d["features_rest"].to(device))

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict_gaussians(), path)

    @classmethod
    def load(cls, path: str | Path, config: dict) -> "GaussianModel":
        model = cls(config)
        d = torch.load(path, map_location="cpu")
        model.load_state_dict_gaussians(d)
        return model
