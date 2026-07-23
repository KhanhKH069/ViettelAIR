"""
Ray utilities: convert camera parameters to ray bundles.

Used by NeRF-based models; not required for 3DGS (which uses
gsplat's rasterizer directly).  Kept as a utility for geometric
debugging and optional hybrid approaches.
"""

from __future__ import annotations

import torch
from torch import Tensor


def get_rays(
    H: int,
    W: int,
    K: Tensor,   # (3, 3) intrinsics
    c2w: Tensor, # (4, 4) camera-to-world
) -> tuple[Tensor, Tensor]:
    """
    Generate ray origins and directions for every pixel.

    Returns:
        rays_o: (H, W, 3) — ray origins in world space (all same = camera center)
        rays_d: (H, W, 3) — unit ray directions in world space
    """
    device = K.device

    # Pixel grid (u, v) — centre of each pixel
    v_coords, u_coords = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=device),
        torch.arange(W, dtype=torch.float32, device=device),
        indexing="ij",
    )

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Direction in camera space (OpenCV: z forward, y down, x right)
    dirs_cam = torch.stack(
        [
            (u_coords - cx) / fx,
            (v_coords - cy) / fy,
            torch.ones_like(u_coords),
        ],
        dim=-1,
    )  # (H, W, 3)

    # Rotate to world space
    R = c2w[:3, :3]  # (3, 3)
    rays_d = (dirs_cam @ R.T)           # (H, W, 3)
    rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)

    # Camera origin in world space
    rays_o = c2w[:3, 3].expand(H, W, 3)  # (H, W, 3)

    return rays_o, rays_d
