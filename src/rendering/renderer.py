"""
Single-view renderer.

Handles the camera convention conversion between our internal c2w
(camera-to-world, OpenCV) and whatever the underlying model expects.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from src.models.gaussian_model import GaussianModel


def render_view(
    model:  GaussianModel,
    c2w:    Tensor,         # (4, 4) camera-to-world, OpenCV convention
    K:      Tensor,         # (3, 3) intrinsic matrix
    width:  int,
    height: int,
    device: str | torch.device = "cuda",
) -> Dict[str, Tensor]:
    """
    Render a single view via the GaussianModel.

    Returns:
        {
          "rgb":   (H, W, 3) float32 Tensor in [0, 1]
          "alpha": (H, W, 1) float32 Tensor in [0, 1]
        }
    """
    model.eval()
    with torch.no_grad():
        c2w = c2w.to(device)
        K   = K.to(device)
        out = model(c2w, K, width, height)
    return out


def render_view_to_pil(
    model:  GaussianModel,
    c2w:    Tensor,
    K:      Tensor,
    width:  int,
    height: int,
    device: str | torch.device = "cuda",
) -> Image.Image:
    """
    Render a single view and return a PIL.Image (RGB).
    """
    out  = render_view(model, c2w, K, width, height, device)
    rgb  = out["rgb"].cpu().numpy()                     # (H, W, 3) float32
    rgb8 = (rgb * 255.0).clip(0, 255).astype(np.uint8) # uint8
    return Image.fromarray(rgb8, mode="RGB")
