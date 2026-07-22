"""
Densification / pruning logic specific to 3D Gaussian Splatting.

- Split/clone gaussians in high-gradient regions (under-reconstructed areas).
- Prune low-opacity gaussians.
- Optional: priority densification in regions flagged in scene config
  (e.g. antenna/cable areas) to combat thin-structure floaters.
"""


def densify_and_prune(model, grad_threshold: float, opacity_threshold: float):
    raise NotImplementedError


def reset_opacity(model):
    raise NotImplementedError
