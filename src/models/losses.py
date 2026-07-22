"""
Loss functions used during training.

- Photometric loss (L1 / L2) between rendered and ground-truth pixels.
- SSIM loss for structural similarity.
- Optional depth regularization (if sparse/MVS depth available).
- Optional scale regularization to discourage elongated/needle gaussians
  (helps reduce floaters around thin structures like cables/antennas).
"""


def photometric_loss(pred, target, mode: str = "l1"):
    raise NotImplementedError


def ssim_loss(pred, target):
    raise NotImplementedError


def depth_regularization(pred_depth, sparse_depth, mask=None):
    raise NotImplementedError


def scale_regularization(gaussian_scales):
    raise NotImplementedError
