"""
Image-quality metrics for internal validation (held-out views).

- PSNR
- SSIM (scikit-image or torch implementation)
- LPIPS (perceptual similarity, needs pretrained network — check network access)
"""


def compute_psnr(pred, target):
    raise NotImplementedError


def compute_ssim(pred, target):
    raise NotImplementedError


def compute_lpips(pred, target, net: str = "alex"):
    raise NotImplementedError
