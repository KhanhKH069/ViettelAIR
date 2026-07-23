"""
Competition evaluation metrics:
    PSNR, SSIM, LPIPS — and the official competition score formula.

Official score:
    Score = 0.4 × (1 - LPIPS) + 0.3 × SSIM + 0.3 × PSNR_norm
    PSNR_norm = clamp(PSNR / PSNR_max, 0, 1)

All functions accept:
  - torch Tensors (B, C, H, W) or (C, H, W) in [0, 1], or
  - numpy arrays  (H, W, C) or (H, W) in [0, 1] / uint8

Images are automatically converted to float [0, 1] as needed.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch import Tensor


# ---------------------------------------------------------------------------
# Metric implementations
# ---------------------------------------------------------------------------

def compute_psnr(pred: Tensor, gt: Tensor) -> float:
    """
    Peak Signal-to-Noise Ratio.
    Inputs: float Tensors in [0, 1], any shape.
    Returns: PSNR value in dB (higher = better).
    """
    mse = ((pred.float() - gt.float()) ** 2).mean().item()
    if mse < 1e-10:
        return 100.0
    return -10.0 * math.log10(mse)


def compute_ssim(pred: Tensor, gt: Tensor) -> float:
    """
    Structural Similarity Index (SSIM).
    Inputs: float Tensors (C,H,W) or (B,C,H,W) in [0,1].
    Returns: SSIM scalar (higher = better, max 1.0).
    """
    from src.models.losses import ssim as ssim_fn
    return ssim_fn(pred.float(), gt.float()).item()


def compute_lpips(
    pred: Tensor,
    gt:   Tensor,
    net:  str = "vgg",
    _cache: dict = {},
) -> float:
    """
    Learned Perceptual Image Patch Similarity (LPIPS).
    Inputs: float Tensors (C,H,W) or (B,C,H,W) in [0,1].
    Returns: LPIPS scalar (lower = better, min 0.0).

    Uses the `lpips` package (pip install lpips).
    The loss network is lazily loaded and cached.
    """
    import lpips as lpips_lib

    device = pred.device
    cache_key = (net, str(device))
    if cache_key not in _cache:
        _cache[cache_key] = lpips_lib.LPIPS(net=net).to(device)
    loss_fn = _cache[cache_key]

    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
        gt   = gt.unsqueeze(0)

    # lpips expects images in [-1, 1]
    p = pred.float() * 2.0 - 1.0
    g = gt.float()   * 2.0 - 1.0

    with torch.no_grad():
        val = loss_fn(p, g)
    return float(val.mean().item())


def compute_psnr_norm(psnr: float, psnr_max: float = 40.0) -> float:
    """Normalise PSNR to [0, 1] per the competition formula."""
    return float(np.clip(psnr / psnr_max, 0.0, 1.0))


def competition_score(
    psnr:     float,
    ssim:     float,
    lpips:    float,
    psnr_max: float = 40.0,
) -> float:
    """
    Official competition scoring formula:
        Score = 0.4 × (1 - LPIPS) + 0.3 × SSIM + 0.3 × PSNR_norm
    """
    psnr_n = compute_psnr_norm(psnr, psnr_max)
    return 0.4 * (1.0 - lpips) + 0.3 * ssim + 0.3 * psnr_n


# ---------------------------------------------------------------------------
# Image I/O helpers
# ---------------------------------------------------------------------------

def load_image_tensor(path: str | Path) -> Tensor:
    """Load PNG/JPG → (3, H, W) float32 Tensor in [0, 1]."""
    img = Image.open(path).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


# ---------------------------------------------------------------------------
# Scene-level evaluation
# ---------------------------------------------------------------------------

def evaluate_pair(
    pred_path: str | Path,
    gt_path:   str | Path,
    device:    str = "cuda",
    psnr_max:  float = 40.0,
) -> Dict[str, float]:
    """
    Evaluate a single (pred, gt) image pair.

    Returns dict with keys: psnr, ssim, lpips, score
    """
    pred = load_image_tensor(pred_path).to(device)
    gt   = load_image_tensor(gt_path).to(device)

    # Resize pred to match gt if needed
    if pred.shape != gt.shape:
        import torch.nn.functional as F
        pred = F.interpolate(
            pred.unsqueeze(0),
            size=(gt.shape[1], gt.shape[2]),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    psnr  = compute_psnr(pred, gt)
    ssim  = compute_ssim(pred, gt)
    lp    = compute_lpips(pred, gt)
    score = competition_score(psnr, ssim, lp, psnr_max)

    return {"psnr": psnr, "ssim": ssim, "lpips": lp, "score": score}


def evaluate_scene(
    pred_dir: str | Path,
    gt_dir:   str | Path,
    device:   str = "cuda",
    psnr_max: float = 40.0,
    extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg"),
) -> Dict[str, float]:
    """
    Evaluate all predicted images against ground-truth images in a scene directory.

    pred_dir and gt_dir must contain matching filenames.

    Returns:
        Dict with mean psnr, ssim, lpips, score + per-image list.
    """
    pred_dir = Path(pred_dir)
    gt_dir   = Path(gt_dir)

    gt_files = sorted(p for p in gt_dir.iterdir() if p.suffix.lower() in extensions)
    if not gt_files:
        raise FileNotFoundError(f"No GT images found in {gt_dir}")

    results: List[Dict[str, float]] = []
    missing = 0
    for gt_path in gt_files:
        pred_path = pred_dir / gt_path.name
        if not pred_path.exists():
            print(f"  [WARN] Missing prediction: {pred_path.name}")
            missing += 1
            continue
        r = evaluate_pair(pred_path, gt_path, device=device, psnr_max=psnr_max)
        results.append(r)

    if not results:
        return {"psnr": 0.0, "ssim": 0.0, "lpips": 1.0, "score": 0.0, "n": 0}

    avg = {
        "psnr":  float(np.mean([r["psnr"]  for r in results])),
        "ssim":  float(np.mean([r["ssim"]  for r in results])),
        "lpips": float(np.mean([r["lpips"] for r in results])),
        "score": float(np.mean([r["score"] for r in results])),
        "n":     len(results),
        "missing": missing,
    }
    return avg


def evaluate_all_scenes(
    pred_root: str | Path,
    gt_root:   str | Path,
    device:    str = "cuda",
    psnr_max:  float = 40.0,
) -> Dict[str, Dict]:
    """
    Evaluate all scenes.  Returns per-scene and overall mean metrics.

    pred_root/
        scene_001/ ← predicted images
        scene_002/
    gt_root/
        scene_001/ ← ground-truth images
        scene_002/
    """
    pred_root = Path(pred_root)
    gt_root   = Path(gt_root)

    scene_dirs = sorted(d for d in gt_root.iterdir() if d.is_dir())
    per_scene: Dict[str, Dict] = {}

    for scene_dir in scene_dirs:
        name      = scene_dir.name
        pred_path = pred_root / name
        if not pred_path.exists():
            print(f"[WARN] Missing scene prediction: {name}")
            continue
        per_scene[name] = evaluate_scene(pred_path, scene_dir, device, psnr_max)
        m = per_scene[name]
        print(f"  {name}: score={m['score']:.4f}  "
              f"psnr={m['psnr']:.2f}  ssim={m['ssim']:.4f}  lpips={m['lpips']:.4f}")

    if not per_scene:
        return {}

    overall = {
        "psnr":  float(np.mean([v["psnr"]  for v in per_scene.values()])),
        "ssim":  float(np.mean([v["ssim"]  for v in per_scene.values()])),
        "lpips": float(np.mean([v["lpips"] for v in per_scene.values()])),
        "score": float(np.mean([v["score"] for v in per_scene.values()])),
    }
    per_scene["__overall__"] = overall
    print(f"\nOverall score: {overall['score']:.4f}")
    return per_scene
