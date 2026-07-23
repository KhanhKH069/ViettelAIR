"""
Single-scene pipeline runner.

Orchestrates: preprocess → train → render → (optional eval) for one scene.
Can be called directly or from train_all_scenes.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch

from src.data.colmap_utils import get_sfm_points, parse_colmap_scene
from src.data.dataset import BTSSceneDataset
from src.models.gaussian_model import GaussianModel
from src.rendering.novel_view import render_test_poses
from src.training.trainer import Trainer


def run_single_scene(
    scene_dir:   str | Path,
    output_dir:  str | Path,
    config:      Dict,
    scene_name:  Optional[str] = None,
    eval_gt_dir: Optional[str | Path] = None,
    resume_ckpt: Optional[str | Path] = None,
) -> Dict:
    """
    Full pipeline for one scene.

    Args:
        scene_dir:   Root of the scene (contains train/ and test/).
        output_dir:  Where to store checkpoints, renders, logs.
        config:      Merged YAML config dict.
        scene_name:  Override scene name (default: scene_dir.name).
        eval_gt_dir: If provided, compute metrics after rendering.
        resume_ckpt: Path to checkpoint to resume training from.

    Returns:
        Dict with paths and optional metrics.
    """
    scene_dir  = Path(scene_dir)
    output_dir = Path(output_dir)
    scene_name = scene_name or scene_dir.name
    device     = config.get("device", "cuda")

    print(f"\n{'='*60}")
    print(f"  Scene: {scene_name}")
    print(f"{'='*60}")

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("\n[1/4] Loading dataset …")
    train_ds = BTSSceneDataset(
        scene_dir  = scene_dir,
        split      = "train",
        val_ratio  = config.get("val_ratio", 0.1),
        max_size   = config.get("max_image_size", 0),
    )
    val_ds = BTSSceneDataset(
        scene_dir  = scene_dir,
        split      = "val",
        val_ratio  = config.get("val_ratio", 0.1),
        max_size   = config.get("max_image_size", 0),
    )
    print(f"   Train: {len(train_ds)} views  |  Val: {len(val_ds)} views")

    # ------------------------------------------------------------------
    # 2. Initialise model
    # ------------------------------------------------------------------
    print("\n[2/4] Initialising Gaussian model …")
    model_cfg = config.get("model", {})
    model = GaussianModel(model_cfg)

    if resume_ckpt and Path(resume_ckpt).exists():
        print(f"   Resuming from {resume_ckpt}")
        ckpt = Trainer.load_checkpoint(resume_ckpt)
        model.load_state_dict_gaussians(ckpt["gaussians"])
        start_iter = ckpt.get("iteration", 0)
    else:
        # Init from SfM points
        sparse_dir = scene_dir / "train" / "sparse" / "0"
        scene_colmap = parse_colmap_scene(sparse_dir)
        xyz, colors  = get_sfm_points(scene_colmap)
        print(f"   SfM points: {len(xyz):,}")
        model.init_from_colmap_points(xyz, colors, device=device)
        start_iter = 0

    # ------------------------------------------------------------------
    # 3. Train
    # ------------------------------------------------------------------
    print(f"\n[3/4] Training ({start_iter} → {config.get('optimizer', {}).get('iterations', 30000)} iters) …")
    scene_out = output_dir / scene_name
    trainer = Trainer(
        model       = model,
        dataset     = train_ds,
        config      = config,
        output_dir  = scene_out,
        val_dataset = val_ds,
    )
    if resume_ckpt and Path(resume_ckpt).exists():
        trainer.optimizer.load_state_dict(ckpt["optimizer"])

    trainer.train(start_iter=start_iter)

    # ------------------------------------------------------------------
    # 4. Render test poses
    # ------------------------------------------------------------------
    print("\n[4/4] Rendering test poses …")
    test_csv = scene_dir / "test" / "test_poses.csv"
    renders_dir = output_dir / "submission_renders"

    saved_paths = render_test_poses(
        model       = model,
        test_csv    = test_csv,
        output_dir  = renders_dir,
        scene_name  = scene_name,
        device      = device,
    )

    result = {
        "scene":       scene_name,
        "checkpoint":  str(scene_out / "checkpoints" / f"ckpt_{trainer.iterations:07d}.pt"),
        "renders":     str(renders_dir / scene_name),
        "num_renders": len(saved_paths),
    }

    # ------------------------------------------------------------------
    # Optional: evaluate against GT (if available)
    # ------------------------------------------------------------------
    if eval_gt_dir is not None:
        from src.evaluation.metrics import evaluate_scene
        gt_scene_dir = Path(eval_gt_dir) / scene_name
        if gt_scene_dir.exists():
            print("\n[Optional] Evaluating renders …")
            psnr_max = config.get("psnr_max", 40.0)
            metrics  = evaluate_scene(
                renders_dir / scene_name,
                gt_scene_dir,
                device   = device,
                psnr_max = psnr_max,
            )
            result["metrics"] = metrics
            print(f"   Score: {metrics['score']:.4f}  PSNR: {metrics['psnr']:.2f} dB  "
                  f"SSIM: {metrics['ssim']:.4f}  LPIPS: {metrics['lpips']:.4f}")

    return result
