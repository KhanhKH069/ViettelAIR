"""
Batch novel-view renderer for competition submission.

Reads test_poses.csv, renders each target view, and saves PNG files
in the correct directory structure for the submission ZIP.

Output layout (matches competition spec):
    output_dir/
        scene_XXX/
            0001.png
            0002.png
            ...
"""

from __future__ import annotations

from pathlib import Path

import torch
from tqdm import tqdm

from src.data.test_dataset import TestPoseDataset
from src.models.gaussian_model import GaussianModel
from src.rendering.renderer import render_view_to_pil


def render_test_poses(
    model:       GaussianModel,
    test_csv:    str | Path,
    output_dir:  str | Path,
    scene_name:  str,
    device:      str | torch.device = "cuda",
) -> list[Path]:
    """
    Render all target poses from test_poses.csv and save PNGs.

    Args:
        model:       Trained GaussianModel.
        test_csv:    Path to test/test_poses.csv.
        output_dir:  Root output directory (submission root).
        scene_name:  Name of the current scene (e.g. "scene_001").
        device:      Torch device.

    Returns:
        List of paths to saved PNG files.
    """
    scene_out = Path(output_dir) / scene_name
    scene_out.mkdir(parents=True, exist_ok=True)

    test_ds = TestPoseDataset(test_csv)
    saved: list[Path] = []

    model = model.to(device)
    model.eval()

    for i, item in enumerate(tqdm(test_ds, desc=f"Rendering {scene_name}", leave=False)):
        c2w    = item["c2w"].to(device)
        K      = item["K"].to(device)
        W      = item["width"]
        H      = item["height"]
        name   = item["image_name"]  # as specified in CSV, e.g. "0001.png"

        # Ensure the filename has .png extension regardless of CSV value
        stem   = Path(name).stem
        out_fn = stem + ".png"
        out_path = scene_out / out_fn

        pil_img = render_view_to_pil(model, c2w, K, W, H, device)
        pil_img.save(out_path, format="PNG")
        saved.append(out_path)

    print(f"  ✓ Rendered {len(saved)} views → {scene_out}")
    return saved
