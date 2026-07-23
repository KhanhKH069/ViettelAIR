"""
Dataset loader for a single BTS scene.

Reads processed scene data (images + COLMAP sparse) and provides
PyTorch-compatible Dataset items for training and validation.

Item format returned by __getitem__:
    {
        "image":   Float Tensor (3, H, W)  — RGB in [0, 1]
        "K":       Float Tensor (3, 3)     — intrinsic matrix
        "c2w":     Float Tensor (4, 4)     — camera-to-world (OpenCV)
        "width":   int
        "height":  int
        "name":    str                     — image filename (stem)
    }
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.colmap_utils import (
    ColmapCamera,
    ColmapImage,
    ColmapScene,
    parse_colmap_scene,
)


class BTSSceneDataset(Dataset):
    """
    Dataset for one BTS scene.

    Directory layout expected:
        scene_dir/
            train/
                images/          ← RGB images (any common format)
                sparse/0/
                    cameras.bin
                    images.bin
                    points3D.bin

    Args:
        scene_dir:  path to the scene root (contains train/ and test/)
        split:      "train" | "val"
        val_ratio:  fraction of views held out for internal validation
        seed:       random seed for reproducible split
        max_size:   if > 0, resize longest edge to this many pixels
    """

    def __init__(
        self,
        scene_dir: str | Path,
        split: str = "train",
        val_ratio: float = 0.1,
        seed: int = 42,
        max_size: int = 0,
    ):
        super().__init__()
        self.scene_dir = Path(scene_dir)
        self.split     = split
        self.max_size  = max_size

        train_dir   = self.scene_dir / "train"
        images_dir  = train_dir / "images"
        sparse_dir  = train_dir / "sparse" / "0"

        if not images_dir.exists():
            raise FileNotFoundError(f"Images dir not found: {images_dir}")
        if not sparse_dir.exists():
            raise FileNotFoundError(f"COLMAP sparse dir not found: {sparse_dir}")

        # Parse COLMAP sparse reconstruction
        self.colmap_scene: ColmapScene = parse_colmap_scene(sparse_dir)

        # Build image → camera lookup
        # Sort by image name for deterministic ordering
        all_colmap_images: List[ColmapImage] = sorted(
            self.colmap_scene.images.values(), key=lambda im: im.name
        )

        # Resolve full image paths (COLMAP names are relative to the images dir)
        self._entries: List[Dict] = []
        for colmap_img in all_colmap_images:
            img_path = images_dir / colmap_img.name
            # Also try just the filename (in case COLMAP stores basename only)
            if not img_path.exists():
                img_path = images_dir / Path(colmap_img.name).name
            if not img_path.exists():
                continue  # skip missing images

            cam = self.colmap_scene.cameras[colmap_img.camera_id]
            self._entries.append({
                "path":   img_path,
                "name":   img_path.stem,
                "colmap_image": colmap_img,
                "camera": cam,
            })

        if not self._entries:
            raise RuntimeError(f"No valid images found in {images_dir}")

        # Train / val split
        rng = random.Random(seed)
        indices = list(range(len(self._entries)))
        rng.shuffle(indices)
        n_val = max(1, int(len(indices) * val_ratio))

        if split == "val":
            chosen = sorted(indices[:n_val])
        else:
            chosen = sorted(indices[n_val:])

        self._entries = [self._entries[i] for i in chosen]

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @property
    def colmap(self) -> ColmapScene:
        return self.colmap_scene

    def get_all_c2w(self) -> torch.Tensor:
        """Return all c2w matrices as (N, 4, 4) tensor — useful for scene scaling."""
        mats = [torch.from_numpy(e["colmap_image"].c2w()).float() for e in self._entries]
        return torch.stack(mats, dim=0)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, idx: int) -> Dict:
        entry       = self._entries[idx]
        colmap_img  = entry["colmap_image"]
        cam: ColmapCamera = entry["camera"]

        # Load image
        image = _load_image(entry["path"], self.max_size)
        actual_h, actual_w = image.shape[1], image.shape[2]

        # Scale intrinsics if image was resized
        orig_w, orig_h = cam.width, cam.height
        sx = actual_w / orig_w
        sy = actual_h / orig_h

        K = torch.from_numpy(cam.K()).float()
        K[0, 0] *= sx   # fx
        K[1, 1] *= sy   # fy
        K[0, 2] *= sx   # cx
        K[1, 2] *= sy   # cy

        c2w = torch.from_numpy(colmap_img.c2w()).float()

        return {
            "image":  image,
            "K":      K,
            "c2w":    c2w,
            "width":  actual_w,
            "height": actual_h,
            "name":   entry["name"],
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_image(path: Path, max_size: int = 0) -> torch.Tensor:
    """
    Load an image as a float32 tensor (3, H, W) in [0, 1].

    Optionally resize so the longest edge ≤ max_size (bicubic, keep aspect).
    """
    img = Image.open(path).convert("RGB")

    if max_size > 0:
        w, h = img.size
        longest = max(w, h)
        if longest > max_size:
            scale = max_size / longest
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            img = img.resize((new_w, new_h), Image.BICUBIC)

    arr = np.array(img, dtype=np.float32) / 255.0   # (H, W, 3)
    tensor = torch.from_numpy(arr).permute(2, 0, 1)  # (3, H, W)
    return tensor
