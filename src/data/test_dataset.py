"""
Test pose dataset — loads test_poses.csv and yields camera parameters
for batch rendering.  Does NOT load any images (ground-truth is unknown).

Item format:
    {
        "image_name": str             — output filename as specified in CSV
        "K":          Float Tensor (3,3)
        "c2w":        Float Tensor (4,4)
        "width":      int
        "height":     int
    }
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import Dataset

from src.data.colmap_utils import TestPose, parse_test_poses_csv


class TestPoseDataset(Dataset):
    """
    Wraps the competition test_poses.csv as a PyTorch Dataset.

    Args:
        csv_path: path to test/test_poses.csv
    """

    def __init__(self, csv_path: str | Path):
        super().__init__()
        self._poses: List[TestPose] = parse_test_poses_csv(csv_path)

        if not self._poses:
            raise RuntimeError(f"No test poses found in {csv_path}")

    def __len__(self) -> int:
        return len(self._poses)

    def __getitem__(self, idx: int) -> Dict:
        pose = self._poses[idx]
        return {
            "image_name": pose.image_name,
            "K":          torch.from_numpy(pose.K()).float(),
            "c2w":        torch.from_numpy(pose.c2w()).float(),
            "width":      pose.width,
            "height":     pose.height,
        }

    @property
    def poses(self) -> List[TestPose]:
        return self._poses
