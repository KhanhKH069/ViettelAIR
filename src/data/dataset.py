"""
Dataset / DataLoader for a single BTS scene.

Responsibilities:
- Load processed images + camera intrinsics/extrinsics for a scene.
- Provide train/val split (hold out ~10-20% of the given views internally
  for validation before submission).
- Return items in the format expected by the chosen model (gsplat / nerfacto).

TODO once real data schema is known:
- Confirm raw pose convention (OpenCV vs OpenGL vs COLMAP) and document here.
- Implement __getitem__ to return image, intrinsics (K), extrinsics (c2w/w2c).
"""

from torch.utils.data import Dataset


class BTSSceneDataset(Dataset):
    def __init__(self, processed_dir: str, split: str = "train", val_ratio: float = 0.1):
        """
        Args:
            processed_dir: path to data/processed/scene_XX
            split: "train" | "val"
            val_ratio: fraction of views held out for internal validation
        """
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError
