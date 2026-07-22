"""
Preprocessing pipeline: raw -> processed.

Steps (in order):
1. Undistort images using intrinsics (if distortion coefficients provided).
2. Check/normalize exposure & white balance consistency across views.
3. Optional masking (sky, transient objects).
4. Compute scene normalization (scale + center) and store in scene config.

Run via: scripts/run_preprocess.sh
"""


def undistort_images(raw_dir: str, out_dir: str):
    raise NotImplementedError


def compute_scene_normalization(poses: dict):
    """Returns (scale, center) to normalize scene into a bounded volume."""
    raise NotImplementedError
