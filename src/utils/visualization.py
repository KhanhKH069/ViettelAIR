"""
Debug visualization helpers:
- Plot camera poses in 3D (sanity-check pose convention / coverage).
- Overlay sparse SfM points with camera frustums.
- Side-by-side render vs ground-truth comparison grids.
"""


def plot_camera_poses(poses: dict, out_path: str = None):
    raise NotImplementedError


def compare_render_vs_gt(pred, gt, out_path: str = None):
    raise NotImplementedError
