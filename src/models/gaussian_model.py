"""
3D Gaussian Splatting model wrapper.

Plan: wrap/fork an existing implementation (e.g. gsplat) rather than
reimplementing the CUDA rasterizer. This module should expose a clean
interface used by training/trainer.py and rendering/renderer.py,
independent of whichever underlying library is vendored in third_party/.

TODO:
- Decide vendored library (gsplat vs original Inria 3DGS vs nerfstudio-splat)
  and add as git submodule under third_party/.
- Implement GaussianModel.init_from_points(sfm_points)
- Implement GaussianModel.forward(camera) -> rendered RGB, depth, alpha
- Implement parameter groups for optimizer (position/feature/opacity/scale/rotation)
"""


class GaussianModel:
    def __init__(self, config: dict):
        raise NotImplementedError

    def init_from_points(self, points, colors=None):
        raise NotImplementedError

    def forward(self, camera):
        """Render RGB (+ optional depth/alpha) for a given camera."""
        raise NotImplementedError

    def get_optimizable_params(self):
        raise NotImplementedError
