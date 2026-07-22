"""
Ray generation utilities — only needed if a NeRF-based model is used
as fallback/comparison to 3DGS. Not required for pure 3DGS rasterization.
"""


def get_rays(camera_intrinsics, camera_pose, image_size):
    raise NotImplementedError
