"""
Render a trained model from an arbitrary camera pose.
Thin wrapper around model.forward(camera) with I/O handling
(saving PNG/JPEG, batching multiple cameras).
"""


def render_view(model, camera_intrinsics, camera_pose, image_size):
    raise NotImplementedError


def render_batch(model, cameras: list, out_dir: str):
    raise NotImplementedError
