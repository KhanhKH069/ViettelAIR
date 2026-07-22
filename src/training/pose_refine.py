"""
Optional bundle-adjustment-style refinement of provided camera poses.

Organizer-provided poses (likely from drone GPS/IMU) may have small
errors (cm to tens of cm). This module optionally treats poses as
learnable parameters during training to refine them jointly with
the scene representation.

Not required for the first working baseline — add if pose noise is
visibly hurting reconstruction quality.
"""


def enable_pose_optimization(cameras, lr: float = 1e-4):
    raise NotImplementedError
