"""
Geometric evaluation — only usable if ground-truth mesh/point cloud
is available (unclear if organizer provides this). Otherwise this
module stays unused and evaluation is purely photometric.
"""


def chamfer_distance(pred_points, gt_points):
    raise NotImplementedError
