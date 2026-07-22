"""
Convert organizer-provided camera/pose format into the internal
standardized format (and/or COLMAP format, if the chosen framework expects it).

TODO once real data schema is known:
- Write parse_cameras(raw_dir) -> dict of intrinsics per image
- Write parse_poses(raw_dir) -> dict of extrinsics per image
- Write export_to_colmap(...) if framework requires COLMAP sparse model
- Write export_to_transforms_json(...) if framework expects Nerfstudio format
"""


def parse_cameras(raw_dir: str):
    raise NotImplementedError


def parse_poses(raw_dir: str):
    raise NotImplementedError
