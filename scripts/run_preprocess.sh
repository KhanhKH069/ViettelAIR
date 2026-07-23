#!/usr/bin/env bash
# Preprocess: verify data layout for a scene before training.
# Usage: ./scripts/run_preprocess.sh scene_001
set -e
SCENE=${1:?Usage: run_preprocess.sh <scene_name>}

python - <<EOF
import sys; sys.path.insert(0, ".")
from pathlib import Path
from src.data.colmap_utils import parse_colmap_scene, get_sfm_points, parse_test_poses_csv

scene_name = "$SCENE"
scene_dir  = Path("data/raw") / scene_name
sparse_dir = scene_dir / "train" / "sparse" / "0"
test_csv   = scene_dir / "test" / "test_poses.csv"

print(f"\n=== {scene_name} ===")

# Parse COLMAP
colmap = parse_colmap_scene(sparse_dir)
print(f"Cameras:   {len(colmap.cameras)}")
print(f"Images:    {len(colmap.images)}")
print(f"Points3D:  {len(colmap.points3D):,}")

xyz, colors = get_sfm_points(colmap)
print(f"SfM XYZ range: x=[{xyz[:,0].min():.2f}, {xyz[:,0].max():.2f}]  "
      f"y=[{xyz[:,1].min():.2f}, {xyz[:,1].max():.2f}]  "
      f"z=[{xyz[:,2].min():.2f}, {xyz[:,2].max():.2f}]")

# Parse test poses
if test_csv.exists():
    poses = parse_test_poses_csv(test_csv)
    print(f"Test poses: {len(poses)}")
    print(f"  First target: {poses[0].image_name}  "
          f"resolution={poses[0].width}x{poses[0].height}")
else:
    print(f"[WARN] test_poses.csv not found: {test_csv}")

print("\n✓ Preprocessing check passed.\n")
EOF
