#!/usr/bin/env bash
# Render the 20-50 target-view submission images from a trained checkpoint.
# Usage: ./scripts/run_render_targets.sh scene_01 outputs/scene_01/exp001_baseline/checkpoints/last.ckpt
set -e
SCENE=$1
CKPT=$2
python -m src.rendering.novel_view --scene "configs/${SCENE}.yaml" --checkpoint "$CKPT"
