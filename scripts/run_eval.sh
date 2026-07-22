#!/usr/bin/env bash
# Run internal validation metrics (PSNR/SSIM/LPIPS) on held-out views.
# Usage: ./scripts/run_eval.sh scene_01 outputs/scene_01/exp001_baseline/checkpoints/last.ckpt
set -e
SCENE=$1
CKPT=$2
python -m src.evaluation.metrics --scene "configs/${SCENE}.yaml" --checkpoint "$CKPT"
