#!/usr/bin/env bash
# Train a scene with a given experiment config.
# Usage: ./scripts/run_train.sh scene_01 exp001_baseline
set -e
SCENE=$1
EXP=$2
python -m src.training.trainer \
    --base configs/base.yaml \
    --scene "configs/${SCENE}.yaml" \
    --exp "configs/experiments/${EXP}.yaml"
