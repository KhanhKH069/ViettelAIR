#!/usr/bin/env bash
# Train one scene (and render test poses).
# Usage: ./scripts/run_train.sh scene_001 [--iters 30000] [--resume]
set -e
SCENE=${1:?Usage: run_train.sh <scene_name> [extra args...]}
shift
python scripts/train_all_scenes.py \
    --scenes "$SCENE" \
    --config configs/base.yaml \
    --data-root data/raw \
    --output-root outputs \
    --no-zip \
    "$@"
