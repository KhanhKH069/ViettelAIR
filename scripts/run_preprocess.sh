#!/usr/bin/env bash
# Convert data/raw/scene_XX -> data/processed/scene_XX
# Usage: ./scripts/run_preprocess.sh scene_01
set -e
SCENE=$1
python -m src.data.preprocess --scene "$SCENE"
