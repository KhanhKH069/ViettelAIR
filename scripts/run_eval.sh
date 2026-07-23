#!/usr/bin/env bash
# Evaluate renders against ground-truth (if GT is available).
# Usage: ./scripts/run_eval.sh <pred_root> <gt_root> [--psnr-max 40]
set -e
PRED_ROOT=${1:?Usage: run_eval.sh <pred_root> <gt_root>}
GT_ROOT=${2:?Usage: run_eval.sh <pred_root> <gt_root>}
PSNR_MAX=${3:-40.0}

python - <<EOF
import sys; sys.path.insert(0, ".")
import json
from src.evaluation.metrics import evaluate_all_scenes

results = evaluate_all_scenes(
    pred_root = "$PRED_ROOT",
    gt_root   = "$GT_ROOT",
    psnr_max  = $PSNR_MAX,
)
import json; print(json.dumps(results, indent=2))
EOF
