#!/usr/bin/env bash
# Render novel views for one scene from an existing checkpoint.
# Usage: ./scripts/run_render_targets.sh scene_001 [path/to/ckpt.pt]
set -e
SCENE=${1:?Usage: run_render_targets.sh <scene_name> [checkpoint]}
CKPT=${2:-""}   # optional checkpoint path

python - <<EOF
import sys; sys.path.insert(0, ".")
from pathlib import Path
import torch, yaml
from src.models.gaussian_model import GaussianModel
from src.training.trainer import Trainer
from src.rendering.novel_view import render_test_poses

scene_name = "$SCENE"
ckpt_path  = "$CKPT" or None
scene_dir  = Path("data/raw") / scene_name
output_dir = Path("outputs/submission_renders")
device     = "cuda" if torch.cuda.is_available() else "cpu"

with open("configs/base.yaml") as f:
    import yaml; config = yaml.safe_load(f)

model_cfg = config.get("model", {})
model     = GaussianModel(model_cfg)

if ckpt_path:
    ckpt = Trainer.load_checkpoint(ckpt_path)
else:
    # Find latest checkpoint
    ckpt_dir = Path("outputs") / scene_name / "checkpoints"
    ckpts    = sorted(ckpt_dir.glob("ckpt_*.pt"))
    if not ckpts:
        print(f"No checkpoint found in {ckpt_dir}"); sys.exit(1)
    ckpt = Trainer.load_checkpoint(ckpts[-1])
    print(f"Using checkpoint: {ckpts[-1].name}")

model.load_state_dict_gaussians(ckpt["gaussians"])
test_csv = scene_dir / "test" / "test_poses.csv"
render_test_poses(model, test_csv, output_dir, scene_name, device)
EOF
