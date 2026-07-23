#!/usr/bin/env python3
"""
Train + render all scenes in data/raw/ and package submission.zip.

Usage:
    python scripts/train_all_scenes.py [--config configs/base.yaml]
                                        [--data-root data/raw]
                                        [--output-root outputs]
                                        [--scenes scene_001 scene_002]
                                        [--device cuda]
                                        [--resume]

    # Quick smoke test (1k iters):
    python scripts/train_all_scenes.py --iters 1000 --scenes scene_001
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml

from src.utils.scene_runner import run_single_scene
from src.utils.submission import build_submission_zip


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def flatten_config(cfg: dict) -> dict:
    """Flatten nested config into a flat dict for convenience."""
    flat = {}
    for section, values in cfg.items():
        if isinstance(values, dict):
            flat.update(values)
        else:
            flat[section] = values
    flat.update(cfg)  # keep nested structure too
    return flat


def main():
    parser = argparse.ArgumentParser(description="BTS Digital Twin — Train all scenes")
    parser.add_argument("--config",      default="configs/base.yaml",  help="Base YAML config")
    parser.add_argument("--data-root",   default="data/raw",           help="Root containing scene_XXX dirs")
    parser.add_argument("--output-root", default="outputs",            help="Root output directory")
    parser.add_argument("--scenes",      nargs="*",                    help="Scene names to process (default: all)")
    parser.add_argument("--device",      default="cuda",               help="Torch device")
    parser.add_argument("--iters",       type=int, default=None,       help="Override training iterations")
    parser.add_argument("--resume",      action="store_true",          help="Resume from latest checkpoint")
    parser.add_argument("--eval-gt",     default=None,                 help="GT dir for optional evaluation")
    parser.add_argument("--no-zip",      action="store_true",          help="Skip building submission.zip")
    args = parser.parse_args()

    # Load config
    config_path = ROOT / args.config
    config = load_config(str(config_path))
    config["device"] = args.device
    if args.iters is not None:
        config.setdefault("optimizer", {})["iterations"] = args.iters

    # Discover scenes
    data_root = ROOT / args.data_root
    if args.scenes:
        scene_dirs = [data_root / s for s in args.scenes]
    else:
        scene_dirs = sorted(d for d in data_root.iterdir() if d.is_dir())

    if not scene_dirs:
        print(f"[ERROR] No scenes found in {data_root}")
        sys.exit(1)

    print(f"\nFound {len(scene_dirs)} scene(s): {[d.name for d in scene_dirs]}")
    output_root = ROOT / args.output_root

    all_results = []
    for scene_dir in scene_dirs:
        if not scene_dir.exists():
            print(f"[WARN] Scene dir not found: {scene_dir}")
            continue

        # Find latest checkpoint for resume
        resume_ckpt = None
        if args.resume:
            ckpt_dir = output_root / scene_dir.name / "checkpoints"
            if ckpt_dir.exists():
                ckpts = sorted(ckpt_dir.glob("ckpt_*.pt"))
                if ckpts:
                    resume_ckpt = ckpts[-1]
                    print(f"  Resuming {scene_dir.name} from {resume_ckpt.name}")

        result = run_single_scene(
            scene_dir   = scene_dir,
            output_dir  = output_root,
            config      = config,
            scene_name  = scene_dir.name,
            eval_gt_dir = args.eval_gt,
            resume_ckpt = resume_ckpt,
        )
        all_results.append(result)

    # Save run summary
    summary_path = output_root / "run_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nRun summary → {summary_path}")

    # Build submission.zip
    if not args.no_zip:
        renders_root = output_root / "submission_renders"
        zip_path     = output_root / "submission.zip"
        build_submission_zip(renders_root, zip_path)

    print("\n🎉 All done!")


if __name__ == "__main__":
    main()
