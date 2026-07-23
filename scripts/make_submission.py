#!/usr/bin/env python3
"""
Build competition submission.zip from rendered images.

Usage:
    python scripts/make_submission.py
    python scripts/make_submission.py --renders-root outputs/submission_renders
                                       --output outputs/submission.zip
                                       --scenes scene_001 scene_002
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.submission import build_submission_zip


def main():
    parser = argparse.ArgumentParser(description="Package competition submission ZIP")
    parser.add_argument(
        "--renders-root",
        default="outputs/submission_renders",
        help="Directory containing per-scene rendered images",
    )
    parser.add_argument(
        "--output",
        default="outputs/submission.zip",
        help="Output ZIP path",
    )
    parser.add_argument(
        "--scenes",
        nargs="*",
        help="Specific scene names to include (default: all subdirectories)",
    )
    args = parser.parse_args()

    renders_root = ROOT / args.renders_root
    output_zip   = ROOT / args.output

    if not renders_root.exists():
        print(f"[ERROR] Renders root not found: {renders_root}")
        sys.exit(1)

    build_submission_zip(
        output_root = renders_root,
        output_zip  = output_zip,
        scene_names = args.scenes,
    )


if __name__ == "__main__":
    main()
