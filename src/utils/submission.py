"""
Submission packager.

Collects rendered scene images and zips them into the competition
submission format:

    submission.zip
    ├── scene_001/
    │   ├── 0001.png
    │   ├── 0002.png
    │   └── ...
    ├── scene_002/
    └── ...
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import List, Optional


def build_submission_zip(
    output_root:  str | Path,
    output_zip:   str | Path,
    scene_names:  Optional[List[str]] = None,
    extensions:   tuple = (".png",),
) -> Path:
    """
    Zip all rendered scene images into a competition-ready submission.

    Args:
        output_root:  Directory containing per-scene output folders.
                      E.g. outputs/submission_renders/
        output_zip:   Destination zip path.
                      E.g. outputs/submission.zip
        scene_names:  If given, only include these scene names.
                      If None, include all subdirectories.
        extensions:   File extensions to include (default: .png only).

    Returns:
        Path to the created zip file.
    """
    output_root = Path(output_root)
    output_zip  = Path(output_zip)
    output_zip.parent.mkdir(parents=True, exist_ok=True)

    # Collect scenes
    if scene_names is not None:
        scene_dirs = [output_root / n for n in scene_names]
    else:
        scene_dirs = sorted(d for d in output_root.iterdir() if d.is_dir())

    total_files = 0
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for scene_dir in scene_dirs:
            if not scene_dir.exists():
                print(f"  [WARN] Scene directory not found: {scene_dir}")
                continue

            scene_name = scene_dir.name
            image_files = sorted(
                f for f in scene_dir.iterdir()
                if f.is_file() and f.suffix.lower() in extensions
            )

            if not image_files:
                print(f"  [WARN] No images in {scene_dir}")
                continue

            for img_path in image_files:
                arcname = f"{scene_name}/{img_path.name}"
                zf.write(img_path, arcname)
                total_files += 1

    size_mb = output_zip.stat().st_size / (1024 ** 2)
    print(f"\n✓ Submission ZIP created: {output_zip}")
    print(f"  Scenes: {len(scene_dirs)}  |  Images: {total_files}  |  Size: {size_mb:.1f} MB")
    return output_zip
