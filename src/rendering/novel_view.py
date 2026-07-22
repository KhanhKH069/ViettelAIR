"""
Pipeline to generate the 20-50 target-view submission images.

Steps:
1. Load trained checkpoint for a scene.
2. Load target_views/scene_XX/poses.json (organizer-provided target poses).
3. Render each target view.
4. Save outputs to outputs/scene_XX/<exp_name>/renders/ in required
   naming convention (TBD once submission format is confirmed).

Usage (planned):
    python -m src.rendering.novel_view --scene configs/scene_01.yaml --checkpoint <path>
"""


def generate_submission_renders(config: dict, checkpoint_path: str):
    raise NotImplementedError
