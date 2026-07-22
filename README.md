# BTS Digital Twin — 3D Reconstruction & Novel View Synthesis

Codebase for the competition problem: reconstruct the implicit 3D structure
of a BTS (telecom base) station from drone RGB images, and synthesize
photorealistic RGB images at unseen target viewpoints.

## Status

Skeleton only — no datasets received yet. Structure and interfaces are in
place; implementations are stubbed with `NotImplementedError` and TODOs.
Fill in `src/` modules once real data format is confirmed.

## Problem recap

- Input per scene: 100–300 RGB images + camera intrinsics/extrinsics.
- Output per scene: 20–50 rendered RGB images at given target poses.
- Evaluation (assumed): photometric quality (PSNR/SSIM/LPIPS) + geometric/
  device-placement correctness.

## Approach (current plan)

- Primary: 3D Gaussian Splatting (via `gsplat` or similar), chosen for fast
  training and strong RGB fidelity.
- Fallback/comparison: NeRF-based (Nerfacto) if thin structures (antennas,
  cables) produce too many floaters under 3DGS.
- Extra regularization planned for thin-structure regions (scale reg,
  priority densification, optional depth supervision).

## Project layout

```
configs/        base + per-scene + per-experiment YAML configs
data/           raw (untouched) / processed / target_views
src/data/       dataset loading, format conversion, preprocessing
src/models/     3DGS / NeRF model wrappers, loss functions
src/training/   training loop, densification, optional pose refinement
src/rendering/  render arbitrary views, generate submission renders
src/evaluation/ PSNR/SSIM/LPIPS, optional geometric eval
src/export/     mesh + point cloud export for Digital Twin deliverable
scripts/        thin CLI wrappers around the src/ pipeline stages
outputs/        per-scene, per-experiment checkpoints/renders/logs
third_party/    vendored/forked base framework (e.g. gsplat) as submodule
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Add the chosen base framework (e.g. `gsplat`) under `third_party/` once
selected — see `src/models/gaussian_model.py` for the integration point.

## Once data arrives

1. Drop raw scene data into `data/raw/scene_XX/` (images + cameras.json + poses.json).
2. Copy `configs/scene_default.yaml` -> `configs/scene_XX.yaml`, fill in paths.
3. Implement `src/data/colmap_utils.py` parsers to match the actual pose format.
4. `./scripts/run_preprocess.sh scene_XX`
5. `./scripts/run_train.sh scene_XX exp001_baseline`
6. `./scripts/run_render_targets.sh scene_XX <checkpoint>`
7. `./scripts/run_eval.sh scene_XX <checkpoint>`

## Open decisions (see TODOs in code)

- [ ] Confirm base framework: gsplat vs Nerfstudio vs original 3DGS repo.
- [ ] Confirm organizer pose/camera format (COLMAP? custom JSON? convention?).
- [ ] Confirm submission format for rendered images (naming, resolution).
- [ ] Confirm whether ground-truth mesh/depth is available for geometric eval.
