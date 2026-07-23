# BTS Digital Twin — 3D Reconstruction & Novel View Synthesis

Competition pipeline for the BTS Digital Twin challenge:
reconstruct the implicit 3D structure of a telecom base station from drone
RGB images and synthesise photorealistic images at unseen target viewpoints.

## Approach

**3D Gaussian Splatting** via [`gsplat`](https://github.com/nerfstudio-project/gsplat):
- Fast per-scene training (~15–30 min on A100)
- Strong RGB fidelity, hardware rasterizer
- Thin-structure regularisation via `scale_reg_weight` (antennas, cables)
- SH degree progressive activation for stable colour learning

## Project layout

```
configs/          base.yaml + per-scene YAML overrides
data/
  raw/            competition data (drop here, read-only)
    scene_001/
      train/
        images/         ← training RGB images
        sparse/0/       ← cameras.bin, images.bin, points3D.bin (COLMAP)
      test/
        test_poses.csv  ← target camera poses
  processed/      (not used — we read COLMAP directly)
  target_views/   (not used)
src/
  data/
    colmap_utils.py     ← COLMAP binary reader + test_poses.csv parser
    dataset.py          ← BTSSceneDataset (train/val split, image loading)
    test_dataset.py     ← TestPoseDataset (target poses only)
    ray_utils.py        ← ray generation (debugging / NeRF fallback)
  models/
    gaussian_model.py   ← GaussianModel wrapping gsplat
    losses.py           ← L1+SSIM photometric loss, scale/opacity reg
  training/
    trainer.py          ← full 3DGS training loop (30k iters)
    densification.py    ← adaptive clone/split/prune controller
    pose_refine.py      ← (stub) optional camera pose refinement
  rendering/
    renderer.py         ← single-view render → PIL Image
    novel_view.py       ← batch render all test poses, save PNGs
  evaluation/
    metrics.py          ← PSNR / SSIM / LPIPS + competition score
    geometry_eval.py    ← (stub) optional geometric evaluation
  export/
    mesh_extract.py     ← (stub) mesh from Gaussians
    pointcloud_export.py← (stub) PLY export
  utils/
    scene_runner.py     ← single-scene pipeline orchestrator
    submission.py       ← ZIP packager
scripts/
  train_all_scenes.py   ← main entry point (train + render + zip)
  make_submission.py    ← standalone ZIP packager
  run_train.sh          ← shell wrapper
  run_render_targets.sh ← shell wrapper
  run_eval.sh           ← shell wrapper
outputs/
  scene_001/
    checkpoints/        ← ckpt_NNNNNNN.pt
    tensorboard/        ← TensorBoard logs
  submission_renders/
    scene_001/
      0001.png
      ...
  submission.zip        ← final submission
third_party/            ← (empty) add gsplat fork here if needed
```

## Setup

### 1. Create virtual environment

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note on `gsplat`**: Requires CUDA. If `pip install gsplat` fails, build from
> source following https://github.com/nerfstudio-project/gsplat#installation.

## Workflow — when data arrives

### Step 1: Drop data

```
data/raw/
  scene_001/
    train/images/        ← all training images
    train/sparse/0/      ← cameras.bin, images.bin, points3D.bin
    test/test_poses.csv  ← target poses
  scene_002/
    ...
```

### Step 2: Train all scenes + build submission

```bash
python scripts/train_all_scenes.py \
    --config configs/base.yaml \
    --data-root data/raw \
    --output-root outputs \
    --device cuda
```

This will:
1. For each scene: load SfM points → init Gaussians → train 30k iters → render test poses
2. Save checkpoints to `outputs/scene_XXX/checkpoints/`
3. Save renders to `outputs/submission_renders/scene_XXX/`
4. Package `outputs/submission.zip`

### Smoke test (quick sanity check with 1k iters)

```bash
python scripts/train_all_scenes.py \
    --scenes scene_001 \
    --iters 1000 \
    --device cuda
```

### Resume interrupted training

```bash
python scripts/train_all_scenes.py --resume
```

### Build submission from existing renders

```bash
python scripts/make_submission.py \
    --renders-root outputs/submission_renders \
    --output outputs/submission.zip
```

### Evaluate against ground-truth (if available)

```python
from src.evaluation.metrics import evaluate_all_scenes

results = evaluate_all_scenes(
    pred_root = "outputs/submission_renders",
    gt_root   = "path/to/gt",
    psnr_max  = 40.0,
)
```

## Competition scoring

```
Score = 0.4 × (1 − LPIPS) + 0.3 × SSIM + 0.3 × PSNR_norm
PSNR_norm = clamp(PSNR / 40.0, 0, 1)
```

## Config tuning tips

| Issue | Fix |
|---|---|
| Floaters / noise around thin structures | Increase `loss.scale_reg_weight` (0.01–0.1) |
| Blurry renders | Increase `optimizer.iterations` to 50k |
| Out of GPU memory | Set `max_image_size: 1920` in config |
| Training too slow | Reduce `densification.end_iter` to 10000 |
| Dark / oversaturated output | Adjust `lr_feature` or reduce `ssim_weight` |

## Data format reference

**COLMAP binary** (`sparse/0/`):
- `cameras.bin`: camera intrinsics models (PINHOLE, SIMPLE_PINHOLE, ...)
- `images.bin`: per-image pose (quaternion w2c + translation) + image name
- `points3D.bin`: sparse 3D point cloud with RGB + reprojection error

**`test_poses.csv`** columns:
```
image_name, qw, qx, qy, qz, tx, ty, tz, fx, fy, cx, cy, width, height
```
Quaternion is COLMAP convention (world→camera rotation, qw first).
