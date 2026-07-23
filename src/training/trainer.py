"""
Training loop for 3D Gaussian Splatting.

Usage:
    from src.training.trainer import Trainer
    trainer = Trainer(model, dataset, config)
    trainer.train()
"""

from __future__ import annotations

import math
import random
import time
from pathlib import Path
from typing import Optional

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.data.dataset import BTSSceneDataset
from src.models.gaussian_model import GaussianModel
from src.models.losses import photometric_loss, scale_regularization, opacity_regularization
from src.training.densification import DensificationController


class Trainer:
    """
    Full 3DGS training loop for one scene.

    Config keys consumed (all from base.yaml):
        optimizer.*
        densification.*
        loss.*
        logging.*
        project.device
    """

    def __init__(
        self,
        model:       GaussianModel,
        dataset:     BTSSceneDataset,
        config:      dict,
        output_dir:  str | Path,
        val_dataset: Optional[BTSSceneDataset] = None,
    ):
        self.model       = model
        self.dataset     = dataset
        self.val_dataset = val_dataset
        self.config      = config
        self.output_dir  = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device(config.get("device", "cuda"))
        self.model  = self.model.to(self.device)

        opt_cfg = config.get("optimizer", {})
        self.iterations    = opt_cfg.get("iterations", 30_000)
        self.lr_scheduler  = opt_cfg.get("lr_scheduler", "exp_decay")

        dens_cfg = config.get("densification", {})
        self.dens_start    = dens_cfg.get("start_iter",  500)
        self.dens_end      = dens_cfg.get("end_iter",   15_000)
        self.dens_interval = dens_cfg.get("interval",   100)
        self.opacity_reset_interval = dens_cfg.get("opacity_reset_interval", 3_000)

        loss_cfg = config.get("loss", {})
        self.lambda_ssim        = loss_cfg.get("ssim_weight",       0.2)
        self.scale_reg_weight   = loss_cfg.get("scale_reg_weight",  0.0)
        self.opacity_reg_weight = loss_cfg.get("opacity_reg_weight", 0.0)

        log_cfg = config.get("logging", {})
        self.log_interval   = log_cfg.get("log_interval",        100)
        self.eval_interval  = log_cfg.get("eval_interval",      1000)
        self.ckpt_interval  = log_cfg.get("checkpoint_interval", 5000)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.get_optimizable_params(opt_cfg),
            eps=1e-15,
        )

        # Densification controller
        self.densifier = DensificationController(
            model          = self.model,
            grad_threshold = dens_cfg.get("grad_threshold",        2e-4),
            min_opacity    = dens_cfg.get("prune_opacity_threshold", 0.005),
        )

        # TensorBoard writer
        tb_dir = self.output_dir / "tensorboard"
        self.writer = SummaryWriter(str(tb_dir))

        # SH degree milestones (one-up every 1000 iters)
        self._sh_oneup_interval = 1000

    # ------------------------------------------------------------------
    # Training entry point
    # ------------------------------------------------------------------

    def train(self, start_iter: int = 0) -> None:
        """Run the full training loop."""
        items = list(range(len(self.dataset)))
        t0    = time.time()

        pbar = tqdm(range(start_iter, self.iterations), desc="Training", dynamic_ncols=True)
        for iteration in pbar:
            # ---------- sample a random training view ----------
            idx   = random.choice(items)
            batch = self.dataset[idx]

            image  = batch["image"].to(self.device)    # (3, H, W)
            K      = batch["K"].to(self.device)         # (3, 3)
            c2w    = batch["c2w"].to(self.device)       # (4, 4)
            H, W   = batch["height"], batch["width"]

            # ---------- render ----------
            self.optimizer.zero_grad(set_to_none=True)
            out = self.model(c2w, K, W, H)
            pred_rgb = out["rgb"].permute(2, 0, 1)  # (H,W,3) → (3,H,W)

            # ---------- loss ----------
            loss = photometric_loss(pred_rgb, image, self.lambda_ssim)

            if self.scale_reg_weight > 0:
                loss = loss + self.scale_reg_weight * scale_regularization(self.model.scales)

            if self.opacity_reg_weight > 0:
                loss = loss + self.opacity_reg_weight * opacity_regularization(self.model.opacities)

            loss.backward()

            # ---------- densification ----------
            if self.dens_start <= iteration < self.dens_end:
                # Accumulate 2D gradient from gsplat's info (if available)
                # gsplat stores the 2D mean gradients in info["means2d"].grad
                # We call update_stats with those
                if self.model._means.grad is not None:
                    # Fallback: use 3D gradient norm projected (approx)
                    grad2d = self.model._means.grad[..., :2]
                    self.densifier.update_stats(grad2d)

                if (iteration + 1) % self.dens_interval == 0:
                    n = self.densifier.step(self.optimizer)
                    pbar.set_postfix({"#G": n, "loss": f"{loss.item():.4f}"})

            if iteration > 0 and iteration % self.opacity_reset_interval == 0:
                self.densifier.reset_opacity(self.optimizer)

            # ---------- step ----------
            self.optimizer.step()
            self._update_lr(iteration)

            # ---------- SH degree schedule ----------
            if (iteration + 1) % self._sh_oneup_interval == 0:
                self.model.oneup_sh_degree()

            # ---------- logging ----------
            if (iteration + 1) % self.log_interval == 0:
                self.writer.add_scalar("train/loss", loss.item(), iteration)
                self.writer.add_scalar("train/num_gaussians", self.model.num_gaussians, iteration)
                elapsed = time.time() - t0
                pbar.set_postfix({
                    "#G": self.model.num_gaussians,
                    "loss": f"{loss.item():.4f}",
                    "min": f"{elapsed/60:.1f}",
                })

            # ---------- checkpoint ----------
            if (iteration + 1) % self.ckpt_interval == 0:
                self.save_checkpoint(iteration + 1)

        # Final checkpoint
        self.save_checkpoint(self.iterations)
        self.writer.close()
        print(f"\n✓ Training done. Final #Gaussians: {self.model.num_gaussians}")

    # ------------------------------------------------------------------
    # LR scheduling
    # ------------------------------------------------------------------

    def _update_lr(self, iteration: int) -> None:
        """Apply exponential decay to position LR (others stay constant)."""
        if self.lr_scheduler != "exp_decay":
            return
        opt_cfg   = self.config.get("optimizer", {})
        lr_init   = opt_cfg.get("lr_position", 1.6e-4)
        lr_final  = lr_init * 0.01
        total     = self.iterations
        # Linear interpolation in log space
        t = min(iteration / total, 1.0)
        lr = math.exp(math.log(lr_init) * (1 - t) + math.log(lr_final) * t)
        for group in self.optimizer.param_groups:
            if group.get("name") == "means":
                group["lr"] = lr

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, iteration: int) -> Path:
        ckpt_path = self.output_dir / "checkpoints" / f"ckpt_{iteration:07d}.pt"
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "iteration":  iteration,
            "gaussians":  self.model.state_dict_gaussians(),
            "optimizer":  self.optimizer.state_dict(),
            "config":     self.config,
        }, ckpt_path)
        print(f"  Saved checkpoint: {ckpt_path}")
        return ckpt_path

    @classmethod
    def load_checkpoint(cls, ckpt_path: str | Path) -> dict:
        return torch.load(ckpt_path, map_location="cpu")
