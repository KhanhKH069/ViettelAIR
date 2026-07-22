"""
Main training loop.

Responsibilities:
- Load config (base + scene + experiment overrides).
- Build dataset, model, optimizer, loss.
- Run training loop with periodic logging / eval / checkpointing.
- Call densification.py logic at configured intervals.

Usage (planned):
    python -m src.training.trainer \
        --base configs/base.yaml \
        --scene configs/scene_01.yaml \
        --exp configs/experiments/exp001_baseline.yaml
"""


def train(config: dict):
    raise NotImplementedError


if __name__ == "__main__":
    raise NotImplementedError("Wire up argparse + config loading, then call train().")
