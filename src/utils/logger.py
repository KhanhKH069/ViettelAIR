"""
Experiment logging: scalar metrics, image samples, checkpoints.
Backend configurable via config.logging.backend (tensorboard | wandb | none).
"""


class ExperimentLogger:
    def __init__(self, log_dir: str, backend: str = "tensorboard"):
        raise NotImplementedError

    def log_scalar(self, name: str, value: float, step: int):
        raise NotImplementedError

    def log_image(self, name: str, image, step: int):
        raise NotImplementedError

    def save_checkpoint(self, model, step: int):
        raise NotImplementedError
