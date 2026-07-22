"""
Load and merge YAML configs: base.yaml + scene_XX.yaml + experiment override.
Consider using OmegaConf for dot-access and easy CLI overrides.
"""


def load_config(base_path: str, scene_path: str, exp_path: str = None) -> dict:
    raise NotImplementedError
