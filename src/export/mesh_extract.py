"""
Extract a mesh from the trained Gaussian representation for the
Digital Twin deliverable (beyond just rendered images).

Approach options:
- Poisson surface reconstruction from gaussian centers/normals.
- TSDF fusion from rendered depth maps across many viewpoints.
"""


def extract_mesh_poisson(model, out_path: str):
    raise NotImplementedError


def extract_mesh_tsdf(model, cameras: list, out_path: str):
    raise NotImplementedError
