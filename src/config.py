from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelineConfig:
    model_name: str = "microsoft/TRELLIS-image-large"
    model_repo_url: str = "https://github.com/microsoft/TRELLIS.git"
    model_repo_dir: Path = Path("external_models/TRELLIS")
    max_image_size: int = 1024
    remove_background: bool = True
    output_format: str = "glb"
    texture_resolution: int = 1024
    remesh_option: str = "none"
    require_cuda: bool = True
    attention_backend: str = "xformers"
    spconv_algo: str = "native"
    seed: int = 1
    sparse_structure_steps: int = 12
    sparse_structure_cfg_strength: float = 7.5
    slat_steps: int = 12
    slat_cfg_strength: float = 3.0
    glb_simplify: float = 0.95
    # Keep the original textured path; only fall back after CUDA texture-bake OOM.
    enable_untextured_fallback: bool = True


DEFAULT_CONFIG = PipelineConfig()
