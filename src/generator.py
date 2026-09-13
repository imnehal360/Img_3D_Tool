from __future__ import annotations

import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from .config import PipelineConfig
from .environment import get_environment_info


@dataclass
class GenerationResult:
    model_path: Path
    inference_time_sec: float
    model_name: str
    model_version: str | None
    command: list[str]


class StableFast3DGenerator:
    """Thin adapter around Stability AI's official Stable Fast 3D CLI."""

    def __init__(self, config: PipelineConfig):
        self.config = config

    def check_runtime(self) -> None:
        env = get_environment_info()
        if self.config.require_cuda and not env.cuda_available:
            raise RuntimeError("CUDA GPU is required for Stable Fast 3D in this project configuration.")

    def check_model_repo(self) -> None:
        repo = self.config.model_repo_dir
        if not (repo / "run.py").exists():
            raise FileNotFoundError(
                "Stable Fast 3D repo is not installed. In Colab, run the setup cell that clones "
                f"{self.config.model_repo_url} into {repo}."
            )

    def generate(self, image_path: str | Path, output_dir: str | Path) -> GenerationResult:
        self.check_runtime()
        self.check_model_repo()

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        command = [
            os.environ.get("PYTHON", "python"),
            "run.py",
            str(Path(image_path).resolve()),
            "--output-dir",
            str(out_dir.resolve()),
            "--texture-resolution",
            str(self.config.texture_resolution),
            "--remesh_option",
            self.config.remesh_option,
        ]

        start = time.perf_counter()
        completed = subprocess.run(
            command,
            cwd=self.config.model_repo_dir,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        elapsed = round(time.perf_counter() - start, 4)
        if completed.returncode != 0:
            raise RuntimeError(f"Stable Fast 3D failed with exit code {completed.returncode}:\n{completed.stdout}")

        glbs = sorted(out_dir.glob("*.glb"), key=lambda path: path.stat().st_mtime, reverse=True)
        if not glbs:
            raise RuntimeError(f"Stable Fast 3D completed but no GLB was found in {out_dir}.")

        final_path = out_dir / "model.glb"
        if glbs[0] != final_path:
            shutil.copy2(glbs[0], final_path)

        return GenerationResult(
            model_path=final_path,
            inference_time_sec=elapsed,
            model_name="Stable Fast 3D",
            model_version=_git_revision(self.config.model_repo_dir),
            command=command,
        )


class TrellisGenerator:
    """Generator adapter for Microsoft's official TRELLIS image-to-3D pipeline."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._pipeline = None

    def check_runtime(self) -> None:
        env = get_environment_info()
        if self.config.require_cuda and not env.cuda_available:
            raise RuntimeError("CUDA GPU with about 16 GB VRAM is required for original TRELLIS inference.")
        if env.gpu_total_memory_gb is not None and env.gpu_total_memory_gb < 15:
            raise RuntimeError(
                f"Detected GPU memory is {env.gpu_total_memory_gb} GB. "
                "Original TRELLIS is expected to need at least 16 GB VRAM."
            )

        # Loading TRELLIS while another service owns the GPU produces an opaque
        # CUDA OOM later in pipeline.run(). Fail early with an actionable error.
        try:
            import torch

            free_bytes, total_bytes = torch.cuda.mem_get_info()
            free_gb = free_bytes / (1024**3)
            if free_gb < 8:
                raise RuntimeError(
                    f"Only {free_gb:.2f} GiB of {total_bytes / (1024**3):.2f} GiB GPU memory is free. "
                    "Stop other GPU processes (for example the existing uvicorn/API server) "
                    "before loading TRELLIS, then retry."
                )
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(f"Unable to query free CUDA memory before TRELLIS startup: {exc}") from exc

    def check_model_repo(self) -> None:
        repo = self.config.model_repo_dir
        if not (repo / "trellis").exists():
            raise FileNotFoundError(
                "TRELLIS repo is not installed. In Colab, run the setup cell that clones "
                f"{self.config.model_repo_url} into {repo}."
            )

    def _load_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline

        import sys

        repo = str(self.config.model_repo_dir.resolve())
        if repo not in sys.path:
            sys.path.insert(0, repo)

        os.environ["ATTN_BACKEND"] = self.config.attention_backend
        os.environ["SPCONV_ALGO"] = self.config.spconv_algo

        from trellis.pipelines import TrellisImageTo3DPipeline

        pipeline = TrellisImageTo3DPipeline.from_pretrained(self.config.model_name)
        pipeline.cuda()
        self._pipeline = pipeline
        return pipeline

    def generate(self, image_path: str | Path, output_dir: str | Path) -> GenerationResult:
        self.check_runtime()
        self.check_model_repo()

        from PIL import Image

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        model_path = out_dir / "model.glb"

        pipeline = self._load_pipeline()
        from trellis.utils import postprocessing_utils

        image = Image.open(image_path)

        start = time.perf_counter()
        outputs = pipeline.run(
            image,
            seed=self.config.seed,
            sparse_structure_sampler_params={
                "steps": self.config.sparse_structure_steps,
                "cfg_strength": self.config.sparse_structure_cfg_strength,
            },
            slat_sampler_params={
                "steps": self.config.slat_steps,
                "cfg_strength": self.config.slat_cfg_strength,
            },
        )
        glb = postprocessing_utils.to_glb(
            outputs["gaussian"][0],
            outputs["mesh"][0],
            simplify=self.config.glb_simplify,
            texture_size=self.config.texture_resolution,
        )
        glb.export(model_path)
        elapsed = round(time.perf_counter() - start, 4)

        return GenerationResult(
            model_path=model_path,
            inference_time_sec=elapsed,
            model_name=self.config.model_name,
            model_version=_git_revision(self.config.model_repo_dir),
            command=[],
        )


class SyntheticTestGenerator:
    """Test-only generator that creates a simple GLB to exercise the pipeline plumbing."""

    def generate(self, image_path: str | Path, output_dir: str | Path) -> GenerationResult:
        import trimesh

        start = time.perf_counter()
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        model_path = out_dir / "model.glb"
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=0.5)
        mesh.visual.vertex_colors = [220, 120, 80, 255]
        mesh.export(model_path)
        return GenerationResult(
            model_path=model_path,
            inference_time_sec=round(time.perf_counter() - start, 4),
            model_name="SyntheticTestGenerator",
            model_version="test-only",
            command=[],
        )


def _git_revision(repo_dir: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        ).strip()
    except Exception:
        return None
