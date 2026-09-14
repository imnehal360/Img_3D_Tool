from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from typing import Any

from .config import DEFAULT_CONFIG, PipelineConfig
from .environment import get_environment_info
from .evaluation import PipelineMetrics, write_metrics
from .generator import TrellisGenerator
from .postprocessing import get_mesh_stats
from .preprocessing import preprocess_image
from .visualization import write_basic_glb_preview


@dataclass
class PipelineOutput:
    input_path: Path
    processed_path: Path
    model_path: Path
    preview_path: Path
    metrics_path: Path
    metrics: dict[str, Any]


def run_single_image_pipeline(
    image_path: str | Path,
    output_dir: str | Path,
    *,
    config: PipelineConfig = DEFAULT_CONFIG,
    generator: Any | None = None,
) -> PipelineOutput:
    total_start = time.perf_counter()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = get_environment_info()
    prep = preprocess_image(
        image_path,
        out_dir,
        remove_background=config.remove_background,
        max_image_size=config.max_image_size,
    )

    active_generator = generator or TrellisGenerator(config)
    generation = active_generator.generate(prep.processed_path, out_dir)
    apply_neutral_grayscale(generation.model_path)

    post_start = time.perf_counter()
    stats = get_mesh_stats(generation.model_path)
    preview_path = write_basic_glb_preview(generation.model_path, out_dir / "preview.html")
    post_time = round(time.perf_counter() - post_start, 4)

    metrics = PipelineMetrics(
        preprocessing_time_sec=prep.preprocessing_time_sec,
        inference_time_sec=generation.inference_time_sec,
        postprocessing_time_sec=post_time,
        total_time_sec=round(time.perf_counter() - total_start, 4),
        model_name=generation.model_name,
        model_version=generation.model_version,
        environment=env.to_dict(),
        mesh=stats.to_dict(),
    )
    metrics_path = write_metrics(metrics, out_dir / "metrics.json")

    if not stats.valid:
        raise RuntimeError(f"Generated GLB failed validation: {stats.error}")

    return PipelineOutput(
        input_path=prep.original_path,
        processed_path=prep.processed_path,
        model_path=generation.model_path,
        preview_path=preview_path,
        metrics_path=metrics_path,
        metrics=metrics.to_dict(),
    )
