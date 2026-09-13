from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class PipelineMetrics:
    preprocessing_time_sec: float
    inference_time_sec: float
    postprocessing_time_sec: float
    total_time_sec: float
    model_name: str
    model_version: str | None
    environment: dict[str, Any]
    mesh: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def write_metrics(metrics: PipelineMetrics, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics.to_dict(), indent=2), encoding="utf-8")
    return path
