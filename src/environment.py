from __future__ import annotations

import platform
import subprocess
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class EnvironmentInfo:
    python: str
    platform: str
    torch_version: str | None
    cuda_available: bool
    cuda_version: str | None
    gpu_name: str | None
    gpu_total_memory_gb: float | None
    nvidia_smi: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def get_environment_info() -> EnvironmentInfo:
    torch_version = None
    cuda_available = False
    cuda_version = None
    gpu_name = None
    gpu_total_memory_gb = None

    try:
        import torch

        torch_version = torch.__version__
        cuda_available = bool(torch.cuda.is_available())
        cuda_version = torch.version.cuda
        if cuda_available:
            props = torch.cuda.get_device_properties(0)
            gpu_name = props.name
            gpu_total_memory_gb = round(props.total_memory / (1024**3), 2)
    except Exception:
        pass

    try:
        nvidia_smi = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=10,
        ).strip()
    except Exception:
        nvidia_smi = None

    return EnvironmentInfo(
        python=platform.python_version(),
        platform=platform.platform(),
        torch_version=torch_version,
        cuda_available=cuda_available,
        cuda_version=cuda_version,
        gpu_name=gpu_name,
        gpu_total_memory_gb=gpu_total_memory_gb,
        nvidia_smi=nvidia_smi,
    )
