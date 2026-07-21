"""Read-only environment diagnostics for hardware-adaptive Cyrus profiles."""

from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DoctorReport:
    os: str
    architecture: str
    python: str
    cpu_count: int
    memory_total_bytes: int | None
    disk_free_bytes: int
    gpu: tuple[str, ...]
    cuda_available: bool
    torch_installed: bool
    selected_profile: str
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _memory_total() -> int | None:
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    return None


def _gpu_names() -> tuple[str, ...]:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return ()
    try:
        result = subprocess.run(
            [executable, "--query-gpu=name,memory.total", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return ()
    return tuple(line.strip() for line in result.stdout.splitlines() if line.strip())


def inspect_environment(path: str | Path = ".") -> DoctorReport:
    torch_installed = importlib.util.find_spec("torch") is not None
    cuda_available = False
    warnings: list[str] = []
    if torch_installed:
        try:
            import torch

            cuda_available = bool(torch.cuda.is_available())
        except (ImportError, RuntimeError) as exc:
            warnings.append(f"PyTorch present but unavailable: {exc}")
    else:
        warnings.append("PyTorch is not installed; model training commands are unavailable")

    gpu = _gpu_names()
    memory = _memory_total()
    free_disk = shutil.disk_usage(Path(path).resolve()).free
    if not gpu:
        warnings.append("No NVIDIA GPU detected; Smoke profile will use CPU")
    if memory is not None and memory < 4 * 1024**3:
        warnings.append("Less than 4 GiB RAM detected; reduce smoke batch/context settings")
    if free_disk < 5 * 1024**3:
        warnings.append("Less than 5 GiB free disk detected")

    return DoctorReport(
        os=f"{platform.system()} {platform.release()}",
        architecture=platform.machine(),
        python=platform.python_version(),
        cpu_count=os.cpu_count() or 1,
        memory_total_bytes=memory,
        disk_free_bytes=free_disk,
        gpu=gpu,
        cuda_available=cuda_available,
        torch_installed=torch_installed,
        selected_profile="smoke",
        warnings=tuple(warnings),
    )

