"""Atomic, hash-verified Cyrus checkpoint persistence."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any

import torch


class CheckpointError(ValueError):
    """Raised when a checkpoint path or artifact fails verification."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_workspace_path(path: Path, workspace_root: Path) -> Path:
    root = workspace_root.resolve()
    resolved = path.resolve()
    if resolved == root or root not in resolved.parents:
        raise CheckpointError("checkpoint must be stored under the repository workspace")
    current = resolved.parent
    while current != root:
        if current.exists() and current.is_symlink():
            raise CheckpointError("checkpoint parent directories may not be symbolic links")
        current = current.parent
    return resolved


def save_checkpoint(
    path: str | Path,
    payload: dict[str, Any],
    *,
    workspace_root: str | Path,
) -> dict[str, Any]:
    destination = _validate_workspace_path(Path(path), Path(workspace_root))
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".partial", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        digest = sha256_file(temporary)
        os.replace(temporary, destination)
        sidecar = destination.with_suffix(destination.suffix + ".sha256")
        sidecar_temporary = sidecar.with_suffix(sidecar.suffix + ".partial")
        sidecar_temporary.write_text(f"{digest}  {destination.name}\n", encoding="ascii")
        os.replace(sidecar_temporary, sidecar)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {"path": str(destination), "sha256": digest, "bytes": destination.stat().st_size}


def load_checkpoint(
    path: str | Path,
    *,
    workspace_root: str | Path,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    checkpoint = _validate_workspace_path(Path(path), Path(workspace_root))
    if not checkpoint.is_file() or checkpoint.is_symlink():
        raise CheckpointError(f"checkpoint is missing or unsafe: {checkpoint}")
    sidecar = checkpoint.with_suffix(checkpoint.suffix + ".sha256")
    try:
        expected = sidecar.read_text(encoding="ascii").split()[0]
    except (OSError, IndexError) as exc:
        raise CheckpointError("checkpoint hash sidecar is missing or invalid") from exc
    actual = sha256_file(checkpoint)
    if actual != expected:
        raise CheckpointError("checkpoint SHA-256 verification failed")
    try:
        payload = torch.load(checkpoint, map_location=map_location, weights_only=True)
    except Exception as exc:
        raise CheckpointError(f"checkpoint could not be safely loaded: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise CheckpointError("unsupported checkpoint schema")
    return payload


def copy_verified_checkpoint(
    source: str | Path,
    destination: str | Path,
    *,
    workspace_root: str | Path,
) -> dict[str, Any]:
    source_path = _validate_workspace_path(Path(source), Path(workspace_root))
    payload = load_checkpoint(
        source_path, workspace_root=workspace_root, map_location="cpu"
    )
    return save_checkpoint(destination, payload, workspace_root=workspace_root)


def enforce_retention(directory: str | Path, *, prefix: str, keep: int) -> None:
    if keep < 1:
        raise CheckpointError("checkpoint retention must keep at least one artifact")
    root = Path(directory)
    candidates = sorted(root.glob(f"{prefix}-step-*.pt"), key=lambda path: path.stat().st_mtime)
    for checkpoint in candidates[:-keep]:
        checkpoint.unlink(missing_ok=True)
        checkpoint.with_suffix(checkpoint.suffix + ".sha256").unlink(missing_ok=True)

