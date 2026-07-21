"""Convert immutable text snapshots into hash-bound local token streams."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from cyrus.data.pipeline import DataError
from cyrus.tokenizer import ByteBPETokenizer


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode(
        "utf-8"
    )


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _atomic_write(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DataError(f"invalid dataset row at {path}:{line_number}") from exc
        if not isinstance(row, dict) or not isinstance(row.get("text"), str):
            raise DataError(f"dataset row lacks text at {path}:{line_number}")
        rows.append(row)
    return rows


def pack_token_shards(
    *,
    dataset_dir: str | Path,
    tokenizer_path: str | Path,
    output_root: str | Path,
    sequence_length: int,
) -> dict[str, Any]:
    if sequence_length < 2:
        raise DataError("token sequence length must be at least 2")
    dataset_root = Path(dataset_dir)
    try:
        dataset_manifest = json.loads(
            (dataset_root / "manifest.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise DataError("dataset manifest is missing or invalid") from exc
    tokenizer = ByteBPETokenizer.load(tokenizer_path)
    document_id = tokenizer.special_to_id.get("<|document|>")
    end_id = tokenizer.special_to_id.get("<|end|>")
    if document_id is None or end_id is None:
        raise DataError("tokenizer must define document and end special tokens")

    basis = {
        "schema_version": 1,
        "dataset_id": dataset_manifest.get("snapshot_id"),
        "tokenizer_sha256": tokenizer.tokenizer_hash,
        "sequence_length": sequence_length,
        "packing": "explicit-document-boundary-v1",
    }
    snapshot_id = _sha256(_canonical_json(basis))
    output_dir = Path(output_root) / f"tokens-{snapshot_id[:16]}"
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_metadata: dict[str, dict[str, Any]] = {}

    for split in ("train", "validation", "test"):
        rows = _read_rows(dataset_root / f"{split}.jsonl")
        stream: list[int] = []
        for row in rows:
            stream.append(document_id)
            stream.extend(tokenizer.encode(row["text"]))
            stream.append(end_id)
        payload_object = {
            "schema_version": 1,
            "split": split,
            "dataset_id": dataset_manifest.get("snapshot_id"),
            "tokenizer_sha256": tokenizer.tokenizer_hash,
            "documents": len(rows),
            "tokens": stream,
        }
        payload = _canonical_json(payload_object)
        shard_path = output_dir / f"{split}.tokens.json"
        _atomic_write(shard_path, payload)
        shard_metadata[split] = {
            "path": shard_path.name,
            "documents": len(rows),
            "tokens": len(stream),
            "usable_windows": max(0, len(stream) - sequence_length),
            "bytes": len(payload),
            "sha256": _sha256(payload),
        }

    if shard_metadata["train"]["usable_windows"] == 0:
        raise DataError("training token stream is too short for the selected sequence length")
    report = {
        **basis,
        "snapshot_id": snapshot_id,
        "vocab_size": tokenizer.vocab_size,
        "shards": shard_metadata,
    }
    _atomic_write(output_dir / "manifest.json", _canonical_json(report))
    return {**report, "path": str(output_dir)}

