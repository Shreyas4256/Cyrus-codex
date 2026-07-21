"""Deterministic, approval-gated local data pipeline for Cyrus."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import unicodedata
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cyrus.config.schema import DataConfig

PARSER_VERSION = "cyrus-text-v1"
SUPPORTED_SUFFIXES = {".txt", ".md", ".jsonl"}

_SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("private-key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
    ("aws-access-key", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
    (
        "generic-secret",
        re.compile(
            r"(?i)\b(?:api[_-]?key|secret|token|password)\s*[:=]\s*['\"]?[A-Za-z0-9_\-/.+=]{12,}"
        ),
    ),
)
_PII_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("email-address", re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)),
    ("indian-mobile-number", re.compile(r"(?<!\d)(?:\+91[- ]?)?[6-9]\d{9}(?!\d)")),
)


class DataError(ValueError):
    """Raised when an input cannot safely enter the Cyrus data pipeline."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode(
        "utf-8"
    )


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _safe_manifest_path(data_root: Path, record_id: str) -> Path:
    if not re.fullmatch(r"[a-f0-9]{64}", record_id):
        raise DataError("record id must be a full lowercase SHA-256 value")
    root = data_root.resolve()
    candidate = (root / "manifests" / f"{record_id}.json").resolve()
    if root not in candidate.parents:
        raise DataError("record path escapes the configured data root")
    return candidate


def _read_supported_text(path: Path, raw: bytes) -> str:
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_SUFFIXES))
        raise DataError(f"unsupported file type '{path.suffix}'; supported: {supported}")
    if b"\x00" in raw:
        raise DataError("NUL byte detected; binary or corrupt files are not accepted")
    try:
        decoded = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DataError("input must be valid UTF-8") from exc

    if path.suffix.lower() != ".jsonl":
        return decoded

    records: list[str] = []
    for line_number, line in enumerate(decoded.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DataError(f"invalid JSONL at line {line_number}: {exc.msg}") from exc
        if isinstance(item, str):
            records.append(item)
        elif isinstance(item, dict) and isinstance(item.get("text"), str):
            records.append(item["text"])
        else:
            raise DataError(
                f"JSONL line {line_number} must be a string or an object with a string 'text' field"
            )
    if not records:
        raise DataError("JSONL contains no usable text records")
    return "\n\n".join(records)


def _normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))
    normalized = "\n".join(line.rstrip() for line in normalized.split("\n"))
    return normalized.strip() + "\n"


def _scan(text: str) -> list[str]:
    warnings: list[str] = []
    for name, pattern in (*_SECRET_PATTERNS, *_PII_PATTERNS):
        if pattern.search(text):
            warnings.append(name)
    if len(text.strip()) < 40:
        warnings.append("very-short-document")
    if text:
        most_common = max((text.count(char) for char in set(text)), default=0)
        if most_common / len(text) > 0.7:
            warnings.append("extremely-repetitive")
    return sorted(set(warnings))


def inspect_file(path: str | Path, *, max_file_bytes: int) -> dict[str, Any]:
    source = Path(path)
    if source.is_symlink():
        raise DataError("symbolic links are not accepted")
    if not source.is_file():
        raise DataError(f"input is not a regular file: {source}")
    size = source.stat().st_size
    if size == 0:
        raise DataError("empty files are not accepted")
    if size > max_file_bytes:
        raise DataError(f"file is {size} bytes; limit is {max_file_bytes} bytes")
    raw = source.read_bytes()
    text = _read_supported_text(source, raw)
    normalized = _normalize(text)
    return {
        "path": str(source.resolve()),
        "name": source.name,
        "suffix": source.suffix.lower(),
        "byte_size": size,
        "content_sha256": _sha256_bytes(raw),
        "normalized_sha256": _sha256_bytes(normalized.encode("utf-8")),
        "characters": len(normalized),
        "warnings": _scan(normalized),
        "parser_version": PARSER_VERSION,
    }


def _append_audit(data_root: Path, event: dict[str, Any]) -> None:
    path = data_root / "audit.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = _canonical_json(event)
    with path.open("ab") as handle:
        handle.write(serialized)
        handle.flush()
        os.fsync(handle.fileno())


def ingest_file(
    path: str | Path,
    *,
    data_root: str | Path,
    license_name: str,
    source_name: str,
    language: str = "und",
    max_file_bytes: int,
) -> dict[str, Any]:
    if not license_name.strip():
        raise DataError("license/permission is required before ingestion")
    if not source_name.strip():
        raise DataError("source is required before ingestion")

    source = Path(path)
    inspection = inspect_file(source, max_file_bytes=max_file_bytes)
    raw = source.read_bytes()
    text = _normalize(_read_supported_text(source, raw))
    record_id = inspection["content_sha256"]
    root = Path(data_root)
    manifest_path = _safe_manifest_path(root, record_id)
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("content_sha256") != record_id:
            raise DataError("existing manifest hash mismatch")
        return existing

    quarantine_source = root / "quarantine" / f"{record_id}.source"
    quarantine_text = root / "quarantine" / f"{record_id}.txt"
    _atomic_write(quarantine_source, raw)
    _atomic_write(quarantine_text, text.encode("utf-8"))
    now = _utc_now()
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "record_id": record_id,
        "content_sha256": record_id,
        "normalized_sha256": _sha256_bytes(text.encode("utf-8")),
        "byte_size": len(raw),
        "original_name": source.name,
        "source": source_name.strip(),
        "license": license_name.strip(),
        "language": language.strip() or "und",
        "ingested_at": now,
        "parser_version": PARSER_VERSION,
        "warnings": inspection["warnings"],
        "state": "quarantined",
        "state_history": [{"state": "quarantined", "at": now}],
        "transformations": ["utf8-parse", "unicode-nfc", "newline-normalize"],
    }
    _atomic_write(manifest_path, _canonical_json(manifest))
    _append_audit(root, {"event": "data-ingested", "record_id": record_id, "at": now})
    return manifest


def approve_record(
    record_id: str,
    *,
    data_root: str | Path,
    allow_flagged: bool = False,
) -> dict[str, Any]:
    root = Path(data_root)
    manifest_path = _safe_manifest_path(root, record_id)
    if not manifest_path.exists():
        raise DataError(f"unknown record: {record_id}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("state") == "approved":
        return manifest
    if manifest.get("state") != "quarantined":
        raise DataError(f"record is in non-approvable state: {manifest.get('state')}")
    blocking = set(manifest.get("warnings", ())) - {"very-short-document"}
    if blocking and not allow_flagged:
        raise DataError(
            "record has review flags; inspect it and pass --allow-flagged to confirm: "
            + ", ".join(sorted(blocking))
        )

    parsed = root / "quarantine" / f"{record_id}.txt"
    if not parsed.is_file() or parsed.is_symlink():
        raise DataError("quarantined parsed text is missing or unsafe")
    payload = parsed.read_bytes()
    if _sha256_bytes(payload) != manifest.get("normalized_sha256"):
        raise DataError("quarantined text hash does not match its manifest")
    approved = root / "approved" / f"{record_id}.txt"
    _atomic_write(approved, payload)

    now = _utc_now()
    manifest["state"] = "approved"
    manifest["approved_at"] = now
    manifest.setdefault("state_history", []).append({"state": "approved", "at": now})
    _atomic_write(manifest_path, _canonical_json(manifest))
    _append_audit(root, {"event": "data-approved", "record_id": record_id, "at": now})
    return manifest


def _word_ngrams(text: str, size: int = 5) -> set[tuple[str, ...]]:
    words = re.findall(r"\w+", text.casefold())
    if len(words) < size:
        return {tuple(words)} if words else set()
    return {tuple(words[index : index + size]) for index in range(len(words) - size + 1)}


def _near_duplicate(left: str, right: str, threshold: float = 0.95) -> bool:
    left_ngrams = _word_ngrams(left)
    right_ngrams = _word_ngrams(right)
    if not left_ngrams or not right_ngrams:
        return left.strip().casefold() == right.strip().casefold()
    union = left_ngrams | right_ngrams
    return len(left_ngrams & right_ngrams) / len(union) >= threshold


def _choose_split(record_id: str, config: DataConfig) -> str:
    digest = hashlib.sha256(f"{config.split_seed}:{record_id}".encode("utf-8")).digest()
    fraction = int.from_bytes(digest[:8], "big") / 2**64
    if fraction < config.train_ratio:
        return "train"
    if fraction < config.train_ratio + config.validation_ratio:
        return "validation"
    return "test"


def _read_manifests(data_root: Path) -> Iterable[dict[str, Any]]:
    manifest_root = data_root / "manifests"
    if not manifest_root.exists():
        return ()
    manifests: list[dict[str, Any]] = []
    for path in sorted(manifest_root.glob("*.json")):
        if path.is_symlink():
            raise DataError(f"manifest may not be a symlink: {path}")
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("record_id") != path.stem:
            raise DataError(f"manifest filename/id mismatch: {path}")
        manifests.append(value)
    return manifests


def prepare_dataset(
    *,
    data_root: str | Path,
    output_root: str | Path,
    config: DataConfig,
) -> dict[str, Any]:
    root = Path(data_root)
    candidates: list[dict[str, Any]] = []
    rejected_duplicates: list[dict[str, str]] = []
    seen_normalized: dict[str, str] = {}
    accepted_texts: list[tuple[str, str]] = []

    for manifest in _read_manifests(root):
        if manifest.get("state") != "approved":
            continue
        record_id = manifest["record_id"]
        text_path = root / "approved" / f"{record_id}.txt"
        if not text_path.is_file() or text_path.is_symlink():
            raise DataError(f"approved text missing or unsafe: {record_id}")
        payload = text_path.read_bytes()
        normalized_hash = _sha256_bytes(payload)
        if normalized_hash != manifest.get("normalized_sha256"):
            raise DataError(f"approved text hash mismatch: {record_id}")
        text = payload.decode("utf-8")
        if normalized_hash in seen_normalized:
            rejected_duplicates.append(
                {"record_id": record_id, "duplicate_of": seen_normalized[normalized_hash], "kind": "exact"}
            )
            continue
        near_match = next(
            (existing_id for existing_id, existing_text in accepted_texts if _near_duplicate(text, existing_text)),
            None,
        )
        if near_match is not None:
            rejected_duplicates.append(
                {"record_id": record_id, "duplicate_of": near_match, "kind": "near"}
            )
            continue
        seen_normalized[normalized_hash] = record_id
        accepted_texts.append((record_id, text))
        candidates.append(
            {
                "record_id": record_id,
                "text": text,
                "source": manifest["source"],
                "license": manifest["license"],
                "language": manifest["language"],
                "normalized_sha256": normalized_hash,
                "split": _choose_split(record_id, config),
            }
        )

    if not candidates:
        raise DataError("no approved, unique records are available")

    snapshot_basis = {
        "schema_version": 1,
        "records": [(item["record_id"], item["split"]) for item in candidates],
        "split_seed": config.split_seed,
        "ratios": [config.train_ratio, config.validation_ratio, config.test_ratio],
        "normalization": PARSER_VERSION,
    }
    snapshot_hash = _sha256_bytes(_canonical_json(snapshot_basis))
    snapshot_dir = Path(output_root) / f"dataset-{snapshot_hash[:16]}"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    shard_metadata: dict[str, dict[str, Any]] = {}
    for split in ("train", "validation", "test"):
        rows = [item for item in candidates if item["split"] == split]
        lines = [
            _canonical_json({key: value for key, value in row.items() if key != "split"})
            for row in rows
        ]
        payload = b"".join(lines)
        shard_path = snapshot_dir / f"{split}.jsonl"
        _atomic_write(shard_path, payload)
        shard_metadata[split] = {
            "path": shard_path.name,
            "records": len(rows),
            "bytes": len(payload),
            "sha256": _sha256_bytes(payload),
        }

    report: dict[str, Any] = {
        "schema_version": 1,
        "snapshot_id": snapshot_hash,
        "normalization": PARSER_VERSION,
        "split_seed": config.split_seed,
        "ratios": {
            "train": config.train_ratio,
            "validation": config.validation_ratio,
            "test": config.test_ratio,
        },
        "records": len(candidates),
        "duplicate_rejections": rejected_duplicates,
        "languages": dict(
            sorted(
                {
                    language: sum(1 for item in candidates if item["language"] == language)
                    for language in {item["language"] for item in candidates}
                }.items()
            )
        ),
        "shards": shard_metadata,
        "record_lineage": [
            {
                "record_id": item["record_id"],
                "normalized_sha256": item["normalized_sha256"],
                "split": item["split"],
                "source": item["source"],
                "license": item["license"],
            }
            for item in candidates
        ],
    }
    _atomic_write(snapshot_dir / "manifest.json", _canonical_json(report))
    _append_audit(
        root,
        {
            "event": "dataset-prepared",
            "snapshot_id": snapshot_hash,
            "records": len(candidates),
            "at": _utc_now(),
        },
    )
    return {**report, "path": str(snapshot_dir)}

