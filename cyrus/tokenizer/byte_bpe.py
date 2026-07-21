"""Auditable byte-level BPE tokenizer trained only from approved Cyrus data."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from cyrus.config.schema import TokenizerConfig

TOKENIZER_SCHEMA_VERSION = 1


class TokenizerError(ValueError):
    """Raised for incompatible, corrupt, or invalid tokenizer artifacts."""


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


class ByteBPETokenizer:
    """A deterministic tokenizer with lossless UTF-8 byte fallback.

    Special-token IDs come first, followed by one ID for every possible byte.
    Learned merge tokens are created from the supplied corpus in deterministic
    frequency/tie-break order. No pretrained tokenizer data is loaded.
    """

    def __init__(
        self,
        *,
        special_tokens: Sequence[str],
        merges: Sequence[tuple[int, int, int]],
        token_bytes: dict[int, bytes] | None = None,
        training_corpus_sha256: str | None = None,
    ) -> None:
        if not special_tokens or len(set(special_tokens)) != len(special_tokens):
            raise TokenizerError("special tokens must be non-empty and unique")
        self.special_tokens = tuple(special_tokens)
        self.special_to_id = {token: index for index, token in enumerate(self.special_tokens)}
        self.byte_offset = len(self.special_tokens)
        vocabulary: dict[int, bytes] = {
            self.byte_offset + byte: bytes([byte]) for byte in range(256)
        }
        if token_bytes:
            vocabulary.update(token_bytes)
        self.token_bytes = vocabulary
        self.merges = tuple(merges)
        self.training_corpus_sha256 = training_corpus_sha256
        self._validate()
        pattern = "|".join(re.escape(token) for token in sorted(self.special_tokens, key=len, reverse=True))
        self._special_pattern = re.compile(f"({pattern})")

    def _validate(self) -> None:
        expected_new_id = self.byte_offset + 256
        for left, right, new in self.merges:
            if left not in self.token_bytes or right not in self.token_bytes:
                raise TokenizerError("merge references an unknown token id")
            if new != expected_new_id:
                raise TokenizerError("merge ids must be contiguous and rank ordered")
            expected_bytes = self.token_bytes[left] + self.token_bytes[right]
            actual = self.token_bytes.get(new)
            if actual is None:
                self.token_bytes[new] = expected_bytes
            elif actual != expected_bytes:
                raise TokenizerError("stored merge bytes do not match their parent tokens")
            expected_new_id += 1
        expected_ids = set(range(self.byte_offset, self.byte_offset + 256 + len(self.merges)))
        if set(self.token_bytes) != expected_ids:
            raise TokenizerError("token vocabulary is not contiguous")

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        *,
        vocab_size: int,
        min_frequency: int,
        special_tokens: Sequence[str],
        training_corpus_sha256: str | None = None,
    ) -> "ByteBPETokenizer":
        byte_offset = len(special_tokens)
        minimum = byte_offset + 256
        if vocab_size < minimum:
            raise TokenizerError(f"vocabulary must contain at least {minimum} tokens")
        if min_frequency < 1:
            raise TokenizerError("min_frequency must be at least 1")
        materialized = list(texts)
        if not materialized or not any(text for text in materialized):
            raise TokenizerError("cannot train a tokenizer on an empty corpus")

        sequences = [
            [byte_offset + value for value in text.encode("utf-8")] for text in materialized
        ]
        token_bytes: dict[int, bytes] = {
            byte_offset + value: bytes([value]) for value in range(256)
        }
        existing_bytes = set(token_bytes.values())
        merges: list[tuple[int, int, int]] = []

        while byte_offset + 256 + len(merges) < vocab_size:
            pair_counts: Counter[tuple[int, int]] = Counter()
            for sequence in sequences:
                pair_counts.update(zip(sequence, sequence[1:]))
            candidates = sorted(
                (pair for pair, count in pair_counts.items() if count >= min_frequency),
                key=lambda pair: (-pair_counts[pair], pair),
            )
            selected: tuple[int, int] | None = None
            for pair in candidates:
                if token_bytes[pair[0]] + token_bytes[pair[1]] not in existing_bytes:
                    selected = pair
                    break
            if selected is None:
                break

            left, right = selected
            new_id = byte_offset + 256 + len(merges)
            merged_bytes = token_bytes[left] + token_bytes[right]
            token_bytes[new_id] = merged_bytes
            existing_bytes.add(merged_bytes)
            merges.append((left, right, new_id))

            updated: list[list[int]] = []
            for sequence in sequences:
                merged: list[int] = []
                index = 0
                while index < len(sequence):
                    if (
                        index + 1 < len(sequence)
                        and sequence[index] == left
                        and sequence[index + 1] == right
                    ):
                        merged.append(new_id)
                        index += 2
                    else:
                        merged.append(sequence[index])
                        index += 1
                updated.append(merged)
            sequences = updated

        learned = {
            token_id: value
            for token_id, value in token_bytes.items()
            if token_id >= byte_offset + 256
        }
        return cls(
            special_tokens=special_tokens,
            merges=merges,
            token_bytes=learned,
            training_corpus_sha256=training_corpus_sha256,
        )

    @property
    def vocab_size(self) -> int:
        return len(self.special_tokens) + len(self.token_bytes)

    @property
    def tokenizer_hash(self) -> str:
        return _sha256(_canonical_json(self._payload(include_hash=False)))

    def _encode_bytes(self, value: bytes) -> list[int]:
        tokens = [self.byte_offset + byte for byte in value]
        for left, right, new_id in self.merges:
            updated: list[int] = []
            index = 0
            while index < len(tokens):
                if (
                    index + 1 < len(tokens)
                    and tokens[index] == left
                    and tokens[index + 1] == right
                ):
                    updated.append(new_id)
                    index += 2
                else:
                    updated.append(tokens[index])
                    index += 1
            tokens = updated
        return tokens

    def encode(self, text: str, *, allowed_special: bool = False) -> list[int]:
        if not allowed_special:
            return self._encode_bytes(text.encode("utf-8"))
        result: list[int] = []
        for part in self._special_pattern.split(text):
            if not part:
                continue
            special_id = self.special_to_id.get(part)
            if special_id is not None:
                result.append(special_id)
            else:
                result.extend(self._encode_bytes(part.encode("utf-8")))
        return result

    def decode(
        self,
        token_ids: Iterable[int],
        *,
        show_special: bool = True,
        errors: str = "strict",
    ) -> str:
        if errors not in {"strict", "replace"}:
            raise TokenizerError("decode errors must be 'strict' or 'replace'")
        output = bytearray()
        chunks: list[str] = []

        def flush() -> None:
            if output:
                try:
                    chunks.append(output.decode("utf-8", errors=errors))
                except UnicodeDecodeError as exc:
                    raise TokenizerError("token sequence is not valid UTF-8") from exc
                output.clear()

        for token_id in token_ids:
            if 0 <= token_id < self.byte_offset:
                flush()
                if show_special:
                    chunks.append(self.special_tokens[token_id])
                continue
            value = self.token_bytes.get(token_id)
            if value is None:
                raise TokenizerError(f"unknown token id: {token_id}")
            output.extend(value)
        flush()
        return "".join(chunks)

    def _payload(self, *, include_hash: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": TOKENIZER_SCHEMA_VERSION,
            "algorithm": "cyrus-byte-bpe-v1",
            "special_tokens": list(self.special_tokens),
            "byte_offset": self.byte_offset,
            "vocab_size": self.vocab_size,
            "training_corpus_sha256": self.training_corpus_sha256,
            "merges": [
                {
                    "left": left,
                    "right": right,
                    "new": new,
                    "bytes_hex": self.token_bytes[new].hex(),
                }
                for left, right, new in self.merges
            ],
        }
        if include_hash:
            payload["tokenizer_sha256"] = self.tokenizer_hash
        return payload

    def save(self, path: str | Path) -> dict[str, Any]:
        payload = self._payload(include_hash=True)
        _atomic_write(Path(path), _canonical_json(payload))
        return payload

    @classmethod
    def load(cls, path: str | Path) -> "ByteBPETokenizer":
        artifact_path = Path(path)
        try:
            payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TokenizerError(f"cannot read tokenizer artifact: {artifact_path}") from exc
        if payload.get("schema_version") != TOKENIZER_SCHEMA_VERSION:
            raise TokenizerError("unsupported tokenizer schema version")
        if payload.get("algorithm") != "cyrus-byte-bpe-v1":
            raise TokenizerError("unsupported tokenizer algorithm")
        merges: list[tuple[int, int, int]] = []
        token_bytes: dict[int, bytes] = {}
        for item in payload.get("merges", []):
            try:
                merge = (int(item["left"]), int(item["right"]), int(item["new"]))
                token_bytes[merge[2]] = bytes.fromhex(item["bytes_hex"])
            except (KeyError, TypeError, ValueError) as exc:
                raise TokenizerError("malformed tokenizer merge") from exc
            merges.append(merge)
        tokenizer = cls(
            special_tokens=payload.get("special_tokens", ()),
            merges=merges,
            token_bytes=token_bytes,
            training_corpus_sha256=payload.get("training_corpus_sha256"),
        )
        if payload.get("byte_offset") != tokenizer.byte_offset:
            raise TokenizerError("tokenizer byte offset mismatch")
        if payload.get("vocab_size") != tokenizer.vocab_size:
            raise TokenizerError("tokenizer vocabulary size mismatch")
        if payload.get("tokenizer_sha256") != tokenizer.tokenizer_hash:
            raise TokenizerError("tokenizer artifact hash mismatch")
        return tokenizer

    def metrics(self, texts: Iterable[str]) -> dict[str, Any]:
        materialized = list(texts)
        characters = sum(len(text) for text in materialized)
        utf8_bytes = sum(len(text.encode("utf-8")) for text in materialized)
        tokens = sum(len(self.encode(text)) for text in materialized)
        return {
            "documents": len(materialized),
            "characters": characters,
            "utf8_bytes": utf8_bytes,
            "tokens": tokens,
            "tokens_per_character": tokens / characters if characters else None,
            "tokens_per_utf8_byte": tokens / utf8_bytes if utf8_bytes else None,
            "compression_ratio": utf8_bytes / tokens if tokens else None,
            "unknown_rate": 0.0,
        }


def _load_shard(path: Path) -> list[str]:
    texts: list[str] = []
    if not path.exists():
        return texts
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TokenizerError(f"invalid dataset shard {path}:{line_number}") from exc
        text = item.get("text") if isinstance(item, dict) else None
        if not isinstance(text, str):
            raise TokenizerError(f"dataset row lacks text at {path}:{line_number}")
        texts.append(text)
    return texts


def train_from_dataset(
    dataset_dir: str | Path,
    *,
    config: TokenizerConfig,
    output_path: str | Path,
) -> dict[str, Any]:
    root = Path(dataset_dir)
    manifest_path = root / "manifest.json"
    try:
        dataset_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TokenizerError(f"dataset manifest is missing or invalid: {manifest_path}") from exc

    training_texts = _load_shard(root / "train.jsonl")
    if not training_texts:
        raise TokenizerError("training shard is empty; approve more data or change the split seed")
    tokenizer = ByteBPETokenizer.train(
        training_texts,
        vocab_size=config.vocab_size,
        min_frequency=config.min_frequency,
        special_tokens=config.special_tokens,
        training_corpus_sha256=dataset_manifest.get("shards", {}).get("train", {}).get("sha256"),
    )
    artifact = tokenizer.save(output_path)
    split_metrics = {
        split: tokenizer.metrics(_load_shard(root / f"{split}.jsonl"))
        for split in ("train", "validation", "test")
    }
    report = {
        "schema_version": 1,
        "tokenizer_sha256": artifact["tokenizer_sha256"],
        "algorithm": artifact["algorithm"],
        "requested_vocab_size": config.vocab_size,
        "actual_vocab_size": tokenizer.vocab_size,
        "learned_merges": len(tokenizer.merges),
        "min_frequency": config.min_frequency,
        "training_dataset_id": dataset_manifest.get("snapshot_id"),
        "training_corpus_sha256": artifact["training_corpus_sha256"],
        "splits": split_metrics,
    }
    report_path = Path(output_path).with_suffix(".report.json")
    _atomic_write(report_path, _canonical_json(report))
    return {**report, "tokenizer_path": str(output_path), "report_path": str(report_path)}
