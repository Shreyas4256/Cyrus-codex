"""Typed, fail-closed configuration for Cyrus development profiles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a Cyrus configuration is missing or internally inconsistent."""


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name)
    if not isinstance(value, dict):
        raise ConfigError(f"missing or invalid '{name}' section")
    return value


def _positive_int(section: dict[str, Any], key: str) -> int:
    value = section.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"'{key}' must be a positive integer")
    return value


def _positive_float(section: dict[str, Any], key: str) -> float:
    value = section.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ConfigError(f"'{key}' must be a positive number")
    return float(value)


@dataclass(frozen=True)
class RuntimeConfig:
    offline: bool
    seed: int
    max_workers: int
    max_ram_gb: float
    max_disk_gb: float
    max_training_minutes: int
    checkpoint_retention: int


@dataclass(frozen=True)
class DataConfig:
    max_file_bytes: int
    train_ratio: float
    validation_ratio: float
    test_ratio: float
    split_seed: str


@dataclass(frozen=True)
class TokenizerConfig:
    vocab_size: int
    min_frequency: int
    special_tokens: tuple[str, ...]


@dataclass(frozen=True)
class ModelConfig:
    context_length: int
    d_model: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    dropout: float
    rope_theta: float


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int
    sequence_length: int
    gradient_accumulation_steps: int
    max_steps: int
    learning_rate: float
    weight_decay: float
    warmup_steps: int
    gradient_clip: float
    eval_interval: int
    eval_batches: int
    checkpoint_interval: int


@dataclass(frozen=True)
class CyrusConfig:
    profile: str
    runtime: RuntimeConfig
    data: DataConfig
    tokenizer: TokenizerConfig
    model: ModelConfig
    training: TrainingConfig


def load_config(path: str | Path) -> CyrusConfig:
    config_path = Path(path)
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"configuration not found: {config_path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {config_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError("configuration root must be a mapping")

    profile = raw.get("profile")
    if not isinstance(profile, str) or not profile.strip():
        raise ConfigError("'profile' must be a non-empty string")

    runtime_raw = _section(raw, "runtime")
    data_raw = _section(raw, "data")
    tokenizer_raw = _section(raw, "tokenizer")
    model_raw = _section(raw, "model")
    training_raw = _section(raw, "training")

    offline = runtime_raw.get("offline")
    if not isinstance(offline, bool):
        raise ConfigError("'runtime.offline' must be true or false")
    seed = runtime_raw.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ConfigError("'runtime.seed' must be a non-negative integer")

    runtime = RuntimeConfig(
        offline=offline,
        seed=seed,
        max_workers=_positive_int(runtime_raw, "max_workers"),
        max_ram_gb=_positive_float(runtime_raw, "max_ram_gb"),
        max_disk_gb=_positive_float(runtime_raw, "max_disk_gb"),
        max_training_minutes=_positive_int(runtime_raw, "max_training_minutes"),
        checkpoint_retention=_positive_int(runtime_raw, "checkpoint_retention"),
    )

    ratios = []
    for key in ("train_ratio", "validation_ratio", "test_ratio"):
        value = data_raw.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise ConfigError(f"'data.{key}' must be a non-negative number")
        ratios.append(float(value))
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ConfigError("data split ratios must sum to exactly 1.0")
    if any(value == 0 for value in ratios):
        raise ConfigError("every data split ratio must be greater than zero")
    split_seed = data_raw.get("split_seed")
    if not isinstance(split_seed, str) or not split_seed:
        raise ConfigError("'data.split_seed' must be a non-empty string")
    data = DataConfig(
        max_file_bytes=_positive_int(data_raw, "max_file_bytes"),
        train_ratio=ratios[0],
        validation_ratio=ratios[1],
        test_ratio=ratios[2],
        split_seed=split_seed,
    )

    special_tokens = tokenizer_raw.get("special_tokens")
    if (
        not isinstance(special_tokens, list)
        or not special_tokens
        or not all(isinstance(token, str) and token for token in special_tokens)
    ):
        raise ConfigError("'tokenizer.special_tokens' must be a non-empty string list")
    if len(set(special_tokens)) != len(special_tokens):
        raise ConfigError("special tokens must be unique")
    vocab_size = _positive_int(tokenizer_raw, "vocab_size")
    minimum_vocab = 256 + len(special_tokens)
    if vocab_size < minimum_vocab:
        raise ConfigError(
            f"tokenizer vocabulary must be at least {minimum_vocab} for byte coverage"
        )
    tokenizer = TokenizerConfig(
        vocab_size=vocab_size,
        min_frequency=_positive_int(tokenizer_raw, "min_frequency"),
        special_tokens=tuple(special_tokens),
    )

    d_model = _positive_int(model_raw, "d_model")
    n_heads = _positive_int(model_raw, "n_heads")
    n_kv_heads = _positive_int(model_raw, "n_kv_heads")
    if d_model % n_heads != 0:
        raise ConfigError("model.d_model must be divisible by model.n_heads")
    if n_heads % n_kv_heads != 0:
        raise ConfigError("model.n_heads must be divisible by model.n_kv_heads")
    dropout = model_raw.get("dropout")
    if not isinstance(dropout, (int, float)) or isinstance(dropout, bool) or not 0 <= dropout < 1:
        raise ConfigError("model.dropout must be in [0, 1)")
    model = ModelConfig(
        context_length=_positive_int(model_raw, "context_length"),
        d_model=d_model,
        n_layers=_positive_int(model_raw, "n_layers"),
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        dropout=float(dropout),
        rope_theta=_positive_float(model_raw, "rope_theta"),
    )

    weight_decay = training_raw.get("weight_decay")
    if not isinstance(weight_decay, (int, float)) or isinstance(weight_decay, bool) or weight_decay < 0:
        raise ConfigError("training.weight_decay must be non-negative")
    training = TrainingConfig(
        batch_size=_positive_int(training_raw, "batch_size"),
        sequence_length=_positive_int(training_raw, "sequence_length"),
        gradient_accumulation_steps=_positive_int(
            training_raw, "gradient_accumulation_steps"
        ),
        max_steps=_positive_int(training_raw, "max_steps"),
        learning_rate=_positive_float(training_raw, "learning_rate"),
        weight_decay=float(weight_decay),
        warmup_steps=_positive_int(training_raw, "warmup_steps"),
        gradient_clip=_positive_float(training_raw, "gradient_clip"),
        eval_interval=_positive_int(training_raw, "eval_interval"),
        eval_batches=_positive_int(training_raw, "eval_batches"),
        checkpoint_interval=_positive_int(training_raw, "checkpoint_interval"),
    )
    if training.warmup_steps >= training.max_steps:
        raise ConfigError("training.warmup_steps must be smaller than training.max_steps")
    if training.sequence_length > model.context_length:
        raise ConfigError("training.sequence_length cannot exceed model.context_length")

    return CyrusConfig(
        profile=profile.strip(),
        runtime=runtime,
        data=data,
        tokenizer=tokenizer,
        model=model,
        training=training,
    )
