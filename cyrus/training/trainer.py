"""Single-device CPU/GPU Smoke trainer with bounded runtime and gated promotion."""

from __future__ import annotations

import json
import math
import os
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from cyrus.config import CyrusConfig
from cyrus.model import CyrusModel, ModelHyperparameters, choose_device
from cyrus.tokenizer import ByteBPETokenizer
from cyrus.training.checkpoint import (
    CheckpointError,
    copy_verified_checkpoint,
    enforce_retention,
    load_checkpoint,
    save_checkpoint,
    sha256_file,
)


class TrainingError(RuntimeError):
    """Raised when a training precondition or promotion gate fails."""


def _read_manifest(directory: Path) -> dict[str, Any]:
    try:
        value = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainingError(f"invalid token snapshot manifest: {directory}") from exc
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise TrainingError("unsupported token snapshot schema")
    return value


def _load_stream(directory: Path, split: str, manifest: dict[str, Any]) -> torch.Tensor:
    shard_info = manifest.get("shards", {}).get(split)
    if not isinstance(shard_info, dict):
        raise TrainingError(f"token snapshot lacks the {split} shard")
    path = directory / str(shard_info.get("path"))
    if not path.is_file() or path.is_symlink():
        raise TrainingError(f"token shard is missing or unsafe: {path}")
    if sha256_file(path) != shard_info.get("sha256"):
        raise TrainingError(f"token shard hash mismatch: {split}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TrainingError(f"token shard JSON is invalid: {split}") from exc
    tokens = payload.get("tokens") if isinstance(payload, dict) else None
    if not isinstance(tokens, list) or not all(isinstance(token, int) for token in tokens):
        raise TrainingError(f"token shard contains invalid IDs: {split}")
    if not tokens:
        raise TrainingError(f"token shard is empty: {split}")
    return torch.tensor(tokens, dtype=torch.long)


def _batch(
    stream: torch.Tensor,
    *,
    batch_size: int,
    sequence_length: int,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    maximum = stream.numel() - sequence_length - 1
    if maximum < 0:
        raise TrainingError(
            f"token stream has {stream.numel()} tokens but needs more than {sequence_length}"
        )
    starts = torch.randint(0, maximum + 1, (batch_size,), generator=generator)
    inputs = torch.stack([stream[start : start + sequence_length] for start in starts.tolist()])
    targets = torch.stack(
        [stream[start + 1 : start + sequence_length + 1] for start in starts.tolist()]
    )
    return inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)


def _model_hyperparameters(config: CyrusConfig, vocab_size: int) -> ModelHyperparameters:
    return ModelHyperparameters(
        vocab_size=vocab_size,
        context_length=config.model.context_length,
        d_model=config.model.d_model,
        n_layers=config.model.n_layers,
        n_heads=config.model.n_heads,
        n_kv_heads=config.model.n_kv_heads,
        dropout=config.model.dropout,
        rope_theta=config.model.rope_theta,
    )


def _learning_rate(step: int, config: CyrusConfig) -> float:
    if step < config.training.warmup_steps:
        return config.training.learning_rate * (step + 1) / config.training.warmup_steps
    progress = (step - config.training.warmup_steps) / max(
        1, config.training.max_steps - config.training.warmup_steps
    )
    return config.training.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def _estimate_loss(
    model: CyrusModel,
    stream: torch.Tensor,
    *,
    config: CyrusConfig,
    device: torch.device,
    seed: int,
) -> float:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    was_training = model.training
    model.eval()
    losses: list[float] = []
    try:
        for _ in range(config.training.eval_batches):
            inputs, targets = _batch(
                stream,
                batch_size=config.training.batch_size,
                sequence_length=config.training.sequence_length,
                generator=generator,
                device=device,
            )
            _, loss = model(inputs, targets)
            assert loss is not None
            losses.append(float(loss.detach().cpu()))
    finally:
        model.train(was_training)
    return sum(losses) / len(losses)


def _source_commit(workspace_root: Path) -> str | None:
    environment_sha = os.environ.get("GITHUB_SHA")
    if environment_sha:
        return environment_sha
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=workspace_root,
            capture_output=True,
            text=True,
            check=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _checkpoint_payload(
    *,
    model: CyrusModel,
    optimizer: torch.optim.Optimizer,
    hyperparameters: ModelHyperparameters,
    config: CyrusConfig,
    tokenizer_hash: str,
    token_snapshot_id: str,
    step: int,
    tokens_seen: int,
    batch_generator: torch.Generator,
    initial_validation_loss: float,
    source_commit: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "family": "cyrus-smoke-base",
        "initialization": "random",
        "model_hyperparameters": hyperparameters.to_dict(),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "step": step,
        "tokens_seen": tokens_seen,
        "torch_rng_state": torch.get_rng_state(),
        "batch_rng_state": batch_generator.get_state(),
        "config": {
            "profile": config.profile,
            "runtime": asdict(config.runtime),
            "training": asdict(config.training),
        },
        "tokenizer_sha256": tokenizer_hash,
        "token_snapshot_id": token_snapshot_id,
        "initial_validation_loss": initial_validation_loss,
        "source_commit": source_commit,
    }


def _overfit_probe(
    hyperparameters: ModelHyperparameters,
    stream: torch.Tensor,
    *,
    config: CyrusConfig,
    device: torch.device,
) -> dict[str, Any]:
    torch.manual_seed(config.runtime.seed + 1)
    model = CyrusModel(hyperparameters).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.0)
    generator = torch.Generator(device="cpu").manual_seed(config.runtime.seed + 1)
    inputs, targets = _batch(
        stream,
        batch_size=1,
        sequence_length=min(32, config.training.sequence_length),
        generator=generator,
        device=device,
    )
    with torch.no_grad():
        _, initial = model(inputs, targets)
    assert initial is not None
    for _ in range(30):
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(inputs, targets)
        assert loss is not None
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip)
        optimizer.step()
    with torch.no_grad():
        _, final = model(inputs, targets)
    assert final is not None
    reduction = 1.0 - final.item() / initial.item()
    return {
        "initial_loss": initial.item(),
        "final_loss": final.item(),
        "relative_reduction": reduction,
        "passed": math.isfinite(final.item()) and reduction >= 0.50,
    }


def train_smoke(
    *,
    config: CyrusConfig,
    token_snapshot_dir: str | Path,
    tokenizer_path: str | Path,
    workspace_root: str | Path,
    device_name: str = "auto",
    resume_path: str | Path | None = None,
    promote: bool = True,
) -> dict[str, Any]:
    if config.profile != "smoke":
        raise TrainingError("this command only permits the Smoke profile")
    if config.runtime.max_training_minutes > 20:
        raise TrainingError("Codespaces experiments may not exceed 20 minutes")
    workspace = Path(workspace_root).resolve()
    token_root = Path(token_snapshot_dir)
    manifest = _read_manifest(token_root)
    tokenizer = ByteBPETokenizer.load(tokenizer_path)
    if manifest.get("tokenizer_sha256") != tokenizer.tokenizer_hash:
        raise TrainingError("token snapshot and tokenizer hashes do not match")
    if manifest.get("vocab_size") != tokenizer.vocab_size:
        raise TrainingError("token snapshot and tokenizer vocabulary sizes do not match")

    train_stream = _load_stream(token_root, "train", manifest)
    validation_stream = _load_stream(token_root, "validation", manifest)
    hyperparameters = _model_hyperparameters(config, tokenizer.vocab_size)
    device = choose_device(device_name)
    torch.manual_seed(config.runtime.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.runtime.seed)
    model = CyrusModel(hyperparameters).to(device)
    parameter_count = model.parameter_count
    if not 1_000_000 <= parameter_count <= 5_000_000:
        raise TrainingError(
            f"Smoke model has {parameter_count:,} parameters; owner limit is 1M-5M"
        )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    batch_generator = torch.Generator(device="cpu").manual_seed(config.runtime.seed)
    starting_step = 0
    tokens_seen = 0
    initial_validation_loss = _estimate_loss(
        model,
        validation_stream,
        config=config,
        device=device,
        seed=config.runtime.seed + 100,
    )

    if resume_path is not None:
        payload = load_checkpoint(
            resume_path, workspace_root=workspace, map_location=device
        )
        if payload.get("tokenizer_sha256") != tokenizer.tokenizer_hash:
            raise TrainingError("resume checkpoint was created with another tokenizer")
        if payload.get("token_snapshot_id") != manifest.get("snapshot_id"):
            raise TrainingError("resume checkpoint was created from another token snapshot")
        if payload.get("model_hyperparameters") != hyperparameters.to_dict():
            raise TrainingError("resume checkpoint architecture is incompatible")
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        starting_step = int(payload["step"])
        tokens_seen = int(payload["tokens_seen"])
        torch.set_rng_state(payload["torch_rng_state"].cpu())
        batch_generator.set_state(payload["batch_rng_state"].cpu())
        initial_validation_loss = float(payload["initial_validation_loss"])

    overfit = _overfit_probe(
        hyperparameters, train_stream, config=config, device=device
    )
    if not overfit["passed"]:
        raise TrainingError(f"tiny-batch overfit gate failed: {overfit}")

    deadline_seconds = config.runtime.max_training_minutes * 60
    started = time.monotonic()
    checkpoint_dir = workspace / "checkpoints" / "candidates"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    source_commit = _source_commit(workspace)
    history: list[dict[str, Any]] = []
    last_checkpoint: dict[str, Any] | None = None
    completed_step = starting_step
    stopped_for_time = False

    model.train()
    for step_index in range(starting_step, config.training.max_steps):
        if time.monotonic() - started >= deadline_seconds:
            stopped_for_time = True
            break
        learning_rate = _learning_rate(step_index, config)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        for _ in range(config.training.gradient_accumulation_steps):
            inputs, targets = _batch(
                train_stream,
                batch_size=config.training.batch_size,
                sequence_length=config.training.sequence_length,
                generator=batch_generator,
                device=device,
            )
            _, loss = model(inputs, targets)
            assert loss is not None
            if not torch.isfinite(loss):
                raise TrainingError(f"non-finite loss at step {step_index + 1}")
            scaled = loss / config.training.gradient_accumulation_steps
            scaled.backward()
            accumulated_loss += float(loss.detach().cpu())
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.training.gradient_clip
        )
        if not torch.isfinite(gradient_norm):
            raise TrainingError(f"non-finite gradient norm at step {step_index + 1}")
        optimizer.step()
        completed_step = step_index + 1
        tokens_seen += (
            config.training.batch_size
            * config.training.sequence_length
            * config.training.gradient_accumulation_steps
        )
        record: dict[str, Any] = {
            "step": completed_step,
            "train_loss": accumulated_loss / config.training.gradient_accumulation_steps,
            "learning_rate": learning_rate,
            "gradient_norm": float(gradient_norm.detach().cpu()),
        }
        if completed_step % config.training.eval_interval == 0:
            record["validation_loss"] = _estimate_loss(
                model,
                validation_stream,
                config=config,
                device=device,
                seed=config.runtime.seed + 100,
            )
        history.append(record)

        if completed_step % config.training.checkpoint_interval == 0:
            path = checkpoint_dir / f"smoke-step-{completed_step:06d}.pt"
            last_checkpoint = save_checkpoint(
                path,
                _checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    hyperparameters=hyperparameters,
                    config=config,
                    tokenizer_hash=tokenizer.tokenizer_hash,
                    token_snapshot_id=str(manifest.get("snapshot_id")),
                    step=completed_step,
                    tokens_seen=tokens_seen,
                    batch_generator=batch_generator,
                    initial_validation_loss=initial_validation_loss,
                    source_commit=source_commit,
                ),
                workspace_root=workspace,
            )
            enforce_retention(
                checkpoint_dir,
                prefix="smoke",
                keep=config.runtime.checkpoint_retention,
            )

    final_path = checkpoint_dir / f"smoke-step-{completed_step:06d}.pt"
    last_checkpoint = save_checkpoint(
        final_path,
        _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            hyperparameters=hyperparameters,
            config=config,
            tokenizer_hash=tokenizer.tokenizer_hash,
            token_snapshot_id=str(manifest.get("snapshot_id")),
            step=completed_step,
            tokens_seen=tokens_seen,
            batch_generator=batch_generator,
            initial_validation_loss=initial_validation_loss,
            source_commit=source_commit,
        ),
        workspace_root=workspace,
    )
    enforce_retention(checkpoint_dir, prefix="smoke", keep=config.runtime.checkpoint_retention)
    final_validation_loss = _estimate_loss(
        model,
        validation_stream,
        config=config,
        device=device,
        seed=config.runtime.seed + 100,
    )
    elapsed = time.monotonic() - started

    reloaded_payload = load_checkpoint(
        last_checkpoint["path"], workspace_root=workspace, map_location=device
    )
    reloaded = CyrusModel(hyperparameters).to(device).eval()
    reloaded.load_state_dict(reloaded_payload["model_state"])
    probe = train_stream[: min(16, config.training.sequence_length)].unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        original_logits, _ = model(probe)
        reloaded_logits, _ = reloaded(probe)
    checkpoint_parity = bool(torch.equal(original_logits, reloaded_logits))
    first_sample = model.generate(probe, max_new_tokens=8, temperature=0.8, top_k=40, seed=7)
    second_sample = model.generate(probe, max_new_tokens=8, temperature=0.8, top_k=40, seed=7)
    deterministic_sampling = bool(torch.equal(first_sample, second_sample))

    completed_budget = completed_step >= config.training.max_steps
    promotion_gates = {
        "parameter_budget": 1_000_000 <= parameter_count <= 5_000_000,
        "tiny_batch_overfit": bool(overfit["passed"]),
        "validation_improved": math.isfinite(final_validation_loss)
        and final_validation_loss < initial_validation_loss,
        "checkpoint_parity": checkpoint_parity,
        "deterministic_sampling": deterministic_sampling,
        "completed_step_budget": completed_budget,
        "within_time_budget": elapsed <= deadline_seconds + 5,
    }
    passed = all(promotion_gates.values())
    stable_checkpoint = None
    if promote and passed:
        stable_checkpoint = copy_verified_checkpoint(
            last_checkpoint["path"],
            workspace / "checkpoints" / "stable" / "cyrus-smoke-base.pt",
            workspace_root=workspace,
        )

    report = {
        "schema_version": 1,
        "profile": "smoke",
        "initialization": "random",
        "device": str(device),
        "torch_version": torch.__version__,
        "parameter_count": parameter_count,
        "model_hyperparameters": hyperparameters.to_dict(),
        "tokenizer_sha256": tokenizer.tokenizer_hash,
        "token_snapshot_id": manifest.get("snapshot_id"),
        "steps": completed_step,
        "tokens_seen": tokens_seen,
        "elapsed_seconds": elapsed,
        "tokens_per_second": tokens_seen / elapsed if elapsed else None,
        "initial_validation_loss": initial_validation_loss,
        "final_validation_loss": final_validation_loss,
        "overfit_probe": overfit,
        "stopped_for_time": stopped_for_time,
        "promotion_gates": promotion_gates,
        "promotion_passed": passed,
        "candidate_checkpoint": last_checkpoint,
        "stable_checkpoint": stable_checkpoint,
        "history": history,
    }
    reports = workspace / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    report_path = reports / "smoke-training.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report["report_path"] = str(report_path)
    return report


def load_trained_model(
    *,
    checkpoint_path: str | Path,
    tokenizer_path: str | Path,
    workspace_root: str | Path,
    device_name: str = "auto",
) -> tuple[CyrusModel, ByteBPETokenizer, dict[str, Any], torch.device]:
    device = choose_device(device_name)
    tokenizer = ByteBPETokenizer.load(tokenizer_path)
    try:
        payload = load_checkpoint(
            checkpoint_path, workspace_root=workspace_root, map_location=device
        )
    except CheckpointError as exc:
        raise TrainingError(str(exc)) from exc
    if payload.get("tokenizer_sha256") != tokenizer.tokenizer_hash:
        raise TrainingError("checkpoint and tokenizer hashes do not match")
    hyperparameters = ModelHyperparameters(**payload["model_hyperparameters"])
    if hyperparameters.vocab_size != tokenizer.vocab_size:
        raise TrainingError("checkpoint and tokenizer vocabulary sizes do not match")
    model = CyrusModel(hyperparameters).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, tokenizer, payload, device

