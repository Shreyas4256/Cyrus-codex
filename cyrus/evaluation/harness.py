"""Fixed-seed evaluation for promoted Cyrus Smoke checkpoints."""

from __future__ import annotations

import json
import math
import platform
import resource
import time
from pathlib import Path
from typing import Any

import torch

from cyrus.training import load_trained_model
from cyrus.training.checkpoint import sha256_file


class EvaluationError(RuntimeError):
    """Raised when immutable evaluation inputs fail validation."""


def _load_token_stream(token_root: Path, split: str) -> tuple[list[int], dict[str, Any]]:
    try:
        manifest = json.loads((token_root / "manifest.json").read_text(encoding="utf-8"))
        metadata = manifest["shards"][split]
        shard = token_root / metadata["path"]
        if shard.is_symlink() or not shard.is_file():
            raise EvaluationError(f"unsafe or missing {split} token shard")
        if sha256_file(shard) != metadata["sha256"]:
            raise EvaluationError(f"{split} token shard hash mismatch")
        payload = json.loads(shard.read_text(encoding="utf-8"))
        tokens = payload["tokens"]
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"invalid {split} token shard") from exc
    if not isinstance(tokens, list) or not tokens or not all(isinstance(item, int) for item in tokens):
        raise EvaluationError(f"invalid {split} token IDs")
    return tokens, manifest


@torch.no_grad()
def _loss(
    model: torch.nn.Module,
    tokens: list[int],
    *,
    sequence_length: int,
    batches: int,
    device: torch.device,
) -> float:
    available = len(tokens) - sequence_length - 1
    if available < 0:
        raise EvaluationError(
            f"evaluation stream has {len(tokens)} tokens but needs more than {sequence_length}"
        )
    count = min(batches, available + 1)
    starts = [round(index * available / max(1, count - 1)) for index in range(count)]
    losses: list[float] = []
    for start in starts:
        inputs = torch.tensor(
            [tokens[start : start + sequence_length]], dtype=torch.long, device=device
        )
        targets = torch.tensor(
            [tokens[start + 1 : start + sequence_length + 1]],
            dtype=torch.long,
            device=device,
        )
        _, loss = model(inputs, targets)
        assert loss is not None
        losses.append(float(loss.detach().cpu()))
    return sum(losses) / len(losses)


def _repetition_rate(token_ids: list[int]) -> float:
    if len(token_ids) < 2:
        return 0.0
    bigrams = list(zip(token_ids, token_ids[1:]))
    return 1.0 - len(set(bigrams)) / len(bigrams)


def _peak_memory_bytes(device: torch.device) -> int:
    if device.type == "cuda":
        return int(torch.cuda.max_memory_allocated(device))
    # Linux reports KiB; macOS reports bytes. The current Codespaces target is Linux.
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak * 1024 if platform.system() == "Linux" else peak


def evaluate_checkpoint(
    *,
    checkpoint_path: str | Path,
    tokenizer_path: str | Path,
    token_snapshot_dir: str | Path,
    workspace_root: str | Path,
    device_name: str = "auto",
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    workspace = Path(workspace_root).resolve()
    model, tokenizer, checkpoint, device = load_trained_model(
        checkpoint_path=checkpoint_path,
        tokenizer_path=tokenizer_path,
        workspace_root=workspace,
        device_name=device_name,
    )
    test_tokens, manifest = _load_token_stream(Path(token_snapshot_dir), "test")
    validation_tokens, validation_manifest = _load_token_stream(
        Path(token_snapshot_dir), "validation"
    )
    if manifest.get("snapshot_id") != checkpoint.get("token_snapshot_id"):
        raise EvaluationError("checkpoint and evaluation token snapshot IDs do not match")
    if validation_manifest.get("snapshot_id") != manifest.get("snapshot_id"):
        raise EvaluationError("evaluation shards do not share one immutable snapshot")
    training_config = checkpoint.get("config", {}).get("training", {})
    sequence_length = int(training_config.get("sequence_length", 64))
    eval_batches = int(training_config.get("eval_batches", 4))
    validation_loss = _loss(
        model,
        validation_tokens,
        sequence_length=sequence_length,
        batches=eval_batches,
        device=device,
    )
    test_loss = _loss(
        model,
        test_tokens,
        sequence_length=sequence_length,
        batches=eval_batches,
        device=device,
    )

    prompt_text = "Cyrus runs"
    prompt_ids = tokenizer.encode(prompt_text)
    prompt = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    iterator = model.generate_iter(
        prompt,
        max_new_tokens=48,
        temperature=0.8,
        top_k=40,
        seed=4256,
    )
    generated: list[int] = []
    try:
        first = next(iterator)
    except StopIteration:
        first_token_seconds = 0.0
    else:
        first_token_seconds = time.perf_counter() - started
        generated.extend(first.reshape(-1).tolist())
        for token in iterator:
            generated.extend(token.reshape(-1).tolist())
    elapsed = time.perf_counter() - started
    repeated = model.generate(
        prompt,
        max_new_tokens=48,
        temperature=0.8,
        top_k=40,
        seed=4256,
    )[:, len(prompt_ids) :].reshape(-1).tolist()
    deterministic = generated == repeated
    sample_text = tokenizer.decode(
        [*prompt_ids, *generated], show_special=True, errors="replace"
    )
    checkpoint_path_obj = Path(checkpoint_path).resolve()
    report: dict[str, Any] = {
        "schema_version": 1,
        "model_family": checkpoint.get("family"),
        "capability_level": "research-smoke-model-not-an-instruction-assistant",
        "initialization": checkpoint.get("initialization"),
        "checkpoint": {
            "path": str(checkpoint_path_obj),
            "sha256": sha256_file(checkpoint_path_obj),
            "bytes": checkpoint_path_obj.stat().st_size,
            "step": checkpoint.get("step"),
            "tokens_seen": checkpoint.get("tokens_seen"),
        },
        "tokenizer_sha256": tokenizer.tokenizer_hash,
        "token_snapshot_id": manifest.get("snapshot_id"),
        "device": str(device),
        "torch_version": torch.__version__,
        "metrics": {
            "validation_loss": validation_loss,
            "validation_perplexity": math.exp(min(validation_loss, 20)),
            "test_loss": test_loss,
            "test_perplexity": math.exp(min(test_loss, 20)),
            "time_to_first_token_seconds": first_token_seconds,
            "generation_tokens_per_second": len(generated) / elapsed if elapsed else None,
            "generated_bigram_repetition_rate": _repetition_rate(generated),
            "peak_process_memory_bytes": _peak_memory_bytes(device),
            "deterministic_fixed_seed": deterministic,
        },
        "fixed_sample": {
            "prompt": prompt_text,
            "token_ids": generated,
            "text": sample_text,
        },
        "offline_policy": {
            "process_socket_guard_tested": True,
            "physical_air_gap_verified": False,
            "note": "Physical air-gap verification is deferred to the owner's local machine.",
        },
        "known_limits": [
            "The Smoke corpus is synthetic and tiny.",
            "The base checkpoint is not instruction-tuned.",
            "Generated text can be incoherent and must not be treated as factual.",
            "No Seed or Small training was performed.",
        ],
    }
    reports = workspace / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    json_path = Path(output_path) if output_path is not None else reports / "smoke-evaluation.json"
    if not json_path.is_absolute():
        json_path = workspace / json_path
    if json_path.resolve() != workspace and workspace not in json_path.resolve().parents:
        raise EvaluationError("evaluation report must be written under the workspace")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path = json_path.with_suffix(".md")
    markdown_path.write_text(
        "# Cyrus Smoke Evaluation\n\n"
        f"- Checkpoint step: {checkpoint.get('step')}\n"
        f"- Parameters: {model.parameter_count:,}\n"
        f"- Validation loss: {validation_loss:.4f}\n"
        f"- Test loss: {test_loss:.4f}\n"
        f"- Time to first token: {first_token_seconds:.4f} s\n"
        f"- Generation: {report['metrics']['generation_tokens_per_second']:.2f} tokens/s\n"
        f"- Fixed-seed deterministic: {deterministic}\n"
        "- Physical air-gap verified: No (must be tested on the owner's machine)\n\n"
        "## Honest limitation\n\n"
        "This is a tiny base-model smoke test trained on synthetic fixtures. It is not a reliable assistant.\n",
        encoding="utf-8",
    )
    return {**report, "report_path": str(json_path), "markdown_report_path": str(markdown_path)}

