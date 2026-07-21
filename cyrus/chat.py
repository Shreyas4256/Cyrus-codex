"""Real local checkpoint chat/completion with optional provenance-bearing memory."""

from __future__ import annotations

import codecs
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch

from cyrus.memory import MemoryStore
from cyrus.training import load_trained_model


def retrieve_evidence(
    prompt: str, *, memory_db: str | Path | None, limit: int = 3
) -> list[dict[str, Any]]:
    if memory_db is None or not Path(memory_db).exists():
        return []
    with MemoryStore(memory_db) as store:
        return store.search(prompt, limit=limit)


def build_prompt(prompt: str, evidence: list[dict[str, Any]]) -> str:
    sections = ["<|begin|><|user|>", prompt]
    if evidence:
        sections.append("\n<|memory|>\n")
        for item in evidence:
            sections.append(
                f"[source:{item['source']} record:{item['record_id']}]\n{item['excerpt']}\n"
            )
    sections.append("\n<|assistant|>")
    return "".join(sections)


def stream_completion(
    *,
    checkpoint_path: str | Path,
    tokenizer_path: str | Path,
    workspace_root: str | Path,
    prompt: str,
    evidence: list[dict[str, Any]],
    device_name: str = "auto",
    max_new_tokens: int = 80,
    temperature: float = 0.8,
    top_k: int = 40,
    seed: int = 4256,
) -> Iterator[str]:
    model, tokenizer, _checkpoint, device = load_trained_model(
        checkpoint_path=checkpoint_path,
        tokenizer_path=tokenizer_path,
        workspace_root=workspace_root,
        device_name=device_name,
    )
    formatted = build_prompt(prompt, evidence)
    input_ids = tokenizer.encode(formatted, allowed_special=True)
    inputs = torch.tensor([input_ids], dtype=torch.long, device=device)
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    end_id = tokenizer.special_to_id.get("<|end|>")
    for next_tensor in model.generate_iter(
        inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        seed=seed,
    ):
        token_id = int(next_tensor.item())
        if token_id == end_id:
            break
        if 0 <= token_id < tokenizer.byte_offset:
            pending = decoder.decode(b"", final=True)
            if pending:
                yield pending
            decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
            yield tokenizer.special_tokens[token_id]
        else:
            chunk = decoder.decode(tokenizer.token_bytes[token_id], final=False)
            if chunk:
                yield chunk
    pending = decoder.decode(b"", final=True)
    if pending:
        yield pending

