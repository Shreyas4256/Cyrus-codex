"""Cyrus decoder-only Transformer, initialized entirely from random weights."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ModelHyperparameters:
    vocab_size: int
    context_length: int
    d_model: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    dropout: float = 0.0
    rope_theta: float = 10_000.0
    rms_norm_eps: float = 1e-5

    def __post_init__(self) -> None:
        if self.vocab_size <= 0 or self.context_length <= 0 or self.d_model <= 0:
            raise ValueError("vocabulary, context length, and model width must be positive")
        if self.n_layers <= 0 or self.n_heads <= 0 or self.n_kv_heads <= 0:
            raise ValueError("layer and head counts must be positive")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if (self.d_model // self.n_heads) % 2:
            raise ValueError("attention head dimension must be even for rotary embeddings")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.rope_theta <= 0 or self.rms_norm_eps <= 0:
            raise ValueError("RoPE theta and RMSNorm epsilon must be positive")

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def choose_device(requested: str = "auto") -> torch.device:
    """Select a portable PyTorch device without assuming NVIDIA hardware."""

    normalized = requested.casefold()
    if normalized != "auto":
        device = torch.device(normalized)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA/ROCm was requested but is not available")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("Apple MPS was requested but is not available")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class RMSNorm(nn.Module):
    def __init__(self, dimension: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dimension))
        self.eps = eps

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        normalized = inputs.float() * torch.rsqrt(inputs.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return normalized.to(dtype=inputs.dtype) * self.weight


def _apply_rotary(inputs: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    even = inputs[..., ::2]
    odd = inputs[..., 1::2]
    rotated = torch.stack((-odd, even), dim=-1).flatten(-2)
    return inputs * cos + rotated * sin


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dimension: int, theta: float) -> None:
        super().__init__()
        inverse_frequency = 1.0 / (
            theta ** (torch.arange(0, head_dimension, 2, dtype=torch.float32) / head_dimension)
        )
        self.register_buffer("inverse_frequency", inverse_frequency, persistent=True)

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, position_offset: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sequence_length = query.size(-2)
        positions = torch.arange(
            position_offset,
            position_offset + sequence_length,
            device=query.device,
            dtype=torch.float32,
        )
        frequencies = torch.outer(positions, self.inverse_frequency.float())
        angles = torch.repeat_interleave(frequencies, repeats=2, dim=-1)[None, None, :, :]
        cos = angles.cos().to(dtype=query.dtype)
        sin = angles.sin().to(dtype=query.dtype)
        return _apply_rotary(query, cos, sin), _apply_rotary(key, cos, sin)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelHyperparameters) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dimension = config.d_model // config.n_heads
        query_size = config.n_heads * self.head_dimension
        kv_size = config.n_kv_heads * self.head_dimension
        self.query = nn.Linear(config.d_model, query_size, bias=False)
        self.key = nn.Linear(config.d_model, kv_size, bias=False)
        self.value = nn.Linear(config.d_model, kv_size, bias=False)
        self.output = nn.Linear(query_size, config.d_model, bias=False)
        self.rotary = RotaryEmbedding(self.head_dimension, config.rope_theta)
        self.dropout = config.dropout

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, sequence, _ = inputs.shape
        query = self.query(inputs).view(batch, sequence, self.n_heads, self.head_dimension).transpose(1, 2)
        key = self.key(inputs).view(batch, sequence, self.n_kv_heads, self.head_dimension).transpose(1, 2)
        value = self.value(inputs).view(batch, sequence, self.n_kv_heads, self.head_dimension).transpose(1, 2)
        query, key = self.rotary(query, key)
        if self.n_kv_heads != self.n_heads:
            repeats = self.n_heads // self.n_kv_heads
            key = key.repeat_interleave(repeats, dim=1)
            value = value.repeat_interleave(repeats, dim=1)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        attended = attended.transpose(1, 2).contiguous().view(batch, sequence, -1)
        return self.output(attended)


class SwiGLU(nn.Module):
    def __init__(self, config: ModelHyperparameters) -> None:
        super().__init__()
        raw_hidden = int((8 * config.d_model) / 3)
        hidden = 64 * math.ceil(raw_hidden / 64)
        self.gate = nn.Linear(config.d_model, hidden, bias=False)
        self.up = nn.Linear(config.d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, config.d_model, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(inputs)) * self.up(inputs))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelHyperparameters) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.attention = CausalSelfAttention(config)
        self.feed_forward_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.feed_forward = SwiGLU(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = inputs + self.dropout(self.attention(self.attention_norm(inputs)))
        return hidden + self.dropout(self.feed_forward(self.feed_forward_norm(hidden)))


class CyrusModel(nn.Module):
    """A decoder-only causal language model with no pretrained components."""

    def __init__(self, config: ModelHyperparameters) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layers))
        self.final_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.size(1) > self.config.context_length:
            raise ValueError("sequence exceeds the configured context length")
        hidden = self.embedding_dropout(self.token_embedding(input_ids))
        for block in self.blocks:
            hidden = block(hidden)
        logits = self.lm_head(self.final_norm(hidden))
        loss = None
        if targets is not None:
            if targets.shape != input_ids.shape:
                raise ValueError("targets must have the same shape as input_ids")
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate_iter(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: int | None = 40,
        seed: int = 0,
    ) -> Iterator[torch.Tensor]:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens cannot be negative")
        generator = torch.Generator(device=input_ids.device)
        generator.manual_seed(seed)
        sequence = input_ids
        was_training = self.training
        self.eval()
        try:
            for _ in range(max_new_tokens):
                conditioned = sequence[:, -self.config.context_length :]
                logits, _ = self(conditioned)
                next_logits = logits[:, -1, :] / temperature
                if top_k is not None:
                    bounded_k = min(max(top_k, 1), next_logits.size(-1))
                    threshold = torch.topk(next_logits, bounded_k).values[:, [-1]]
                    next_logits = next_logits.masked_fill(next_logits < threshold, float("-inf"))
                probabilities = F.softmax(next_logits, dim=-1)
                next_id = torch.multinomial(probabilities, num_samples=1, generator=generator)
                sequence = torch.cat((sequence, next_id), dim=1)
                yield next_id
        finally:
            self.train(was_training)

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, **kwargs: int | float | None) -> torch.Tensor:
        sequence = input_ids
        for next_id in self.generate_iter(input_ids, **kwargs):
            sequence = torch.cat((sequence, next_id), dim=1)
        return sequence

