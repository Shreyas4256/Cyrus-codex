# Cyrus

Cyrus is a fully local language-model research stack. Its tokenizer is trained from the repository's approved corpus, its Transformer weights start from random initialization, and runtime commands do not call hosted models or require API keys.

The current checkpoint family is a **2.72M-parameter Smoke model**. It proves the pipeline, trainer, recovery, evaluation, memory, and chat plumbing. It is not an intelligent general assistant: its generated text is usually incoherent because it has seen only a tiny synthetic fixture corpus.

## What works

- Hardware and safety-limit inspection with `cyrus doctor`
- Approval-gated UTF-8/Markdown/JSONL ingestion with hashes, provenance, scanner flags, and deterministic splits
- A from-scratch, lossless Unicode byte-BPE tokenizer
- A repository-owned decoder-only Transformer with RMSNorm, RoPE, causal SDPA, GQA support, SwiGLU, and tied embeddings
- Bounded CPU Smoke training, atomic resumable checkpoints, hash verification, retention, and promotion gates
- Fixed-seed evaluation reports with loss, perplexity, latency, throughput, repetition, and honest limits
- SQLite FTS5 local memory with source citations, inspection, search, and verified hard deletion
- Streaming CLI completion from the real checkpoint with optional retrieved evidence
- A process-level socket guard and tests for offline runtime behavior

## GitHub Codespaces quick start

The initial run uses CPU PyTorch even when the host has no GPU.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python -m pip install -r requirements-ml.lock \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install --no-build-isolation --no-deps -e .

cyrus doctor --config configs/smoke.yaml
python scripts/bootstrap_smoke.py
```

The bootstrap command prints the content-addressed token snapshot path. Use that exact path below:

```bash
cyrus train pretrain \
  --tokens data/processed/tokens-<snapshot> \
  --tokenizer artifacts/tokenizers/cyrus-smoke.json \
  --config configs/smoke.yaml \
  --workspace . \
  --device cpu

cyrus eval \
  --checkpoint checkpoints/stable/cyrus-smoke-base.pt \
  --tokenizer artifacts/tokenizers/cyrus-smoke.json \
  --tokens data/processed/tokens-<snapshot> \
  --config configs/smoke.yaml \
  --workspace .

cyrus memory index --data-root data --db data/memory.sqlite3

cyrus chat \
  --checkpoint checkpoints/stable/cyrus-smoke-base.pt \
  --tokenizer artifacts/tokenizers/cyrus-smoke.json \
  --memory-db data/memory.sqlite3 \
  --workspace .
```

Run the complete test suite with:

```bash
python -m unittest discover -s tests -v
```

## Local GPU portability

The model chooses CUDA/ROCm, Apple MPS, or CPU through PyTorch. The Codespaces lock intentionally installs the CPU wheel. On a physical NVIDIA or AMD machine, install the PyTorch build matching that machine from the official selector, then install Cyrus with `--no-deps`. Do not copy a CUDA wheel between incompatible systems.

## Private artifacts

`data/`, `artifacts/`, `checkpoints/`, SQLite memory, and generated reports are ignored by Git except for empty directory markers. Do not force-add them. Only the tiny synthetic test fixtures belong in version control.

Read [docs/STATUS.md](docs/STATUS.md) for measured results and [docs/SEED_PLAN.md](docs/SEED_PLAN.md) before considering any larger training run.

