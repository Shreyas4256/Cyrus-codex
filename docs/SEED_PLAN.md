# Cyrus Seed Hardware and Data Plan

No Seed or Small training is authorized by the Codespaces override. This document defines the decisions required before a Seed experiment can begin.

## Proposed research target—not approval

| Item | Proposed planning range |
|---|---:|
| Parameters | 20M–50M initially |
| Context | 512–1,024 tokens |
| Vocabulary | Compare 8K, 12K, and 16K |
| Training tokens | Determine only after corpus audit; target substantially more tokens than parameters |
| Precision | BF16 where verified; otherwise FP32/FP16 with measured stability |
| Checkpoint storage | At least 5× one full training-state checkpoint plus processed data |
| Promotion | Manual approval required |

## Hardware measurements required

Run `cyrus doctor`, then record exact CPU, RAM, free disk, GPU/accelerator model, VRAM, driver, CUDA/ROCm/MPS version, PyTorch build, sustained training throughput, peak memory, thermals, and power stability. Run a 10–30 minute profiling job first; extrapolate wall time and energy from measured tokens/second. Do not infer feasibility from advertised GPU specifications.

A practical single-GPU starting point is a modern accelerator with at least 12–16 GiB VRAM, 32 GiB system RAM, and ample SSD space, but the actual configuration must be chosen from profiling. CPU-only Seed training is technically portable but may be impractically slow. Multi-GPU work remains out of scope until the single-device trainer is reproducible.

## Data approval checklist

- List every source, owner/permission, license, acquisition date, and immutable hash.
- Exclude personal conversations and private documents unless the owner gives explicit, informed approval for that exact training use.
- Measure English, Hindi, and Marathi coverage rather than claiming multilingual ability from a few examples.
- Run secret/PII review, exact and near-deduplication, quality filtering, and document-level split leakage checks.
- Freeze development and final evaluation sets before training.
- Compare tokenizer vocabulary sizes on the training sample and record fertility by language.
- Calculate tokens, bytes, shard count, checkpoint size, expected training steps, wall time, and recovery storage.
- Obtain explicit owner approval for the corpus snapshot, compute budget, and maximum training duration.

## Go/no-go gates

Seed remains **no-go** until Smoke reproduces from a clean checkout on the target physical machine, physical offline operation is verified, checkpoint interruption/resume is tested during an active run, the data card is signed, the compute estimate is approved, and a rollback-safe promotion policy exists. Cyrus Small is a separate future decision.

