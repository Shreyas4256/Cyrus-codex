# Project Status

## Outcome

Milestones 0, 1, and 2 are complete for the CPU Smoke profile. The requested local-memory and streaming chat paths also work. Seed and Small training were not started.

## Verified evidence (2026-07-21)

- Model: 2,718,144 parameters, random initialization, 6 layers, width 192, 256-token context.
- Data: 12 synthetic/non-private records; deterministic snapshot `6e28fe0d46ea6648…`.
- Tokenizer: 320-token from-scratch byte-BPE; SHA-256 `3217ed75bf625755…`; lossless English/Hindi/Marathi/emoji tests.
- Token snapshot: `f8471b2d4abd81b6…`, explicit document boundaries, separate held-out shards.
- Smoke training: 100 steps and 12,800 tokens in 5.58 seconds on CPU; 2,295.70 training tokens/s.
- Validation loss: 5.8272 before training to 4.0608 after training.
- Tiny-batch overfit: 5.7757 to 1.3315 loss, a 76.95% reduction.
- Promotion: parameter budget, overfit, validation improvement, checkpoint parity, fixed-seed sampling, completed-step budget, and time budget all passed.
- Independent held-out evaluation: validation loss 4.1027; test loss 4.8658; test perplexity 129.77.
- Inference: 3.37 ms observed time to first token and 237.03 tokens/s for the fixed 48-token CPU sample.
- Recovery: hash-verified checkpoint loaded with optimizer/RNG state; a resume command completed without modifying the stable checkpoint.
- Memory: 12 approved records indexed locally; relevant source retrieval, idempotency, injection-shaped query handling, and verified deletion pass tests.
- Offline: process socket denial tests pass. Physical air-gap testing is pending on the owner's machine.

Generated datasets, tokenizers, reports, memory databases, and 32.7 MB checkpoints are present only in ignored workspace paths and are not intended for Git.

## Honest capability limit

The fixed sample is character-level-looking, partially word-like text, not a useful answer. This is expected from a 2.72M base model trained for 12,800 tokens on a tiny synthetic corpus. Local retrieval provides correct evidence, but the base model cannot reliably synthesize it. No instruction behavior, factual reliability, broad language competence, continual weight learning, or production safety is claimed.

## Stopping boundary

The Codespaces request is satisfied at the Smoke boundary: skeleton, data pipeline, tokenizer, model, training, evaluation, local memory, and streaming chat are implemented. The next engineering milestone would be a small approved instruction-tuning/evidence-grounding slice, but no larger training should start until the physical-machine and data decisions in `docs/SEED_PLAN.md` are approved.

