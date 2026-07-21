# Model Card: Cyrus Smoke Base

- **Purpose:** Validate the Cyrus stack end to end.
- **Architecture:** 2,718,144-parameter decoder-only causal Transformer with RMSNorm, RoPE, SDPA, SwiGLU, and tied embeddings.
- **Initialization:** Random; no pretrained weights, distillation, merging, or hosted-model calls.
- **Tokenizer:** Repository-trained 320-token byte-BPE with full UTF-8 byte coverage.
- **Training data:** Twelve repository-authored synthetic fixtures, with held-out documents excluded from training.
- **Observed training:** 100 CPU steps, 12,800 tokens, final validation loss 4.0608.
- **Intended use:** Software correctness, profiling, checkpoint/recovery testing, and local interface development.
- **Not intended for:** Advice, factual answers, private-data memorization, autonomous action, or any high-stakes use.
- **Known behavior:** Generated text is frequently incoherent; the model is not instruction-tuned.
- **Distribution:** Checkpoints are ignored by Git and must be reproduced locally.

