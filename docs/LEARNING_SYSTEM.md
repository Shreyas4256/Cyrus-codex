# Learning System

Cyrus currently has two safe learning speeds:

1. Approved local documents become searchable immediately through SQLite FTS5 memory. Their source and record ID accompany every retrieved excerpt.
2. Weight changes happen only in a bounded training job that freezes a token snapshot, writes candidate checkpoints, evaluates gates, and preserves the prior stable checkpoint.

The current Smoke trainer does not silently learn from conversations, model outputs, or the memory database. It does not update weights after each message. Continual-learning consolidation, replay buffers, correction memory, candidate comparison against a mature task suite, rollback history, and instruction tuning are intentionally not claimed as complete.

The CLI chat path retrieves up to a configured number of excerpts, marks them as memory context, and streams actual tokens from the stable model. Because the Smoke base model is not instruction-tuned, retrieved evidence being correct does not mean its generated completion will answer correctly. The CLI therefore prints evidence separately and labels model output as unreliable.

