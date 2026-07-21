# Data Governance

Codespaces uses only the twelve synthetic, non-private fixtures under `tests/fixtures/corpus`. They exist to verify software behavior, not to provide meaningful language capability.

Every ingestion requires an explicit source and license/permission label. Cyrus records the original SHA-256, normalized SHA-256, byte size, language tag, parser version, transformation list, timestamps, warnings, and state history. Re-ingesting identical bytes is idempotent.

Supported initial formats are UTF-8 text, Markdown, and JSONL containing strings or objects with a string `text` field. NUL bytes, unsupported formats, oversized inputs, malformed JSONL, symlinks, missing provenance, and invalid UTF-8 fail closed. Scanner flags cover common secrets, private keys, email addresses, Indian mobile numbers, extreme repetition, and very short documents. Flagged material needs an explicit review override.

Documents are split before token packing using a stable hash of the split seed and record ID. Exact normalized duplicates and high-overlap word five-gram near-duplicates are rejected before the split. Each generated shard has a content hash and lineage list. The tokenizer learns merges from the training shard only; validation and test text never influence merge selection.

Private datasets, manifests, normalized records, token shards, model artifacts, and memory databases are ignored by Git. Never use `git add -f` on these paths.

Before Seed training, replace synthetic fixtures with an owner-approved corpus inventory and complete the license, consent, language-balance, PII, duplication, quality, held-out evaluation, token-budget, and deletion review described in `docs/SEED_PLAN.md`.

