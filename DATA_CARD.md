# Data Card: Cyrus Synthetic Smoke Fixtures

- **Records:** 12 short UTF-8 text documents.
- **Origin:** Created specifically as non-private repository test fixtures.
- **Permission label used by the pipeline:** `project-synthetic-test-fixture`.
- **Languages represented for plumbing tests:** English, Marathi, Hindi, plus a Unicode mixed-language fixture.
- **Purpose:** Exercise provenance, approval, splitting, tokenization, training, evaluation, and retrieval code.
- **Limitations:** Far too small and artificial to establish linguistic, factual, cultural, or safety capability.
- **Privacy:** Contains no intended personal data. Scanner tests use a separate temporary fake-secret fixture that is never committed to a dataset.
- **Versioning:** Source bytes live under `tests/fixtures/corpus`; generated manifests and shards are content-addressed and ignored by Git.

