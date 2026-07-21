# Security and Offline Operation

## Implemented controls

- Runtime data, training, evaluation, memory, and chat commands enter a socket-denial context when `runtime.offline` is true.
- The test suite verifies that `socket.create_connection`, `socket.connect`, and `socket.connect_ex` fail before a connection attempt.
- Ingestion rejects symlinks, binary/NUL inputs, invalid UTF-8, unsupported formats, and files over the configured limit.
- Record IDs are full SHA-256 values and path construction is constrained to the configured data root.
- Approved bytes and token shards are re-hashed before use.
- Checkpoints must live under the repository workspace, may not traverse symlinked parents, require a SHA-256 sidecar, and load through PyTorch's restricted `weights_only=True` path.
- The local memory index accepts only hash-verified approved records. Search syntax is reconstructed from Unicode word tokens, preventing raw FTS query injection.
- Memory deletion removes both the record and search entry, then verifies the record is no longer retrievable. The audit row retains only event metadata and the record ID.
- No telemetry, analytics, hosted inference, remote fonts, CDN assets, or API keys are used.

## Important limit

The Python socket guard is defense in depth; it is not a physical air gap and cannot constrain a compromised native dependency or a separately launched process. Final verification must run on the owner's machine with networking disabled at the operating-system, container-network, firewall, or physical-interface layer. Dependencies should be acquired and verified first, then the runtime should be exercised with no route to the internet.

Do not load unknown checkpoints. A valid hash proves that bytes match the sidecar, not that an untrusted artifact is benevolent. Cyrus checkpoints should be produced inside this repository from approved snapshots.

