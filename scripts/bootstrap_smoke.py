#!/usr/bin/env python3
"""Create a reproducible Smoke data/tokenizer snapshot from repository fixtures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cyrus.config import load_config
from cyrus.data import approve_record, ingest_file, pack_token_shards, prepare_dataset
from cyrus.tokenizer import train_from_dataset


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--config", default="configs/smoke.yaml")
    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = workspace / config_path
    config = load_config(config_path)
    data_root = workspace / "data"
    processed_root = data_root / "processed"
    fixtures = workspace / "tests" / "fixtures" / "corpus"
    if not fixtures.is_dir():
        raise SystemExit(f"fixture directory not found: {fixtures}")

    record_ids: list[str] = []
    for path in sorted(fixtures.glob("*.txt")):
        language = "mr" if "marathi" in path.name else "hi" if "hindi" in path.name else "und"
        manifest = ingest_file(
            path,
            data_root=data_root,
            license_name="project-synthetic-test-fixture",
            source_name=f"repository fixture:{path.name}",
            language=language,
            max_file_bytes=config.data.max_file_bytes,
        )
        approve_record(str(manifest["record_id"]), data_root=data_root)
        record_ids.append(str(manifest["record_id"]))

    dataset = prepare_dataset(
        data_root=data_root,
        output_root=processed_root,
        config=config.data,
    )
    tokenizer_path = workspace / "artifacts" / "tokenizers" / "cyrus-smoke.json"
    tokenizer_report = train_from_dataset(
        dataset["path"],
        config=config.tokenizer,
        output_path=tokenizer_path,
    )
    tokens = pack_token_shards(
        dataset_dir=dataset["path"],
        tokenizer_path=tokenizer_path,
        output_root=processed_root,
        sequence_length=config.training.sequence_length,
    )
    print(
        json.dumps(
            {
                "approved_records": len(record_ids),
                "dataset": dataset["path"],
                "dataset_id": dataset["snapshot_id"],
                "tokenizer": str(tokenizer_path),
                "tokenizer_sha256": tokenizer_report["tokenizer_sha256"],
                "token_snapshot": tokens["path"],
                "token_snapshot_id": tokens["snapshot_id"],
                "next_command": (
                    f"cyrus train pretrain --tokens {tokens['path']} "
                    f"--tokenizer {tokenizer_path} --config {config_path}"
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

