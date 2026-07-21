"""Local-first Cyrus command-line application."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from cyrus import __version__
from cyrus.chat import retrieve_evidence, stream_completion
from cyrus.config import ConfigError, load_config
from cyrus.data import (
    DataError,
    approve_record,
    ingest_file,
    inspect_file,
    pack_token_shards,
    prepare_dataset,
)
from cyrus.doctor import inspect_environment
from cyrus.evaluation import EvaluationError, evaluate_checkpoint
from cyrus.memory import MemoryError as CyrusMemoryError
from cyrus.memory import MemoryStore
from cyrus.security import OfflineViolation, offline_guard
from cyrus.tokenizer import ByteBPETokenizer, TokenizerError, train_from_dataset
from cyrus.training import TrainingError, train_smoke
from cyrus.training.checkpoint import CheckpointError


def _print(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False))


def _add_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        default="configs/smoke.yaml",
        help="versioned Cyrus YAML profile (default: configs/smoke.yaml)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cyrus",
        description="Build and operate Cyrus entirely on local resources.",
    )
    parser.add_argument("--version", action="version", version=f"Cyrus {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    doctor = commands.add_parser("doctor", help="inspect hardware and validate a profile")
    _add_config(doctor)
    doctor.add_argument("--path", default=".", help="filesystem used for free-space checks")

    data = commands.add_parser("data", help="inspect and process owner-approved local data")
    data_commands = data.add_subparsers(dest="data_command", required=True)

    inspect = data_commands.add_parser("inspect", help="safely inspect a local input file")
    inspect.add_argument("path")
    _add_config(inspect)

    ingest = data_commands.add_parser("ingest", help="place a local file in quarantine")
    ingest.add_argument("path")
    ingest.add_argument("--license", required=True, dest="license_name")
    ingest.add_argument("--source", required=True, dest="source_name")
    ingest.add_argument("--language", default="und")
    ingest.add_argument("--data-root", default="data")
    _add_config(ingest)

    approve = data_commands.add_parser("approve", help="approve one quarantined record")
    approve.add_argument("record_id")
    approve.add_argument("--data-root", default="data")
    approve.add_argument(
        "--allow-flagged",
        action="store_true",
        help="explicitly approve after manually reviewing scanner flags",
    )
    _add_config(approve)

    prepare = data_commands.add_parser(
        "prepare", help="deduplicate and split all approved records deterministically"
    )
    prepare.add_argument("--data-root", default="data")
    prepare.add_argument("--output-root", default="data/processed")
    _add_config(prepare)

    pack = data_commands.add_parser(
        "pack", help="encode a dataset snapshot into tokenizer-bound token streams"
    )
    pack.add_argument("--dataset", required=True)
    pack.add_argument("--tokenizer", required=True)
    pack.add_argument("--output-root", default="data/processed")
    _add_config(pack)

    tokenizer = commands.add_parser("tokenizer", help="train and inspect the Cyrus tokenizer")
    tokenizer_commands = tokenizer.add_subparsers(dest="tokenizer_command", required=True)

    tokenizer_train = tokenizer_commands.add_parser(
        "train", help="train a byte-BPE tokenizer from an approved dataset snapshot"
    )
    tokenizer_train.add_argument("--dataset", required=True)
    tokenizer_train.add_argument("--output", default="reports/tokenizer-smoke.json")
    _add_config(tokenizer_train)

    tokenizer_encode = tokenizer_commands.add_parser("encode", help="encode text locally")
    tokenizer_encode.add_argument("--tokenizer", required=True)
    tokenizer_encode.add_argument("text")
    tokenizer_encode.add_argument("--allowed-special", action="store_true")

    tokenizer_decode = tokenizer_commands.add_parser("decode", help="decode comma-separated IDs")
    tokenizer_decode.add_argument("--tokenizer", required=True)
    tokenizer_decode.add_argument("ids", help="for example: 5,271,272,1")
    tokenizer_decode.add_argument("--hide-special", action="store_true")

    train = commands.add_parser("train", help="run bounded local Cyrus training jobs")
    train_commands = train.add_subparsers(dest="train_command", required=True)
    pretrain = train_commands.add_parser(
        "pretrain", help="train the random-initialized CPU-compatible Smoke model"
    )
    pretrain.add_argument("--tokens", required=True, help="immutable token snapshot directory")
    pretrain.add_argument("--tokenizer", required=True)
    pretrain.add_argument("--workspace", default=".")
    pretrain.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    pretrain.add_argument("--resume", default=None)
    pretrain.add_argument("--no-promote", action="store_true")
    _add_config(pretrain)

    evaluate = commands.add_parser("eval", help="evaluate a real local Cyrus checkpoint")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--tokenizer", required=True)
    evaluate.add_argument("--tokens", required=True)
    evaluate.add_argument("--workspace", default=".")
    evaluate.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    evaluate.add_argument("--output", default=None)
    _add_config(evaluate)

    memory = commands.add_parser("memory", help="manage provenance-bearing local memory")
    memory_commands = memory.add_subparsers(dest="memory_command", required=True)
    memory_index = memory_commands.add_parser("index", help="index all approved local records")
    memory_index.add_argument("--data-root", default="data")
    memory_index.add_argument("--db", default="data/memory.sqlite3")
    _add_config(memory_index)
    memory_search = memory_commands.add_parser("search", help="search approved local memory")
    memory_search.add_argument("query")
    memory_search.add_argument("--db", default="data/memory.sqlite3")
    memory_search.add_argument("--limit", type=int, default=5)
    _add_config(memory_search)
    memory_list = memory_commands.add_parser("list", help="list indexed memory records")
    memory_list.add_argument("--db", default="data/memory.sqlite3")
    memory_list.add_argument("--limit", type=int, default=100)
    _add_config(memory_list)
    memory_forget = memory_commands.add_parser("forget", help="hard-delete one memory record")
    memory_forget.add_argument("record_id")
    memory_forget.add_argument("--db", default="data/memory.sqlite3")
    _add_config(memory_forget)

    chat = commands.add_parser("chat", help="stream tokens from a real local Cyrus checkpoint")
    chat.add_argument("--checkpoint", required=True)
    chat.add_argument("--tokenizer", required=True)
    chat.add_argument("--prompt", default=None, help="omit to enter one prompt interactively")
    chat.add_argument("--memory-db", default=None)
    chat.add_argument("--memory-limit", type=int, default=3)
    chat.add_argument("--workspace", default=".")
    chat.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    chat.add_argument("--max-new-tokens", type=int, default=80)
    chat.add_argument("--temperature", type=float, default=0.8)
    chat.add_argument("--top-k", type=int, default=40)
    chat.add_argument("--seed", type=int, default=4256)
    _add_config(chat)

    return parser


def _run(args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "doctor":
        config = load_config(args.config)
        report = inspect_environment(args.path).to_dict()
        report["config"] = {
            "path": str(Path(args.config).resolve()),
            "profile": config.profile,
            "offline": config.runtime.offline,
            "limits": {
                "max_workers": config.runtime.max_workers,
                "max_ram_gb": config.runtime.max_ram_gb,
                "max_disk_gb": config.runtime.max_disk_gb,
                "max_training_minutes": config.runtime.max_training_minutes,
                "checkpoint_retention": config.runtime.checkpoint_retention,
            },
        }
        return report

    if args.command == "data":
        config = load_config(args.config)
        with offline_guard(config.runtime.offline):
            if args.data_command == "inspect":
                return inspect_file(args.path, max_file_bytes=config.data.max_file_bytes)
            if args.data_command == "ingest":
                return ingest_file(
                    args.path,
                    data_root=args.data_root,
                    license_name=args.license_name,
                    source_name=args.source_name,
                    language=args.language,
                    max_file_bytes=config.data.max_file_bytes,
                )
            if args.data_command == "approve":
                return approve_record(
                    args.record_id,
                    data_root=args.data_root,
                    allow_flagged=args.allow_flagged,
                )
            if args.data_command == "prepare":
                return prepare_dataset(
                    data_root=args.data_root,
                    output_root=args.output_root,
                    config=config.data,
                )
            if args.data_command == "pack":
                return pack_token_shards(
                    dataset_dir=args.dataset,
                    tokenizer_path=args.tokenizer,
                    output_root=args.output_root,
                    sequence_length=config.training.sequence_length,
                )

    if args.command == "tokenizer":
        if args.tokenizer_command == "train":
            config = load_config(args.config)
            with offline_guard(config.runtime.offline):
                return train_from_dataset(
                    args.dataset,
                    config=config.tokenizer,
                    output_path=args.output,
                )
        tokenizer = ByteBPETokenizer.load(args.tokenizer)
        if args.tokenizer_command == "encode":
            token_ids = tokenizer.encode(args.text, allowed_special=args.allowed_special)
            return {"token_ids": token_ids, "count": len(token_ids)}
        if args.tokenizer_command == "decode":
            try:
                token_ids = [int(value.strip()) for value in args.ids.split(",") if value.strip()]
            except ValueError as exc:
                raise TokenizerError("ids must be comma-separated integers") from exc
            return {
                "text": tokenizer.decode(token_ids, show_special=not args.hide_special),
                "count": len(token_ids),
            }

    if args.command == "train" and args.train_command == "pretrain":
        config = load_config(args.config)
        with offline_guard(config.runtime.offline):
            return train_smoke(
                config=config,
                token_snapshot_dir=args.tokens,
                tokenizer_path=args.tokenizer,
                workspace_root=args.workspace,
                device_name=args.device,
                resume_path=args.resume,
                promote=not args.no_promote,
            )

    if args.command == "eval":
        config = load_config(args.config)
        with offline_guard(config.runtime.offline):
            return evaluate_checkpoint(
                checkpoint_path=args.checkpoint,
                tokenizer_path=args.tokenizer,
                token_snapshot_dir=args.tokens,
                workspace_root=args.workspace,
                device_name=args.device,
                output_path=args.output,
            )

    if args.command == "memory":
        config = load_config(args.config)
        with offline_guard(config.runtime.offline), MemoryStore(args.db) as store:
            if args.memory_command == "index":
                return store.index_approved(args.data_root)
            if args.memory_command == "search":
                return {"results": store.search(args.query, limit=args.limit)}
            if args.memory_command == "list":
                return {"records": store.list_records(limit=args.limit), "count": store.count()}
            if args.memory_command == "forget":
                return {"record_id": args.record_id, "deleted": store.forget(args.record_id)}

    if args.command == "chat":
        config = load_config(args.config)
        prompt = args.prompt if args.prompt is not None else input("You: ")
        with offline_guard(config.runtime.offline):
            evidence = retrieve_evidence(
                prompt, memory_db=args.memory_db, limit=args.memory_limit
            )
            if evidence:
                print("Retrieved local evidence (untrusted context):")
                for item in evidence:
                    print(f"- {item['source']} [{item['record_id'][:12]}]: {item['excerpt']}")
            else:
                print("Retrieved local evidence: none")
            print(
                "Cyrus Smoke (real base-model tokens; not instruction-tuned or fact-reliable): ",
                end="",
                flush=True,
            )
            chunks = 0
            for chunk in stream_completion(
                checkpoint_path=args.checkpoint,
                tokenizer_path=args.tokenizer,
                workspace_root=args.workspace,
                prompt=prompt,
                evidence=evidence,
                device_name=args.device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                seed=args.seed,
            ):
                print(chunk, end="", flush=True)
                chunks += 1
            print()
            return {
                "status": "complete",
                "streamed_chunks": chunks,
                "sources": [
                    {"record_id": item["record_id"], "source": item["source"]}
                    for item in evidence
                ],
            }

    raise RuntimeError("unreachable command state")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = _run(args)
    except (
        ConfigError,
        DataError,
        TokenizerError,
        OfflineViolation,
        TrainingError,
        CheckpointError,
        EvaluationError,
        CyrusMemoryError,
    ) as exc:
        print(json.dumps({"error": type(exc).__name__, "message": str(exc)}), file=sys.stderr)
        return 2
    _print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
