from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from cyrus.config import load_config
from cyrus.data import approve_record, ingest_file, pack_token_shards, prepare_dataset
from cyrus.tokenizer import ByteBPETokenizer, TokenizerError, train_from_dataset


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "corpus"


class TokenizerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config(ROOT / "configs" / "smoke.yaml")
        self.texts = [path.read_text(encoding="utf-8") for path in sorted(FIXTURES.glob("*.txt"))]

    def test_unicode_round_trip_and_deterministic_training(self) -> None:
        first = ByteBPETokenizer.train(
            self.texts,
            vocab_size=self.config.tokenizer.vocab_size,
            min_frequency=self.config.tokenizer.min_frequency,
            special_tokens=self.config.tokenizer.special_tokens,
        )
        second = ByteBPETokenizer.train(
            self.texts,
            vocab_size=self.config.tokenizer.vocab_size,
            min_frequency=self.config.tokenizer.min_frequency,
            special_tokens=self.config.tokenizer.special_tokens,
        )
        self.assertEqual(first.tokenizer_hash, second.tokenizer_hash)
        self.assertGreater(first.vocab_size, 256 + len(first.special_tokens))
        samples = [
            "plain English",
            "मराठी मजकूर",
            "हिन्दी पाठ",
            "café Ελληνικά 日本語 🚀",
            "line one\nline two\twith a tab",
        ]
        for sample in samples:
            self.assertEqual(first.decode(first.encode(sample)), sample)

    def test_special_tokens_are_opt_in(self) -> None:
        tokenizer = ByteBPETokenizer.train(
            self.texts,
            vocab_size=280,
            min_frequency=2,
            special_tokens=self.config.tokenizer.special_tokens,
        )
        text = "<|user|>hello<|assistant|>"
        ordinary = tokenizer.encode(text)
        structured = tokenizer.encode(text, allowed_special=True)
        self.assertNotEqual(ordinary, structured)
        self.assertIn(tokenizer.special_to_id["<|user|>"], structured)
        self.assertEqual(tokenizer.decode(structured), text)

    def test_saved_artifact_is_hash_verified(self) -> None:
        tokenizer = ByteBPETokenizer.train(
            self.texts,
            vocab_size=280,
            min_frequency=2,
            special_tokens=self.config.tokenizer.special_tokens,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            tokenizer.save(path)
            loaded = ByteBPETokenizer.load(path)
            self.assertEqual(loaded.tokenizer_hash, tokenizer.tokenizer_hash)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["training_corpus_sha256"] = "tampered"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(TokenizerError, "hash mismatch"):
                ByteBPETokenizer.load(path)

    def test_approved_dataset_to_tokenizer_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for fixture in sorted(FIXTURES.glob("*.txt")):
                manifest = ingest_file(
                    fixture,
                    data_root=root / "data",
                    license_name="project-synthetic-test-fixture",
                    source_name="Cyrus repository test fixture",
                    max_file_bytes=self.config.data.max_file_bytes,
                )
                approve_record(str(manifest["record_id"]), data_root=root / "data")
            dataset = prepare_dataset(
                data_root=root / "data",
                output_root=root / "processed",
                config=self.config.data,
            )
            output = root / "reports" / "tokenizer.json"
            report = train_from_dataset(
                dataset["path"], config=self.config.tokenizer, output_path=output
            )
            self.assertTrue(output.exists())
            self.assertTrue(Path(report["report_path"]).exists())
            self.assertEqual(report["unknown_rate"] if "unknown_rate" in report else 0.0, 0.0)
            loaded = ByteBPETokenizer.load(output)
            self.assertEqual(loaded.tokenizer_hash, report["tokenizer_sha256"])
            packed = pack_token_shards(
                dataset_dir=dataset["path"],
                tokenizer_path=output,
                output_root=root / "processed",
                sequence_length=self.config.training.sequence_length,
            )
            packed_again = pack_token_shards(
                dataset_dir=dataset["path"],
                tokenizer_path=output,
                output_root=root / "processed",
                sequence_length=self.config.training.sequence_length,
            )
            self.assertEqual(packed["snapshot_id"], packed_again["snapshot_id"])
            self.assertGreater(packed["shards"]["train"]["usable_windows"], 0)
            train_tokens = json.loads(
                (Path(packed["path"]) / "train.tokens.json").read_text(encoding="utf-8")
            )["tokens"]
            self.assertIn(loaded.special_to_id["<|document|>"], train_tokens)


if __name__ == "__main__":
    unittest.main()
