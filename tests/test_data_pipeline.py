from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from cyrus.config import load_config
from cyrus.data import DataError, approve_record, ingest_file, inspect_file, prepare_dataset


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "corpus"


class DataPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.data_root = self.root / "data"
        self.output_root = self.root / "processed"
        self.config = load_config(ROOT / "configs" / "smoke.yaml")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _ingest_all(self) -> list[dict[str, object]]:
        manifests = []
        for path in sorted(FIXTURES.glob("*.txt")):
            manifest = ingest_file(
                path,
                data_root=self.data_root,
                license_name="project-synthetic-test-fixture",
                source_name="Cyrus repository test fixture",
                language="mr" if "marathi" in path.name else "und",
                max_file_bytes=self.config.data.max_file_bytes,
            )
            manifests.append(manifest)
            approve_record(str(manifest["record_id"]), data_root=self.data_root)
        return manifests

    def test_inspection_and_ingestion_are_content_addressed_and_idempotent(self) -> None:
        path = FIXTURES / "01_locality.txt"
        inspection = inspect_file(path, max_file_bytes=self.config.data.max_file_bytes)
        first = ingest_file(
            path,
            data_root=self.data_root,
            license_name="project-synthetic-test-fixture",
            source_name="Cyrus repository test fixture",
            max_file_bytes=self.config.data.max_file_bytes,
        )
        second = ingest_file(
            path,
            data_root=self.data_root,
            license_name="a different claim cannot rewrite the manifest",
            source_name="different source",
            max_file_bytes=self.config.data.max_file_bytes,
        )
        self.assertEqual(first, second)
        self.assertEqual(first["record_id"], inspection["content_sha256"])
        self.assertEqual(first["state"], "quarantined")
        manifest_files = list((self.data_root / "manifests").glob("*.json"))
        self.assertEqual(len(manifest_files), 1)

    def test_secret_flag_requires_explicit_review_override(self) -> None:
        path = self.root / "flagged.txt"
        path.write_text(
            "This review fixture deliberately contains api_key=abcdefghijklmnop and enough surrounding text.",
            encoding="utf-8",
        )
        manifest = ingest_file(
            path,
            data_root=self.data_root,
            license_name="owner-provided",
            source_name="security test",
            max_file_bytes=self.config.data.max_file_bytes,
        )
        self.assertIn("generic-secret", manifest["warnings"])
        with self.assertRaisesRegex(DataError, "review flags"):
            approve_record(str(manifest["record_id"]), data_root=self.data_root)
        approved = approve_record(
            str(manifest["record_id"]), data_root=self.data_root, allow_flagged=True
        )
        self.assertEqual(approved["state"], "approved")

    def test_dataset_snapshot_is_deterministic_and_split_before_sharding(self) -> None:
        manifests = self._ingest_all()
        first = prepare_dataset(
            data_root=self.data_root,
            output_root=self.output_root,
            config=self.config.data,
        )
        second = prepare_dataset(
            data_root=self.data_root,
            output_root=self.output_root,
            config=self.config.data,
        )
        self.assertEqual(first["snapshot_id"], second["snapshot_id"])
        self.assertEqual(first["shards"], second["shards"])
        self.assertEqual(first["records"], len(manifests))

        seen: set[str] = set()
        snapshot = Path(first["path"])
        for split in ("train", "validation", "test"):
            shard = snapshot / f"{split}.jsonl"
            for line in shard.read_text(encoding="utf-8").splitlines():
                item = json.loads(line)
                self.assertNotIn(item["record_id"], seen)
                seen.add(item["record_id"])
        self.assertEqual(len(seen), len(manifests))
        stored_manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(stored_manifest["snapshot_id"], first["snapshot_id"])

    def test_path_traversal_record_id_is_rejected(self) -> None:
        with self.assertRaisesRegex(DataError, "SHA-256"):
            approve_record("../../outside", data_root=self.data_root)


if __name__ == "__main__":
    unittest.main()

