from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from cyrus.training.checkpoint import CheckpointError, load_checkpoint, save_checkpoint


ROOT = Path(__file__).resolve().parents[1]


class CheckpointTests(unittest.TestCase):
    def test_checkpoint_is_atomic_hash_verified_and_safely_loaded(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "checkpoints") as directory:
            path = Path(directory) / "candidate.pt"
            saved = save_checkpoint(
                path,
                {"schema_version": 1, "weights": {"x": torch.arange(4)}},
                workspace_root=ROOT,
            )
            self.assertTrue(path.exists())
            self.assertEqual(len(saved["sha256"]), 64)
            loaded = load_checkpoint(path, workspace_root=ROOT)
            torch.testing.assert_close(loaded["weights"]["x"], torch.arange(4))
            with path.open("ab") as handle:
                handle.write(b"tamper")
            with self.assertRaisesRegex(CheckpointError, "verification failed"):
                load_checkpoint(path, workspace_root=ROOT)

    def test_checkpoint_outside_workspace_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(CheckpointError, "repository workspace"):
                save_checkpoint(
                    Path(directory) / "outside.pt",
                    {"schema_version": 1},
                    workspace_root=ROOT,
                )


if __name__ == "__main__":
    unittest.main()

