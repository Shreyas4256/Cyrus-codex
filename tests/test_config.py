from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cyrus.config import ConfigError, load_config


ROOT = Path(__file__).resolve().parents[1]


class ConfigTests(unittest.TestCase):
    def test_smoke_config_is_valid_and_safely_bounded(self) -> None:
        config = load_config(ROOT / "configs" / "smoke.yaml")
        self.assertEqual(config.profile, "smoke")
        self.assertTrue(config.runtime.offline)
        self.assertLessEqual(config.runtime.max_ram_gb, 4.0)
        self.assertEqual(config.tokenizer.vocab_size, 320)
        self.assertEqual(config.model.d_model % config.model.n_heads, 0)

    def test_invalid_split_is_rejected(self) -> None:
        original = (ROOT / "configs" / "smoke.yaml").read_text(encoding="utf-8")
        invalid = original.replace("test_ratio: 0.1", "test_ratio: 0.2")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.yaml"
            path.write_text(invalid, encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "sum to exactly 1.0"):
                load_config(path)

    def test_invalid_attention_shape_is_rejected(self) -> None:
        original = (ROOT / "configs" / "smoke.yaml").read_text(encoding="utf-8")
        invalid = original.replace("d_model: 192", "d_model: 191")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.yaml"
            path.write_text(invalid, encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "divisible"):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
