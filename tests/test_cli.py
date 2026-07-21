from __future__ import annotations

import contextlib
import io
import json
import unittest
from pathlib import Path

from cyrus.cli.main import main


ROOT = Path(__file__).resolve().parents[1]


class CliTests(unittest.TestCase):
    def test_doctor_command_emits_machine_readable_report(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = main(["doctor", "--config", str(ROOT / "configs" / "smoke.yaml")])
        self.assertEqual(result, 0)
        report = json.loads(output.getvalue())
        self.assertEqual(report["selected_profile"], "smoke")
        self.assertTrue(report["config"]["offline"])


if __name__ == "__main__":
    unittest.main()

