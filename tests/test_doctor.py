from __future__ import annotations

import unittest

from cyrus.doctor import inspect_environment


class DoctorTests(unittest.TestCase):
    def test_doctor_returns_measured_smoke_profile(self) -> None:
        report = inspect_environment()
        self.assertEqual(report.selected_profile, "smoke")
        self.assertGreaterEqual(report.cpu_count, 1)
        self.assertGreater(report.disk_free_bytes, 0)
        self.assertTrue(report.python)
        self.assertIsInstance(report.warnings, tuple)


if __name__ == "__main__":
    unittest.main()

