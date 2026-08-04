from __future__ import annotations

import unittest

from ml.training.drawback_ml.path_validation import (
    is_portable_safe_basename,
    portable_basename_key,
)


class PortableBasenameTests(unittest.TestCase):
    def test_accepts_canonical_release_artifact_names(self) -> None:
        for value in (
            "release.json",
            "sealed-report-2026_08_04.json",
            "checkpoint.pt",
            "A1",
        ):
            with self.subTest(value=value):
                self.assertTrue(is_portable_safe_basename(value))

    def test_rejects_cross_platform_aliases_and_paths(self) -> None:
        for value in (
            "",
            ".",
            "..",
            ".hidden",
            "report.json.",
            "report.json ",
            "report.json:secret",
            "folder/report.json",
            r"folder\report.json",
            r"C:\report.json",
            "NUL",
            "nul.json",
            "CON",
            "aux.txt",
            "COM1.log",
            "lpt9",
            "REPORT~1.JSON",
            "report name.json",
            "réport.json",
            "report\x00.json",
            "a" * 101,
        ):
            with self.subTest(value=value):
                self.assertFalse(is_portable_safe_basename(value))

    def test_collision_key_is_case_insensitive(self) -> None:
        self.assertEqual(
            portable_basename_key("Sealed-Report.json"),
            portable_basename_key("sealed-report.JSON"),
        )
        with self.assertRaisesRegex(ValueError, "safe basename"):
            portable_basename_key("report.json:secret")


if __name__ == "__main__":
    unittest.main()
