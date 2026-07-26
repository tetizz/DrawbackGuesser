from __future__ import annotations

import unittest

from ml.evaluation import (
    SplitManifest,
    SplitOverlapError,
    validate_split_manifest,
)


class SplitManifestTest(unittest.TestCase):
    def test_accepts_disjoint_seed_sets(self) -> None:
        manifest = SplitManifest.from_mapping(
            {
                "train": [1, 2, 3],
                "validation": [101, 102],
                "test": [201, 202],
            }
        )
        self.assertEqual(manifest.validation, (101, 102))

    def test_rejects_cross_split_overlap(self) -> None:
        with self.assertRaisesRegex(SplitOverlapError, "train and validation"):
            validate_split_manifest(
                {"train": [1, 2], "validation": [2, 3], "test": [4]}
            )

    def test_rejects_duplicates_within_a_split(self) -> None:
        with self.assertRaisesRegex(SplitOverlapError, "duplicate"):
            SplitManifest(train=(1, 1), validation=(2,), test=(3,))

    def test_rejects_missing_or_invalid_seeds(self) -> None:
        with self.assertRaises(ValueError):
            SplitManifest.from_mapping({"train": [1], "validation": [2]})
        with self.assertRaises(ValueError):
            SplitManifest(train=(-1,), validation=(2,), test=(3,))
        with self.assertRaises(ValueError):
            SplitManifest(train=(True,), validation=(2,), test=(3,))


if __name__ == "__main__":
    unittest.main()
