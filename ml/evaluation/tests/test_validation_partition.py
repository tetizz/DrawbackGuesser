from __future__ import annotations

import unittest

from ml.evaluation.validation_partition import (
    VALIDATION_PARTITION_IDENTITY,
    ValidationPartition,
    assign_validation_partition,
    validation_partition_unit,
    validation_seed_sha256,
)


class ValidationPartitionTest(unittest.TestCase):
    def test_frozen_identity_and_golden_assignments(self) -> None:
        self.assertEqual(
            VALIDATION_PARTITION_IDENTITY,
            "BLAKE2b-64(current-catalog-182-v2:validation:gameSeed)",
        )
        self.assertEqual(
            [assign_validation_partition(seed) for seed in range(10)],
            [
                ValidationPartition.SELECTION,
                ValidationPartition.SELECTION,
                ValidationPartition.SELECTION,
                ValidationPartition.SELECTION,
                ValidationPartition.VALIDATION_GATE,
                ValidationPartition.SELECTION,
                ValidationPartition.SELECTION,
                ValidationPartition.SELECTION,
                ValidationPartition.SELECTION,
                ValidationPartition.SELECTION,
            ],
        )

    def test_unit_is_stable_and_inside_half_open_interval(self) -> None:
        first = validation_partition_unit(20260811)
        self.assertEqual(first, validation_partition_unit(20260811))
        self.assertGreaterEqual(first, 0.0)
        self.assertLess(first, 1.0)

    def test_seed_set_hash_is_order_independent_but_duplicate_strict(self) -> None:
        self.assertEqual(
            validation_seed_sha256((3, 1, 2)),
            validation_seed_sha256((1, 2, 3)),
        )
        with self.assertRaisesRegex(ValueError, "duplicates"):
            validation_seed_sha256((1, 1))

    def test_invalid_seeds_are_rejected(self) -> None:
        for value in (-1, True, 1.5, "1"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    assign_validation_partition(value)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
