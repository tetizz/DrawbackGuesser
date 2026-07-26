from __future__ import annotations

from types import SimpleNamespace
import unittest

from ml.evaluation.cli import (
    FULL_VALIDATION_PARTITION_NAME,
    _filter_validation_partition_rows,
    _validation_evaluation_context,
)
from ml.evaluation.splits import SplitManifest
from ml.evaluation.validation_partition import (
    VALIDATION_PARTITION_IDENTITY,
    ValidationPartition,
    assign_validation_partition,
    validation_seed_sha256,
)


def seeds_for(partition: ValidationPartition, count: int) -> tuple[int, ...]:
    result: list[int] = []
    seed = 0
    while len(result) < count:
        if assign_validation_partition(seed) is partition:
            result.append(seed)
        seed += 1
    return tuple(result)


class ValidationPartitionCliTest(unittest.TestCase):
    def setUp(self) -> None:
        self.selection = seeds_for(ValidationPartition.SELECTION, 2)
        self.calibration = seeds_for(ValidationPartition.CALIBRATION_FIT, 1)
        self.gate = seeds_for(ValidationPartition.VALIDATION_GATE, 1)
        validation = (
            *self.selection,
            *self.calibration,
            *self.gate,
        )
        self.manifest = SplitManifest(
            train=(10_000,),
            validation=validation,
            test=(20_000,),
        )
        self.assignments = tuple(
            (f"game-{seed}", "A", "B") for seed in validation
        )
        self.audited = SimpleNamespace(
            seeds=validation,
            observed_seeds=validation,
            game_assignments=self.assignments,
        )

    def test_context_uses_complete_partition_seed_set_and_assignments(self) -> None:
        context = _validation_evaluation_context(
            self.audited,
            self.manifest,
            ValidationPartition.SELECTION.value,
        )
        self.assertEqual(context.allowed_seeds, frozenset(self.selection))
        self.assertEqual(
            context.metadata,
            {
                "identity": VALIDATION_PARTITION_IDENTITY,
                "name": "selection",
                "seedSha256": validation_seed_sha256(self.selection),
            },
        )
        self.assertEqual(
            set(context.game_assignments),
            {f"game-{seed}" for seed in self.selection},
        )

    def test_filter_never_splits_a_complete_game_across_partitions(self) -> None:
        context = _validation_evaluation_context(
            self.audited,
            self.manifest,
            "selection",
        )
        rows = [
            {"gameId": f"game-{self.selection[0]}", "seed": self.selection[0]},
            {"gameId": f"game-{self.calibration[0]}", "seed": self.calibration[0]},
            {"gameId": f"game-{self.selection[1]}", "seed": self.selection[1]},
        ]
        self.assertEqual(
            list(_filter_validation_partition_rows(rows, context)),
            [rows[0], rows[2]],
        )

        inconsistent = [
            {"gameId": f"game-{self.selection[0]}", "seed": self.selection[0]},
            {"gameId": f"game-{self.selection[0]}", "seed": self.calibration[0]},
        ]
        with self.assertRaisesRegex(ValueError, "split across"):
            list(_filter_validation_partition_rows(inconsistent, context))

    def test_empty_and_zero_ply_only_partitions_fail_closed(self) -> None:
        selection_only = SimpleNamespace(
            seeds=self.selection,
            observed_seeds=self.selection,
            game_assignments=tuple(
                (f"game-{seed}", "A", "B") for seed in self.selection
            ),
        )
        selection_manifest = SplitManifest(
            train=(10_000,),
            validation=self.selection,
            test=(20_000,),
        )
        with self.assertRaisesRegex(ValueError, "contains no games"):
            _validation_evaluation_context(
                selection_only,
                selection_manifest,
                "calibration-fit",
            )

        zero_ply_gate = SimpleNamespace(
            seeds=self.manifest.validation,
            observed_seeds=(*self.selection, *self.calibration),
            game_assignments=self.assignments,
        )
        with self.assertRaisesRegex(ValueError, "no move-bearing"):
            _validation_evaluation_context(
                zero_ply_gate,
                self.manifest,
                "validation-gate",
            )

    def test_full_validation_is_explicitly_non_selection(self) -> None:
        context = _validation_evaluation_context(
            self.audited,
            self.manifest,
            None,
        )
        self.assertEqual(
            context.metadata["name"],
            FULL_VALIDATION_PARTITION_NAME,
        )
        self.assertNotEqual(context.metadata["name"], "selection")

    def test_rejects_manifest_assignment_misalignment(self) -> None:
        broken = SimpleNamespace(
            seeds=self.manifest.validation,
            observed_seeds=self.manifest.validation,
            game_assignments=self.assignments[:-1],
        )
        with self.assertRaisesRegex(ValueError, "do not align"):
            _validation_evaluation_context(broken, self.manifest, "selection")


if __name__ == "__main__":
    unittest.main()
