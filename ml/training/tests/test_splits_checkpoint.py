from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import _bootstrap  # noqa: F401

from drawback_ml.checkpoint import (
    checkpoint_metadata,
    checkpoint_path,
    write_run_metadata,
)
from drawback_ml.splits import Split, SplitConfig, assign_split


class SplitTests(unittest.TestCase):
    def test_assignment_is_order_independent_and_seed_isolated(self) -> None:
        seeds = list(range(500))
        forward = {seed: assign_split(seed) for seed in seeds}
        reverse = {seed: assign_split(seed) for seed in reversed(seeds)}
        self.assertEqual(forward, reverse)
        self.assertEqual(set(forward.values()), set(Split))
        for seed in seeds:
            self.assertIs(assign_split(seed), assign_split(seed))

    def test_salt_is_reproducible_and_config_is_validated(self) -> None:
        config = SplitConfig(salt="evaluation-v2")
        self.assertEqual(assign_split(123, config), assign_split(123, config))
        with self.assertRaisesRegex(ValueError, "sum to one"):
            SplitConfig(train_fraction=0.5, validation_fraction=0.5, test_fraction=0.5)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            assign_split(-1)

    def test_split_config_rejects_non_finite_and_non_numeric_values(self) -> None:
        for train_fraction in (float("nan"), float("inf"), True, "0.8"):
            with self.subTest(train_fraction=train_fraction):
                with self.assertRaisesRegex(ValueError, "finite numbers"):
                    SplitConfig(
                        train_fraction=train_fraction,  # type: ignore[arg-type]
                        validation_fraction=0.1,
                        test_fraction=0.1,
                    )
        with self.assertRaisesRegex(ValueError, "non-empty string"):
            SplitConfig(salt=1)  # type: ignore[arg-type]


class CheckpointTests(unittest.TestCase):
    def test_checkpoint_name_and_metadata_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.assertEqual(
                checkpoint_path(directory, 7, 3).name,
                "baseline-seed-0000000007-epoch-0003.pt",
            )
            path = write_run_metadata(directory, {"seed": 7, "format_version": 1})
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")),
                {"format_version": 1, "seed": 7},
            )
            self.assertFalse((directory / "run.json.tmp").exists())

    def test_parameter_vocabulary_is_part_of_versioned_checkpoint_metadata(self) -> None:
        metadata = checkpoint_metadata(
            seed=7,
            epoch=3,
            drawback_vocabulary=["checkers", "gambler"],
            parameter_vocabulary=['{"seed":9}'],
            model_config={
                "input_dimension": 792,
                "drawback_classes": 2,
                "parameter_classes": 1,
                "legal_mask_dimension": 20480,
                "hidden_dimension": 128,
            },
            training_metadata={
                "feature_schema_version": 1,
                "loss_weights": {"white_drawback": 1.0},
                "split": {"salt": "drawbacktrainer-v1"},
            },
        )
        self.assertEqual(metadata["format_version"], 3)
        self.assertEqual(metadata["parameter_vocabulary"], ['{"seed":9}'])
        self.assertEqual(metadata["drawback_vocabulary"], ["checkers", "gambler"])
        self.assertEqual(metadata["model_config"]["hidden_dimension"], 128)
        self.assertEqual(
            metadata["training_metadata"]["split"]["salt"],
            "drawbacktrainer-v1",
        )


if __name__ == "__main__":
    unittest.main()
