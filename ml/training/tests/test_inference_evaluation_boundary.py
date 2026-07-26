from __future__ import annotations

import builtins
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import _bootstrap  # noqa: F401

from drawback_ml.features import FEATURE_DIMENSION, MOVE_VOCABULARY_SIZE
from drawback_ml.inference import (
    CheckpointPredictor,
    _normalized_probabilities,
    load_checkpoint_predictor,
)
from drawback_ml.model import ModelConfig, create_model
from drawback_ml.records import group_training_examples, parse_dataset_row
from drawback_ml.splits import Split, SplitConfig, assign_split
from drawback_ml.sequence import SanTokenizer
from drawback_ml.training import prepare_training_labels

from test_records import row


class TrainingEvaluationBoundaryTests(unittest.TestCase):
    def test_inference_normalizes_float32_like_softmax_values(self) -> None:
        class FakeTensor:
            def detach(self) -> "FakeTensor":
                return self

            def cpu(self) -> "FakeTensor":
                return self

            def tolist(self) -> list[float]:
                return [0.3333333, 0.3333333, 0.3333333]

        probabilities = _normalized_probabilities(("a", "b", "c"), FakeTensor())
        self.assertAlmostEqual(sum(probabilities.values()), 1.0, places=12)

    def test_vocabulary_is_fitted_only_from_training_seed_labels(self) -> None:
        config = SplitConfig()
        train_seed = next(
            seed for seed in range(10_000) if assign_split(seed, config) is Split.TRAIN
        )
        test_seed = next(
            seed for seed in range(10_000) if assign_split(seed, config) is Split.TEST
        )
        train_rows = [row("white", "train-a"), row("black", "train-b")]
        test_rows = [row("white", "test-only"), row("black", "test-only")]
        for value in train_rows:
            value["gameId"] = "train-game"
            value["seed"] = train_seed
            value["hiddenParameters"] = {"rank": 3}
            value["historySan"] = ["e4", "training-only"]
        for value in test_rows:
            value["gameId"] = "test-game"
            value["seed"] = test_seed
            value["hiddenParameters"] = {"square": "h8"}
            value["historySan"] = ["d4", "held-out-only"]
        examples = group_training_examples([*train_rows, *test_rows])
        prepared = prepare_training_labels(examples, config)
        self.assertEqual(prepared.drawback_vocabulary, ("train-a", "train-b"))
        self.assertEqual(prepared.parameter_vocabulary.tokens, ('{"rank":3}',))
        self.assertTrue(all(example.seed == train_seed for example in prepared.examples))
        tokenizer = SanTokenizer.fit(
            (example.features.history_san for example in prepared.examples),
            max_history=8,
        )
        self.assertIn("training-only", tokenizer.vocabulary)
        self.assertNotIn("held-out-only", tokenizer.vocabulary)

    def test_required_vocabulary_is_ordered_and_requires_both_color_coverage(
        self,
    ) -> None:
        config = SplitConfig()
        train_seed = next(
            seed
            for seed in range(10_000)
            if assign_split(seed, config) is Split.TRAIN
        )
        values = [
            row("white", "vegan"),
            row("black", "checkers"),
            row("white", "checkers"),
            row("black", "vegan"),
        ]
        values[2]["gameId"] = "game-2"
        values[3]["gameId"] = "game-2"
        for value in values:
            value["seed"] = train_seed
        examples = group_training_examples(values)

        prepared = prepare_training_labels(
            examples,
            config,
            ("checkers", "vegan"),
        )
        self.assertEqual(
            prepared.drawback_vocabulary,
            ("checkers", "vegan"),
        )
        with self.assertRaisesRegex(ValueError, "White missing truant"):
            prepare_training_labels(
                examples,
                config,
                ("vegan", "checkers", "truant"),
            )
        with self.assertRaisesRegex(ValueError, "outside the required"):
            prepare_training_labels(examples, config, ("vegan",))

    def test_checkpoint_loader_is_lazy_when_torch_is_unavailable(self) -> None:
        original_import = builtins.__import__

        def without_torch(name: str, *args: object, **kwargs: object) -> object:
            if name == "torch" or name.startswith("torch."):
                raise ImportError("simulated missing torch")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=without_torch):
            with self.assertRaisesRegex(RuntimeError, "PyTorch is required"):
                load_checkpoint_predictor(Path("missing.pt"))

    @unittest.skipUnless(
        importlib.util.find_spec("torch") is not None,
        "PyTorch is not installed",
    )
    def test_batch_inference_matches_scalar_outputs(self) -> None:
        import torch

        torch.manual_seed(71)
        model = create_model(
            ModelConfig(
                input_dimension=FEATURE_DIMENSION,
                drawback_classes=2,
                parameter_classes=1,
                legal_mask_dimension=MOVE_VOCABULARY_SIZE,
                hidden_dimension=8,
            )
        )
        predictor = CheckpointPredictor(
            torch_module=torch,
            model=model,
            drawback_vocabulary=("A", "B"),
            parameter_vocabulary=("none",),
            legal_mask_dimension=MOVE_VOCABULARY_SIZE,
            device="cpu",
            checkpoint_seed=71,
            checkpoint_epoch=1,
            training_run_id="a" * 64,
        )
        first = parse_dataset_row(row("white", "A")).features
        second_row = row("black", "B")
        second_row["move"] = "e7e5"
        second = parse_dataset_row(second_row).features
        scalar = (predictor.predict(first), predictor.predict(second))
        batched = predictor.predict_batch((first, second))
        self.assertEqual(len(batched), 2)
        for expected, actual in zip(scalar, batched, strict=True):
            for expected_map, actual_map in (
                (
                    expected.white_drawback_probabilities,
                    actual.white_drawback_probabilities,
                ),
                (
                    expected.black_drawback_probabilities,
                    actual.black_drawback_probabilities,
                ),
                (
                    expected.white_parameter_probabilities,
                    actual.white_parameter_probabilities,
                ),
                (
                    expected.black_parameter_probabilities,
                    actual.black_parameter_probabilities,
                ),
            ):
                self.assertEqual(expected_map.keys(), actual_map.keys())
                for key in expected_map:
                    self.assertAlmostEqual(
                        expected_map[key], actual_map[key], places=6
                    )
            for left, right in zip(
                expected.legal_mask_probabilities,
                actual.legal_mask_probabilities,
                strict=True,
            ):
                self.assertAlmostEqual(left, right, places=6)
            self.assertAlmostEqual(
                expected.trigger_probability,
                actual.trigger_probability,
                places=6,
            )


if __name__ == "__main__":
    unittest.main()
