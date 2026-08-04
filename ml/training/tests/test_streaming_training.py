from __future__ import annotations

import hashlib
import json
from pathlib import Path
import importlib.util
import tempfile
import unittest
from unittest.mock import patch
import builtins

import _bootstrap  # noqa: F401

from drawback_ml.checkpoint import checkpoint_path, verify_checkpoint_index
from drawback_ml.records import group_training_examples
from drawback_ml.streaming import build_player_game_sampling_plan
from drawback_ml.streaming_training import (
    _claim_run,
    staged_output_directory,
    train_streaming_baseline,
    validate_training_device,
)
from drawback_ml.symbolic import (
    SYMBOLIC_FEATURE_VERSION,
    SYMBOLIC_RULE_IDS,
    fusion_aware_loss_metadata,
)
from drawback_ml.training import (
    TrainingConfig,
    drawback_observation_masks,
    drawback_supervision_masks,
    normalized_drawback_head_weights,
)
from test_records import row


class _FakeCuda:
    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def get_device_capability(_index: int) -> tuple[int, int]:
        return (12, 0)

    @staticmethod
    def get_arch_list() -> list[str]:
        return ["sm_90"]


class _FakeTorch:
    cuda = _FakeCuda()


class StreamingTrainingTests(unittest.TestCase):
    def test_run_claim_binds_execution_source_revision(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "run"
            revision = "a" * 40
            config = TrainingConfig(
                seed=1,
                execution_source_revision=revision,
            )
            _claim_run(
                directory,
                config,
                {"device": "cpu"},
                {"policy": "fixture"},
            )
            claim = json.loads(
                (directory / "run.claim.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                claim["config"]["execution_source_revision"],
                revision,
            )

    def test_drawback_labels_require_an_observed_move_by_that_color(self) -> None:
        white, black = drawback_observation_masks(
            ("white", "black", "white", "black"),
            (0, 1, 2, 3),
        )
        self.assertEqual(white, (True, True, True, True))
        self.assertEqual(black, (False, True, True, True))

    def test_observation_masks_follow_the_current_mover(self) -> None:
        white, black = drawback_observation_masks(
            ("black", "white"),
            (0, 0),
        )
        self.assertEqual(white, (False, True))
        self.assertEqual(black, (True, False))

    def test_explicit_supervision_policies_have_exact_masks(self) -> None:
        histories = ((), ("white",), ("white", "black"))
        available = drawback_supervision_masks(
            "available-history-v1",
            ("white", "black", "white"),
            histories,
        )
        mover_only = drawback_supervision_masks(
            "moving-color-only-v1",
            ("white", "black", "white"),
            histories,
        )

        self.assertEqual(available, ((True, True, True), (False, True, True)))
        self.assertEqual(
            mover_only,
            ((True, False, True), (False, True, False)),
        )

    def test_available_history_handles_black_to_move_setup(self) -> None:
        white, black = drawback_supervision_masks(
            "available-history-v1",
            ("black", "white", "black"),
            ((), ("black",), ("black", "white")),
        )

        self.assertEqual(white, (False, True, True))
        self.assertEqual(black, (True, True, True))

    def test_supervision_policy_rejects_invalid_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported"):
            drawback_supervision_masks(
                "other",  # type: ignore[arg-type]
                ("white",),
                ((),),
            )
        with self.assertRaisesRegex(ValueError, "align"):
            drawback_supervision_masks(
                "available-history-v1",
                ("white",),
                (),
            )
        with self.assertRaisesRegex(ValueError, "player color"):
            drawback_supervision_masks(
                "available-history-v1",
                ("green",),
                ((),),
            )
        with self.assertRaisesRegex(ValueError, "history colors"):
            drawback_supervision_masks(
                "available-history-v1",
                ("white",),
                (("green",),),
            )

    def test_normalized_head_weights_have_unit_weight_per_row(self) -> None:
        white, black = normalized_drawback_head_weights(
            (True, True, False),
            (False, True, True),
        )

        self.assertEqual(white, (1.0, 0.5, 0.0))
        self.assertEqual(black, (0.0, 0.5, 1.0))
        self.assertEqual(
            tuple(
                left + right
                for left, right in zip(white, black, strict=True)
            ),
            (1.0, 1.0, 1.0),
        )
        with self.assertRaisesRegex(ValueError, "at least one"):
            normalized_drawback_head_weights((False,), (False,))
        with self.assertRaisesRegex(ValueError, "booleans"):
            normalized_drawback_head_weights(
                (1,),  # type: ignore[arg-type]
                (False,),
            )

    def test_game_balancing_configuration_must_be_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "game_examples_per_epoch"):
            TrainingConfig(seed=1, game_examples_per_epoch=0)
        with self.assertRaisesRegex(
            ValueError, "player_game_examples_per_epoch"
        ):
            TrainingConfig(seed=1, player_game_examples_per_epoch=0)
        with self.assertRaisesRegex(ValueError, "must be even"):
            TrainingConfig(seed=1, game_examples_per_epoch=15)

    def test_auxiliary_loss_weights_must_be_non_negative(self) -> None:
        with self.assertRaisesRegex(ValueError, "auxiliary loss weights"):
            TrainingConfig(seed=1, legal_mask_loss_weight=-0.01)
        with self.assertRaisesRegex(ValueError, "auxiliary loss weights"):
            TrainingConfig(seed=1, trigger_loss_weight=float("nan"))

    def test_seed_and_learning_rate_reject_invalid_numeric_values(self) -> None:
        for seed in (True, -1):
            with self.subTest(seed=seed):
                with self.assertRaisesRegex(ValueError, "non-negative integer"):
                    TrainingConfig(seed=seed)  # type: ignore[arg-type]
        for learning_rate in (True, float("nan"), float("inf"), 0.0):
            with self.subTest(learning_rate=learning_rate):
                with self.assertRaisesRegex(ValueError, "finite and positive"):
                    TrainingConfig(
                        seed=1,
                        learning_rate=learning_rate,  # type: ignore[arg-type]
                    )

    def test_staged_output_is_atomic_and_failure_is_retryable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "run"
            with self.assertRaisesRegex(ValueError, "validation failed"):
                with staged_output_directory(output) as staging:
                    (staging / "checkpoint.pt").write_bytes(b"partial")
                    raise ValueError("validation failed")
            self.assertFalse(output.exists())
            self.assertFalse(any(root.glob(".run.staging-*")))

            with staged_output_directory(output) as staging:
                (staging / "checkpoint.pt").write_bytes(b"complete")
            self.assertEqual((output / "checkpoint.pt").read_bytes(), b"complete")

    def test_staged_output_does_not_replace_a_racing_destination(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "run"
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                with staged_output_directory(output) as staging:
                    (staging / "checkpoint.pt").write_bytes(b"candidate")
                    output.mkdir()
                    (output / "owner.txt").write_text("other run", encoding="utf-8")
            self.assertEqual(
                (output / "owner.txt").read_text(encoding="utf-8"),
                "other run",
            )
            self.assertFalse((output / "checkpoint.pt").exists())
            self.assertFalse(any(root.glob(".run.staging-*")))

    def test_device_validation_fails_before_execution(self) -> None:
        self.assertEqual(validate_training_device(_FakeTorch(), "cpu"), "cpu")
        with self.assertRaisesRegex(RuntimeError, "sm_120"):
            validate_training_device(_FakeTorch(), "cuda")
        with self.assertRaisesRegex(ValueError, "cpu or cuda"):
            validate_training_device(_FakeTorch(), "other")

    def test_cuda_requires_a_deterministic_cublas_workspace(self) -> None:
        class CompatibleCuda(_FakeCuda):
            @staticmethod
            def get_arch_list() -> list[str]:
                return ["sm_120"]

        class CompatibleTorch:
            cuda = CompatibleCuda()

        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "CUBLAS_WORKSPACE_CONFIG"):
                validate_training_device(CompatibleTorch(), "cuda")
        with patch.dict(
            "os.environ", {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}, clear=True
        ):
            self.assertEqual(
                validate_training_device(CompatibleTorch(), "cuda"), "cuda"
            )

    def test_refuses_to_reuse_a_claimed_output_directory(self) -> None:
        calls = 0

        def factory():
            nonlocal calls
            calls += 1
            return iter(())

        config = TrainingConfig(
            seed=7,
            required_drawback_vocabulary=("vegan", "checkers"),
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            output.mkdir()
            original_import = builtins.__import__

            def no_torch(name: str, *args: object, **kwargs: object) -> object:
                if name == "torch" or name.startswith("torch."):
                    raise ImportError("torch must not be imported")
                return original_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=no_torch):
                with self.assertRaisesRegex(FileExistsError, "already exists"):
                    train_streaming_baseline(factory, output, config)
            self.assertEqual(calls, 0)

    def test_training_failure_immediately_removes_sampling_index(self) -> None:
        values = group_training_examples(
            [
                row("white", "vegan"),
                row("black", "checkers"),
                {**row("white", "checkers"), "gameId": "game-2"},
                {**row("black", "vegan"), "gameId": "game-2"},
            ]
        )
        paths: list[Path] = []

        def capture_plan(*args: object, **kwargs: object):
            plan = build_player_game_sampling_plan(*args, **kwargs)
            paths.append(plan.temporary_directory)
            return plan

        config = TrainingConfig(
            seed=7,
            required_drawback_vocabulary=("vegan", "checkers"),
        )
        original_import = builtins.__import__

        def no_torch(name: str, *args: object, **kwargs: object) -> object:
            if name == "torch" or name.startswith("torch."):
                raise ImportError("torch unavailable")
            return original_import(name, *args, **kwargs)

        retained_error: RuntimeError | None = None
        with tempfile.TemporaryDirectory() as temporary, patch(
            "drawback_ml.streaming_training.build_player_game_sampling_plan",
            side_effect=capture_plan,
        ), patch("builtins.__import__", side_effect=no_torch):
            try:
                train_streaming_baseline(
                    lambda: iter(values),
                    Path(temporary) / "run",
                    config,
                )
            except RuntimeError as error:
                retained_error = error
            self.assertIsNotNone(retained_error)
            self.assertEqual(len(paths), 1)
            self.assertFalse(paths[0].exists())

    @unittest.skipUnless(
        importlib.util.find_spec("torch") is not None,
        "PyTorch is not installed",
    )
    def test_real_training_publishes_only_the_completed_run(self) -> None:
        first_white = row("white", "vegan")
        first_black = row("black", "checkers")
        second_white = {**row("white", "checkers"), "gameId": "game-2"}
        second_black = {**row("black", "vegan"), "gameId": "game-2"}
        values = group_training_examples(
            [first_white, first_black, second_white, second_black]
        )
        calls = 0

        def factory():
            nonlocal calls
            calls += 1
            return iter(values)

        config = TrainingConfig(
            seed=7,
            epochs=1,
            batch_size=2,
            hidden_dimension=8,
            required_drawback_vocabulary=("vegan", "checkers"),
            shuffle_buffer_size=2,
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            validated: list[Path] = []

            def validate_completed_staging(staging_directory: Path) -> None:
                self.assertTrue(
                    (staging_directory / "checkpoint-index.claim.json").is_file()
                )
                validated.append(staging_directory)

            train_streaming_baseline(
                factory,
                output,
                config,
                final_validation=validate_completed_staging,
            )
            self.assertEqual(len(validated), 1)
            self.assertGreaterEqual(calls, 2)
            self.assertTrue((output / "run.json").is_file())
            run_metadata = json.loads(
                (output / "run.json").read_text(encoding="utf-8")
            )
            sampling = run_metadata["sampling"]
            self.assertEqual(
                sampling["examples_setting"],
                "legacy-game-examples-divided-by-two",
            )
            self.assertEqual(sampling["examples_per_player_game"], 8)
            self.assertEqual(sampling["effective_examples_per_epoch"], 32)
            self.assertEqual(
                sampling["configured_game_examples_per_epoch"], 16
            )
            self.assertIsNone(
                sampling["configured_player_game_examples_per_epoch"]
            )
            self.assertTrue(
                (output / "checkpoint-index.claim.json").is_file()
            )
            index = output / "checkpoint-index.claim.json"
            verify_checkpoint_index(
                index,
                hashlib.sha256(index.read_bytes()).hexdigest(),
            )
            self.assertFalse(
                any(Path(temporary).glob(".run.staging-*"))
            )
            rejected = Path(temporary) / "rejected"

            def reject_changed_corpus(_staging_directory: Path) -> None:
                raise ValueError("corpus changed during streaming training")

            with self.assertRaisesRegex(ValueError, "corpus changed"):
                train_streaming_baseline(
                    factory,
                    rejected,
                    config,
                    final_validation=reject_changed_corpus,
                )
            self.assertFalse(rejected.exists())
            self.assertFalse(
                any(Path(temporary).glob(".rejected.staging-*"))
            )

    @unittest.skipUnless(
        importlib.util.find_spec("torch") is not None,
        "PyTorch is not installed",
    )
    def test_full_vocabulary_v21_streaming_epoch_records_fusion_objective(
        self,
    ) -> None:
        import torch

        probability = 1.0 / len(SYMBOLIC_RULE_IDS)
        probabilities = [probability] * len(SYMBOLIC_RULE_IDS)
        eliminated = [False] * len(SYMBOLIC_RULE_IDS)
        rows: list[dict[str, object]] = []
        for index, drawback_id in enumerate(SYMBOLIC_RULE_IDS):
            game_id = f"full-vocabulary-{index:03d}"
            for color, history in (("white", []), ("black", ["e4"])):
                value = row(color, drawback_id)
                value.update(
                    {
                        "gameId": game_id,
                        "seed": 10_000 + index,
                        "historySan": history,
                        "symbolicFeatureVersion": SYMBOLIC_FEATURE_VERSION,
                        "symbolicWhiteRuleProbabilities": probabilities,
                        "symbolicBlackRuleProbabilities": probabilities,
                        "symbolicWhiteEliminated": eliminated,
                        "symbolicBlackEliminated": eliminated,
                    }
                )
                rows.append(value)
        examples = group_training_examples(rows)
        config = TrainingConfig(
            seed=23,
            epochs=1,
            batch_size=len(examples),
            hidden_dimension=4,
            model_variant="v21-hybrid",
            max_history=2,
            san_embedding_dimension=2,
            sequence_hidden_dimension=2,
            symbolic_hidden_dimension=2,
            required_drawback_vocabulary=SYMBOLIC_RULE_IDS,
            shuffle_buffer_size=16,
            player_game_examples_per_epoch=1,
            trigger_loss_weight=0.0,
            parameter_loss_weight=0.0,
            legal_mask_loss_weight=0.0,
        )
        expected_objective = fusion_aware_loss_metadata("v21-hybrid")
        self.assertIsNotNone(expected_objective)

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "v21-run"
            completed = train_streaming_baseline(
                lambda: iter(examples),
                output,
                config,
            )

            self.assertEqual(completed, output)
            run_metadata = json.loads(
                (output / "run.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                run_metadata["drawback_vocabulary"],
                list(SYMBOLIC_RULE_IDS),
            )
            self.assertEqual(
                run_metadata["drawback_loss_objective"],
                expected_objective,
            )
            checkpoint = torch.load(
                checkpoint_path(output, config.seed, 1),
                map_location="cpu",
                weights_only=True,
            )
            self.assertEqual(
                checkpoint["training_metadata"][
                    "drawback_loss_objective"
                ],
                expected_objective,
            )


if __name__ == "__main__":
    unittest.main()
