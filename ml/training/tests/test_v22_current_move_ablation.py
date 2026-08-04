from __future__ import annotations

from contextlib import redirect_stderr
from dataclasses import asdict
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import _bootstrap  # noqa: F401

from drawback_ml.checkpoint import (
    FUSION_GRID_DRAWBACK_OBJECTIVE,
    checkpoint_path,
    parse_training_run_drawback_objective,
)
from drawback_ml.cli import build_parser
from drawback_ml.inference import load_checkpoint_predictor
from drawback_ml.model import ModelConfig
from drawback_ml.records import TrainingExample, group_training_examples
from drawback_ml.sequence import (
    MASKED_CURRENT_MOVE_TOKEN,
    ObservationTokenizerV2,
    encode_public_sequence,
    public_sequence_observation,
)
from drawback_ml.splits import SplitConfig
from drawback_ml.streaming_training import (
    _claim_run,
    _scan,
    train_streaming_baseline,
)
from drawback_ml.symbolic import (
    SYMBOLIC_FEATURE_VERSION,
    SYMBOLIC_RULE_IDS,
    fusion_aware_loss_metadata,
)
from drawback_ml.training import (
    TrainingConfig,
    _in_memory_v22_run_id,
    train_baseline,
)
from test_records import row

try:
    import torch
except ImportError:
    torch = None


def _examples() -> list[TrainingExample]:
    probability = 1.0 / len(SYMBOLIC_RULE_IDS)
    probabilities = [probability] * len(SYMBOLIC_RULE_IDS)
    eliminated = [False] * len(SYMBOLIC_RULE_IDS)
    rows: list[dict[str, object]] = []
    assignments = (
        ("a1-control", "vegan", "checkers"),
        ("a1-treatment", "checkers", "vegan"),
    )
    for offset, (game_id, white_drawback, black_drawback) in enumerate(
        assignments
    ):
        for color, drawback, move, history in (
            ("white", white_drawback, "e2e4", []),
            ("black", black_drawback, "e7e5", ["e4"]),
        ):
            value = row(color, drawback)
            value.update(
                {
                    "gameId": game_id,
                    "seed": 800 + offset,
                    "move": move,
                    "historySan": history,
                    "symbolicFeatureVersion": SYMBOLIC_FEATURE_VERSION,
                    "symbolicWhiteRuleProbabilities": probabilities,
                    "symbolicBlackRuleProbabilities": probabilities,
                    "symbolicWhiteEliminated": eliminated,
                    "symbolicBlackEliminated": eliminated,
                }
            )
            rows.append(value)
    return group_training_examples(rows)


def _full_vocabulary_examples() -> list[TrainingExample]:
    probability = 1.0 / len(SYMBOLIC_RULE_IDS)
    probabilities = [probability] * len(SYMBOLIC_RULE_IDS)
    eliminated = [False] * len(SYMBOLIC_RULE_IDS)
    rows: list[dict[str, object]] = []
    for index, drawback_id in enumerate(SYMBOLIC_RULE_IDS):
        for color, move, history in (
            ("white", "e2e4", []),
            ("black", "e7e5", ["e4"]),
        ):
            value = row(color, drawback_id)
            value.update(
                {
                    "gameId": f"a1-full-{index:03d}",
                    "seed": 20_000 + index,
                    "move": move,
                    "historySan": history,
                    "symbolicFeatureVersion": SYMBOLIC_FEATURE_VERSION,
                    "symbolicWhiteRuleProbabilities": probabilities,
                    "symbolicBlackRuleProbabilities": probabilities,
                    "symbolicWhiteEliminated": eliminated,
                    "symbolicBlackEliminated": eliminated,
                }
            )
            rows.append(value)
    return group_training_examples(rows)


def _zero_fusion_loss(
    _torch: object,
    logits: object,
    *_arguments: object,
) -> object:
    return logits.sum() * 0.0  # type: ignore[union-attr]


class V22ConfigurationTests(unittest.TestCase):
    def test_v22_requires_one_explicit_mode_and_legacy_variants_reject_it(
        self,
    ) -> None:
        for config_type, arguments in (
            (TrainingConfig, {"seed": 1}),
            (
                ModelConfig,
                {
                    "input_dimension": 2,
                    "drawback_classes": 2,
                    "legal_mask_dimension": 2,
                    "san_vocabulary_size": 4,
                    "symbolic_dimension": 4,
                },
            ),
        ):
            with self.subTest(config=config_type.__name__):
                with self.assertRaisesRegex(ValueError, "explicit"):
                    config_type(  # type: ignore[call-arg]
                        **arguments,
                        model_variant="v22-hybrid",
                    )
                with self.assertRaisesRegex(ValueError, "exclusive"):
                    config_type(  # type: ignore[call-arg]
                        **arguments,
                        model_variant="v21-hybrid",
                        sequence_observation_mode="exact-current-v2",
                    )
                for mode in ("masked-current-v2", "exact-current-v2"):
                    configured = config_type(  # type: ignore[call-arg]
                        **arguments,
                        model_variant="v22-hybrid",
                        sequence_observation_mode=mode,
                    )
                    self.assertEqual(configured.sequence_observation_mode, mode)
        with self.assertRaisesRegex(ValueError, "reserved tokens"):
            ModelConfig(
                input_dimension=2,
                drawback_classes=2,
                legal_mask_dimension=2,
                model_variant="v22-hybrid",
                sequence_observation_mode="exact-current-v2",
                san_vocabulary_size=3,
                symbolic_dimension=4,
            )

    def test_cli_exposes_only_the_two_preregistered_modes(self) -> None:
        parser = build_parser()
        arguments = parser.parse_args(
            [
                "train",
                "fixture.ndjson",
                "output",
                "--seed",
                "7",
                "--model-variant",
                "v22-hybrid",
                "--sequence-observation-mode",
                "exact-current-v2",
            ]
        )
        self.assertEqual(arguments.model_variant, "v22-hybrid")
        self.assertEqual(
            arguments.sequence_observation_mode,
            "exact-current-v2",
        )
        with redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args(
                    [
                        "train",
                        "fixture.ndjson",
                        "output",
                        "--seed",
                        "7",
                        "--model-variant",
                        "v22-hybrid",
                        "--sequence-observation-mode",
                        "unregistered",
                    ]
                )

    def test_control_and_treatment_fit_identical_training_only_vocabulary(
        self,
    ) -> None:
        examples = _examples()
        fitted = ObservationTokenizerV2.fit(
            (public_sequence_observation(item.features) for item in examples),
            max_sequence=4,
        )
        _parameters, streamed, _rows, _games = _scan(
            lambda: iter(examples),
            ("checkers", "vegan"),
            model_variant="v22-hybrid",
            max_history=3,
        )
        self.assertEqual(streamed, fitted)
        self.assertEqual(fitted.max_sequence, 4)

        feature = examples[1].features
        masked, masked_length = encode_public_sequence(
            fitted,
            feature,
            "masked-current-v2",
        )
        exact, exact_length = encode_public_sequence(
            fitted,
            feature,
            "exact-current-v2",
        )
        self.assertEqual(masked_length, exact_length)
        self.assertEqual(masked[: masked_length - 1], exact[: exact_length - 1])
        self.assertEqual(
            fitted.vocabulary[masked[masked_length - 1]],
            MASKED_CURRENT_MOVE_TOKEN,
        )
        self.assertEqual(
            fitted.vocabulary[exact[exact_length - 1]],
            "<move:e7e5>",
        )

    def test_secret_label_mutation_cannot_change_v22_observation(self) -> None:
        original = row("white", "vegan")
        mutated = dict(original)
        mutated["trueDrawback"] = "checkers"
        mutated["hiddenParameters"] = {"different": 42}
        mutated["drawbackInternalState"] = {"private": "changed"}
        original_feature = group_training_examples(
            [original, row("black", "checkers")]
        )[0].features
        mutated_black = row("black", "vegan")
        mutated_feature = group_training_examples(
            [mutated, mutated_black]
        )[0].features
        self.assertEqual(original_feature, mutated_feature)
        self.assertEqual(
            public_sequence_observation(original_feature),
            public_sequence_observation(mutated_feature),
        )

    def test_run_claim_binds_mode_without_changing_legacy_claim_shape(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = {"device": "cpu"}
            sampling = {"policy": "fixture"}
            masked = TrainingConfig(
                seed=3,
                model_variant="v22-hybrid",
                sequence_observation_mode="masked-current-v2",
            )
            exact = TrainingConfig(
                seed=3,
                model_variant="v22-hybrid",
                sequence_observation_mode="exact-current-v2",
            )
            masked_id = _claim_run(root / "masked", masked, runtime, sampling)
            exact_id = _claim_run(root / "exact", exact, runtime, sampling)
            self.assertNotEqual(masked_id, exact_id)
            for directory, mode in (
                ("masked", "masked-current-v2"),
                ("exact", "exact-current-v2"),
            ):
                claim = json.loads(
                    (root / directory / "run.claim.json").read_text(
                        encoding="utf-8"
                    )
                )
                self.assertEqual(
                    claim["config"]["sequence_observation_mode"],
                    mode,
                )

            _claim_run(
                root / "v21",
                TrainingConfig(seed=3, model_variant="v21-hybrid"),
                runtime,
                sampling,
            )
            legacy = json.loads(
                (root / "v21" / "run.claim.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertNotIn(
                "sequence_observation_mode",
                legacy["config"],
            )

    def test_in_memory_run_identity_is_order_independent_and_binds_mode(
        self,
    ) -> None:
        examples = _examples()
        masked = TrainingConfig(
            seed=9,
            model_variant="v22-hybrid",
            sequence_observation_mode="masked-current-v2",
        )
        exact = TrainingConfig(
            seed=9,
            model_variant="v22-hybrid",
            sequence_observation_mode="exact-current-v2",
        )
        self.assertEqual(
            _in_memory_v22_run_id(exact, examples),
            _in_memory_v22_run_id(exact, tuple(reversed(examples))),
        )
        self.assertNotEqual(
            _in_memory_v22_run_id(masked, examples),
            _in_memory_v22_run_id(exact, examples),
        )

    def test_v22_run_claim_requires_the_frozen_fusion_objective(self) -> None:
        config = asdict(
            TrainingConfig(
                seed=11,
                model_variant="v22-hybrid",
                sequence_observation_mode="exact-current-v2",
            )
        )
        self.assertEqual(
            parse_training_run_drawback_objective(config),
            FUSION_GRID_DRAWBACK_OBJECTIVE,
        )
        config.pop("fusion_aware_loss_method")
        with self.assertRaisesRegex(ValueError, "partial"):
            parse_training_run_drawback_objective(config)


@unittest.skipUnless(torch is not None, "PyTorch is not installed")
class V22TrainerIntegrationTests(unittest.TestCase):
    def _config(self, mode: str) -> TrainingConfig:
        return TrainingConfig(
            seed=19,
            epochs=1,
            batch_size=4,
            hidden_dimension=2,
            model_variant="v22-hybrid",
            sequence_observation_mode=mode,  # type: ignore[arg-type]
            max_history=3,
            san_embedding_dimension=2,
            sequence_hidden_dimension=2,
            symbolic_hidden_dimension=2,
            split=SplitConfig(1.0, 0.0, 0.0),
            required_drawback_vocabulary=("checkers", "vegan"),
            shuffle_buffer_size=4,
            player_game_examples_per_epoch=1,
            trigger_loss_weight=0.0,
            parameter_loss_weight=0.0,
            legal_mask_loss_weight=0.0,
        )

    def _assert_artifacts(
        self,
        output: Path,
        config: TrainingConfig,
    ) -> None:
        assert torch is not None
        run = json.loads((output / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(
            run["sequence_observation_mode"],
            config.sequence_observation_mode,
        )
        self.assertEqual(len(run["run_id"]), 64)
        self.assertEqual(run["san_tokenizer"]["version"], 2)
        self.assertEqual(run["san_tokenizer"]["max_sequence"], 4)
        self.assertEqual(
            run["drawback_loss_objective"],
            fusion_aware_loss_metadata("v22-hybrid"),
        )
        checkpoint = torch.load(
            checkpoint_path(output, config.seed, 1),
            map_location="cpu",
            weights_only=True,
        )
        self.assertEqual(
            checkpoint["model_config"]["sequence_observation_mode"],
            config.sequence_observation_mode,
        )
        self.assertEqual(
            checkpoint["training_metadata"]["sequence_observation_mode"],
            config.sequence_observation_mode,
        )
        self.assertEqual(
            checkpoint["training_metadata"]["run_id"],
            run["run_id"],
        )
        self.assertEqual(
            checkpoint["training_metadata"]["san_tokenizer"],
            run["san_tokenizer"],
        )

    def test_in_memory_trainer_emits_exact_a1_contract(self) -> None:
        examples = _examples()
        config = self._config("exact-current-v2")
        with tempfile.TemporaryDirectory() as temporary, patch(
            "drawback_ml.training.fusion_aware_drawback_loss",
            side_effect=_zero_fusion_loss,
        ):
            output = Path(temporary) / "in-memory"
            train_baseline(examples, output, config)
            self._assert_artifacts(output, config)
            before = {
                path.relative_to(output): path.read_bytes()
                for path in output.rglob("*")
                if path.is_file()
            }
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                train_baseline(examples, output, config)
            after = {
                path.relative_to(output): path.read_bytes()
                for path in output.rglob("*")
                if path.is_file()
            }
            self.assertEqual(after, before)

    def test_streaming_trainer_emits_masked_a1_contract(self) -> None:
        examples = _examples()
        config = self._config("masked-current-v2")
        with tempfile.TemporaryDirectory() as temporary, patch(
            "drawback_ml.streaming_training.fusion_aware_drawback_loss",
            side_effect=_zero_fusion_loss,
        ):
            output = Path(temporary) / "streaming"
            train_streaming_baseline(lambda: iter(examples), output, config)
            self._assert_artifacts(output, config)

    def test_full_vocabulary_in_memory_checkpoint_loads_for_inference(
        self,
    ) -> None:
        examples = _full_vocabulary_examples()
        config = TrainingConfig(
            seed=29,
            epochs=1,
            batch_size=len(examples),
            hidden_dimension=2,
            model_variant="v22-hybrid",
            sequence_observation_mode="exact-current-v2",
            max_history=3,
            san_embedding_dimension=2,
            sequence_hidden_dimension=2,
            symbolic_hidden_dimension=2,
            split=SplitConfig(1.0, 0.0, 0.0),
            required_drawback_vocabulary=SYMBOLIC_RULE_IDS,
            trigger_loss_weight=0.0,
            parameter_loss_weight=0.0,
            legal_mask_loss_weight=0.0,
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "full"
            train_baseline(examples, output, config)
            predictor = load_checkpoint_predictor(
                checkpoint_path(output, config.seed, 1)
            )
            prediction = predictor.predict(examples[0].features)
            self.assertEqual(
                tuple(prediction.white_drawback_probabilities),
                SYMBOLIC_RULE_IDS,
            )
            self.assertEqual(
                predictor.model_config.sequence_observation_mode,
                "exact-current-v2",
            )


if __name__ == "__main__":
    unittest.main()
