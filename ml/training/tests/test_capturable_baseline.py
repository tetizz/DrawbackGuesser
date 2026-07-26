from __future__ import annotations

import unittest

import _bootstrap  # noqa: F401

from capturable_fixture import capturable_row
from drawback_ml.capturable_baseline import (
    CapturableTrainingConfig,
    _hard_mask_fusion,
    _validation_selection_metrics,
    create_capturable_model,
    evaluate_capturable,
    tensorize,
    train_capturable_baseline,
)
from drawback_ml.capturable_records import (
    CAPTURABLE_FEATURE_DIMENSION,
    parse_capturable_dataset_row,
)


class CapturableBaselineTests(unittest.TestCase):
    def test_model_emits_both_color_and_auxiliary_heads(self) -> None:
        import torch

        model = create_capturable_model(8)
        outputs = model(torch.zeros((3, CAPTURABLE_FEATURE_DIMENSION)))

        self.assertEqual(tuple(outputs["white_drawback"].shape), (3, 10))
        self.assertEqual(tuple(outputs["black_drawback"].shape), (3, 10))
        self.assertEqual(tuple(outputs["trigger"].shape), (3,))
        self.assertEqual(tuple(outputs["forced"].shape), (3,))
        self.assertEqual(
            tuple(outputs["triple_play_parameter"].shape),
            (3, 2),
        )

    def test_tensor_weights_each_player_game_equally(self) -> None:
        rows = (
            parse_capturable_dataset_row(
                capturable_row(game_id="long")
            ),
            parse_capturable_dataset_row(
                capturable_row(game_id="long")
            ),
            parse_capturable_dataset_row(
                capturable_row(game_id="short", color="black")
            ),
        )

        weights = tensorize(rows).player_game_weights.tolist()

        self.assertEqual(weights, [0.5, 0.5, 1.0])

    def test_hybrid_never_restores_a_hard_eliminated_rule(self) -> None:
        rows = (
            parse_capturable_dataset_row(
                capturable_row(eliminated_rule="checkers")
            ),
        )
        config = CapturableTrainingConfig(
            epochs=1,
            batch_size=1,
            hidden_dimension=8,
            torch_threads=1,
        )
        report = evaluate_capturable(
            create_capturable_model(8),
            rows,
            tensorize(rows),
            config,
            1.0,
            0.1,
        )

        diagnostics = report["hybrid"]["probability_diagnostics"]
        self.assertEqual(
            diagnostics["hard_elimination_violation_count"],
            0,
        )
        self.assertEqual(
            diagnostics["maximum_eliminated_probability"],
            0.0,
        )

    def test_neural_signal_can_rerank_only_surviving_soft_hypotheses(
        self,
    ) -> None:
        probabilities = _hard_mask_fusion(
            (-10.0, 10.0, 1000.0),
            (0.8, 0.2, 0.0),
            (False, False, True),
            alpha=1.0,
            prior_smoothing=0.1,
        )

        self.assertGreater(probabilities[1], probabilities[0])
        self.assertEqual(probabilities[2], 0.0)

    def test_lightweight_validation_selection_matches_full_metrics(
        self,
    ) -> None:
        rows = self._split("selection")
        residuals = [
            [float(index) for index in range(10)],
            [float(-index) for index in range(10)],
        ]
        selection = _validation_selection_metrics(
            rows,
            residuals,
            0.5,
            0.1,
        )
        config = CapturableTrainingConfig(
            epochs=1,
            batch_size=2,
            hidden_dimension=8,
            torch_threads=1,
        )
        report = evaluate_capturable(
            create_capturable_model(8),
            rows,
            tensorize(rows),
            config,
            0.5,
            0.1,
            (
                residuals,
                [0.5, 0.5],
                [0.5, 0.5],
                [[0.5, 0.5], [0.5, 0.5]],
            ),
        )["hybrid"]

        self.assertAlmostEqual(
            selection[0],
            report["game_normalized_top_1_accuracy"],
        )
        self.assertAlmostEqual(
            selection[1],
            report["game_normalized_top_3_accuracy"],
        )
        self.assertAlmostEqual(
            selection[2],
            report["game_normalized_negative_log_likelihood"],
        )

    def test_fresh_training_is_deterministic_and_holds_test_games_back(
        self,
    ) -> None:
        train = self._split("train")
        validation = self._split("validation")
        test = self._split("test")
        config = CapturableTrainingConfig(
            seed=20260726,
            epochs=2,
            batch_size=2,
            hidden_dimension=8,
            torch_threads=1,
        )

        first_model, first_report = train_capturable_baseline(
            train, validation, test, config
        )
        second_model, second_report = train_capturable_baseline(
            train, validation, test, config
        )

        self.assertEqual(first_report, second_report)
        self.assertTrue(first_report["freshStart"])
        self.assertEqual(
            first_report["selectionMetric"],
            "validation game_normalized_top_1_accuracy, then "
            "game_normalized_top_3_accuracy, then "
            "game_normalized_negative_log_likelihood",
        )
        for key, first_value in first_model.state_dict().items():
            self.assertTrue(
                first_value.equal(second_model.state_dict()[key])
            )

    def test_rejects_invalid_training_configuration(self) -> None:
        for arguments in (
            {"epochs": 0},
            {"batch_size": 0},
            {"learning_rate": 0.0},
            {"fusion_alpha_grid": (0.25, 1.0)},
            {"fusion_alpha_grid": (0.0, 1.0, 0.5)},
            {"fusion_alpha_grid": (0.0,)},
            {"prior_smoothing_grid": (0.1,)},
            {"prior_smoothing_grid": (0.0, 0.2, 0.1)},
            {"training_prior_smoothing": 1.0},
            {"torch_threads": 0},
        ):
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError):
                    CapturableTrainingConfig(**arguments)

    @staticmethod
    def _split(prefix: str):
        return (
            parse_capturable_dataset_row(
                capturable_row(
                    game_id=f"{prefix}-white",
                    drawback="vegan",
                )
            ),
            parse_capturable_dataset_row(
                capturable_row(
                    game_id=f"{prefix}-black",
                    color="black",
                    drawback="checkers",
                )
            ),
        )


if __name__ == "__main__":
    unittest.main()
