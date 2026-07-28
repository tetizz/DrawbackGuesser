from __future__ import annotations

import unittest

import _bootstrap  # noqa: F401

from capturable_fixture import capturable_row
from drawback_ml.capturable_baseline import (
    CapturableTrainingConfig,
    _hard_mask_fusion,
    _validation_selection_metrics,
    _weighted_mean,
    create_capturable_model,
    evaluate_capturable,
    evaluate_capturable_posteriors,
    tensorize,
    train_capturable_baseline,
)
from drawback_ml.capturable_records import (
    CAPTURABLE_FEATURE_DIMENSION,
    CAPTURABLE_RULE_COUNT,
    parse_capturable_dataset_row,
)


class CapturableBaselineTests(unittest.TestCase):
    def test_model_emits_both_color_and_auxiliary_heads(self) -> None:
        import torch

        model = create_capturable_model(8)
        outputs = model(torch.zeros((3, CAPTURABLE_FEATURE_DIMENSION)))

        self.assertEqual(
            tuple(outputs["white_drawback"].shape),
            (3, CAPTURABLE_RULE_COUNT),
        )
        self.assertEqual(
            tuple(outputs["black_drawback"].shape),
            (3, CAPTURABLE_RULE_COUNT),
        )
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

    def test_trigger_weighting_preserves_each_player_game_total(self) -> None:
        rows = (
            parse_capturable_dataset_row(
                capturable_row(game_id="mixed", triggered=True)
            ),
            parse_capturable_dataset_row(
                capturable_row(game_id="mixed", triggered=False)
            ),
            parse_capturable_dataset_row(
                capturable_row(game_id="quiet", triggered=False)
            ),
        )

        weights = tensorize(
            rows,
            trigger_row_multiplier=3.0,
        ).player_game_weights.tolist()

        self.assertEqual(weights, [0.75, 0.25, 1.0])
        self.assertEqual(sum(weights[:2]), weights[2])

    def test_source_weighting_scales_each_player_game_total(self) -> None:
        rows = (
            parse_capturable_dataset_row(
                capturable_row(game_id="diagnostic")
            ),
            parse_capturable_dataset_row(
                capturable_row(game_id="diagnostic")
            ),
            parse_capturable_dataset_row(
                capturable_row(game_id="baseline", color="black")
            ),
        )

        weights = tensorize(
            rows,
            game_source_weights={
                "diagnostic": 0.1,
                "baseline": 1.0,
            },
        ).player_game_weights.tolist()

        diagnostic_total = sum(weights[:2])
        baseline_total = weights[2]
        self.assertAlmostEqual(weights[0], weights[1])
        self.assertAlmostEqual(diagnostic_total, baseline_total * 0.1)
        self.assertAlmostEqual(
            (diagnostic_total + baseline_total) / 2.0,
            1.0,
        )

    def test_source_weighting_rejects_incomplete_or_invalid_maps(self) -> None:
        rows = (
            parse_capturable_dataset_row(
                capturable_row(game_id="baseline")
            ),
        )

        for weights in (
            {},
            {"baseline": 0.0},
            {"baseline": float("nan")},
            {"baseline": 10**10000},
            {"baseline": 1.0, "unknown": 1.0},
        ):
            with self.subTest(weights=weights):
                with self.assertRaises(ValueError):
                    tensorize(rows, game_source_weights=weights)

    def test_source_weighting_survives_homogeneous_minibatches(self) -> None:
        import torch

        rows = (
            parse_capturable_dataset_row(
                capturable_row(game_id="diagnostic")
            ),
            parse_capturable_dataset_row(
                capturable_row(game_id="diagnostic")
            ),
            parse_capturable_dataset_row(
                capturable_row(game_id="baseline", color="black")
            ),
            parse_capturable_dataset_row(
                capturable_row(game_id="baseline", color="black")
            ),
        )
        weighted = tensorize(
            rows,
            game_source_weights={
                "diagnostic": 0.1,
                "baseline": 1.0,
            },
        )
        diagnostic = torch.tensor([0, 1])
        baseline = torch.tensor([2, 3])
        one = torch.ones(2)
        diagnostic_loss = _weighted_mean(
            one,
            weighted.player_game_weights[diagnostic],
            weighted.player_game_normalization_weights[diagnostic],
        )
        baseline_loss = _weighted_mean(
            one,
            weighted.player_game_weights[baseline],
            weighted.player_game_normalization_weights[baseline],
        )
        singleton_diagnostic_loss = _weighted_mean(
            one[:1],
            weighted.player_game_weights[:1],
            weighted.player_game_normalization_weights[:1],
        )
        singleton_baseline_loss = _weighted_mean(
            one[:1],
            weighted.player_game_weights[2:3],
            weighted.player_game_normalization_weights[2:3],
        )

        self.assertAlmostEqual(
            float(diagnostic_loss),
            float(baseline_loss) * 0.1,
        )
        self.assertAlmostEqual(
            float(singleton_diagnostic_loss),
            float(singleton_baseline_loss) * 0.1,
        )

        unweighted = tensorize(rows)
        self.assertIs(
            unweighted.player_game_weights,
            unweighted.player_game_normalization_weights,
        )
        all_one = tensorize(
            rows,
            game_source_weights={
                "diagnostic": 1.0,
                "baseline": 1.0,
            },
        )
        self.assertTrue(
            torch.equal(
                all_one.player_game_weights,
                unweighted.player_game_weights,
            )
        )
        self.assertTrue(
            torch.equal(
                all_one.player_game_normalization_weights,
                unweighted.player_game_weights,
            )
        )

        rescaled = tensorize(
            rows,
            game_source_weights={
                "diagnostic": 1.0,
                "baseline": 10.0,
            },
        )
        values = torch.tensor([1.0, 2.0, 3.0, 4.0])
        weighted_loss = _weighted_mean(
            values,
            weighted.player_game_weights,
            weighted.player_game_normalization_weights,
        )
        rescaled_loss = _weighted_mean(
            values,
            rescaled.player_game_weights,
            rescaled.player_game_normalization_weights,
        )
        self.assertAlmostEqual(float(weighted_loss), float(rescaled_loss))
        weighted_values = values.clone().requires_grad_(True)
        rescaled_values = values.clone().requires_grad_(True)
        _weighted_mean(
            weighted_values,
            weighted.player_game_weights,
            weighted.player_game_normalization_weights,
        ).backward()
        _weighted_mean(
            rescaled_values,
            rescaled.player_game_weights,
            rescaled.player_game_normalization_weights,
        ).backward()
        torch.testing.assert_close(
            weighted_values.grad,
            rescaled_values.grad,
        )
        for common_scale in (1e300, 1e-300):
            extreme = tensorize(
                rows,
                game_source_weights={
                    "diagnostic": common_scale,
                    "baseline": common_scale,
                },
            )
            self.assertTrue(torch.all(torch.isfinite(extreme.player_game_weights)))
            self.assertTrue(torch.all(extreme.player_game_weights > 0.0))
            torch.testing.assert_close(
                extreme.player_game_weights,
                unweighted.player_game_weights,
            )
            torch.testing.assert_close(
                extreme.player_game_normalization_weights,
                unweighted.player_game_weights,
            )
        with self.assertRaisesRegex(ValueError, "relative ratios"):
            tensorize(
                rows,
                game_source_weights={
                    "diagnostic": 1e300,
                    "baseline": 1e-300,
                },
            )

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

    def test_final_posterior_evaluator_rejects_misaligned_rows(self) -> None:
        rows = self._split("posterior")
        probabilities = [
            [1.0 / CAPTURABLE_RULE_COUNT] * CAPTURABLE_RULE_COUNT
            for _ in rows
        ]

        with self.assertRaisesRegex(ValueError, "must align"):
            evaluate_capturable_posteriors(
                rows,
                probabilities[:-1],
                [0.5] * len(rows),
                [0.5] * len(rows),
                [[0.5, 0.5]] * len(rows),
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
            [float(index) for index in range(CAPTURABLE_RULE_COUNT)],
            [float(-index) for index in range(CAPTURABLE_RULE_COUNT)],
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
        self.assertEqual(first_report["version"], 2)
        self.assertEqual(
            first_report["config"]["trigger_row_multiplier"],
            1.0,
        )
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

    def test_weighted_batch_size_one_training_is_deterministic(self) -> None:
        train = self._split("weighted-train")
        validation = self._split("weighted-validation")
        weights = {
            "weighted-train-white": 1.0,
            "weighted-train-black": 0.1,
        }
        config = CapturableTrainingConfig(
            seed=20260728,
            epochs=1,
            batch_size=1,
            hidden_dimension=8,
            torch_threads=1,
        )

        first_model, first_report = train_capturable_baseline(
            train,
            validation,
            None,
            config,
            train_game_source_weights=weights,
        )
        second_model, second_report = train_capturable_baseline(
            train,
            validation,
            None,
            config,
            train_game_source_weights=weights,
        )

        self.assertEqual(first_report, second_report)
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
            {"trigger_row_multiplier": 0.99},
            {"trigger_row_multiplier": 101.0},
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
