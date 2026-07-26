from __future__ import annotations

import math
import importlib.util
import unittest

from ml.evaluation import (
    PredictionExample,
    evaluate,
    evaluate_binary,
    evaluate_legal_masks,
    expected_calibration_error,
    negative_log_likelihood,
)
from ml.evaluation.metrics import (
    StreamingBinaryEvaluation,
    StreamingEvaluation,
    StreamingLegalMaskEvaluation,
)


class EvaluationMetricsTest(unittest.TestCase):
    @unittest.skipUnless(
        importlib.util.find_spec("torch") is not None,
        "PyTorch is optional for metric-only environments",
    )
    def test_vectorized_legal_statistics_match_scalar_with_zeroes(self) -> None:
        import torch

        from ml.training.drawback_ml.inference import (
            _legal_mask_batch_statistics,
        )

        probabilities = torch.tensor(
            [[0.0, 1.0, 0.5], [1.0, 0.0, 0.4]],
            dtype=torch.float32,
        )
        targets = ((0,), (0,))
        statistics = _legal_mask_batch_statistics(
            torch,
            probabilities,
            targets,
            3,
        )
        vectorized = StreamingLegalMaskEvaluation(3)
        vectorized.add_batch_statistics(
            example_count=statistics.example_count,
            dimension=statistics.dimension,
            exact_matches=statistics.exact_matches,
            true_positives=statistics.true_positives,
            false_positives=statistics.false_positives,
            false_negatives=statistics.false_negatives,
            binary_cross_entropy_sum=statistics.binary_cross_entropy_sum,
            has_infinite_loss=statistics.has_infinite_loss,
        )
        scalar = StreamingLegalMaskEvaluation(3)
        for indices, row in zip(
            targets,
            probabilities.tolist(),
            strict=True,
        ):
            scalar.add(indices, row)
        self.assertEqual(vectorized.report(), scalar.report())

        finite = torch.tensor(
            [[0.1, 0.9, 0.5], [0.8, 0.2, 0.4]],
            dtype=torch.float32,
        )
        finite_statistics = _legal_mask_batch_statistics(
            torch, finite, ((1,), (0,)), 3
        )
        finite_vectorized = StreamingLegalMaskEvaluation(3)
        finite_vectorized.add_batch_statistics(
            example_count=finite_statistics.example_count,
            dimension=finite_statistics.dimension,
            exact_matches=finite_statistics.exact_matches,
            true_positives=finite_statistics.true_positives,
            false_positives=finite_statistics.false_positives,
            false_negatives=finite_statistics.false_negatives,
            binary_cross_entropy_sum=finite_statistics.binary_cross_entropy_sum,
            has_infinite_loss=finite_statistics.has_infinite_loss,
        )
        finite_scalar = StreamingLegalMaskEvaluation(3)
        for indices, row in zip(
            ((1,), (0,)), finite.tolist(), strict=True
        ):
            finite_scalar.add(indices, row)
        left = finite_vectorized.report()
        right = finite_scalar.report()
        self.assertEqual(
            (
                left.exact_match_accuracy,
                left.micro_precision,
                left.micro_recall,
                left.micro_f1,
            ),
            (
                right.exact_match_accuracy,
                right.micro_precision,
                right.micro_recall,
                right.micro_f1,
            ),
        )
        self.assertAlmostEqual(
            left.binary_cross_entropy,
            right.binary_cross_entropy,
            places=12,
        )

    def test_streaming_accumulators_match_batch_reports(self) -> None:
        examples = (
            PredictionExample(
                game_id="g1",
                move_number=1,
                observed_ply=1,
                player_color="white",
                true_drawback="A",
                probabilities={"A": 0.8, "B": 0.2},
                rule_family="family-a",
                true_parameters={"rank": 3},
                predicted_parameters={"rank": 3},
            ),
            PredictionExample(
                game_id="g1",
                move_number=2,
                observed_ply=2,
                player_color="white",
                true_drawback="B",
                probabilities={"A": 0.6, "B": 0.4},
                rule_family="family-b",
            ),
        )
        multiclass = StreamingEvaluation()
        for example in examples:
            multiclass.add(example)
        self.assertEqual(multiclass.report(), evaluate(examples))

        labels = [True, False]
        probabilities = [0.8, 0.3]
        binary = StreamingBinaryEvaluation()
        for label, probability in zip(labels, probabilities, strict=True):
            binary.add(label, probability)
        self.assertEqual(
            binary.report(),
            evaluate_binary(labels, probabilities),
        )

        masks = [[True, False], [False, True]]
        mask_probabilities = [[0.9, 0.2], [0.4, 0.7]]
        legal = StreamingLegalMaskEvaluation(2)
        for mask, probabilities in zip(
            masks, mask_probabilities, strict=True
        ):
            legal.add(
                (index for index, value in enumerate(mask) if value),
                probabilities,
            )
        self.assertEqual(
            legal.report(),
            evaluate_legal_masks(masks, mask_probabilities),
        )

    def setUp(self) -> None:
        self.examples = (
            PredictionExample(
                game_id="game-a",
                move_number=5,
                true_drawback="A",
                probabilities={"A": 0.7, "B": 0.2, "C": 0.1},
                rule_family="capture",
                true_parameters={"square": "a1"},
                predicted_parameters={"square": "a1"},
                entropy_before=2.0,
                entropy_after=1.4,
                diagnostic_information_gain=0.4,
            ),
            PredictionExample(
                game_id="game-a",
                move_number=10,
                true_drawback="A",
                probabilities={"A": 0.3, "B": 0.6, "C": 0.1},
                rule_family="capture",
                true_parameters={"square": "a1"},
                predicted_parameters={"square": "b2"},
                entropy_before=1.4,
                entropy_after=1.2,
                diagnostic_information_gain=0.2,
            ),
            PredictionExample(
                game_id="game-b",
                move_number=5,
                true_drawback="B",
                probabilities={"A": 0.5, "B": 0.4, "C": 0.1},
                rule_family="history",
                true_parameters={"piece": "knight"},
                predicted_parameters={"piece": "bishop"},
                entropy_before=2.2,
                entropy_after=1.7,
                diagnostic_information_gain=0.3,
            ),
            PredictionExample(
                game_id="game-b",
                move_number=10,
                true_drawback="B",
                probabilities={"A": 0.1, "B": 0.8, "C": 0.1},
                rule_family="history",
                true_parameters={"piece": "knight"},
                predicted_parameters={"piece": "knight"},
                entropy_before=1.7,
                entropy_after=0.7,
                diagnostic_information_gain=0.5,
            ),
        )

    def test_complete_report_uses_hand_computable_examples(self) -> None:
        report = evaluate(self.examples, calibration_bins=10)

        self.assertEqual(report.count, 4)
        self.assertEqual(report.top_1_accuracy, 0.5)
        self.assertEqual(report.top_3_accuracy, 1.0)
        self.assertEqual(report.top_5_accuracy, 1.0)
        self.assertAlmostEqual(
            report.negative_log_likelihood,
            -math.log(0.7 * 0.3 * 0.4 * 0.8) / 4,
        )
        expected_brier = (
            (0.3**2 + 0.2**2 + 0.1**2)
            + (0.7**2 + 0.6**2 + 0.1**2)
            + (0.5**2 + 0.6**2 + 0.1**2)
            + (0.1**2 + 0.2**2 + 0.1**2)
        ) / 4
        self.assertAlmostEqual(report.brier_score, expected_brier)
        self.assertAlmostEqual(report.expected_calibration_error, 0.4)
        self.assertEqual(
            dict(report.accuracy_after_moves),
            {5: 0.5, 10: 0.5, 15: 0.5, 20: 0.5},
        )
        self.assertEqual(report.mean_first_rank_one_move, 7.5)
        self.assertEqual(dict(report.accuracy_per_drawback), {"A": 0.5, "B": 0.5})
        self.assertEqual(
            dict(report.accuracy_per_rule_family),
            {"capture": 0.5, "history": 0.5},
        )
        self.assertEqual(report.hidden_parameter_accuracy, 0.5)
        self.assertEqual(
            dict(report.hidden_parameter_accuracy_by_name),
            {"piece": 0.5, "square": 0.5},
        )
        self.assertAlmostEqual(report.mean_entropy_reduction or 0.0, 0.575)
        self.assertAlmostEqual(
            report.mean_diagnostic_information_gain or 0.0, 0.35
        )
        self.assertEqual(
            {label: dict(row) for label, row in report.confusion_matrix.items()},
            {"A": {"A": 1, "B": 1}, "B": {"A": 1, "B": 1}},
        )
        self.assertEqual(
            {label: dict(row) for label, row in report.confusion_counts.items()},
            {"A": {"A": 1, "B": 1}, "B": {"A": 1, "B": 1}},
        )
        self.assertEqual(report.probability_diagnostics.checked_count, 4)
        self.assertLessEqual(
            report.probability_diagnostics.maximum_absolute_sum_error,
            1e-15,
        )
        capture = report.metrics_per_rule_family["capture"]
        self.assertEqual(capture.support, 2)
        self.assertEqual(capture.top_1_accuracy, 0.5)
        self.assertEqual(capture.top_3_accuracy, 1.0)
        self.assertEqual(capture.top_5_accuracy, 1.0)
        self.assertAlmostEqual(
            capture.negative_log_likelihood,
            -math.log(0.7 * 0.3) / 2,
        )
        self.assertAlmostEqual(
            capture.brier_score,
            (
                (0.3**2 + 0.2**2 + 0.1**2)
                + (0.7**2 + 0.6**2 + 0.1**2)
            )
            / 2,
        )

    def test_zero_true_probability_has_infinite_nll(self) -> None:
        row = PredictionExample(
            game_id="zero",
            move_number=1,
            true_drawback="A",
            probabilities={"A": 0.0, "B": 1.0},
        )
        self.assertTrue(math.isinf(negative_log_likelihood([row])))

    def test_ece_uses_top_class_confidence_bins(self) -> None:
        self.assertAlmostEqual(
            expected_calibration_error(self.examples, bin_count=10),
            0.4,
        )

    def test_invalid_probability_distribution_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            PredictionExample(
                game_id="bad",
                move_number=1,
                true_drawback="A",
                probabilities={"A": 0.8, "B": 0.3},
            )

    def test_empty_evaluation_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            evaluate([])

    def test_binary_and_legal_mask_outputs_are_measured(self) -> None:
        binary = evaluate_binary([True, False], [0.8, 0.25])
        self.assertEqual(binary.accuracy, 1.0)
        self.assertAlmostEqual(binary.brier_score, (0.2**2 + 0.25**2) / 2)
        masks = evaluate_legal_masks(
            [[True, False, True], [False, True, False]],
            [[0.9, 0.1, 0.8], [0.2, 0.7, 0.1]],
        )
        self.assertEqual(masks.exact_match_accuracy, 1.0)
        self.assertEqual(masks.micro_precision, 1.0)
        self.assertEqual(masks.micro_recall, 1.0)
        self.assertEqual(masks.micro_f1, 1.0)

    def test_player_identity_prevents_rank_one_timing_collapse(self) -> None:
        rows = (
            PredictionExample(
                game_id="same-game",
                move_number=1,
                observed_ply=5,
                player_color="white",
                true_drawback="A",
                probabilities={"A": 1.0, "B": 0.0},
            ),
            PredictionExample(
                game_id="same-game",
                move_number=1,
                observed_ply=10,
                player_color="black",
                true_drawback="B",
                probabilities={"A": 0.0, "B": 1.0},
            ),
        )
        report = evaluate(rows)
        self.assertEqual(report.mean_first_rank_one_move, 7.5)
        self.assertEqual(report.median_first_rank_one_move, 7.5)
        self.assertEqual(report.rank_one_player_games, 2)
        self.assertEqual(report.never_rank_one_player_games, 0)
        self.assertEqual(
            dict(report.accuracy_after_moves),
            {5: 1.0, 10: 1.0, 15: 1.0, 20: 1.0},
        )

    def test_top_k_ties_are_permutation_invariant(self) -> None:
        first = PredictionExample(
            game_id="tie",
            move_number=1,
            true_drawback="B",
            probabilities={"A": 0.5, "B": 0.5, "C": 0.0},
        )
        second = PredictionExample(
            game_id="tie",
            move_number=1,
            true_drawback="B",
            probabilities={"C": 0.0, "B": 0.5, "A": 0.5},
        )

        first_report = evaluate((first,))
        second_report = evaluate((second,))
        self.assertEqual(first_report.top_1_accuracy, 0.5)
        self.assertEqual(first_report.top_3_accuracy, 1.0)
        self.assertEqual(
            first_report.top_1_accuracy,
            second_report.top_1_accuracy,
        )
        self.assertEqual(
            first_report.top_3_accuracy,
            second_report.top_3_accuracy,
        )
        self.assertEqual(
            dict(first_report.accuracy_per_drawback),
            {"B": 0.5},
        )
        self.assertEqual(
            dict(first_report.confusion_matrix["B"]),
            {"A": 0.5, "B": 0.5},
        )
        self.assertEqual(
            first_report.expected_calibration_error,
            second_report.expected_calibration_error,
        )

    def test_tied_maximum_does_not_claim_first_rank_one(self) -> None:
        report = evaluate(
            (
                PredictionExample(
                    game_id="uniform",
                    move_number=1,
                    observed_ply=1,
                    player_color="white",
                    true_drawback="B",
                    probabilities={"A": 0.5, "B": 0.5},
                ),
            )
        )
        self.assertIsNone(report.mean_first_rank_one_move)
        self.assertIsNone(report.median_first_rank_one_move)
        self.assertEqual(report.rank_one_player_games, 0)
        self.assertEqual(report.never_rank_one_player_games, 1)
        uniform = PredictionExample(
            game_id="top-three-tie",
            move_number=1,
            true_drawback="D",
            probabilities={"D": 0.25, "B": 0.25, "A": 0.25, "C": 0.25},
        )
        uniform_report = evaluate((uniform,))
        self.assertEqual(uniform_report.top_1_accuracy, 0.25)
        self.assertEqual(uniform_report.top_3_accuracy, 0.75)
        self.assertEqual(uniform_report.top_5_accuracy, 1.0)

    def test_horizons_use_last_available_player_prefix(self) -> None:
        rows = (
            PredictionExample(
                game_id="short",
                move_number=1,
                observed_ply=2,
                player_color="white",
                true_drawback="A",
                probabilities={"A": 0.8, "B": 0.2},
            ),
            PredictionExample(
                game_id="short",
                move_number=2,
                observed_ply=4,
                player_color="white",
                true_drawback="A",
                probabilities={"A": 0.2, "B": 0.8},
            ),
            PredictionExample(
                game_id="long",
                move_number=1,
                observed_ply=5,
                player_color="black",
                true_drawback="B",
                probabilities={"A": 0.1, "B": 0.9},
            ),
            PredictionExample(
                game_id="long",
                move_number=2,
                observed_ply=11,
                player_color="black",
                true_drawback="B",
                probabilities={"A": 0.9, "B": 0.1},
            ),
        )

        report = evaluate(rows)
        self.assertEqual(
            dict(report.accuracy_after_moves),
            {
                5: 0.5,
                10: 0.5,
                15: 0.0,
                20: 0.0,
            },
        )
        self.assertEqual(
            dict(report.top_1_accuracy_at_observed_plies),
            dict(report.accuracy_after_moves),
        )
        self.assertEqual(
            dict(report.top_3_accuracy_at_observed_plies),
            {5: 1.0, 10: 1.0, 15: 1.0, 20: 1.0},
        )

    def test_default_ece_uses_fifteen_deterministic_bins(self) -> None:
        rows = (
            PredictionExample(
                game_id="correct",
                move_number=1,
                true_drawback="A",
                probabilities={"A": 0.51, "B": 0.49},
            ),
            PredictionExample(
                game_id="wrong",
                move_number=1,
                true_drawback="B",
                probabilities={"A": 0.54, "B": 0.46},
            ),
        )
        # Both confidences share a 10-bin bucket but occupy adjacent 15-bin
        # buckets, making the default observable rather than incidental.
        self.assertAlmostEqual(expected_calibration_error(rows), 0.515)
        self.assertAlmostEqual(
            evaluate(rows).expected_calibration_error,
            0.515,
        )
        self.assertAlmostEqual(
            expected_calibration_error(rows, bin_count=10),
            0.025,
        )

    def test_exact_confusion_counts_break_ties_by_label_without_changing_credit(
        self,
    ) -> None:
        row = PredictionExample(
            game_id="tie-count",
            move_number=1,
            true_drawback="B",
            probabilities={"B": 0.5, "A": 0.5},
        )
        report = evaluate((row,))
        self.assertEqual(dict(report.confusion_matrix["B"]), {"A": 0.5, "B": 0.5})
        self.assertEqual(dict(report.confusion_counts["B"]), {"A": 1})
        self.assertEqual(report.metrics_per_drawback["B"].support, 1)
        self.assertEqual(report.metrics_per_drawback["B"].top_1_accuracy, 0.5)

    def test_reports_probability_on_hard_eliminated_hypotheses(self) -> None:
        rows = (
            PredictionExample(
                game_id="clean",
                move_number=1,
                true_drawback="A",
                probabilities={"A": 1.0, "B": 0.0},
                hard_eliminated={"A": False, "B": True},
            ),
            PredictionExample(
                game_id="violation",
                move_number=1,
                true_drawback="A",
                probabilities={"A": 0.75, "B": 0.25},
                hard_eliminated={"A": False, "B": True},
            ),
            PredictionExample(
                game_id="missing",
                move_number=1,
                true_drawback="A",
                probabilities={"A": 1.0, "B": 0.0},
            ),
        )
        diagnostics = evaluate(rows).probability_diagnostics
        self.assertEqual(diagnostics.checked_count, 3)
        self.assertEqual(diagnostics.hard_mask_checked_count, 2)
        self.assertEqual(diagnostics.missing_hard_mask_count, 1)
        self.assertEqual(diagnostics.hard_elimination_violation_count, 1)
        self.assertEqual(diagnostics.maximum_eliminated_probability, 0.25)

    def test_rejects_incomplete_hard_elimination_mapping(self) -> None:
        with self.assertRaisesRegex(ValueError, "one boolean"):
            PredictionExample(
                game_id="bad-mask",
                move_number=1,
                true_drawback="A",
                probabilities={"A": 1.0, "B": 0.0},
                hard_eliminated={"A": False},
            )

    def test_never_rank_one_player_games_are_counted(self) -> None:
        rows = (
            PredictionExample(
                game_id="found",
                move_number=1,
                observed_ply=3,
                player_color="white",
                true_drawback="A",
                probabilities={"A": 0.7, "B": 0.3},
            ),
            PredictionExample(
                game_id="never",
                move_number=1,
                observed_ply=4,
                player_color="black",
                true_drawback="B",
                probabilities={"A": 0.8, "B": 0.2},
            ),
        )

        report = evaluate(rows)
        self.assertEqual(report.mean_first_rank_one_move, 3.0)
        self.assertEqual(report.median_first_rank_one_move, 3.0)
        self.assertEqual(report.rank_one_player_games, 1)
        self.assertEqual(report.never_rank_one_player_games, 1)

    def test_game_normalized_metrics_give_short_and_long_games_equal_weight(
        self,
    ) -> None:
        long_wrong = [
            PredictionExample(
                game_id="long",
                move_number=index + 1,
                observed_ply=index + 1,
                player_color="white",
                true_drawback="A",
                probabilities={"A": 0.1, "B": 0.9},
            )
            for index in range(9)
        ]
        short_correct = PredictionExample(
            game_id="short",
            move_number=1,
            observed_ply=1,
            player_color="white",
            true_drawback="A",
            probabilities={"A": 0.9, "B": 0.1},
        )
        report = evaluate((*long_wrong, short_correct))
        self.assertEqual(report.count, 10)
        self.assertEqual(report.player_game_count, 2)
        self.assertAlmostEqual(report.top_1_accuracy, 0.1)
        self.assertAlmostEqual(report.game_normalized_top_1_accuracy, 0.5)
        expected_nll = (-math.log(0.1) + -math.log(0.9)) / 2
        self.assertAlmostEqual(
            report.game_normalized_negative_log_likelihood,
            expected_nll,
        )


if __name__ == "__main__":
    unittest.main()
