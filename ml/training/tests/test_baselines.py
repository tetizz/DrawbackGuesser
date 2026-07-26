from __future__ import annotations

import math
import unittest

from ml.training.tests import _bootstrap  # noqa: F401

from drawback_ml.baselines import (
    GameAssignment,
    MoverObservation,
    PredictionInput,
    PublicSymbolicEvidence,
    empirical_game_prior,
    evaluate_mover_only,
    make_constant_predictor,
    make_symbolic_only_predictor,
    symbolic_only_distribution,
    uniform_prior,
)


VOCABULARY = ("alpha", "beta", "gamma", "delta", "epsilon", "zeta")


class BaselineTest(unittest.TestCase):
    def test_uniform_prior_is_normalized(self) -> None:
        self.assertEqual(uniform_prior(VOCABULARY), (1.0 / 6.0,) * 6)

    def test_uniform_top_k_uses_tie_aware_expected_credit(self) -> None:
        report = evaluate_mover_only(
            [
                MoverObservation("g1", "white", drawback)
                for drawback in VOCABULARY
            ],
            VOCABULARY,
            make_constant_predictor(uniform_prior(VOCABULARY)),
        )
        self.assertAlmostEqual(report.overall.top1_accuracy, 1 / 6)
        self.assertAlmostEqual(report.overall.top3_accuracy, 3 / 6)
        self.assertAlmostEqual(report.overall.top5_accuracy, 5 / 6)

    def test_tied_top_k_is_invariant_to_vocabulary_order(self) -> None:
        reversed_vocabulary = tuple(reversed(VOCABULARY))
        first = evaluate_mover_only(
            [MoverObservation("g1", "white", "gamma")],
            VOCABULARY,
            make_constant_predictor(uniform_prior(VOCABULARY)),
        )
        second = evaluate_mover_only(
            [MoverObservation("g1", "white", "gamma")],
            reversed_vocabulary,
            make_constant_predictor(uniform_prior(reversed_vocabulary)),
        )
        self.assertEqual(first.overall, second.overall)

    def test_empirical_prior_counts_game_assignments_not_move_rows(self) -> None:
        assignments = [
            GameAssignment("long", "alpha", "beta"),
            GameAssignment("long", "alpha", "beta"),
            GameAssignment("zero-ply", "beta", "beta"),
        ]
        self.assertEqual(
            empirical_game_prior(assignments, VOCABULARY),
            (0.25, 0.75, 0.0, 0.0, 0.0, 0.0),
        )

    def test_empirical_prior_rejects_inconsistent_game_assignment(self) -> None:
        with self.assertRaisesRegex(ValueError, "inconsistent drawback assignment"):
            empirical_game_prior(
                [
                    GameAssignment("game", "alpha", "beta"),
                    GameAssignment("game", "beta", "beta"),
                ],
                VOCABULARY,
            )

    def test_symbolic_only_applies_exact_elimination_and_renormalizes(self) -> None:
        result = symbolic_only_distribution(
            PublicSymbolicEvidence(
                probabilities=(0.1, 0.2, 0.3, 0.1, 0.2, 0.1),
                eliminated=(False, True, False, True, False, False),
            ),
            VOCABULARY,
        )
        self.assertEqual(result[1], 0.0)
        self.assertEqual(result[3], 0.0)
        self.assertAlmostEqual(sum(result), 1.0)
        for actual, expected in zip(
            result,
            (1 / 7, 0.0, 3 / 7, 0.0, 2 / 7, 1 / 7),
            strict=True,
        ):
            self.assertAlmostEqual(actual, expected)

    def test_symbolic_only_fails_without_surviving_mass(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive probability mass"):
            symbolic_only_distribution(
                PublicSymbolicEvidence(
                    probabilities=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                    eliminated=(True, False, False, False, False, False),
                ),
                VOCABULARY,
            )

    def test_mover_only_report_evaluates_one_label_per_move(self) -> None:
        seen: list[PredictionInput] = []

        def predictor(item: PredictionInput) -> tuple[float, ...]:
            seen.append(item)
            return (
                (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                if item.mover == "white"
                else (0.0, 1.0, 0.0, 0.0, 0.0, 0.0)
            )

        report = evaluate_mover_only(
            [
                MoverObservation("g1", "white", "alpha"),
                MoverObservation("g1", "black", "beta"),
                MoverObservation("g1", "white", "alpha"),
            ],
            VOCABULARY,
            predictor,
        )
        self.assertEqual(len(seen), 3)
        self.assertEqual(report.overall.observations, 3)
        self.assertEqual(report.white.observations, 2)
        self.assertEqual(report.black.observations, 1)
        self.assertEqual(report.overall.top1_accuracy, 1.0)
        self.assertEqual(report.overall.negative_log_likelihood, 0.0)
        self.assertEqual(report.overall.brier_score, 0.0)

    def test_predictor_never_receives_label_or_game_identity(self) -> None:
        received: list[PredictionInput] = []

        def predictor(item: PredictionInput) -> tuple[float, ...]:
            received.append(item)
            return uniform_prior(VOCABULARY)

        evaluate_mover_only(
            [MoverObservation("secret-game", "white", "gamma")],
            VOCABULARY,
            predictor,
        )
        self.assertEqual(received, [PredictionInput(mover="white")])
        self.assertFalse(hasattr(received[0], "true_drawback"))
        self.assertFalse(hasattr(received[0], "game_id"))

    def test_evaluation_fails_if_truth_was_symbolically_eliminated(self) -> None:
        evidence = PublicSymbolicEvidence(
            probabilities=(0.2, 0.2, 0.2, 0.2, 0.1, 0.1),
            eliminated=(False, False, True, False, False, False),
        )
        with self.assertRaisesRegex(
            ValueError, "true drawback is symbolically eliminated"
        ):
            evaluate_mover_only(
                [MoverObservation("g1", "white", "gamma", evidence)],
                VOCABULARY,
                make_symbolic_only_predictor(VOCABULARY),
            )

    def test_symbolic_predictor_uses_only_public_evidence(self) -> None:
        evidence = PublicSymbolicEvidence(
            probabilities=(0.0, 0.6, 0.1, 0.1, 0.1, 0.1),
            eliminated=(True, False, False, False, False, False),
        )
        report = evaluate_mover_only(
            [MoverObservation("g1", "black", "beta", evidence)],
            VOCABULARY,
            make_symbolic_only_predictor(VOCABULARY),
        )
        self.assertEqual(report.overall.top1_accuracy, 1.0)
        self.assertTrue(math.isfinite(report.overall.negative_log_likelihood))

    def test_constant_predictor_is_checked_at_evaluation_boundary(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite non-negative"):
            evaluate_mover_only(
                [MoverObservation("g1", "white", "alpha")],
                VOCABULARY,
                make_constant_predictor((math.nan,) * len(VOCABULARY)),
            )


if __name__ == "__main__":
    unittest.main()
