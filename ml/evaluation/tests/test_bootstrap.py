from __future__ import annotations

import unittest

from ml.evaluation.bootstrap import (
    DEFAULT_REPLICATES,
    PlayerGamePredictions,
    paired_comparator_bootstrap,
    player_game_bootstrap_ci,
)
from ml.evaluation.metrics import PredictionExample


def example(
    game_id: str,
    color: str,
    *,
    correct: bool,
    ply: int = 1,
) -> PredictionExample:
    true = "A" if color == "white" else "B"
    other = "B" if color == "white" else "A"
    return PredictionExample(
        game_id=game_id,
        move_number=ply,
        observed_ply=ply,
        player_color=color,
        true_drawback=true,
        probabilities=(
            {true: 0.8, other: 0.2}
            if correct
            else {true: 0.2, other: 0.8}
        ),
    )


def corpus(correct_by_seed: dict[int, tuple[bool, bool]]) -> tuple[PlayerGamePredictions, ...]:
    rows = []
    for seed, (white_correct, black_correct) in correct_by_seed.items():
        game_id = f"game-{seed}"
        rows.extend(
            (
                PlayerGamePredictions(
                    seed,
                    game_id,
                    "white",
                    (
                        example(game_id, "white", correct=white_correct, ply=1),
                        example(game_id, "white", correct=white_correct, ply=3),
                    ),
                ),
                PlayerGamePredictions(
                    seed,
                    game_id,
                    "black",
                    (
                        example(game_id, "black", correct=black_correct, ply=2),
                        example(game_id, "black", correct=black_correct, ply=4),
                    ),
                ),
            )
        )
    return tuple(rows)


def trajectory_accuracy(rows: list[PlayerGamePredictions] | tuple[PlayerGamePredictions, ...]) -> float:
    correct = 0
    for row in rows:
        final = row.examples[-1]
        predicted = max(final.probabilities, key=final.probabilities.__getitem__)
        correct += predicted == final.true_drawback
    return correct / len(rows)


class PlayerGameBootstrapTests(unittest.TestCase):
    def test_is_deterministic_and_preserves_complete_seed_clusters(self) -> None:
        rows = corpus({11: (True, False), 12: (True, True), 13: (False, False)})
        observed_samples: list[tuple[tuple[int, str], ...]] = []

        def audited_statistic(
            sample: list[PlayerGamePredictions]
            | tuple[PlayerGamePredictions, ...],
        ) -> float:
            counts: dict[int, dict[str, int]] = {}
            for row in sample:
                colors = counts.setdefault(row.simulation_seed, {})
                colors[row.player_color] = colors.get(row.player_color, 0) + 1
                self.assertEqual(len(row.examples), 2)
            for colors in counts.values():
                self.assertEqual(colors.get("white"), colors.get("black"))
            observed_samples.append(
                tuple((row.simulation_seed, row.player_color) for row in sample)
            )
            return trajectory_accuracy(sample)

        first = player_game_bootstrap_ci(
            rows,
            audited_statistic,
            replicates=50,
            random_seed=99,
        )
        second = player_game_bootstrap_ci(
            rows,
            trajectory_accuracy,
            replicates=50,
            random_seed=99,
        )
        self.assertEqual(first, second)
        self.assertEqual(first.estimate, 0.5)
        self.assertEqual(len(observed_samples), 51)

    def test_paired_difference_uses_aligned_resamples(self) -> None:
        candidate = corpus({1: (True, True), 2: (True, True), 3: (False, False)})
        comparator = corpus({1: (False, False), 2: (True, True), 3: (False, False)})
        summary = paired_comparator_bootstrap(
            candidate,
            comparator,
            trajectory_accuracy,
            replicates=200,
            random_seed=7,
        )
        self.assertAlmostEqual(summary.candidate.estimate, 2 / 3)
        self.assertAlmostEqual(summary.comparator.estimate, 1 / 3)
        self.assertAlmostEqual(summary.difference.estimate, 1 / 3)
        self.assertEqual(summary.difference.replicates, 200)
        self.assertGreaterEqual(summary.difference.lower, 0.0)

    def test_supports_the_preregistered_ten_thousand_replicates(self) -> None:
        rows = corpus({1: (True, False), 2: (False, True)})
        result = player_game_bootstrap_ci(
            rows,
            trajectory_accuracy,
            replicates=DEFAULT_REPLICATES,
            random_seed=1234,
        )
        self.assertEqual(result.replicates, 10_000)
        self.assertEqual((result.estimate, result.lower, result.upper), (0.5, 0.5, 0.5))

    def test_rejects_incomplete_or_misaligned_pairs_and_invalid_options(self) -> None:
        complete = corpus({1: (True, False)})
        with self.assertRaisesRegex(ValueError, "complete White and Black"):
            player_game_bootstrap_ci(
                complete[:1],
                trajectory_accuracy,
                replicates=10,
            )
        misaligned = corpus({2: (True, False)})
        with self.assertRaisesRegex(ValueError, "identities must align"):
            paired_comparator_bootstrap(
                complete,
                misaligned,
                trajectory_accuracy,
                replicates=10,
            )
        with self.assertRaisesRegex(ValueError, "replicates"):
            player_game_bootstrap_ci(complete, trajectory_accuracy, replicates=0)
        with self.assertRaisesRegex(ValueError, "confidence_level"):
            player_game_bootstrap_ci(
                complete,
                trajectory_accuracy,
                replicates=10,
                confidence_level=1.0,
            )

    def test_rejects_trajectory_target_mismatch(self) -> None:
        candidate = corpus({1: (True, False)})
        comparator_rows = list(corpus({1: (False, True)}))
        black = comparator_rows[1]
        comparator_rows[1] = PlayerGamePredictions(
            black.simulation_seed,
            black.game_id,
            black.player_color,
            (
                example(black.game_id, "black", correct=True, ply=2),
                PredictionExample(
                    game_id=black.game_id,
                    move_number=4,
                    observed_ply=4,
                    player_color="black",
                    true_drawback="different",
                    probabilities={"different": 0.8, "B": 0.2},
                ),
            ),
        )
        with self.assertRaisesRegex(ValueError, "horizons and targets"):
            paired_comparator_bootstrap(
                candidate,
                comparator_rows,
                trajectory_accuracy,
                replicates=10,
            )


if __name__ == "__main__":
    unittest.main()
