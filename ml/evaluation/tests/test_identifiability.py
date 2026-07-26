from __future__ import annotations

import math
import unittest

from ml.evaluation.identifiability import (
    HypothesisLegality,
    IdentifiabilityAccumulator,
    IdentifiabilityObservation,
    evaluate_identifiability,
    score_identifiability,
)


def hypothesis(
    drawback_id: str,
    histories: tuple[tuple[str, ...], ...],
    probability: float | None,
    *,
    hypothesis_id: str | None = None,
    eliminated: bool = False,
) -> HypothesisLegality:
    return HypothesisLegality(
        hypothesis_id=hypothesis_id or drawback_id,
        drawback_id=drawback_id,
        eliminated=eliminated,
        permitted_move_history=tuple(frozenset(moves) for moves in histories),
        posterior_probability=probability,
    )


def observation(
    *,
    game_id: str = "game",
    color: str = "white",
    horizon: int = 5,
    turn_indices: tuple[int, ...] = (1,),
    truth: str = "a",
    true_hypothesis_id: str = "a",
    ordinary: tuple[tuple[str, ...], ...] = (("e2e4", "d2d4"),),
    hypotheses: tuple[HypothesisLegality, ...],
) -> IdentifiabilityObservation:
    return IdentifiabilityObservation(
        game_id=game_id,
        color=color,
        horizon=horizon,
        turn_indices=turn_indices,
        true_drawback=truth,
        true_hypothesis_id=true_hypothesis_id,
        ordinary_legal_history=tuple(frozenset(moves) for moves in ordinary),
        hypotheses=hypotheses,
    )


class IdentifiabilityTest(unittest.TestCase):
    def test_quiet_move_with_different_oracle_masks_is_publicly_unidentifiable(self) -> None:
        row = observation(
            hypotheses=(
                hypothesis("a", (("e2e4",),), 0.5),
                hypothesis("b", (("d2d4",),), 0.5),
            ),
        )

        scored = score_identifiability(row)

        self.assertEqual(scored.survivor_label_count, 2)
        self.assertFalse(scored.exact_publicly_identifiable)
        self.assertEqual(scored.uniform_hard_legal_top_1_tie_credit, 0.5)
        self.assertTrue(scored.full_mask_diagnostic_separability)
        self.assertEqual(scored.full_mask_diagnostic_class_count, 2)
        self.assertEqual(scored.true_variant_mask_distinct_label_count, 1)
        self.assertGreater(scored.variant_mask_partition_entropy, 0.0)

    def test_parameter_variants_use_label_counts_and_canonical_probability_sum(self) -> None:
        shared = (
            ("e2e4",),
            ("g1f3", "b1c3"),
            ("f1b5",),
        )
        row = observation(
            horizon=9,
            turn_indices=(1, 4, 8),
            truth="truth",
            true_hypothesis_id="truth:2",
            ordinary=(
                ("e2e4", "d2d4", "c2c4"),
                ("g1f3", "b1c3"),
                ("f1b5", "f1c4"),
            ),
            hypotheses=(
                hypothesis("truth", shared, 0.20, hypothesis_id="truth:2"),
                hypothesis("truth", shared, 0.25, hypothesis_id="truth:1"),
                hypothesis("other", shared, 0.30, hypothesis_id="other:1"),
                hypothesis(
                    "other",
                    (("d2d4",), ("g1f3",), ("f1c4",)),
                    0.10,
                    hypothesis_id="other:2",
                ),
                hypothesis(
                    "third",
                    (("c2c4",), ("b1c3",), ("f1c4",)),
                    0.15,
                    hypothesis_id="third:1",
                ),
            ),
        )

        scored = score_identifiability(row)

        self.assertEqual(scored.survivor_hypothesis_count, 5)
        self.assertEqual(scored.survivor_label_count, 3)
        self.assertEqual(scored.true_variant_mask_class_size, 3)
        self.assertEqual(scored.true_variant_mask_distinct_label_count, 2)
        self.assertAlmostEqual(scored.uniform_hard_legal_top_1_tie_credit, 1 / 3)
        # Each truth variant is below 0.30, but canonical label aggregation is 0.45.
        self.assertTrue(scored.model_unique_top_correct)
        self.assertFalse(scored.model_unique_top_error)
        self.assertEqual(scored.model_top_max_tie_credit, 1.0)
        self.assertEqual(scored.cumulative_truth_opportunity_count, 2)
        self.assertAlmostEqual(scored.cumulative_truth_opportunity_rate, 2 / 3)
        self.assertEqual(scored.first_truth_opportunity_index, 1)
        self.assertEqual(scored.truth_restriction_count, 3)
        self.assertEqual(scored.truth_addition_count, 0)
        self.assertAlmostEqual(scored.truth_restriction_fraction, 3 / 7)
        self.assertEqual(scored.truth_addition_fraction, 0.0)

    def test_model_top_ties_report_fractional_credit_not_unique_error(self) -> None:
        row = observation(
            hypotheses=(
                hypothesis("a", (("e2e4",),), 0.5),
                hypothesis("b", (("e2e4",),), 0.5),
            ),
        )

        scored = score_identifiability(row)
        report = evaluate_identifiability((row,))

        self.assertEqual(scored.model_top_max_label_count, 2)
        self.assertEqual(scored.model_top_max_tie_credit, 0.5)
        self.assertFalse(scored.model_unique_top_correct)
        self.assertFalse(scored.model_unique_top_error)
        self.assertEqual(report.overall.mean_model_top_max_tie_credit, 0.5)
        self.assertEqual(report.overall.model_unique_top_correct_rate, 0.0)
        self.assertEqual(report.overall.model_unique_top_error_rate, 0.0)
        self.assertFalse(report.provenance_verified)

    def test_model_wrong_top_tie_explicitly_excludes_truth(self) -> None:
        row = observation(
            hypotheses=(
                hypothesis("a", (("e2e4",),), 0.1),
                hypothesis("b", (("d2d4",),), 0.45),
                hypothesis("c", (("c2c4",),), 0.45),
            ),
        )

        scored = score_identifiability(row)
        report = evaluate_identifiability((row,))

        self.assertEqual(scored.model_top_max_label_count, 2)
        self.assertEqual(scored.model_top_max_tie_credit, 0.0)
        self.assertTrue(scored.model_top_max_excludes_truth)
        self.assertFalse(scored.model_unique_top_correct)
        self.assertFalse(scored.model_unique_top_error)
        self.assertEqual(report.overall.model_top_max_excludes_truth_count, 1)
        self.assertEqual(report.overall.model_top_max_excludes_truth_rate, 1.0)

    def test_hypothesis_reordering_does_not_change_probability_aggregation(self) -> None:
        hypotheses = (
            hypothesis("a", (("e2e4",),), 0.1, hypothesis_id="a:2"),
            hypothesis("b", (("d2d4",),), 0.4, hypothesis_id="b:1"),
            hypothesis("a", (("e2e4",),), 0.5, hypothesis_id="a:1"),
        )
        first = score_identifiability(
            observation(hypotheses=hypotheses, true_hypothesis_id="a:1")
        )
        second = score_identifiability(
            observation(
                hypotheses=tuple(reversed(hypotheses)),
                true_hypothesis_id="a:1",
            )
        )

        self.assertEqual(first, second)
        self.assertTrue(first.model_unique_top_correct)

    def test_exact_identifiability_uses_hard_survivor_labels_only(self) -> None:
        # Eliminated probabilities are required to be exactly zero.
        corrected = observation(
            hypotheses=(
                hypothesis("a", (("e2e4",),), 1.0),
                hypothesis("b", (("d2d4",),), 0.0, eliminated=True),
            ),
        )

        with self.assertRaisesRegex(ValueError, "eliminated hypothesis"):
            hypothesis("b", (("d2d4",),), 0.8, eliminated=True)
        scored = score_identifiability(corrected)
        self.assertTrue(scored.exact_publicly_identifiable)
        self.assertEqual(scored.survivor_label_count, 1)

    def test_indices_and_addition_restriction_magnitude_are_exact(self) -> None:
        row = observation(
            horizon=12,
            turn_indices=(2, 7, 11),
            ordinary=(("a", "b"), ("c",), ("d", "e")),
            hypotheses=(
                hypothesis(
                    "a",
                    (("a", "x"), ("c", "y"), ("d",)),
                    1.0,
                ),
            ),
        )

        scored = score_identifiability(row)

        self.assertTrue(scored.truth_current_opportunity)
        self.assertEqual(scored.cumulative_truth_opportunity_count, 3)
        self.assertEqual(scored.first_truth_opportunity_index, 2)
        self.assertEqual(scored.truth_restriction_count, 2)
        self.assertEqual(scored.truth_addition_count, 2)
        self.assertEqual(scored.truth_restriction_fraction, 2 / 5)
        self.assertEqual(scored.truth_addition_fraction, 2 / 5)

    def test_accumulator_rejects_duplicate_and_longitudinal_contradictions(self) -> None:
        first = observation(
            horizon=3,
            turn_indices=(1,),
            hypotheses=(
                hypothesis("a", (("e2e4",),), 0.5),
                hypothesis("b", (("d2d4",),), 0.5),
            ),
        )
        accumulator = IdentifiabilityAccumulator()
        accumulator.add(first)
        with self.assertRaisesRegex(ValueError, "duplicate game/color/horizon"):
            accumulator.add(first)

        cases = (
            observation(
                horizon=6,
                turn_indices=(1, 5),
                truth="b",
                true_hypothesis_id="b",
                ordinary=(("e2e4", "d2d4"), ("g1f3",)),
                hypotheses=(
                    hypothesis("a", (("e2e4",), ("g1f3",)), 0.5),
                    hypothesis("b", (("d2d4",), ("g1f3",)), 0.5),
                ),
            ),
            observation(
                horizon=6,
                turn_indices=(2, 5),
                ordinary=(("e2e4", "d2d4"), ("g1f3",)),
                hypotheses=(
                    hypothesis("a", (("e2e4",), ("g1f3",)), 0.5),
                    hypothesis("b", (("d2d4",), ("g1f3",)), 0.5),
                ),
            ),
            observation(
                horizon=6,
                turn_indices=(1, 5),
                ordinary=(("e2e4",), ("g1f3",)),
                hypotheses=(
                    hypothesis("a", (("e2e4",), ("g1f3",)), 0.5),
                    hypothesis("b", (("d2d4",), ("g1f3",)), 0.5),
                ),
            ),
            observation(
                horizon=6,
                turn_indices=(1, 5),
                ordinary=(("e2e4", "d2d4"), ("g1f3",)),
                hypotheses=(
                    hypothesis("a", (("d2d4",), ("g1f3",)), 0.5),
                    hypothesis("b", (("d2d4",), ("g1f3",)), 0.5),
                ),
            ),
        )
        messages = (
            "truth label or variant changed",
            "turn indices are not exact prefixes",
            "ordinary legal histories are not exact prefixes",
            "per-hypothesis legal histories are not exact prefixes",
        )
        for later, message in zip(cases, messages, strict=True):
            candidate = IdentifiabilityAccumulator()
            candidate.add(first)
            candidate.add(later)
            with self.assertRaisesRegex(ValueError, message):
                candidate.report()

    def test_same_player_rows_are_input_order_invariant(self) -> None:
        early = observation(
            horizon=3,
            turn_indices=(1,),
            hypotheses=(
                hypothesis("a", (("e2e4",),), 0.6),
                hypothesis("b", (("d2d4",),), 0.4),
            ),
        )
        late = observation(
            horizon=6,
            turn_indices=(1, 5),
            ordinary=(("e2e4", "d2d4"), ("g1f3",)),
            hypotheses=(
                hypothesis("a", (("e2e4",), ("g1f3",)), 0.7),
                hypothesis("b", (("d2d4",), ("g1f3",)), 0.3),
            ),
        )

        self.assertEqual(
            evaluate_identifiability((early, late)),
            evaluate_identifiability((late, early)),
        )

    def test_rejects_invalid_indices_and_truth_binding(self) -> None:
        with self.assertRaisesRegex(ValueError, "turn_indices"):
            observation(
                horizon=3,
                turn_indices=(2, 2),
                ordinary=(("a",), ("b",)),
                hypotheses=(hypothesis("a", (("a",), ("b",)), 1.0),),
            )
        with self.assertRaisesRegex(ValueError, "true hypothesis drawback"):
            observation(
                truth="wrong",
                true_hypothesis_id="a",
                hypotheses=(hypothesis("a", (("e2e4",),), 1.0),),
            )


if __name__ == "__main__":
    unittest.main()
