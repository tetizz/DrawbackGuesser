from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
import random
import unittest

import _bootstrap  # noqa: F401

from drawback_ml.rank_preserving_fusion import (
    MAXIMUM_NEURAL_SCALE,
    RANK_PRESERVING_FUSION_METHOD,
    RankPreservingFusionError,
    apply_rank_preserving_fusion,
    prepare_rank_preserving_fusion,
    prepare_rank_preserving_symbolic,
    rank_preserving_fusion,
)

PARITY_FIXTURE = (
    Path(__file__).parents[3]
    / "apps"
    / "web"
    / "src"
    / "fixtures"
    / "rank-preserving-fusion-v1.json"
)


class RankPreservingFusionTests(unittest.TestCase):
    def test_reused_full_preparation_matches_one_shot_fusion(self) -> None:
        residuals = (2.0, -3.0, 9.0, 0.0)
        prior = (0.45, 0.45, 0.1, 0.0)
        eliminated = (False, False, False, True)
        prepared = prepare_rank_preserving_fusion(
            residuals,
            prior,
            eliminated,
        )
        for alpha in (0.0, 0.125, 0.25, 0.5, 1.0):
            self.assertEqual(
                apply_rank_preserving_fusion(
                    prepared,
                    alpha=alpha,
                ),
                rank_preserving_fusion(
                    residuals,
                    prior,
                    eliminated,
                    alpha=alpha,
                ),
            )

    def test_shared_symbolic_preparation_matches_alpha_zero_fusion(self) -> None:
        prior = (0.4, 0.4, 0.2, 0.0, 0.0)
        eliminated = (False, False, False, False, True)
        prepared = prepare_rank_preserving_symbolic(prior, eliminated)
        fused = rank_preserving_fusion(
            (0.0,) * len(prior),
            prior,
            eliminated,
            alpha=0.0,
        )

        self.assertEqual(prepared.base_logits, fused.logits)
        self.assertEqual(prepared.neural_scales, fused.neural_scales)
        self.assertEqual(prepared.prior, prior)
        self.assertEqual(prepared.eliminated, eliminated)
        self.assertEqual(prepared.base_logits[-1], 0.0)
        self.assertEqual(prepared.neural_scales[-1], 0.0)

    def test_shared_symbolic_preparation_matches_random_scalar_rows(self) -> None:
        rng = random.Random(20260726)
        for _ in range(200):
            dimension = rng.randint(2, 32)
            raw_prior = [
                rng.choice((0.0, 0.1, 0.2, rng.random()))
                for _ in range(dimension)
            ]
            if not any(raw_prior):
                raw_prior[rng.randrange(dimension)] = 1.0
            total = math.fsum(raw_prior)
            prior = tuple(value / total for value in raw_prior)
            eliminated = tuple(
                value == 0.0 and rng.random() < 0.5 for value in prior
            )
            prepared = prepare_rank_preserving_symbolic(prior, eliminated)
            fused = rank_preserving_fusion(
                (0.0,) * dimension,
                prior,
                eliminated,
                alpha=0.0,
            )

            self.assertEqual(prepared.base_logits, fused.logits)
            self.assertEqual(prepared.neural_scales, fused.neural_scales)

    def test_matches_shared_browser_parity_vectors(self) -> None:
        fixture = json.loads(PARITY_FIXTURE.read_text(encoding="utf-8"))
        self.assertEqual(fixture["method"], RANK_PRESERVING_FUSION_METHOD)
        for case in fixture["cases"]:
            with self.subTest(case=case["name"]):
                result = rank_preserving_fusion(
                    case["residuals"],
                    case["prior"],
                    case["eliminated"],
                    alpha=case["alpha"],
                )
                expected = case["expected"]
                for actual_values, expected_values in (
                    (result.logits, expected["logits"]),
                    (result.probabilities, expected["probabilities"]),
                    (
                        result.bounded_neural_signal,
                        expected["boundedNeuralSignal"],
                    ),
                    (result.neural_scales, expected["neuralScales"]),
                ):
                    for actual, target in zip(
                        actual_values,
                        expected_values,
                        strict=True,
                    ):
                        self.assertAlmostEqual(actual, target, places=14)

    def test_extreme_residuals_cannot_reverse_symbolic_order(self) -> None:
        result = rank_preserving_fusion(
            (-1000.0, 1000.0, 1e300),
            (0.9, 0.1, 0.0),
            (False, False, True),
        )

        self.assertGreater(result.logits[0], result.logits[1])
        self.assertGreater(result.probabilities[0], result.probabilities[1])
        self.assertEqual(result.probabilities[2], 0.0)
        self.assertEqual(result.bounded_neural_signal[2], 0.0)

    def test_hard_mask_discards_stale_symbolic_mass(self) -> None:
        result = rank_preserving_fusion(
            (0.0, 1000.0),
            (0.6, 0.4),
            (False, True),
        )

        self.assertEqual(result.probabilities, (1.0, 0.0))

    def test_zero_prior_survivor_cannot_be_restored(self) -> None:
        result = rank_preserving_fusion(
            (-1000.0, 1000.0),
            (1.0, 0.0),
            (False, False),
        )

        self.assertGreater(result.logits[0], result.logits[1])
        self.assertGreater(result.probabilities[0], result.probabilities[1])
        self.assertEqual(result.probabilities[1], 0.0)

    def test_equal_symbolic_tier_uses_neural_order_and_preserves_ties(self) -> None:
        ordered = rank_preserving_fusion(
            (-4.0, 2.0, 1.0),
            (0.5, 0.5, 0.0),
            (False, False, True),
        )
        tied = rank_preserving_fusion(
            (7.0, 7.0),
            (0.5, 0.5),
            (False, False),
        )

        self.assertGreater(ordered.probabilities[1], ordered.probabilities[0])
        self.assertEqual(tied.logits[0], tied.logits[1])
        self.assertEqual(tied.probabilities[0], tied.probabilities[1])

    def test_constant_residuals_are_an_exact_no_op(self) -> None:
        prior = (0.5, 0.3, 0.2)
        result = rank_preserving_fusion(
            (7.0, 7.0, 7.0),
            prior,
            (False, False, False),
        )

        self.assertEqual(result.bounded_neural_signal, (0.0, 0.0, 0.0))
        for actual, expected in zip(
            result.probabilities,
            prior,
            strict=True,
        ):
            self.assertAlmostEqual(actual, expected, places=15)

    def test_alpha_zero_is_symbolic_only_and_shift_is_invariant(self) -> None:
        symbolic_only = rank_preserving_fusion(
            (-100.0, 100.0),
            (0.75, 0.25),
            (False, False),
            alpha=0.0,
        )
        first = rank_preserving_fusion(
            (-2.0, 3.0, 1.0),
            (0.4, 0.4, 0.2),
            (False, False, False),
        )
        shifted = rank_preserving_fusion(
            (998.0, 1003.0, 1001.0),
            (0.4, 0.4, 0.2),
            (False, False, False),
        )

        self.assertAlmostEqual(
            symbolic_only.probabilities[0],
            0.75,
        )
        self.assertAlmostEqual(
            symbolic_only.probabilities[1],
            0.25,
        )
        for left, right in zip(
            first.probabilities, shifted.probabilities, strict=True
        ):
            self.assertAlmostEqual(left, right, places=15)

    def test_permutation_is_equivariant(self) -> None:
        residual = (3.0, -1.0, 9.0, 0.5)
        prior = (0.5, 0.3, 0.0, 0.2)
        eliminated = (False, False, True, False)
        expected = rank_preserving_fusion(residual, prior, eliminated)

        for permutation in itertools.permutations(range(4)):
            actual = rank_preserving_fusion(
                tuple(residual[index] for index in permutation),
                tuple(prior[index] for index in permutation),
                tuple(eliminated[index] for index in permutation),
            )
            restored = [0.0] * 4
            for permuted_index, original_index in enumerate(permutation):
                restored[original_index] = actual.probabilities[permuted_index]
            for left, right in zip(
                expected.probabilities, restored, strict=True
            ):
                self.assertAlmostEqual(left, right, places=15)

    def test_nextafter_close_priors_remain_strict(self) -> None:
        higher = 0.5
        lower = math.nextafter(higher, 0.0)
        result = rank_preserving_fusion(
            (-1e100, 1e100),
            (higher, lower),
            (False, False),
        )

        self.assertGreater(result.logits[0], result.logits[1])
        self.assertTrue(
            all(
                0.0 <= scale <= MAXIMUM_NEURAL_SCALE
                for scale in result.neural_scales
            )
        )

    def test_log_collapsed_valid_priors_are_projected_monotonically(self) -> None:
        prior = (
            0.9999999998,
            9.999999998e-11,
            9.999999997999999e-11,
        )
        result = rank_preserving_fusion(
            (0.0, 0.0, 0.0),
            prior,
            (False, False, False),
        )

        self.assertGreater(result.logits[0], result.logits[1])
        self.assertGreater(result.logits[1], result.logits[2])

    def test_unrelated_near_tie_does_not_suppress_neural_tie_break(self) -> None:
        result = rank_preserving_fusion(
            (-1000.0, 1000.0, 1000.0, -1000.0),
            (0.4, 0.4, 0.100000000000001, 0.1),
            (False, False, False, False),
        )

        self.assertGreater(result.neural_scales[0], 0.1)
        self.assertEqual(result.neural_scales[0], result.neural_scales[1])
        self.assertGreater(result.probabilities[1], result.probabilities[0])
        self.assertGreater(result.logits[0], result.logits[2])
        self.assertGreater(result.logits[1], result.logits[2])

    def test_random_adversarial_rows_never_invert(self) -> None:
        rng = random.Random(20260725)
        for _ in range(500):
            dimension = rng.randint(2, 24)
            raw_prior = [rng.choice((0.0, rng.random())) for _ in range(dimension)]
            if not any(raw_prior):
                raw_prior[rng.randrange(dimension)] = 1.0
            total = math.fsum(raw_prior)
            prior = tuple(value / total for value in raw_prior)
            eliminated = tuple(
                value == 0.0 and rng.random() < 0.5 for value in prior
            )
            residual = tuple(
                rng.uniform(-1e6, 1e6) for _ in range(dimension)
            )
            result = rank_preserving_fusion(
                residual,
                prior,
                eliminated,
                alpha=rng.random(),
            )
            self.assertAlmostEqual(math.fsum(result.probabilities), 1.0)
            for left in range(dimension):
                for right in range(dimension):
                    if eliminated[left] or eliminated[right]:
                        continue
                    if prior[left] > prior[right]:
                        self.assertGreater(
                            result.logits[left],
                            result.logits[right],
                        )
            for index, is_eliminated in enumerate(eliminated):
                if is_eliminated:
                    self.assertEqual(result.probabilities[index], 0.0)

    def test_rejects_invalid_contracts(self) -> None:
        invalid_calls = (
            lambda: rank_preserving_fusion((), (), ()),
            lambda: rank_preserving_fusion((0.0,), (1.0, 0.0), (False,)),
            lambda: rank_preserving_fusion((0.0,), (1.0,), (False,), alpha=-0.1),
            lambda: rank_preserving_fusion((0.0,), (1.0,), (False,), alpha=1.1),
            lambda: rank_preserving_fusion(
                (0.0,), (1.0,), (False,), alpha=float("nan")
            ),
            lambda: rank_preserving_fusion(
                (0.0,), (1.0,), (False,), alpha=float("inf")
            ),
            lambda: rank_preserving_fusion((0.0,), (1.0,), (False,), alpha=True),
            lambda: rank_preserving_fusion((float("nan"),), (1.0,), (False,)),
            lambda: rank_preserving_fusion((0.0,), (-1.0,), (False,)),
            lambda: rank_preserving_fusion((0.0,), (1.0,), (1,)),  # type: ignore[arg-type]
            lambda: rank_preserving_fusion((0.0,), (1.0,), (True,)),
            lambda: rank_preserving_fusion((0.0,), (0.0,), (False,)),
        )
        for call in invalid_calls:
            with self.subTest(call=call), self.assertRaises(
                RankPreservingFusionError
            ):
                call()


if __name__ == "__main__":
    unittest.main()
