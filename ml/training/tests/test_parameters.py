from __future__ import annotations

import math
import unittest

import _bootstrap  # noqa: F401

from drawback_ml.parameters import (
    MASKED_PARAMETER_TOKEN,
    ParameterEncodingError,
    ParameterVocabulary,
    canonical_hidden_parameters,
    encode_parameter_targets,
)


class CanonicalParameterTests(unittest.TestCase):
    def test_encoding_is_recursive_compact_and_key_order_independent(self) -> None:
        first = {
            "square": "é4",
            "nested": {"rank": 3, "enabled": True},
            "choices": ["queen", None, 1.5],
        }
        second = {
            "choices": ["queen", None, 1.5],
            "nested": {"enabled": True, "rank": 3},
            "square": "é4",
        }
        expected = (
            '{"choices":["queen",null,1.5],'
            '"nested":{"enabled":true,"rank":3},"square":"é4"}'
        )
        self.assertEqual(canonical_hidden_parameters(first), expected)
        self.assertEqual(canonical_hidden_parameters(second), expected)
        self.assertIsNone(canonical_hidden_parameters(None))
        self.assertIsNone(canonical_hidden_parameters({}))

    def test_rejects_non_objects_and_non_deterministic_json_values(self) -> None:
        for invalid in (
            ["not", "an", "object"],
            {"value": math.nan},
            {"value": math.inf},
            {"value": object()},
            {1: "non-string key"},
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ParameterEncodingError):
                    canonical_hidden_parameters(invalid)


class ParameterVocabularyTests(unittest.TestCase):
    def test_vocabulary_is_sorted_deduplicated_and_deterministic(self) -> None:
        labels = ['{"square":"h8"}', None, '{"rank":3}', '{"square":"h8"}']
        first = ParameterVocabulary.build(labels)
        second = ParameterVocabulary.build(reversed(labels))
        self.assertEqual(first, second)
        self.assertEqual(first.tokens, ('{"rank":3}', '{"square":"h8"}'))
        self.assertEqual(first.encode('{"rank":3}'), (0, True))
        self.assertEqual(first.encode('{"square":"h8"}'), (1, True))
        self.assertEqual(first.encode(None), (0, False))

    def test_all_masked_vocabulary_still_has_a_valid_head_dimension(self) -> None:
        vocabulary = ParameterVocabulary.build([None, None])
        self.assertEqual(vocabulary.tokens, (MASKED_PARAMETER_TOKEN,))
        self.assertEqual(vocabulary.encode(None), (0, False))
        with self.assertRaisesRegex(ParameterEncodingError, "absent"):
            vocabulary.encode('{"unknown":true}')

    def test_raw_rng_seeds_are_masked_without_discarding_other_parameters(self) -> None:
        training_seed = '{"seed":9}'
        held_out_seed = '{"seed":4294967295}'
        vocabulary = ParameterVocabulary.build(
            [training_seed, '{"rank":3}', '{"rank":3,"seed":10}']
        )

        self.assertEqual(vocabulary.tokens, ('{"rank":3}',))
        self.assertEqual(vocabulary.encode(training_seed), (0, False))
        self.assertEqual(vocabulary.encode(held_out_seed), (0, False))
        self.assertEqual(vocabulary.encode('{"rank":3,"seed":999}'), (0, True))
        self.assertEqual(vocabulary.encode('{"rank":3}'), (0, True))

    def test_unseen_seed_parameters_create_no_loss_targets(self) -> None:
        vocabulary = ParameterVocabulary.build(['{"seed":1}'])
        targets = encode_parameter_targets(
            vocabulary,
            ['{"seed":2}', None],
            [None, '{"seed":3}'],
        )

        self.assertEqual(vocabulary.tokens, (MASKED_PARAMETER_TOKEN,))
        self.assertEqual(targets.white_indices, (0, 0))
        self.assertEqual(targets.white_mask, (False, False))
        self.assertEqual(targets.black_indices, (0, 0))
        self.assertEqual(targets.black_mask, (False, False))

    def test_builds_separate_white_and_black_masks(self) -> None:
        vocabulary = ParameterVocabulary.build(['{"rank":3}', '{"square":"h8"}'])
        targets = encode_parameter_targets(
            vocabulary,
            ['{"rank":3}', None],
            [None, '{"square":"h8"}'],
        )
        self.assertEqual(targets.white_indices, (0, 0))
        self.assertEqual(targets.white_mask, (True, False))
        self.assertEqual(targets.black_indices, (0, 1))
        self.assertEqual(targets.black_mask, (False, True))
        with self.assertRaisesRegex(ParameterEncodingError, "must align"):
            encode_parameter_targets(vocabulary, [None], [None, None])


if __name__ == "__main__":
    unittest.main()
