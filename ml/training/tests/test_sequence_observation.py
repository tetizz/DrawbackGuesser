from __future__ import annotations

import unittest

import _bootstrap  # noqa: F401

from drawback_ml.sequence import (
    MASKED_CURRENT_MOVE_TOKEN,
    OBSERVATION_TOKENIZER_VERSION,
    UNKNOWN_CURRENT_MOVE_TOKEN,
    ObservationTokenizerV2,
    PublicSequenceObservation,
    SanTokenizer,
    current_move_token,
    observation_tokens,
)


class ObservationTokenizerV2Tests(unittest.TestCase):
    def test_namespaces_exact_current_move_and_promotion(self) -> None:
        self.assertEqual(current_move_token("e2e4"), "<move:e2e4>")
        self.assertEqual(current_move_token("a7a8q"), "<move:a7a8q>")
        self.assertNotEqual(
            current_move_token("a7a8n"),
            current_move_token("a7a8q"),
        )
        for malformed in (
            "",
            "e2e9",
            "e2e4Q",
            "e2-e4",
            "e2e4qq",
            "e2e2",
            "e2e4q",
            "a2a3q",
        ):
            with self.subTest(move=malformed):
                with self.assertRaisesRegex(ValueError, "canonical UCI"):
                    current_move_token(malformed)

    def test_appends_exactly_one_current_token_after_prior_san(self) -> None:
        observation = PublicSequenceObservation(("e4", "e5"), "g1f3")
        self.assertEqual(
            observation_tokens(observation),
            ("e4", "e5", "<move:g1f3>"),
        )
        self.assertEqual(
            observation_tokens(observation, mask_current=True),
            ("e4", "e5", MASKED_CURRENT_MOVE_TOKEN),
        )

    def test_fit_is_stable_roundtrippable_and_includes_mask_control(self) -> None:
        observations = (
            PublicSequenceObservation(("e4",), "e7e5"),
            PublicSequenceObservation((), "e2e4"),
        )
        first = ObservationTokenizerV2.fit(observations, max_sequence=4)
        second = ObservationTokenizerV2.fit(
            tuple(reversed(observations)),
            max_sequence=4,
        )
        self.assertEqual(first, second)
        self.assertIn(MASKED_CURRENT_MOVE_TOKEN, first.vocabulary)
        self.assertIn(UNKNOWN_CURRENT_MOVE_TOKEN, first.vocabulary)
        self.assertIn("<move:e2e4>", first.vocabulary)
        self.assertEqual(
            first,
            ObservationTokenizerV2.from_metadata(first.metadata()),
        )
        self.assertEqual(
            first.metadata()["version"],
            OBSERVATION_TOKENIZER_VERSION,
        )

    def test_keep_most_recent_always_retains_current_move(self) -> None:
        observation = PublicSequenceObservation(
            ("e4", "e5", "Nf3", "Nc6"),
            "f1b5",
        )
        tokenizer = ObservationTokenizerV2.fit(
            (observation,),
            max_sequence=3,
        )
        encoded, length = tokenizer.encode(observation)
        decoded = tuple(tokenizer.vocabulary[index] for index in encoded[:length])
        self.assertEqual(decoded, ("Nf3", "Nc6", "<move:f1b5>"))
        masked, masked_length = tokenizer.encode(
            observation,
            mask_current=True,
        )
        self.assertEqual(masked_length, 3)
        self.assertEqual(
            tokenizer.vocabulary[masked[masked_length - 1]],
            MASKED_CURRENT_MOVE_TOKEN,
        )

    def test_unknown_public_tokens_do_not_collide_with_padding_or_mask(self) -> None:
        fitted = ObservationTokenizerV2.fit(
            (PublicSequenceObservation(("e4",), "e7e5"),),
            max_sequence=3,
        )
        encoded, length = fitted.encode(
            PublicSequenceObservation(("d4",), "d7d5")
        )
        self.assertEqual(length, 2)
        self.assertEqual(encoded[:length], (1, 2))
        masked, _ = fitted.encode(
            PublicSequenceObservation(("d4",), "d7d5"),
            mask_current=True,
        )
        self.assertEqual(masked[1], 3)

    def test_typed_boundary_cannot_accept_label_or_hidden_state_mapping(self) -> None:
        leaked = {
            "prior_san": ("e4",),
            "current_move_uci": "e7e5",
            "true_drawback": "vegan",
            "hidden_state": {"secret": 1},
        }
        with self.assertRaisesRegex(TypeError, "PublicSequenceObservation"):
            observation_tokens(leaked)  # type: ignore[arg-type]
        with self.assertRaisesRegex(TypeError, "PublicSequenceObservation"):
            ObservationTokenizerV2.fit(
                (leaked,),  # type: ignore[arg-type]
                max_sequence=4,
            )
        with self.assertRaises(TypeError):
            PublicSequenceObservation(**leaked)  # type: ignore[arg-type]

    def test_reserved_tokens_cannot_be_smuggled_as_prior_san(self) -> None:
        for reserved in (
            "<pad>",
            "<unk>",
            UNKNOWN_CURRENT_MOVE_TOKEN,
            MASKED_CURRENT_MOVE_TOKEN,
            "<move:e2e4>",
        ):
            with self.subTest(token=reserved):
                with self.assertRaisesRegex(ValueError, "reserved"):
                    PublicSequenceObservation((reserved,), "e7e5")

        with self.assertRaisesRegex(TypeError, "tuple"):
            PublicSequenceObservation("e4", "e7e5")  # type: ignore[arg-type]

    def test_browser_bounded_san_tokens_fail_before_fitting(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid"):
            PublicSequenceObservation(("N" * 33,), "e7e5")

    def test_metadata_rejects_tokens_fit_cannot_produce(self) -> None:
        tokenizer = ObservationTokenizerV2.fit(
            (PublicSequenceObservation(("e4",), "e7e5"),),
            max_sequence=3,
        )
        for invalid in ("<move:e2e2>", "<move:e2e4q>", "<reserved>"):
            metadata = tokenizer.metadata()
            metadata["vocabulary"] = [*tokenizer.vocabulary, invalid]
            with self.subTest(token=invalid):
                with self.assertRaisesRegex(ValueError, "invalid"):
                    ObservationTokenizerV2.from_metadata(metadata)

    def test_max_sequence_must_be_a_positive_non_boolean_integer(self) -> None:
        observation = PublicSequenceObservation(("e4",), "e7e5")
        for invalid in (True, 1.5, 0, -1):
            with self.subTest(max_sequence=invalid):
                with self.assertRaisesRegex(ValueError, "positive"):
                    ObservationTokenizerV2.fit(
                        (observation,),
                        max_sequence=invalid,  # type: ignore[arg-type]
                    )

    def test_v1_behavior_remains_unchanged(self) -> None:
        tokenizer = SanTokenizer.fit(
            (("e4", "e5"), ("Nf3", "e5")),
            max_history=3,
        )
        self.assertEqual(
            tokenizer.vocabulary,
            ("<pad>", "<unk>", "Nf3", "e4", "e5"),
        )
        self.assertEqual(tokenizer.encode(("d4", "e4", "Nf3")), ((1, 3, 2), 3))


if __name__ == "__main__":
    unittest.main()
