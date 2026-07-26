from __future__ import annotations

import unittest

import _bootstrap  # noqa: F401

from drawback_ml.model import ModelConfig, create_sequence_model
from drawback_ml.records import group_training_examples
from drawback_ml.sequence import (
    PAD_TOKEN,
    TOKENIZER_VERSION,
    UNKNOWN_TOKEN,
    SanTokenizer,
)
from drawback_ml.training import TrainingConfig
from test_records import row

try:
    import torch
except ImportError:
    torch = None


class SanTokenizerTests(unittest.TestCase):
    def test_vocabulary_is_exact_stable_and_reconstructable(self) -> None:
        tokenizer = SanTokenizer.fit(
            (("e4", "e5"), ("Nf3", "e5")),
            max_history=3,
        )
        self.assertEqual(
            tokenizer.vocabulary,
            (PAD_TOKEN, UNKNOWN_TOKEN, "Nf3", "e4", "e5"),
        )
        self.assertEqual(tokenizer, SanTokenizer.from_metadata(tokenizer.metadata()))
        self.assertEqual(tokenizer.metadata()["version"], TOKENIZER_VERSION)

    def test_encoding_is_bounded_and_unknown_is_not_a_collision(self) -> None:
        tokenizer = SanTokenizer.fit((("e4", "e5", "Nf3"),), max_history=2)
        tokens, length = tokenizer.encode(("d4", "e4", "Nf3"))
        self.assertEqual(
            tokens,
            (
                tokenizer.vocabulary.index("e4"),
                tokenizer.vocabulary.index("Nf3"),
            ),
        )
        self.assertEqual(length, 2)
        unknown, unknown_length = tokenizer.encode(("secret-never-seen",))
        self.assertEqual(unknown, (1, 0))
        self.assertEqual(unknown_length, 1)
        empty, empty_length = tokenizer.encode(())
        self.assertEqual(empty, (0, 0))
        self.assertEqual(empty_length, 0)

    def test_secret_mutation_cannot_change_public_sequence_encoding(self) -> None:
        first = [row("white", "vegan"), row("black", "truant")]
        second = [dict(value) for value in first]
        for values in (first, second):
            values[0]["historySan"] = ["e4", "e5"]
            values[1]["historySan"] = ["e4", "e5", "Nf3"]
        second[0]["trueDrawback"] = "checkers"
        second[0]["hiddenParameters"] = {"secret": "changed"}
        second[0]["drawbackInternalState"] = {"private": 999}
        original = group_training_examples(first)[0].features.history_san
        mutated = group_training_examples(second)[0].features.history_san
        tokenizer = SanTokenizer.fit((original,), max_history=4)
        self.assertEqual(tokenizer.encode(original), tokenizer.encode(mutated))

    def test_future_moves_are_absent_from_prefix_encoding(self) -> None:
        tokenizer = SanTokenizer.fit(
            (("e4", "e5", "Nf3", "Nc6"),),
            max_history=6,
        )
        prefix = ("e4", "e5")
        encoded_prefix, prefix_length = tokenizer.encode(prefix)
        encoded_complete, _ = tokenizer.encode((*prefix, "Nf3", "Nc6"))
        self.assertEqual(prefix_length, 2)
        self.assertEqual(encoded_prefix[:prefix_length], encoded_complete[:prefix_length])
        self.assertTrue(all(token == 0 for token in encoded_prefix[prefix_length:]))

    def test_right_padding_does_not_change_the_encoded_history(self) -> None:
        vocabulary = (PAD_TOKEN, UNKNOWN_TOKEN, "e4", "e5")
        short = SanTokenizer(vocabulary, max_history=3)
        long = SanTokenizer(vocabulary, max_history=7)
        short_tokens, short_length = short.encode(("e4", "e5"))
        long_tokens, long_length = long.encode(("e4", "e5"))
        self.assertEqual(short_length, long_length)
        self.assertEqual(
            short_tokens[:short_length],
            long_tokens[:long_length],
        )
        self.assertTrue(all(token == 0 for token in short_tokens[short_length:]))
        self.assertTrue(all(token == 0 for token in long_tokens[long_length:]))

    def test_invalid_metadata_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "version"):
            metadata = SanTokenizer(
                (PAD_TOKEN, UNKNOWN_TOKEN), max_history=8
            ).metadata()
            metadata["version"] = 999
            SanTokenizer.from_metadata(metadata)
        with self.assertRaisesRegex(ValueError, "reserved"):
            SanTokenizer(("bad", UNKNOWN_TOKEN), max_history=8)


class SequenceConfigurationTests(unittest.TestCase):
    def test_v1_remains_the_default_control(self) -> None:
        self.assertEqual(TrainingConfig(seed=7).model_variant, "v1")
        config = ModelConfig(
            input_dimension=10,
            drawback_classes=2,
            legal_mask_dimension=20,
        )
        self.assertEqual(config.model_variant, "v1")
        self.assertIsNone(config.san_vocabulary_size)

    def test_v2_requires_a_real_vocabulary(self) -> None:
        with self.assertRaisesRegex(ValueError, "SAN vocabulary"):
            ModelConfig(
                input_dimension=10,
                drawback_classes=2,
                legal_mask_dimension=20,
                model_variant="v2-gru",
            )
        config = ModelConfig(
            input_dimension=10,
            drawback_classes=2,
            legal_mask_dimension=20,
            model_variant="v2-gru",
            san_vocabulary_size=4,
        )
        self.assertEqual(config.sequence_hidden_dimension, 64)

    @unittest.skipUnless(torch is not None, "PyTorch is not installed")
    def test_packed_gru_is_invariant_to_extra_right_padding(self) -> None:
        assert torch is not None
        torch.manual_seed(11)
        config = ModelConfig(
            input_dimension=3,
            drawback_classes=2,
            legal_mask_dimension=4,
            parameter_classes=2,
            hidden_dimension=5,
            model_variant="v2-gru",
            san_vocabulary_size=5,
            san_embedding_dimension=3,
            sequence_hidden_dimension=4,
        )
        model = create_sequence_model(config)
        model.eval()
        board = torch.tensor([[0.1, 0.2, 0.3]])
        short = torch.tensor([[2, 3, 0]])
        padded = torch.tensor([[2, 3, 0, 0, 0]])
        lengths = torch.tensor([2])
        with torch.inference_mode():
            first = model(board, short, lengths)
            second = model(board, padded, lengths)
        for head in first:
            self.assertTrue(torch.equal(first[head], second[head]), head)


if __name__ == "__main__":
    unittest.main()
