from __future__ import annotations

import unittest

import _bootstrap  # noqa: F401

from drawback_ml.model import ModelConfig, create_sequence_model
from drawback_ml.records import DatasetSchemaError, parse_dataset_row
from drawback_ml.symbolic import (
    SYMBOLIC_FEATURE_DIMENSION,
    SYMBOLIC_RULE_COUNT,
    SYMBOLIC_RULE_IDS,
    build_symbolic_feature_vector,
    combine_with_symbolic_prior,
    fused_logits_with_symbolic_prior,
)
from drawback_ml.symbolic_schema import SYMBOLIC_FEATURE_VERSION
from test_records import row

try:
    import torch
except ImportError:
    torch = None


def symbolic_row() -> dict[str, object]:
    value = row("white", "vegan")
    value.update(
        {
            "symbolicFeatureVersion": SYMBOLIC_FEATURE_VERSION,
            "symbolicWhiteRuleProbabilities": [1.0]
            + [0.0] * (SYMBOLIC_RULE_COUNT - 1),
            "symbolicBlackRuleProbabilities": [
                1.0 / SYMBOLIC_RULE_COUNT
            ]
            * SYMBOLIC_RULE_COUNT,
            "symbolicWhiteEliminated": [False]
            + [True] * (SYMBOLIC_RULE_COUNT - 1),
            "symbolicBlackEliminated": [False] * SYMBOLIC_RULE_COUNT,
        }
    )
    return value


class SymbolicFeatureTests(unittest.TestCase):
    def test_parses_versioned_public_symbolic_vectors(self) -> None:
        features = parse_dataset_row(symbolic_row()).features
        vector = build_symbolic_feature_vector(features)
        self.assertEqual(len(vector), SYMBOLIC_FEATURE_DIMENSION)
        self.assertEqual(vector[0], 1.0)
        self.assertEqual(vector[-1], 0.0)

    def test_hybrid_fails_closed_on_legacy_or_malformed_rows(self) -> None:
        legacy = parse_dataset_row(row("white", "vegan")).features
        with self.assertRaisesRegex(ValueError, "requires symbolic"):
            build_symbolic_feature_vector(legacy)
        malformed = symbolic_row()
        malformed["symbolicWhiteRuleProbabilities"] = [1.0]
        with self.assertRaisesRegex(DatasetSchemaError, "probabilities"):
            parse_dataset_row(malformed)

    def test_secret_mutation_does_not_change_symbolic_features(self) -> None:
        original = symbolic_row()
        changed = dict(original)
        changed["trueDrawback"] = "checkers"
        changed["hiddenParameters"] = {"secret": "changed"}
        changed["drawbackInternalState"] = {"secret": "changed"}
        first = parse_dataset_row(original).features
        second = parse_dataset_row(changed).features
        self.assertEqual(
            build_symbolic_feature_vector(first),
            build_symbolic_feature_vector(second),
        )

    @unittest.skipUnless(torch is not None, "PyTorch is not installed")
    def test_hybrid_requires_and_uses_symbolic_tensor(self) -> None:
        assert torch is not None
        config = ModelConfig(
            input_dimension=3,
            drawback_classes=2,
            parameter_classes=2,
            legal_mask_dimension=4,
            hidden_dimension=5,
            model_variant="v21-hybrid",
            san_vocabulary_size=5,
            san_embedding_dimension=3,
            sequence_hidden_dimension=4,
            symbolic_dimension=SYMBOLIC_FEATURE_DIMENSION,
            symbolic_hidden_dimension=6,
        )
        model = create_sequence_model(config)
        board = torch.tensor([[0.1, 0.2, 0.3]])
        tokens = torch.tensor([[2, 0]])
        lengths = torch.tensor([1])
        with self.assertRaisesRegex(ValueError, "symbolic inputs"):
            model(board, tokens, lengths)
        output = model(
            board,
            tokens,
            lengths,
            torch.zeros((1, SYMBOLIC_FEATURE_DIMENSION)),
        )
        self.assertEqual(tuple(output["white_drawback"].shape), (1, 2))

    @unittest.skipUnless(torch is not None, "PyTorch is not installed")
    def test_neural_logits_cannot_restore_eliminated_hypothesis(self) -> None:
        assert torch is not None
        features = parse_dataset_row(symbolic_row()).features
        vocabulary = tuple(sorted(SYMBOLIC_RULE_IDS))
        logits = torch.full((1, len(vocabulary)), 1000.0)
        combined = combine_with_symbolic_prior(
            torch, logits, [features], vocabulary, "white"
        )
        vegan_index = vocabulary.index("vegan")
        self.assertTrue(torch.isfinite(combined[0, vegan_index]))
        for index in range(len(vocabulary)):
            if index != vegan_index:
                self.assertTrue(torch.isneginf(combined[0, index]))

    @unittest.skipUnless(torch is not None, "PyTorch is not installed")
    def test_fused_logits_and_mask_reproduce_existing_probabilities(self) -> None:
        assert torch is not None
        features = parse_dataset_row(symbolic_row()).features
        vocabulary = tuple(sorted(SYMBOLIC_RULE_IDS))
        neural = torch.linspace(
            -1.0, 1.0, len(vocabulary)
        ).reshape(1, -1)
        fused, eliminated = fused_logits_with_symbolic_prior(
            torch, neural, [features], vocabulary, "white"
        )
        existing = torch.softmax(
            combine_with_symbolic_prior(
                torch, neural, [features], vocabulary, "white"
            ),
            dim=-1,
        )
        reconstructed = torch.softmax(
            fused.masked_fill(eliminated, float("-inf")),
            dim=-1,
        )
        self.assertTrue(torch.equal(existing, reconstructed))

    @unittest.skipUnless(torch is not None, "PyTorch is not installed")
    def test_training_fails_if_symbolic_engine_eliminates_truth(self) -> None:
        assert torch is not None
        features = parse_dataset_row(symbolic_row()).features
        vocabulary = tuple(sorted(SYMBOLIC_RULE_IDS))
        logits = torch.zeros((1, len(vocabulary)))
        with self.assertRaisesRegex(
            ValueError, "eliminated true white drawback checkers"
        ):
            combine_with_symbolic_prior(
                torch,
                logits,
                [features],
                vocabulary,
                "white",
                true_drawbacks=["checkers"],
            )


if __name__ == "__main__":
    unittest.main()
