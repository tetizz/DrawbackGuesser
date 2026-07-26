from __future__ import annotations

import unittest

import _bootstrap  # noqa: F401

from drawback_ml.features import FEATURE_DIMENSION, build_feature_vector, encode_move
from drawback_ml.records import (
    DatasetSchemaError,
    group_training_examples,
    parse_dataset_row,
    parse_feature_mapping,
)


def row(color: str, drawback: str) -> dict[str, object]:
    return {
        "gameId": "game-1",
        "seed": 42,
        "fenBefore": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "move": "e2e4",
        "san": "e4",
        "moveNumber": 1,
        "ply": 0 if color == "white" else 1,
        "playerColor": color,
        "historySan": [],
        "trueDrawback": drawback,
        "hiddenParameters": {"secret": "never-an-input"},
        "drawbackInternalState": {"secret": "never-an-input"},
        "ordinaryLegalMoves": ["e2e4", "g1f3"],
        "drawbackLegalMoves": ["e2e4"],
        "ruleTriggered": True,
        "forced": True,
        "clockMs": None,
        "result": {"kind": "active"},
    }


def public_constraint() -> dict[str, object]:
    return {
        "provider": "uci-best-move",
        "policyId": "stockfish-bestmove-v1",
        "positionKey": "public-position-key",
        "requestDigest": "ab" * 32,
        "bestMoveUci": "e2e4",
        "engineFingerprint": "stockfish:17:options",
    }


class RecordBoundaryTests(unittest.TestCase):
    def test_rejects_every_secret_or_label_as_direct_feature_input(self) -> None:
        public = {
            "fenBefore": row("white", "vegan")["fenBefore"],
            "move": "e2e4",
            "moveNumber": 1,
            "ply": 0,
            "playerColor": "white",
            "historySan": [],
            "ordinaryLegalMoves": ["e2e4"],
            "clockMs": None,
        }
        for forbidden in (
            "trueDrawback",
            "hiddenParameters",
            "drawbackInternalState",
            "result",
            "drawbackLegalMoves",
            "ruleTriggered",
            "forced",
        ):
            with self.subTest(forbidden=forbidden):
                with self.assertRaisesRegex(
                    DatasetSchemaError, "cannot be feature inputs"
                ):
                    parse_feature_mapping({**public, forbidden: "leak"})

    def test_parses_labels_separately_and_groups_both_color_heads(self) -> None:
        white = row("white", "vegan")
        black = row("black", "checkers")
        examples = group_training_examples([white, black])
        self.assertEqual(len(examples), 2)
        self.assertEqual(examples[0].white_drawback, "vegan")
        self.assertEqual(examples[0].black_drawback, "checkers")
        self.assertEqual(
            examples[0].white_parameters,
            '{"secret":"never-an-input"}',
        )
        self.assertEqual(
            examples[0].black_parameters,
            '{"secret":"never-an-input"}',
        )
        self.assertNotIn("secret", repr(examples[0].features))
        self.assertEqual(white["hiddenParameters"], {"secret": "never-an-input"})

    def test_secret_values_cannot_change_features(self) -> None:
        first = row("white", "vegan")
        second = dict(first)
        second["trueDrawback"] = "checkers"
        second["hiddenParameters"] = {"different": 99}
        second["drawbackInternalState"] = ["different"]
        second["result"] = {"kind": "checkmate"}
        first_features = parse_dataset_row(first).features
        first_labels = parse_dataset_row(first).labels
        second_parsed = parse_dataset_row(second)
        second_features = second_parsed.features
        self.assertEqual(first_features, second_features)
        self.assertNotEqual(
            first_labels.hidden_parameters,
            second_parsed.labels.hidden_parameters,
        )
        self.assertEqual(
            build_feature_vector(first_features),
            build_feature_vector(second_features),
        )

    def test_agent_and_trigger_scoring_metadata_cannot_change_features(
        self,
    ) -> None:
        first = row("white", "vegan")
        first.update(
            {
                "botAgentId": "random-legal",
                "botStyle": "random",
                "botStrength": 100,
            }
        )
        second = dict(first)
        second.update(
            {
                "botAgentId": "human-like-strong",
                "botStyle": "human-like",
                "botStrength": 2000,
                "ruleTriggered": False,
            }
        )
        first_parsed = parse_dataset_row(first)
        second_parsed = parse_dataset_row(second)
        self.assertEqual(first_parsed.features, second_parsed.features)
        self.assertEqual(
            build_feature_vector(first_parsed.features),
            build_feature_vector(second_parsed.features),
        )
        self.assertNotEqual(
            first_parsed.evaluation,
            second_parsed.evaluation,
        )
        self.assertNotEqual(
            first_parsed.labels.rule_triggered,
            second_parsed.labels.rule_triggered,
        )
        with self.assertRaisesRegex(
            DatasetSchemaError,
            "evaluation-only fields",
        ):
            parse_feature_mapping(
                {
                    key: first.get(key)
                    for key in (
                        "fenBefore",
                        "move",
                        "moveNumber",
                        "ply",
                        "playerColor",
                        "historySan",
                        "ordinaryLegalMoves",
                        "clockMs",
                    )
                }
                | {"botAgentId": "random-legal"}
            )

    def test_raw_seed_remains_in_trusted_replay_labels(self) -> None:
        seeded = row("white", "gambler")
        seeded["hiddenParameters"] = {"seed": 4_294_967_295}
        parsed = parse_dataset_row(seeded)

        self.assertEqual(
            parsed.labels.hidden_parameters,
            '{"seed":4294967295}',
        )
        self.assertNotIn("4294967295", repr(parsed.features))

    def test_parses_the_public_evaluator_constraint_without_label_leakage(
        self,
    ) -> None:
        enriched = row("white", "hand-and-gigabrain")
        enriched["publicEvaluatorConstraint"] = public_constraint()
        parsed = parse_dataset_row(enriched)

        constraint = parsed.features.public_evaluator_constraint
        self.assertIsNotNone(constraint)
        assert constraint is not None
        self.assertEqual(constraint.best_move_uci, "e2e4")
        self.assertEqual(constraint.request_digest, "ab" * 32)
        self.assertNotIn("hand-and-gigabrain", repr(parsed.features))

    def test_rejects_malformed_public_evaluator_constraints(self) -> None:
        invalid_values = [
            "not-an-object",
            {**public_constraint(), "requestDigest": "not-a-digest"},
            {**public_constraint(), "bestMoveUci": "e9e4"},
            {**public_constraint(), "secretDrawback": "vegan"},
        ]
        for value in invalid_values:
            with self.subTest(value=value):
                enriched = row("white", "hand-and-gigabrain")
                enriched["publicEvaluatorConstraint"] = value
                with self.assertRaisesRegex(
                    DatasetSchemaError, "publicEvaluatorConstraint"
                ):
                    parse_dataset_row(enriched)

    def test_requires_both_color_labels_and_consistent_game_labels(self) -> None:
        with self.assertRaisesRegex(DatasetSchemaError, "both colors"):
            group_training_examples([row("white", "vegan")])
        changed = row("white", "checkers")
        with self.assertRaisesRegex(DatasetSchemaError, "inconsistent"):
            group_training_examples(
                [row("white", "vegan"), changed, row("black", "checkers")]
            )

    def test_uses_authenticated_assignment_for_a_one_sided_game(self) -> None:
        examples = group_training_examples(
            [row("white", "vegan")],
            {"game-1": ("vegan", "checkers")},
        )
        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0].white_drawback, "vegan")
        self.assertEqual(examples[0].black_drawback, "checkers")
        self.assertIsNotNone(examples[0].white_parameters)
        self.assertIsNone(examples[0].black_parameters)
        self.assertTrue(examples[0].white_parameters_observed)
        self.assertFalse(examples[0].black_parameters_observed)
        with self.assertRaisesRegex(DatasetSchemaError, "disagree"):
            group_training_examples(
                [row("white", "vegan")],
                {"game-1": ("checkers", "vegan")},
            )

    def test_requires_consistent_parameters_for_each_game_color(self) -> None:
        changed = row("white", "vegan")
        changed["hiddenParameters"] = {"square": "a1"}
        with self.assertRaisesRegex(DatasetSchemaError, "parameters"):
            group_training_examples(
                [row("white", "vegan"), changed, row("black", "checkers")]
            )


class FeatureTests(unittest.TestCase):
    def test_feature_vector_is_fixed_and_deterministic(self) -> None:
        features = parse_dataset_row(row("white", "vegan")).features
        first = build_feature_vector(features)
        self.assertEqual(first, build_feature_vector(features))
        self.assertEqual(len(first), FEATURE_DIMENSION)
        self.assertTrue(all(isinstance(value, float) for value in first))

    def test_move_encoding_is_stable_and_promotion_sensitive(self) -> None:
        self.assertEqual(encode_move("a1a1"), 0)
        self.assertNotEqual(encode_move("a7a8"), encode_move("a7a8q"))
        self.assertNotEqual(encode_move("a7a8n"), encode_move("a7a8q"))
        with self.assertRaisesRegex(ValueError, "invalid move code"):
            encode_move("e9e4")


if __name__ == "__main__":
    unittest.main()
