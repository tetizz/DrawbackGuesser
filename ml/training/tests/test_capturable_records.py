from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import _bootstrap  # noqa: F401

from capturable_fixture import capturable_row
from drawback_ml.capturable_records import (
    CAPTURABLE_FEATURE_DIMENSION,
    CAPTURABLE_RULE_COUNT,
    CAPTURABLE_RULE_INDEX,
    CapturableDatasetError,
    assert_disjoint_games,
    capturable_feature_vector,
    parse_capturable_dataset_row,
    read_capturable_dataset,
)


class CapturableRecordTests(unittest.TestCase):
    def test_separates_private_labels_before_building_features(self) -> None:
        first = capturable_row(drawback="vegan")
        second = {
            **first,
            "trueDrawback": "lame-duck",
            "drawbackInternalState": {"different": 7},
            "result": {"kind": "drawback-loss"},
        }

        parsed_first = parse_capturable_dataset_row(first)
        parsed_second = parse_capturable_dataset_row(second)

        self.assertEqual(parsed_first.features, parsed_second.features)
        self.assertEqual(
            capturable_feature_vector(parsed_first.features),
            capturable_feature_vector(parsed_second.features),
        )
        self.assertNotEqual(parsed_first.labels, parsed_second.labels)
        self.assertNotIn("label-only", repr(parsed_first.features))

    def test_has_the_declared_exact_feature_dimension(self) -> None:
        parsed = parse_capturable_dataset_row(capturable_row())
        self.assertEqual(
            len(capturable_feature_vector(parsed.features)),
            CAPTURABLE_FEATURE_DIMENSION,
        )

    def test_rejects_legacy_ten_label_symbolic_rows(self) -> None:
        legacy_version = {
            **capturable_row(),
            "symbolicFeatureVersion": 7,
        }
        with self.assertRaisesRegex(
            CapturableDatasetError, "symbolicFeatureVersion must be 8"
        ):
            parse_capturable_dataset_row(legacy_version)

        legacy_vocabulary = {
            **capturable_row(),
            "symbolicWhiteRuleProbabilities": [0.1] * 10,
            "symbolicBlackRuleProbabilities": [0.1] * 10,
            "symbolicWhiteEliminated": [False] * 10,
            "symbolicBlackEliminated": [False] * 10,
        }
        with self.assertRaisesRegex(
            CapturableDatasetError,
            f"must contain {CAPTURABLE_RULE_COUNT} probabilities",
        ):
            parse_capturable_dataset_row(legacy_vocabulary)

    def test_same_length_histories_preserve_public_piece_behavior(self) -> None:
        pawn_history = capturable_row(color="black")
        knight_history = {
            **pawn_history,
            "historySan": ["Nf3"],
        }

        pawn_features = capturable_feature_vector(
            parse_capturable_dataset_row(pawn_history).features
        )
        knight_features = capturable_feature_vector(
            parse_capturable_dataset_row(knight_history).features
        )

        self.assertNotEqual(pawn_features, knight_features)

    def test_rejects_unknown_secret_fields_and_true_elimination(self) -> None:
        secret_snapshot = capturable_row()
        secret_snapshot["publicAuthorityPositionBefore"] = {
            **secret_snapshot["publicAuthorityPositionBefore"],
            "secretDrawback": "vegan",
        }
        with self.assertRaisesRegex(
            CapturableDatasetError, "unknown secretDrawback"
        ):
            parse_capturable_dataset_row(secret_snapshot)

        top_level = {
            **capturable_row(),
            "privateSeed": 123,
        }
        with self.assertRaisesRegex(
            CapturableDatasetError, "unknown privateSeed"
        ):
            parse_capturable_dataset_row(top_level)

        contradicted = capturable_row(eliminated_rule="vegan")
        self.assertTrue(
            contradicted["symbolicWhiteEliminated"][
                CAPTURABLE_RULE_INDEX["vegan"]
            ]
        )
        with self.assertRaisesRegex(
            CapturableDatasetError, "eliminated the true drawback"
        ):
            parse_capturable_dataset_row(contradicted)

    def test_requires_canonical_framing_and_rejects_duplicate_json_keys(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.ndjson"
            encoded = json.dumps(
                capturable_row(),
                allow_nan=False,
                separators=(",", ":"),
            )
            path.write_bytes(encoded.encode("utf-8"))
            with self.assertRaisesRegex(
                CapturableDatasetError, "canonical LF"
            ):
                tuple(read_capturable_dataset(path))

            duplicate = encoded[:-1] + ',"seed":43}\n'
            path.write_bytes(duplicate.encode("utf-8"))
            with self.assertRaisesRegex(
                CapturableDatasetError, "duplicate key seed"
            ):
                tuple(read_capturable_dataset(path))

    def test_requires_disjoint_games_across_splits(self) -> None:
        first = parse_capturable_dataset_row(
            capturable_row(game_id="shared")
        )
        second = parse_capturable_dataset_row(
            capturable_row(game_id="shared", color="black")
        )
        with self.assertRaisesRegex(
            CapturableDatasetError, "overlaps prior games"
        ):
            assert_disjoint_games((first,), (second,))


if __name__ == "__main__":
    unittest.main()
