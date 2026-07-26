from __future__ import annotations

from collections import Counter
from dataclasses import FrozenInstanceError
import hashlib
import unittest

import _bootstrap  # noqa: F401

from drawback_ml.capturable_fixed_schedule import (
    ENGINE_REVISION,
    ENGINE_RULE_IDS,
    EXPECTED_BLACK_RULE_ORDER,
    EXPECTED_FIXED_ASSIGNMENTS,
    EXPECTED_FIXED_ASSIGNMENTS_BY_GAME_ID,
    EXPECTED_WHITE_RULE_ORDER,
    FIXED_GAME_COUNT,
    SCHEDULE_SHA256,
    FixedScheduleAssignment,
    _schedule_payload,
)


class CapturableFixedScheduleTests(unittest.TestCase):
    def test_matches_pinned_engine_permutations_and_sentinels(self) -> None:
        self.assertEqual(
            ENGINE_REVISION,
            "74eb6fc95571994bd96b7a351278f3f74f0972e3",
        )
        self.assertEqual(len(ENGINE_RULE_IDS), 25)
        self.assertEqual(
            EXPECTED_WHITE_RULE_ORDER[:3],
            ("forward-march", "checkers", "truant"),
        )
        self.assertEqual(
            EXPECTED_BLACK_RULE_ORDER[:3],
            ("trophy-wife", "triple-play", "checkers"),
        )
        self.assertEqual(
            EXPECTED_FIXED_ASSIGNMENTS[0],
            FixedScheduleAssignment(
                game_index=0,
                game_id=(
                    "player-private-v1-35c406ed-000000-"
                    "24debaee-05392f4a"
                ),
                gameplay_seed=902_039_277,
                white_parameter_seed=618_576_622,
                black_parameter_seed=87_633_738,
                white_rule_id="forward-march",
                black_rule_id="trophy-wife",
            ),
        )
        self.assertEqual(
            EXPECTED_FIXED_ASSIGNMENTS[-1],
            FixedScheduleAssignment(
                game_index=624,
                game_id=(
                    "player-private-v1-88e07315-000624-"
                    "429fc786-067b4919"
                ),
                gameplay_seed=2_296_410_901,
                white_parameter_seed=1_117_767_558,
                black_parameter_seed=108_742_937,
                white_rule_id="even-keeled",
                black_rule_id="barbarian-rage",
            ),
        )

    def test_exact_vector_is_balanced_unique_and_content_addressed(self) -> None:
        assignments = EXPECTED_FIXED_ASSIGNMENTS
        self.assertEqual(len(assignments), FIXED_GAME_COUNT)
        self.assertEqual(
            [assignment.game_index for assignment in assignments],
            list(range(FIXED_GAME_COUNT)),
        )
        for field in (
            "game_id",
            "gameplay_seed",
            "white_parameter_seed",
            "black_parameter_seed",
        ):
            self.assertEqual(
                len(
                    {
                        getattr(assignment, field)
                        for assignment in assignments
                    }
                ),
                FIXED_GAME_COUNT,
            )
        pairs = Counter(
            (assignment.white_rule_id, assignment.black_rule_id)
            for assignment in assignments
        )
        self.assertEqual(len(pairs), 625)
        self.assertEqual(set(pairs.values()), {1})
        white = Counter(
            assignment.white_rule_id for assignment in assignments
        )
        black = Counter(
            assignment.black_rule_id for assignment in assignments
        )
        self.assertEqual(set(white), set(ENGINE_RULE_IDS))
        self.assertEqual(set(black), set(ENGINE_RULE_IDS))
        self.assertEqual(set(white.values()), {25})
        self.assertEqual(set(black.values()), {25})
        self.assertEqual(
            hashlib.sha256(_schedule_payload(assignments)).hexdigest(),
            SCHEDULE_SHA256,
        )

    def test_latin_round_and_game_id_vectors_match_engine(self) -> None:
        vectors = {
            25: (
                "forward-march",
                "triple-play",
                "player-private-v1-2109313d-000025-"
                "3bde11c6-e9879d99",
            ),
            26: (
                "checkers",
                "checkers",
                "player-private-v1-b2f5834b-000026-"
                "6e4adab3-0f14ee69",
            ),
            50: (
                "forward-march",
                "checkers",
                "player-private-v1-0df67435-000050-"
                "9735ba1e-b649ffa2",
            ),
        }
        for index, expected in vectors.items():
            assignment = EXPECTED_FIXED_ASSIGNMENTS[index]
            self.assertEqual(
                (
                    assignment.white_rule_id,
                    assignment.black_rule_id,
                    assignment.game_id,
                ),
                expected,
            )
            self.assertIs(
                EXPECTED_FIXED_ASSIGNMENTS_BY_GAME_ID[
                    assignment.game_id
                ],
                assignment,
            )

    def test_exports_are_immutable(self) -> None:
        with self.assertRaises(FrozenInstanceError):
            EXPECTED_FIXED_ASSIGNMENTS[0].gameplay_seed = 0  # type: ignore[misc]
        with self.assertRaises(TypeError):
            mapping = EXPECTED_FIXED_ASSIGNMENTS_BY_GAME_ID
            mapping["replacement"] = (  # type: ignore[index]
                EXPECTED_FIXED_ASSIGNMENTS[0]
            )


if __name__ == "__main__":
    unittest.main()
