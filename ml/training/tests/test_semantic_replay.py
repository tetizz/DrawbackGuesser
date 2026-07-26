from __future__ import annotations

from copy import deepcopy
import unittest

import _bootstrap  # noqa: F401

from drawback_ml.semantic_replay import (
    SEMANTIC_REPLAY_PUBLIC_KEYS,
    SemanticReplayError,
    StreamingSemanticReplayVerifier,
)
from test_corpus_contract import AFTER_E4_E5_FEN, row


def public(value: dict[str, object]) -> dict[str, object]:
    return {key: value[key] for key in SEMANTIC_REPLAY_PUBLIC_KEYS}


class StreamingSemanticReplayTests(unittest.TestCase):
    def rows(self) -> list[dict[str, object]]:
        return [
            row(1, "white", "vegan"),
            row(1, "black", "checkers"),
        ]

    def verifier(self, final_fen: str = AFTER_E4_E5_FEN) -> StreamingSemanticReplayVerifier:
        del final_fen
        return StreamingSemanticReplayVerifier(max_plies=80)

    def test_replays_public_game_and_terminal_fen(self) -> None:
        verifier = self.verifier()
        for line_number, item in enumerate(self.rows(), start=1):
            verifier.observe(
                public(item),
                line_number=line_number,
                expected_final_fen=AFTER_E4_E5_FEN,
            )
        verifier.finish(expected_game_count=1)

    def test_rejects_secret_or_label_input(self) -> None:
        item = public(self.rows()[0])
        item["trueDrawback"] = "vegan"
        with self.assertRaisesRegex(SemanticReplayError, "only.*public"):
            self.verifier().observe(
                item, line_number=1, expected_final_fen=AFTER_E4_E5_FEN
            )

    def test_rejects_fen_history_san_and_legal_move_tampering(self) -> None:
        mutations = (
            ("FEN", lambda rows: rows[1].__setitem__("fenBefore", rows[0]["fenBefore"])),
            ("history", lambda rows: rows[1].__setitem__("historySan", [])),
            ("SAN", lambda rows: rows[0].__setitem__("san", "e3")),
            (
                "legal moves",
                lambda rows: rows[0].__setitem__(
                    "ordinaryLegalMoves",
                    list(rows[0]["ordinaryLegalMoves"])[:-1],
                ),
            ),
        )
        for label, mutate in mutations:
            with self.subTest(label=label):
                rows = deepcopy(self.rows())
                mutate(rows)
                verifier = self.verifier()
                with self.assertRaises(SemanticReplayError):
                    for line_number, item in enumerate(rows, start=1):
                        verifier.observe(
                            public(item),
                            line_number=line_number,
                            expected_final_fen=AFTER_E4_E5_FEN,
                        )

    def test_rejects_symbolic_contract_tampering(self) -> None:
        rows = self.rows()
        probabilities = list(rows[0]["symbolicWhiteRuleProbabilities"])
        eliminated = list(rows[0]["symbolicWhiteEliminated"])
        eliminated[0] = True
        rows[0]["symbolicWhiteEliminated"] = eliminated
        rows[0]["symbolicWhiteRuleProbabilities"] = probabilities
        with self.assertRaisesRegex(SemanticReplayError, "hard-eliminated"):
            self.verifier().observe(
                public(rows[0]),
                line_number=1,
                expected_final_fen=AFTER_E4_E5_FEN,
            )

    def test_rejects_replayed_final_fen_tampering(self) -> None:
        verifier = self.verifier()
        for line_number, item in enumerate(self.rows(), start=1):
            verifier.observe(
                public(item),
                line_number=line_number,
                expected_final_fen=str(self.rows()[0]["fenBefore"]),
            )
        with self.assertRaisesRegex(SemanticReplayError, "final FEN"):
            verifier.finish(expected_game_count=1)


if __name__ == "__main__":
    unittest.main()
