"""Streaming, label-blind semantic verification of public chess observations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any

try:
    import chess
except ImportError as error:  # pragma: no cover - exercised by deployment failures
    raise RuntimeError(
        "semantic corpus replay requires chess>=1.11,<2; "
        "install ml/requirements.txt"
    ) from error

from .symbolic_schema import SYMBOLIC_RULE_COUNT

SEMANTIC_REPLAY_PUBLIC_KEYS = frozenset(
    {
        "gameId",
        "fenBefore",
        "move",
        "san",
        "moveNumber",
        "ply",
        "playerColor",
        "historySan",
        "ordinaryLegalMoves",
        "symbolicWhiteRuleProbabilities",
        "symbolicBlackRuleProbabilities",
        "symbolicWhiteEliminated",
        "symbolicBlackEliminated",
    }
)


class SemanticReplayError(ValueError):
    """Raised when public corpus observations cannot be independently replayed."""


def _uci_moves(board: chess.Board) -> tuple[str, ...]:
    return tuple(sorted(move.uci() for move in board.legal_moves))


def _exact_string_list(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise SemanticReplayError(f"{label} must be a list of strings")
    return tuple(value)


def _exact_bool_list(value: object, label: str) -> tuple[bool, ...]:
    if (
        not isinstance(value, list)
        or len(value) != SYMBOLIC_RULE_COUNT
        or any(not isinstance(item, bool) for item in value)
    ):
        raise SemanticReplayError(
            f"{label} must contain {SYMBOLIC_RULE_COUNT} booleans"
        )
    return tuple(value)


def _probabilities(value: object, label: str) -> tuple[float, ...]:
    if (
        not isinstance(value, list)
        or len(value) != SYMBOLIC_RULE_COUNT
        or any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            or float(item) < 0.0
            or float(item) > 1.0
            for item in value
        )
    ):
        raise SemanticReplayError(
            f"{label} must contain {SYMBOLIC_RULE_COUNT} finite probabilities"
        )
    result = tuple(float(item) for item in value)
    if not math.isclose(sum(result), 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise SemanticReplayError(f"{label} must sum to one")
    return result


def _verify_symbolic_head(
    probabilities: tuple[float, ...],
    eliminated: tuple[bool, ...],
    label: str,
) -> None:
    for index, is_eliminated in enumerate(eliminated):
        if is_eliminated and probabilities[index] != 0.0:
            raise SemanticReplayError(
                f"{label} gives probability to a hard-eliminated hypothesis"
            )
    if all(eliminated):
        raise SemanticReplayError(f"{label} hard-eliminates every hypothesis")


@dataclass
class _GameReplay:
    game_id: str
    board: chess.Board
    history_san: list[str]
    next_ply: int
    expected_final_fen: str
    white_probabilities: tuple[float, ...] | None = None
    black_probabilities: tuple[float, ...] | None = None
    white_eliminated: tuple[bool, ...] | None = None
    black_eliminated: tuple[bool, ...] | None = None


class StreamingSemanticReplayVerifier:
    """Replay one canonically ordered game at a time from public NDJSON fields.

    Secret drawback fields, labels, authoritative rule state, and terminal
    results are deliberately not inputs to :meth:`observe`. Callers must pass
    exactly :data:`SEMANTIC_REPLAY_PUBLIC_KEYS`. The verifier keeps
    only the current board, at most ``max_plies`` SAN strings, and four fixed
    182-element symbolic vectors.
    """

    def __init__(
        self,
        *,
        max_plies: int,
    ) -> None:
        if not isinstance(max_plies, int) or isinstance(max_plies, bool) or max_plies <= 0:
            raise ValueError("max_plies must be a positive integer")
        self._max_plies = max_plies
        self._current: _GameReplay | None = None
        self._sealed_games = 0

    def observe(
        self,
        row: Mapping[str, Any],
        *,
        line_number: int,
        expected_final_fen: str,
    ) -> None:
        if set(row) != SEMANTIC_REPLAY_PUBLIC_KEYS:
            raise SemanticReplayError(
                "semantic replay input must contain only its exact public fields"
            )
        if not isinstance(expected_final_fen, str) or not expected_final_fen:
            raise SemanticReplayError(
                "expected_final_fen must be a non-empty string"
            )
        game_id = row.get("gameId")
        if not isinstance(game_id, str) or not game_id:
            raise SemanticReplayError("gameId must be a non-empty string")
        ply = row.get("ply")
        if not isinstance(ply, int) or isinstance(ply, bool) or ply < 0:
            raise SemanticReplayError("ply must be a non-negative integer")
        if ply >= self._max_plies:
            raise SemanticReplayError("row ply exceeds the configured replay bound")

        if self._current is None or self._current.game_id != game_id:
            self._seal_current()
            if ply != 0:
                raise SemanticReplayError("a replayed game must begin at ply zero")
            fen = row.get("fenBefore")
            if not isinstance(fen, str):
                raise SemanticReplayError("fenBefore must be a string")
            try:
                board = chess.Board(fen)
            except ValueError as error:
                raise SemanticReplayError("initial fenBefore is not legal FEN") from error
            if not board.is_valid():
                raise SemanticReplayError("initial fenBefore is not a valid chess position")
            if board.fen(en_passant="legal") != chess.STARTING_FEN:
                raise SemanticReplayError(
                    "a corpus game must begin from the standard initial position"
                )
            self._current = _GameReplay(
                game_id, board, [], 0, expected_final_fen
            )

        replay = self._current
        assert replay is not None
        prefix = f"line {line_number}, game {game_id}, ply {ply}"
        if replay.expected_final_fen != expected_final_fen:
            raise SemanticReplayError(
                f"{prefix}: outcome-ledger final FEN changed within one game"
            )
        if ply != replay.next_ply:
            raise SemanticReplayError(f"{prefix}: non-consecutive replay ply")
        expected_color = "white" if replay.board.turn == chess.WHITE else "black"
        if row.get("playerColor") != expected_color:
            raise SemanticReplayError(f"{prefix}: playerColor disagrees with position")
        if row.get("moveNumber") != replay.board.fullmove_number:
            raise SemanticReplayError(f"{prefix}: moveNumber disagrees with position")
        if row.get("fenBefore") != replay.board.fen(en_passant="legal"):
            raise SemanticReplayError(f"{prefix}: fenBefore disagrees with replay")

        history = _exact_string_list(row.get("historySan"), "historySan")
        if history != tuple(replay.history_san):
            raise SemanticReplayError(f"{prefix}: historySan disagrees with replay")
        ordinary = _exact_string_list(
            row.get("ordinaryLegalMoves"), "ordinaryLegalMoves"
        )
        if len(set(ordinary)) != len(ordinary):
            raise SemanticReplayError(f"{prefix}: ordinaryLegalMoves contains duplicates")
        if tuple(sorted(ordinary)) != _uci_moves(replay.board):
            raise SemanticReplayError(
                f"{prefix}: ordinaryLegalMoves disagrees with standard chess"
            )

        move_code = row.get("move")
        if not isinstance(move_code, str):
            raise SemanticReplayError("move must be a string")
        try:
            move = chess.Move.from_uci(move_code)
        except ValueError as error:
            raise SemanticReplayError(f"{prefix}: move is not valid UCI") from error
        if move not in replay.board.legal_moves:
            raise SemanticReplayError(f"{prefix}: move is not standard-chess legal")
        expected_san = replay.board.san(move)
        if row.get("san") != expected_san:
            raise SemanticReplayError(f"{prefix}: SAN disagrees with replayed move")

        white_probabilities = _probabilities(
            row.get("symbolicWhiteRuleProbabilities"),
            "symbolicWhiteRuleProbabilities",
        )
        black_probabilities = _probabilities(
            row.get("symbolicBlackRuleProbabilities"),
            "symbolicBlackRuleProbabilities",
        )
        white_eliminated = _exact_bool_list(
            row.get("symbolicWhiteEliminated"), "symbolicWhiteEliminated"
        )
        black_eliminated = _exact_bool_list(
            row.get("symbolicBlackEliminated"), "symbolicBlackEliminated"
        )
        _verify_symbolic_head(white_probabilities, white_eliminated, "white symbolic")
        _verify_symbolic_head(black_probabilities, black_eliminated, "black symbolic")

        if expected_color == "white":
            self._verify_unchanged_opponent(
                replay.black_probabilities,
                replay.black_eliminated,
                black_probabilities,
                black_eliminated,
                prefix,
            )
        else:
            self._verify_unchanged_opponent(
                replay.white_probabilities,
                replay.white_eliminated,
                white_probabilities,
                white_eliminated,
                prefix,
            )
        self._verify_eliminations_monotone(
            replay.white_eliminated, white_eliminated, "white", prefix
        )
        self._verify_eliminations_monotone(
            replay.black_eliminated, black_eliminated, "black", prefix
        )

        replay.white_probabilities = white_probabilities
        replay.black_probabilities = black_probabilities
        replay.white_eliminated = white_eliminated
        replay.black_eliminated = black_eliminated
        replay.board.push(move)
        replay.history_san.append(expected_san)
        replay.next_ply += 1

    @staticmethod
    def _verify_unchanged_opponent(
        previous_probabilities: tuple[float, ...] | None,
        previous_eliminated: tuple[bool, ...] | None,
        current_probabilities: tuple[float, ...],
        current_eliminated: tuple[bool, ...],
        prefix: str,
    ) -> None:
        if previous_probabilities is None:
            return
        if (
            current_probabilities != previous_probabilities
            or current_eliminated != previous_eliminated
        ):
            raise SemanticReplayError(
                f"{prefix}: observing one player changed the opponent symbolic head"
            )

    @staticmethod
    def _verify_eliminations_monotone(
        previous: tuple[bool, ...] | None,
        current: tuple[bool, ...],
        color: str,
        prefix: str,
    ) -> None:
        if previous is not None and any(
            was_eliminated and not is_eliminated
            for was_eliminated, is_eliminated in zip(previous, current, strict=True)
        ):
            raise SemanticReplayError(
                f"{prefix}: {color} symbolic elimination was reversed"
            )

    def _seal_current(self) -> None:
        if self._current is None:
            return
        actual = self._current.board.fen(en_passant="legal")
        expected = self._current.expected_final_fen
        if expected != actual:
            raise SemanticReplayError(
                f"game {self._current.game_id}: replayed final FEN disagrees "
                "with outcome ledger"
            )
        self._sealed_games += 1
        self._current = None

    def finish(self, *, expected_game_count: int) -> None:
        if (
            not isinstance(expected_game_count, int)
            or isinstance(expected_game_count, bool)
            or expected_game_count < 0
        ):
            raise ValueError("expected_game_count must be a non-negative integer")
        self._seal_current()
        if self._sealed_games != expected_game_count:
            raise SemanticReplayError(
                "semantic replay games disagree with the outcome ledger"
            )
