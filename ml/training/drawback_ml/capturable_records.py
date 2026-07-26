"""Strict schema-8 records for capturable-king self-play.

This module is deliberately additive. The historical schema-6 release
pipeline remains frozen, while capturable training uses its own exact public
feature vocabulary and never passes labels or provenance to feature builders.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Iterator, Mapping, Sequence

from .features import FEATURE_DIMENSION, build_feature_vector
from .records import FeatureRecord


CAPTURABLE_SYMBOLIC_FEATURE_VERSION = 8
CAPTURABLE_RULE_IDS = (
    "vegan",
    "true-gentleman",
    "false-prophets",
    "trophy-wife",
    "lame-duck",
    "cess",
    "forward-march",
    "checkers",
    "pacman",
    "oddball",
    "even-keeled",
    "truant",
    "spice-of-life",
    "quit-horsing-around",
    "remorseful",
    "battle-fatigue",
    "eye-for-an-eye",
    "barbarian-rage",
    "conscientious-objectors",
    "horse-tranquilizer",
    "femme-fatale",
    "nurturer",
    "triple-play",
    "you-best-not-miss",
    "irresistible",
)
CAPTURABLE_RULE_COUNT = len(CAPTURABLE_RULE_IDS)
CAPTURABLE_RULE_INDEX = {
    rule_id: index for index, rule_id in enumerate(CAPTURABLE_RULE_IDS)
}
CAPTURABLE_BEHAVIOR_FEATURES = 73
CAPTURABLE_FEATURE_DIMENSION = (
    FEATURE_DIMENSION
    + 2 * CAPTURABLE_RULE_COUNT
    + 7
    + CAPTURABLE_BEHAVIOR_FEATURES
)

_UCI_MOVE = re.compile(r"^[a-h][1-8][a-h][1-8][nbrq]?$")
_BOARD_SQUARE = re.compile(r"^[a-h][1-8]$")
_PUBLIC_KEYS = frozenset(
    {
        "authorityId",
        "publicAuthorityPositionBefore",
        "fenBefore",
        "move",
        "moveNumber",
        "ply",
        "playerColor",
        "historySan",
        "ordinaryLegalMoves",
        "clockMs",
        "symbolicFeatureVersion",
        "symbolicWhiteRuleProbabilities",
        "symbolicBlackRuleProbabilities",
        "symbolicWhiteEliminated",
        "symbolicBlackEliminated",
        "publicEvaluatorConstraint",
    }
)
_LABEL_KEYS = frozenset(
    {
        "trueDrawback",
        "hiddenParameters",
        "drawbackInternalState",
        "drawbackLegalMoves",
        "ruleTriggered",
        "forced",
        "result",
    }
)
_EVALUATION_KEYS = frozenset(
    {
        "gameId",
        "seed",
        "san",
        "botAgentId",
        "botStyle",
        "botStrength",
    }
)
_ROW_KEYS = _PUBLIC_KEYS | _LABEL_KEYS | _EVALUATION_KEYS
_SNAPSHOT_KEYS = frozenset(
    {
        "format",
        "version",
        "authorityId",
        "fen",
        "orthodoxCompatible",
        "kingPassant",
        "terminal",
    }
)
_KING_PASSANT_KEYS = frozenset({"victim", "kingSquare", "targets"})
_PIECE_NAMES = ("pawn", "knight", "bishop", "rook", "queen", "king")
_PIECE_FROM_FEN = {
    "p": "pawn",
    "n": "knight",
    "b": "bishop",
    "r": "rook",
    "q": "queen",
    "k": "king",
}
_SAN_PIECE = {
    "N": "knight",
    "B": "bishop",
    "R": "rook",
    "Q": "queen",
    "K": "king",
}


class CapturableDatasetError(ValueError):
    """Raised when schema-8 data fails the public/private contract."""


@dataclass(frozen=True)
class CapturableAuthorityFeature:
    fen: str
    orthodox_compatible: bool
    king_passant_victim: str | None
    king_passant_king_square: str | None
    king_passant_targets: tuple[str, ...]


@dataclass(frozen=True)
class CapturablePublicFeatures:
    """The only values accepted by model feature construction."""

    authority: CapturableAuthorityFeature
    fen_before: str
    move: str
    move_number: int
    ply: int
    player_color: str
    history_san: tuple[str, ...]
    authority_legal_moves: tuple[str, ...]
    symbolic_white_probabilities: tuple[float, ...]
    symbolic_black_probabilities: tuple[float, ...]
    symbolic_white_eliminated: tuple[bool, ...]
    symbolic_black_eliminated: tuple[bool, ...]


@dataclass(frozen=True)
class CapturableLabels:
    true_drawback: str
    hidden_parameters: Mapping[str, Any]
    drawback_internal_state: Any
    drawback_legal_moves: tuple[str, ...]
    rule_triggered: bool
    forced: bool
    result: Any


@dataclass(frozen=True)
class CapturableEvaluation:
    game_id: str
    seed: int
    san: str
    bot_agent_id: str
    bot_style: str | None
    bot_strength: int | None


@dataclass(frozen=True)
class CapturableDatasetRow:
    features: CapturablePublicFeatures
    labels: CapturableLabels
    evaluation: CapturableEvaluation


def _exact_keys(
    value: Mapping[str, Any],
    expected: frozenset[str],
    label: str,
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unknown:
            details.append("unknown " + ", ".join(unknown))
        raise CapturableDatasetError(f"{label} keys are invalid: {'; '.join(details)}")


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CapturableDatasetError(f"{label} must be an object")
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise CapturableDatasetError(f"{label} must be a non-empty string")
    return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise CapturableDatasetError(
            f"{label} must be an integer of at least {minimum}"
        )
    return value


def _string_tuple(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise CapturableDatasetError(f"{label} must be a list of strings")
    return tuple(value)


def _moves(value: Any, label: str) -> tuple[str, ...]:
    moves = _string_tuple(value, label)
    if (
        any(_UCI_MOVE.fullmatch(move) is None for move in moves)
        or len(set(moves)) != len(moves)
    ):
        raise CapturableDatasetError(f"{label} must contain unique UCI moves")
    return moves


def _probabilities(value: Any, label: str) -> tuple[float, ...]:
    if (
        not isinstance(value, list)
        or len(value) != CAPTURABLE_RULE_COUNT
        or any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            or not 0.0 <= float(item) <= 1.0
            for item in value
        )
    ):
        raise CapturableDatasetError(
            f"{label} must contain {CAPTURABLE_RULE_COUNT} probabilities"
        )
    result = tuple(float(item) for item in value)
    total = sum(result)
    if total != 0.0 and abs(total - 1.0) > 1e-6:
        raise CapturableDatasetError(f"{label} must sum to one or zero")
    return result


def _booleans(value: Any, label: str) -> tuple[bool, ...]:
    if (
        not isinstance(value, list)
        or len(value) != CAPTURABLE_RULE_COUNT
        or any(type(item) is not bool for item in value)
    ):
        raise CapturableDatasetError(
            f"{label} must contain {CAPTURABLE_RULE_COUNT} booleans"
        )
    return tuple(value)


def _authority_snapshot(
    value: Any,
    fen_before: str,
) -> CapturableAuthorityFeature:
    snapshot = _mapping(value, "publicAuthorityPositionBefore")
    _exact_keys(snapshot, _SNAPSHOT_KEYS, "publicAuthorityPositionBefore")
    if (
        snapshot.get("format") != "drawbacktrainer-public-position"
        or snapshot.get("version") != 1
        or snapshot.get("authorityId") != "capturable-king/v1"
        or snapshot.get("fen") != fen_before
        or type(snapshot.get("orthodoxCompatible")) is not bool
        or snapshot.get("terminal") is not None
    ):
        raise CapturableDatasetError(
            "publicAuthorityPositionBefore is not a matching non-terminal snapshot"
        )
    king_passant = snapshot.get("kingPassant")
    if king_passant is None:
        return CapturableAuthorityFeature(
            fen=fen_before,
            orthodox_compatible=snapshot["orthodoxCompatible"],
            king_passant_victim=None,
            king_passant_king_square=None,
            king_passant_targets=(),
        )
    right = _mapping(king_passant, "publicAuthorityPositionBefore.kingPassant")
    _exact_keys(
        right,
        _KING_PASSANT_KEYS,
        "publicAuthorityPositionBefore.kingPassant",
    )
    victim = right.get("victim")
    king_square = right.get("kingSquare")
    targets = _string_tuple(
        right.get("targets"),
        "publicAuthorityPositionBefore.kingPassant.targets",
    )
    if (
        victim not in {"white", "black"}
        or not isinstance(king_square, str)
        or _BOARD_SQUARE.fullmatch(king_square) is None
        or not targets
        or any(_BOARD_SQUARE.fullmatch(target) is None for target in targets)
        or len(set(targets)) != len(targets)
    ):
        raise CapturableDatasetError(
            "publicAuthorityPositionBefore.kingPassant is invalid"
        )
    return CapturableAuthorityFeature(
        fen=fen_before,
        orthodox_compatible=snapshot["orthodoxCompatible"],
        king_passant_victim=victim,
        king_passant_king_square=king_square,
        king_passant_targets=targets,
    )


def parse_capturable_dataset_row(row: Mapping[str, Any]) -> CapturableDatasetRow:
    """Parse one combined storage row and separate features before labels."""

    _exact_keys(row, _ROW_KEYS, "capturable dataset row")
    if row.get("authorityId") != "capturable-king/v1":
        raise CapturableDatasetError("authorityId must be capturable-king/v1")
    if row.get("symbolicFeatureVersion") != CAPTURABLE_SYMBOLIC_FEATURE_VERSION:
        raise CapturableDatasetError("symbolicFeatureVersion must be 8")
    if row.get("publicEvaluatorConstraint") is not None:
        raise CapturableDatasetError(
            "capturable schema 8 does not accept evaluator constraints"
        )
    fen_before = _string(row.get("fenBefore"), "fenBefore")
    color = row.get("playerColor")
    if color not in {"white", "black"}:
        raise CapturableDatasetError("playerColor must be white or black")
    fen_parts = fen_before.split()
    if len(fen_parts) != 6 or fen_parts[1] != ("w" if color == "white" else "b"):
        raise CapturableDatasetError("playerColor must match the six-field FEN")
    move = _string(row.get("move"), "move")
    if _UCI_MOVE.fullmatch(move) is None:
        raise CapturableDatasetError("move must be canonical UCI")
    ply = _integer(row.get("ply"), "ply")
    history = _string_tuple(row.get("historySan"), "historySan")
    if len(history) != ply:
        raise CapturableDatasetError("historySan length must equal ply")
    if row.get("clockMs") is not None:
        _integer(row.get("clockMs"), "clockMs")
    authority_moves = _moves(row.get("ordinaryLegalMoves"), "ordinaryLegalMoves")
    if move not in authority_moves:
        raise CapturableDatasetError("observed move must be authority legal")
    drawback_moves = _moves(row.get("drawbackLegalMoves"), "drawbackLegalMoves")
    if not set(drawback_moves).issubset(authority_moves) or move not in drawback_moves:
        raise CapturableDatasetError(
            "drawbackLegalMoves must be an authority-legal subset containing the move"
        )
    triggered = row.get("ruleTriggered")
    forced = row.get("forced")
    if type(triggered) is not bool or type(forced) is not bool:
        raise CapturableDatasetError("ruleTriggered and forced must be booleans")
    if triggered != (len(drawback_moves) != len(authority_moves)):
        raise CapturableDatasetError("ruleTriggered disagrees with exact masks")
    if forced != (len(drawback_moves) == 1):
        raise CapturableDatasetError("forced disagrees with exact mask")

    white_probabilities = _probabilities(
        row.get("symbolicWhiteRuleProbabilities"),
        "symbolicWhiteRuleProbabilities",
    )
    black_probabilities = _probabilities(
        row.get("symbolicBlackRuleProbabilities"),
        "symbolicBlackRuleProbabilities",
    )
    white_eliminated = _booleans(
        row.get("symbolicWhiteEliminated"),
        "symbolicWhiteEliminated",
    )
    black_eliminated = _booleans(
        row.get("symbolicBlackEliminated"),
        "symbolicBlackEliminated",
    )
    true_drawback = _string(row.get("trueDrawback"), "trueDrawback")
    true_index = CAPTURABLE_RULE_INDEX.get(true_drawback)
    if true_index is None:
        raise CapturableDatasetError("trueDrawback is outside the capturable catalog")
    active_eliminated = (
        white_eliminated if color == "white" else black_eliminated
    )
    if active_eliminated[true_index]:
        raise CapturableDatasetError(
            "the exact symbolic engine eliminated the true drawback"
        )
    parameters = _mapping(row.get("hiddenParameters"), "hiddenParameters")
    if true_drawback == "triple-play":
        _exact_keys(
            parameters,
            frozenset({"requiredType"}),
            "hiddenParameters",
        )
        if parameters.get("requiredType") not in {"bishop", "knight"}:
            raise CapturableDatasetError(
                "Triple Play requiredType must be bishop or knight"
            )
    elif parameters:
        raise CapturableDatasetError(
            f"{true_drawback} hiddenParameters must be empty"
        )

    game_id = _string(row.get("gameId"), "gameId")
    seed = _integer(row.get("seed"), "seed")
    if seed > 0xFFFF_FFFF:
        raise CapturableDatasetError("seed must be uint32")
    bot_style = row.get("botStyle")
    if bot_style is not None and (not isinstance(bot_style, str) or not bot_style):
        raise CapturableDatasetError("botStyle must be a non-empty string or null")
    bot_strength = row.get("botStrength")
    if bot_strength is not None:
        _integer(bot_strength, "botStrength")

    features = CapturablePublicFeatures(
        authority=_authority_snapshot(
            row.get("publicAuthorityPositionBefore"),
            fen_before,
        ),
        fen_before=fen_before,
        move=move,
        move_number=_integer(row.get("moveNumber"), "moveNumber", minimum=1),
        ply=ply,
        player_color=color,
        history_san=history,
        authority_legal_moves=authority_moves,
        symbolic_white_probabilities=white_probabilities,
        symbolic_black_probabilities=black_probabilities,
        symbolic_white_eliminated=white_eliminated,
        symbolic_black_eliminated=black_eliminated,
    )
    labels = CapturableLabels(
        true_drawback=true_drawback,
        hidden_parameters=dict(parameters),
        drawback_internal_state=row.get("drawbackInternalState"),
        drawback_legal_moves=drawback_moves,
        rule_triggered=triggered,
        forced=forced,
        result=row.get("result"),
    )
    evaluation = CapturableEvaluation(
        game_id=game_id,
        seed=seed,
        san=_string(row.get("san"), "san"),
        bot_agent_id=_string(row.get("botAgentId"), "botAgentId"),
        bot_style=bot_style,
        bot_strength=bot_strength,
    )
    return CapturableDatasetRow(features, labels, evaluation)


def active_symbolic(
    features: CapturablePublicFeatures,
) -> tuple[tuple[float, ...], tuple[bool, ...]]:
    if features.player_color == "white":
        return (
            features.symbolic_white_probabilities,
            features.symbolic_white_eliminated,
        )
    return (
        features.symbolic_black_probabilities,
        features.symbolic_black_eliminated,
    )


def capturable_feature_vector(
    features: CapturablePublicFeatures,
) -> tuple[float, ...]:
    """Encode public board, move, authority, and symbolic evidence only."""

    legacy_shape = FeatureRecord(
        fen_before=features.fen_before,
        move=features.move,
        move_number=features.move_number,
        ply=features.ply,
        player_color=features.player_color,
        history_san=features.history_san,
        ordinary_legal_moves=features.authority_legal_moves,
        clock_ms=None,
        symbolic_feature_version=None,
        symbolic_white_rule_probabilities=(),
        symbolic_black_rule_probabilities=(),
        symbolic_white_eliminated=(),
        symbolic_black_eliminated=(),
        public_evaluator_constraint=None,
    )
    board_move = build_feature_vector(legacy_shape)
    probabilities, eliminated = active_symbolic(features)
    right = features.authority
    king_square = _square_scalars(right.king_passant_king_square)
    first_target = _square_scalars(
        right.king_passant_targets[0] if right.king_passant_targets else None
    )
    authority = (
        1.0 if right.orthodox_compatible else 0.0,
        1.0 if right.king_passant_victim is not None else 0.0,
        1.0 if right.king_passant_victim == features.player_color else 0.0,
        *king_square,
        *first_target,
    )
    vector = tuple(
        board_move
        + probabilities
        + tuple(1.0 if item else 0.0 for item in eliminated)
        + authority
        + _behavior_features(features)
    )
    if len(vector) != CAPTURABLE_FEATURE_DIMENSION:
        raise RuntimeError("capturable feature dimension invariant violated")
    return vector


def _behavior_features(
    features: CapturablePublicFeatures,
) -> tuple[float, ...]:
    board = _board_roles(features.fen_before)
    mover = board.get(features.move[:2])
    if mover is None:
        raise CapturableDatasetError("observed move source is empty in FEN")
    target = board.get(features.move[2:4])
    current_piece = _one_hot(_PIECE_NAMES.index(mover), len(_PIECE_NAMES))
    captured_piece = _one_hot(
        0 if target is None else _PIECE_NAMES.index(target) + 1,
        len(_PIECE_NAMES) + 1,
    )
    fen_en_passant = features.fen_before.split()[3]
    authority_targets = set(features.authority.king_passant_targets)
    capture = (
        target is not None
        or features.move[2:4] == fen_en_passant
        or features.move[2:4] in authority_targets
    )
    current_flags = (
        1.0 if capture else 0.0,
        1.0 if len(features.move) == 5 else 0.0,
        1.0
        if mover == "king"
        and abs(ord(features.move[0]) - ord(features.move[2])) == 2
        else 0.0,
    )

    legal_piece_counts = [0] * len(_PIECE_NAMES)
    legal_capture_counts = [0] * len(_PIECE_NAMES)
    for move in features.authority_legal_moves:
        piece = board.get(move[:2])
        if piece is None:
            raise CapturableDatasetError(
                "authority move source is empty in FEN"
            )
        index = _PIECE_NAMES.index(piece)
        legal_piece_counts[index] += 1
        destination = move[2:4]
        if (
            board.get(destination) is not None
            or destination == fen_en_passant
            or destination in authority_targets
        ):
            legal_capture_counts[index] += 1
    legal_total = max(1, len(features.authority_legal_moves))
    legal_composition = tuple(
        value / legal_total
        for value in legal_piece_counts + legal_capture_counts
    )

    active_history: list[tuple[int, tuple[float, ...]]] = []
    opponent_history: list[tuple[int, tuple[float, ...]]] = []
    initial_color = (
        features.player_color
        if features.ply % 2 == 0
        else _opposite(features.player_color)
    )
    for index, san in enumerate(features.history_san):
        color = initial_color if index % 2 == 0 else _opposite(initial_color)
        facts = _san_facts(san)
        target_history = (
            active_history
            if color == features.player_color
            else opponent_history
        )
        target_history.append(facts)
    history = (
        _history_summary(active_history)
        + _history_summary(opponent_history)
        + _recent_piece_types(active_history, 2)
        + _recent_piece_types(opponent_history, 2)
        + (_same_piece_streak(active_history),)
    )
    result = current_piece + captured_piece + current_flags + legal_composition + history
    if len(result) != CAPTURABLE_BEHAVIOR_FEATURES:
        raise RuntimeError("capturable behavior feature invariant violated")
    return result


def _board_roles(fen: str) -> dict[str, str]:
    board: dict[str, str] = {}
    ranks = fen.split()[0].split("/")
    if len(ranks) != 8:
        raise CapturableDatasetError("FEN board must contain eight ranks")
    for rank_offset, contents in enumerate(ranks):
        file_index = 0
        rank = 8 - rank_offset
        for token in contents:
            if token.isdigit():
                file_index += int(token)
                continue
            role = _PIECE_FROM_FEN.get(token.lower())
            if role is None or file_index >= 8:
                raise CapturableDatasetError("FEN board token is invalid")
            board[f"{chr(ord('a') + file_index)}{rank}"] = role
            file_index += 1
        if file_index != 8:
            raise CapturableDatasetError("FEN rank must contain eight squares")
    return board


def _san_facts(san: str) -> tuple[int, tuple[float, ...]]:
    piece = (
        "king"
        if san.startswith("O-O")
        else _SAN_PIECE.get(san[:1], "pawn")
    )
    return (
        _PIECE_NAMES.index(piece),
        (
            1.0 if "x" in san else 0.0,
            1.0 if "=" in san else 0.0,
            1.0 if san.startswith("O-O") else 0.0,
            1.0 if "+" in san or "#" in san else 0.0,
        ),
    )


def _history_summary(
    history: Sequence[tuple[int, tuple[float, ...]]],
) -> tuple[float, ...]:
    if not history:
        return (0.0,) * (len(_PIECE_NAMES) + 4)
    piece_counts = [0] * len(_PIECE_NAMES)
    flag_counts = [0.0] * 4
    for piece, flags in history:
        piece_counts[piece] += 1
        for index, value in enumerate(flags):
            flag_counts[index] += value
    denominator = len(history)
    return tuple(
        value / denominator
        for value in piece_counts + flag_counts
    )


def _recent_piece_types(
    history: Sequence[tuple[int, tuple[float, ...]]],
    count: int,
) -> tuple[float, ...]:
    result: tuple[float, ...] = ()
    for offset in range(count):
        index = len(history) - 1 - offset
        result += (
            (0.0,) * len(_PIECE_NAMES)
            if index < 0
            else _one_hot(history[index][0], len(_PIECE_NAMES))
        )
    return result


def _same_piece_streak(
    history: Sequence[tuple[int, tuple[float, ...]]],
) -> float:
    if not history:
        return 0.0
    latest = history[-1][0]
    streak = 0
    for piece, _flags in reversed(history):
        if piece != latest:
            break
        streak += 1
    return streak / len(history)


def _one_hot(index: int, dimension: int) -> tuple[float, ...]:
    return tuple(1.0 if item == index else 0.0 for item in range(dimension))


def _opposite(color: str) -> str:
    return "black" if color == "white" else "white"


def _square_scalars(square: str | None) -> tuple[float, float]:
    if square is None:
        return (0.0, 0.0)
    return (
        (ord(square[0]) - ord("a")) / 7.0,
        (int(square[1]) - 1) / 7.0,
    )


def strict_json_object(line: str, label: str) -> Mapping[str, Any]:
    def pairs(items: Sequence[tuple[str, Any]]) -> Mapping[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise CapturableDatasetError(f"{label} contains duplicate key {key}")
            result[key] = value
        return result

    def reject_constant(token: str) -> Any:
        raise CapturableDatasetError(
            f"{label} contains non-finite number {token}"
        )

    try:
        value = json.loads(
            line,
            object_pairs_hook=pairs,
            parse_constant=reject_constant,
        )
    except (json.JSONDecodeError, UnicodeError) as error:
        raise CapturableDatasetError(f"{label} is invalid JSON") from error
    return _mapping(value, label)


def read_capturable_dataset(path: Path) -> Iterator[CapturableDatasetRow]:
    """Stream strict LF-framed rows without accepting partial final records."""

    with path.resolve().open("rb") as source:
        for line_number, raw_line in enumerate(source, start=1):
            label = f"{path.name}:{line_number}"
            if not raw_line.endswith(b"\n") or raw_line.endswith(b"\r\n"):
                raise CapturableDatasetError(f"{label} must use canonical LF framing")
            try:
                line = raw_line[:-1].decode("utf-8", errors="strict")
            except UnicodeDecodeError as error:
                raise CapturableDatasetError(f"{label} is not UTF-8") from error
            if not line:
                raise CapturableDatasetError(f"{label} is blank")
            yield parse_capturable_dataset_row(strict_json_object(line, label))


def load_capturable_dataset(path: Path) -> tuple[CapturableDatasetRow, ...]:
    rows = tuple(read_capturable_dataset(path))
    if not rows:
        raise CapturableDatasetError("capturable dataset must contain rows")
    return rows


def assert_disjoint_games(
    *splits: Iterable[CapturableDatasetRow],
) -> None:
    seen: set[str] = set()
    for split_index, rows in enumerate(splits):
        games = {row.evaluation.game_id for row in rows}
        overlap = seen & games
        if overlap:
            raise CapturableDatasetError(
                f"dataset split {split_index} overlaps prior games"
            )
        seen.update(games)
