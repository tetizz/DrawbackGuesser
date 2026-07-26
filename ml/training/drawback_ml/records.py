"""Dataset parsing with an explicit feature/label trust boundary."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable, Mapping

from .parameters import canonical_hidden_parameters
from .symbolic_schema import SYMBOLIC_FEATURE_VERSION, SYMBOLIC_RULE_COUNT


FORBIDDEN_FEATURE_KEYS = frozenset(
    {
        "trueDrawback",
        "hiddenParameters",
        "drawbackInternalState",
        "result",
        "drawbackLegalMoves",
        "ruleTriggered",
        "forced",
    }
)

PUBLIC_FEATURE_KEYS = frozenset(
    {
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
EVALUATION_ONLY_KEYS = frozenset(
    {"botAgentId", "botStyle", "botStrength"}
)


class DatasetSchemaError(ValueError):
    """Raised when a row crosses the feature/label boundary."""


@dataclass(frozen=True)
class PublicEvaluatorConstraint:
    """Public, uniformly collected evaluator observation.

    This is intentionally separate from authoritative drawback state. The
    baseline does not encode it directly; evaluator-aware symbolic features
    may consume the same public fact before export.
    """

    provider: str
    policy_id: str
    position_key: str
    request_digest: str
    best_move_uci: str
    engine_fingerprint: str


@dataclass(frozen=True)
class FeatureRecord:
    fen_before: str
    move: str
    move_number: int
    ply: int
    player_color: str
    history_san: tuple[str, ...]
    ordinary_legal_moves: tuple[str, ...]
    clock_ms: int | None
    symbolic_feature_version: int | None
    symbolic_white_rule_probabilities: tuple[float, ...]
    symbolic_black_rule_probabilities: tuple[float, ...]
    symbolic_white_eliminated: tuple[bool, ...]
    symbolic_black_eliminated: tuple[bool, ...]
    public_evaluator_constraint: PublicEvaluatorConstraint | None


@dataclass(frozen=True)
class LabelRecord:
    player_color: str
    true_drawback: str
    hidden_parameters: str | None
    rule_triggered: bool
    drawback_legal_moves: tuple[str, ...]


@dataclass(frozen=True)
class EvaluationMetadata:
    """Trusted scoring metadata that is never part of ``FeatureRecord``."""

    bot_agent_id: str | None
    bot_style: str | None
    bot_strength: int | None
    agent_metadata_present: bool


@dataclass(frozen=True)
class ParsedRow:
    game_id: str
    seed: int
    features: FeatureRecord
    labels: LabelRecord
    evaluation: EvaluationMetadata


@dataclass(frozen=True)
class TrainingExample:
    game_id: str
    seed: int
    features: FeatureRecord
    white_drawback: str
    black_drawback: str
    white_parameters: str | None
    black_parameters: str | None
    white_parameters_observed: bool
    black_parameters_observed: bool
    rule_triggered: bool
    drawback_legal_moves: tuple[str, ...]
    evaluation: EvaluationMetadata = EvaluationMetadata(
        bot_agent_id=None,
        bot_style=None,
        bot_strength=None,
        agent_metadata_present=False,
    )


def _require(mapping: Mapping[str, Any], key: str, expected: type) -> Any:
    value = mapping.get(key)
    if not isinstance(value, expected):
        raise DatasetSchemaError(f"{key} must be {expected.__name__}")
    return value


def _string_tuple(value: Any, key: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise DatasetSchemaError(f"{key} must be a list of strings")
    return tuple(value)


def _probability_tuple(value: Any, key: str) -> tuple[float, ...]:
    if value is None:
        return ()
    if (
        not isinstance(value, list)
        or len(value) != SYMBOLIC_RULE_COUNT
        or any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not 0.0 <= float(item) <= 1.0
            for item in value
        )
    ):
        raise DatasetSchemaError(
            f"{key} must contain {SYMBOLIC_RULE_COUNT} probabilities"
        )
    probabilities = tuple(float(item) for item in value)
    total = sum(probabilities)
    if not (abs(total - 1.0) <= 1e-6 or total == 0.0):
        raise DatasetSchemaError(f"{key} probabilities must sum to one or zero")
    return probabilities


def _boolean_tuple(value: Any, key: str) -> tuple[bool, ...]:
    if value is None:
        return ()
    if (
        not isinstance(value, list)
        or len(value) != SYMBOLIC_RULE_COUNT
        or any(not isinstance(item, bool) for item in value)
    ):
        raise DatasetSchemaError(
            f"{key} must contain {SYMBOLIC_RULE_COUNT} booleans"
        )
    return tuple(value)


def _public_evaluator_constraint(
    value: Any,
) -> PublicEvaluatorConstraint | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise DatasetSchemaError(
            "publicEvaluatorConstraint must be an object or null"
        )
    expected = {
        "provider",
        "policyId",
        "positionKey",
        "requestDigest",
        "bestMoveUci",
        "engineFingerprint",
    }
    unknown = set(value).difference(expected)
    missing = expected.difference(value)
    if unknown or missing:
        details = []
        if missing:
            details.append(f"missing {', '.join(sorted(missing))}")
        if unknown:
            details.append(f"unknown {', '.join(sorted(unknown))}")
        raise DatasetSchemaError(
            "invalid publicEvaluatorConstraint fields: " + "; ".join(details)
        )

    def non_empty_string(key: str) -> str:
        item = value.get(key)
        if not isinstance(item, str) or not item:
            raise DatasetSchemaError(
                f"publicEvaluatorConstraint.{key} must be a non-empty string"
            )
        return item

    provider = non_empty_string("provider")
    if provider != "uci-best-move":
        raise DatasetSchemaError(
            "publicEvaluatorConstraint.provider must be uci-best-move"
        )
    best_move = non_empty_string("bestMoveUci")
    if re.fullmatch(r"[a-h][1-8][a-h][1-8][nbrq]?", best_move) is None:
        raise DatasetSchemaError(
            "publicEvaluatorConstraint.bestMoveUci must be a UCI move"
        )
    request_digest = non_empty_string("requestDigest")
    if re.fullmatch(r"[0-9a-f]{64}", request_digest) is None:
        raise DatasetSchemaError(
            "publicEvaluatorConstraint.requestDigest must be a lowercase SHA-256 digest"
        )
    return PublicEvaluatorConstraint(
        provider=provider,
        policy_id=non_empty_string("policyId"),
        position_key=non_empty_string("positionKey"),
        request_digest=request_digest,
        best_move_uci=best_move,
        engine_fingerprint=non_empty_string("engineFingerprint"),
    )


def parse_feature_mapping(mapping: Mapping[str, Any]) -> FeatureRecord:
    """Parse only a pre-separated public feature mapping.

    Calling this function with labels or authoritative engine state is an error,
    even if those values would otherwise be ignored.
    """

    leaked = FORBIDDEN_FEATURE_KEYS.intersection(mapping)
    if leaked:
        names = ", ".join(sorted(leaked))
        raise DatasetSchemaError(f"secret or label fields cannot be feature inputs: {names}")
    evaluation_only = EVALUATION_ONLY_KEYS.intersection(mapping)
    if evaluation_only:
        names = ", ".join(sorted(evaluation_only))
        raise DatasetSchemaError(
            f"evaluation-only fields cannot be feature inputs: {names}"
        )
    unknown = set(mapping).difference(PUBLIC_FEATURE_KEYS)
    if unknown:
        raise DatasetSchemaError(f"unknown feature fields: {', '.join(sorted(unknown))}")
    color = _require(mapping, "playerColor", str)
    if color not in {"white", "black"}:
        raise DatasetSchemaError("playerColor must be white or black")
    clock = mapping.get("clockMs")
    if clock is not None and (not isinstance(clock, int) or isinstance(clock, bool) or clock < 0):
        raise DatasetSchemaError("clockMs must be a non-negative integer or null")
    symbolic_version = mapping.get("symbolicFeatureVersion")
    if (
        symbolic_version is not None
        and symbolic_version != SYMBOLIC_FEATURE_VERSION
    ):
        raise DatasetSchemaError(
            f"symbolicFeatureVersion must be {SYMBOLIC_FEATURE_VERSION} or null"
        )
    white_probabilities = _probability_tuple(
        mapping.get("symbolicWhiteRuleProbabilities"),
        "symbolicWhiteRuleProbabilities",
    )
    black_probabilities = _probability_tuple(
        mapping.get("symbolicBlackRuleProbabilities"),
        "symbolicBlackRuleProbabilities",
    )
    white_eliminated = _boolean_tuple(
        mapping.get("symbolicWhiteEliminated"), "symbolicWhiteEliminated"
    )
    black_eliminated = _boolean_tuple(
        mapping.get("symbolicBlackEliminated"), "symbolicBlackEliminated"
    )
    symbolic_lengths = {
        len(white_probabilities),
        len(black_probabilities),
        len(white_eliminated),
        len(black_eliminated),
    }
    if symbolic_version is None:
        if symbolic_lengths != {0}:
            raise DatasetSchemaError(
                "symbolic arrays require symbolicFeatureVersion"
            )
    elif symbolic_lengths != {SYMBOLIC_RULE_COUNT}:
        raise DatasetSchemaError(
            f"symbolic feature version {SYMBOLIC_FEATURE_VERSION} "
            "requires all four arrays"
        )
    return FeatureRecord(
        fen_before=_require(mapping, "fenBefore", str),
        move=_require(mapping, "move", str),
        move_number=_require_int(mapping, "moveNumber"),
        ply=_require_int(mapping, "ply"),
        player_color=color,
        history_san=_string_tuple(mapping.get("historySan"), "historySan"),
        ordinary_legal_moves=_string_tuple(
            mapping.get("ordinaryLegalMoves"), "ordinaryLegalMoves"
        ),
        clock_ms=clock,
        symbolic_feature_version=symbolic_version,
        symbolic_white_rule_probabilities=white_probabilities,
        symbolic_black_rule_probabilities=black_probabilities,
        symbolic_white_eliminated=white_eliminated,
        symbolic_black_eliminated=black_eliminated,
        public_evaluator_constraint=_public_evaluator_constraint(
            mapping.get("publicEvaluatorConstraint")
        ),
    )


def _require_int(mapping: Mapping[str, Any], key: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise DatasetSchemaError(f"{key} must be a non-negative integer")
    return value


def parse_dataset_row(row: Mapping[str, Any]) -> ParsedRow:
    """Separate public features from labels before feature parsing."""

    feature_mapping = {key: row.get(key) for key in PUBLIC_FEATURE_KEYS}
    features = parse_feature_mapping(feature_mapping)
    color = features.player_color
    drawback = _require(row, "trueDrawback", str)
    if "hiddenParameters" not in row:
        raise DatasetSchemaError("missing required dataset field: hiddenParameters")
    triggered = row.get("ruleTriggered")
    if not isinstance(triggered, bool):
        raise DatasetSchemaError("ruleTriggered must be boolean")
    labels = LabelRecord(
        player_color=color,
        true_drawback=drawback,
        hidden_parameters=canonical_hidden_parameters(row.get("hiddenParameters")),
        rule_triggered=triggered,
        drawback_legal_moves=_string_tuple(
            row.get("drawbackLegalMoves"), "drawbackLegalMoves"
        ),
    )
    agent_fields = ("botAgentId", "botStyle", "botStrength")
    present = tuple(field in row for field in agent_fields)
    if any(present) and not all(present):
        raise DatasetSchemaError(
            "botAgentId, botStyle, and botStrength must appear together"
        )
    if all(present):
        agent_id = row.get("botAgentId")
        style = row.get("botStyle")
        strength = row.get("botStrength")
        if not isinstance(agent_id, str) or not agent_id:
            raise DatasetSchemaError("botAgentId must be a non-empty string")
        if style is not None and (
            not isinstance(style, str) or not style
        ):
            raise DatasetSchemaError(
                "botStyle must be a non-empty string or null"
            )
        if strength is not None and (
            isinstance(strength, bool)
            or not isinstance(strength, int)
            or strength < 0
        ):
            raise DatasetSchemaError(
                "botStrength must be a non-negative integer or null"
            )
        evaluation = EvaluationMetadata(agent_id, style, strength, True)
    else:
        evaluation = EvaluationMetadata(None, None, None, False)
    # Authoritative internal state and outcome are intentionally not labels.
    for key in ("drawbackInternalState", "result"):
        if key not in row:
            raise DatasetSchemaError(f"missing required dataset field: {key}")
    return ParsedRow(
        game_id=_require(row, "gameId", str),
        seed=_require_int(row, "seed"),
        features=features,
        labels=labels,
        evaluation=evaluation,
    )


def group_training_examples(
    rows: Iterable[Mapping[str, Any]],
    game_assignments: Mapping[str, tuple[str, str]] | None = None,
) -> list[TrainingExample]:
    """Attach both game-level drawback labels to each move without input leakage."""

    parsed = [parse_dataset_row(row) for row in rows]
    labels_by_game: dict[str, dict[str, tuple[str, str | None]]] = {}
    for row in parsed:
        color_labels = labels_by_game.setdefault(row.game_id, {})
        current = (row.labels.true_drawback, row.labels.hidden_parameters)
        previous = color_labels.get(row.labels.player_color)
        if previous is not None and previous != current:
            raise DatasetSchemaError(
                f"inconsistent {row.labels.player_color} drawback or parameters "
                f"in {row.game_id}"
            )
        color_labels[row.labels.player_color] = current

    examples: list[TrainingExample] = []
    for row in parsed:
        game_labels = labels_by_game[row.game_id]
        if game_assignments is None:
            if "white" not in game_labels or "black" not in game_labels:
                raise DatasetSchemaError(
                    f"game {row.game_id} must contain labeled moves by both colors"
                )
            white_drawback, white_parameters = game_labels["white"]
            black_drawback, black_parameters = game_labels["black"]
            white_parameters_observed = True
            black_parameters_observed = True
        else:
            assignment = game_assignments.get(row.game_id)
            if assignment is None:
                raise DatasetSchemaError(
                    f"game {row.game_id} has no authenticated assignment"
                )
            white_drawback, black_drawback = assignment
            observed_white = game_labels.get("white")
            observed_black = game_labels.get("black")
            if (
                observed_white is not None
                and observed_white[0] != white_drawback
            ) or (
                observed_black is not None
                and observed_black[0] != black_drawback
            ):
                raise DatasetSchemaError(
                    f"game {row.game_id} labels disagree with its assignment"
                )
            white_parameters = (
                None if observed_white is None else observed_white[1]
            )
            black_parameters = (
                None if observed_black is None else observed_black[1]
            )
            white_parameters_observed = observed_white is not None
            black_parameters_observed = observed_black is not None
        examples.append(
            TrainingExample(
                game_id=row.game_id,
                seed=row.seed,
                features=row.features,
                white_drawback=white_drawback,
                black_drawback=black_drawback,
                white_parameters=white_parameters,
                black_parameters=black_parameters,
                white_parameters_observed=white_parameters_observed,
                black_parameters_observed=black_parameters_observed,
                rule_triggered=row.labels.rule_triggered,
                drawback_legal_moves=row.labels.drawback_legal_moves,
                evaluation=row.evaluation,
            )
        )
    return examples
