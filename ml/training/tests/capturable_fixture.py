from __future__ import annotations

from typing import Any

from drawback_ml.capturable_records import CAPTURABLE_RULE_IDS


def capturable_row(
    *,
    game_id: str = "capturable-game-1",
    color: str = "white",
    drawback: str = "vegan",
    eliminated_rule: str | None = None,
) -> dict[str, Any]:
    if color == "white":
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        move = "e2e4"
        san = "e4"
        ply = 0
        history: list[str] = []
    else:
        fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
        move = "e7e5"
        san = "e5"
        ply = 1
        history = ["e4"]
    eliminated = [
        rule_id == eliminated_rule for rule_id in CAPTURABLE_RULE_IDS
    ]
    survivor_count = len(CAPTURABLE_RULE_IDS) - sum(eliminated)
    probabilities = [
        0.0 if is_eliminated else 1.0 / survivor_count
        for is_eliminated in eliminated
    ]
    parameters: dict[str, Any] = (
        {"requiredType": "bishop"} if drawback == "triple-play" else {}
    )
    return {
        "authorityId": "capturable-king/v1",
        "publicAuthorityPositionBefore": {
            "format": "drawbacktrainer-public-position",
            "version": 1,
            "authorityId": "capturable-king/v1",
            "fen": fen,
            "orthodoxCompatible": True,
            "kingPassant": None,
            "terminal": None,
        },
        "fenBefore": fen,
        "move": move,
        "moveNumber": 1,
        "ply": ply,
        "playerColor": color,
        "historySan": history,
        "ordinaryLegalMoves": [move, "g1f3" if color == "white" else "g8f6"],
        "clockMs": None,
        "symbolicFeatureVersion": 7,
        "symbolicWhiteRuleProbabilities": probabilities,
        "symbolicBlackRuleProbabilities": probabilities,
        "symbolicWhiteEliminated": eliminated,
        "symbolicBlackEliminated": eliminated,
        "publicEvaluatorConstraint": None,
        "trueDrawback": drawback,
        "hiddenParameters": parameters,
        "drawbackInternalState": {"private": "label-only"},
        "drawbackLegalMoves": [move],
        "ruleTriggered": True,
        "forced": True,
        "result": {"kind": "active"},
        "gameId": game_id,
        "seed": 42,
        "san": san,
        "botAgentId": "material-player-private-corpus/v1",
        "botStyle": "drawback-search",
        "botStrength": None,
    }
