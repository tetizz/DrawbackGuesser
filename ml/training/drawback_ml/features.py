"""Deterministic hand-engineered features from public observations only."""

from __future__ import annotations

import re

from .records import FeatureRecord


PIECES = "PNBRQKpnbrqk"
PIECE_INDEX = {piece: index for index, piece in enumerate(PIECES)}
BOARD_FEATURES = 64 * len(PIECES)
MOVE_VOCABULARY_SIZE = 64 * 64 * 5
FEATURE_SCHEMA_VERSION = 1
SCALAR_FEATURES = 2 + 4 + 9 + 2 + 7
FEATURE_DIMENSION = BOARD_FEATURES + SCALAR_FEATURES
MOVE_PATTERN = re.compile(r"^[a-h][1-8][a-h][1-8][nbrq]?$")


def _square_index(square: str) -> int:
    return (int(square[1]) - 1) * 8 + (ord(square[0]) - ord("a"))


def encode_move(move: str) -> int:
    """Map UCI-like move codes to a stable legal-mask index."""

    if not MOVE_PATTERN.fullmatch(move):
        raise ValueError(f"invalid move code: {move}")
    promotion = {"n": 1, "b": 2, "r": 3, "q": 4}.get(move[4:] or "", 0)
    return ((_square_index(move[:2]) * 64) + _square_index(move[2:4])) * 5 + promotion


def _parse_fen(fen: str) -> tuple[list[float], list[float]]:
    fields = fen.split()
    if len(fields) != 6:
        raise ValueError("FEN must contain six fields")
    board, turn, castling, en_passant, halfmove, fullmove = fields
    board_features = [0.0] * BOARD_FEATURES
    ranks = board.split("/")
    if len(ranks) != 8:
        raise ValueError("FEN board must contain eight ranks")
    for fen_rank, contents in enumerate(ranks):
        file_index = 0
        for token in contents:
            if token.isdigit():
                file_index += int(token)
                continue
            piece_index = PIECE_INDEX.get(token)
            if piece_index is None or file_index >= 8:
                raise ValueError("FEN contains an invalid board token")
            board_rank = 7 - fen_rank
            square = board_rank * 8 + file_index
            board_features[piece_index * 64 + square] = 1.0
            file_index += 1
        if file_index != 8:
            raise ValueError("FEN rank does not contain eight squares")
    if turn not in {"w", "b"}:
        raise ValueError("FEN side to move must be w or b")
    try:
        halfmove_value = int(halfmove)
        fullmove_value = int(fullmove)
    except ValueError as error:
        raise ValueError("FEN counters must be integers") from error
    if halfmove_value < 0 or fullmove_value < 1:
        raise ValueError("FEN counters are out of range")
    ep = [0.0] * 9
    if en_passant == "-":
        ep[8] = 1.0
    elif re.fullmatch(r"[a-h][36]", en_passant):
        ep[ord(en_passant[0]) - ord("a")] = 1.0
    else:
        raise ValueError("FEN en-passant square is invalid")
    scalars = [
        1.0 if turn == "w" else 0.0,
        1.0 if turn == "b" else 0.0,
        *(1.0 if right in castling else 0.0 for right in "KQkq"),
        *ep,
        min(halfmove_value, 100) / 100.0,
        min(fullmove_value, 300) / 300.0,
    ]
    return board_features, scalars


def build_feature_vector(record: FeatureRecord) -> tuple[float, ...]:
    """Build features without authoritative rule state or labels."""

    board, fen_scalars = _parse_fen(record.fen_before)
    move_index = encode_move(record.move)
    move_from = move_index // (64 * 5)
    move_to = (move_index // 5) % 64
    scalars = [
        *fen_scalars,
        1.0 if record.player_color == "white" else 0.0,
        min(record.ply, 600) / 600.0,
        min(record.move_number, 300) / 300.0,
        min(len(record.history_san), 600) / 600.0,
        min(len(record.ordinary_legal_moves), 218) / 218.0,
        move_from / 63.0,
        move_to / 63.0,
    ]
    vector = tuple(board + scalars)
    if len(vector) != FEATURE_DIMENSION:
        raise RuntimeError("feature dimension invariant violated")
    return vector
