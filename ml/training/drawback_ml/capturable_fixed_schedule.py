"""Frozen Engine-74eb assignment schedule for fixed-blend confirmation.

The arithmetic in this module is a direct Python transcription of the
unsigned-32-bit scheduler committed in DrawbackEngine revision
``74eb6fc95571994bd96b7a351278f3f74f0972e3``.  Keeping the transcription
here lets the sealed evaluator authenticate every assignment independently
of the separate byte-identical Engine regeneration check.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType
from typing import Mapping


ENGINE_REVISION = "74eb6fc95571994bd96b7a351278f3f74f0972e3"
LABEL_SEED = 633_442_320
GAMEPLAY_SEED = 633_446_417
PARAMETER_SEED = 633_450_514
FIXED_GAME_COUNT = 625

ENGINE_RULE_IDS = (
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

EXPECTED_WHITE_RULE_ORDER = (
    "forward-march",
    "checkers",
    "truant",
    "eye-for-an-eye",
    "battle-fatigue",
    "remorseful",
    "true-gentleman",
    "barbarian-rage",
    "spice-of-life",
    "triple-play",
    "false-prophets",
    "you-best-not-miss",
    "conscientious-objectors",
    "horse-tranquilizer",
    "nurturer",
    "oddball",
    "pacman",
    "cess",
    "femme-fatale",
    "irresistible",
    "quit-horsing-around",
    "trophy-wife",
    "vegan",
    "lame-duck",
    "even-keeled",
)

EXPECTED_BLACK_RULE_ORDER = (
    "trophy-wife",
    "triple-play",
    "checkers",
    "cess",
    "forward-march",
    "conscientious-objectors",
    "lame-duck",
    "even-keeled",
    "spice-of-life",
    "femme-fatale",
    "nurturer",
    "you-best-not-miss",
    "quit-horsing-around",
    "irresistible",
    "pacman",
    "oddball",
    "false-prophets",
    "horse-tranquilizer",
    "truant",
    "remorseful",
    "vegan",
    "eye-for-an-eye",
    "battle-fatigue",
    "barbarian-rage",
    "true-gentleman",
)

SCHEDULE_SHA256 = (
    "283f818736da3ce8673d608a75b46158e11268a1b01b112047071930f8534cb0"
)

_UINT32_MASK = (1 << 32) - 1
_UINT32_RANGE = 1 << 32
_GOLDEN_RATIO_32 = 0x9E37_79B9
_MIX_MULTIPLIER_ONE = 0x21F0_AAAD
_MIX_MULTIPLIER_TWO = 0x735A_2D97
_MULBERRY_INCREMENT = 0x6D2B_79F5

_WHITE_LABEL_DOMAIN = 0xA91F_0B21
_BLACK_LABEL_DOMAIN = 0x76E3_4CD5
_WHITE_PARAMETER_DOMAIN = 0x1C69_AE77
_BLACK_PARAMETER_DOMAIN = 0xD432_508B


@dataclass(frozen=True)
class FixedScheduleAssignment:
    """One exact test assignment emitted by the pinned Engine scheduler."""

    game_index: int
    game_id: str
    gameplay_seed: int
    white_parameter_seed: int
    black_parameter_seed: int
    white_rule_id: str
    black_rule_id: str


def _uint32(value: int) -> int:
    return value & _UINT32_MASK


def _mix_uint32(value: int) -> int:
    value = _uint32(value ^ (value >> 16))
    value = _uint32(value * _MIX_MULTIPLIER_ONE)
    value = _uint32(value ^ (value >> 15))
    value = _uint32(value * _MIX_MULTIPLIER_TWO)
    return _uint32(value ^ (value >> 15))


def _derive_game_seed(seed: int, game_index: int) -> int:
    return _mix_uint32(
        _uint32(
            seed
            ^ _uint32(_uint32(game_index + 1) * _GOLDEN_RATIO_32)
        )
    )


def _derive_stream_seed(seed: int, domain: int, index: int) -> int:
    return _mix_uint32(
        _uint32(
            seed
            ^ domain
            ^ _uint32(_uint32(index + 1) * _GOLDEN_RATIO_32)
        )
    )


class _Mulberry32:
    def __init__(self, seed: int) -> None:
        self._state = _uint32(seed)

    def _word(self) -> int:
        self._state = _uint32(self._state + _MULBERRY_INCREMENT)
        value = self._state
        value = _uint32((value ^ (value >> 15)) * (value | 1))
        value = _uint32(
            value
            ^ _uint32(
                value
                + _uint32((value ^ (value >> 7)) * (value | 61))
            )
        )
        return _uint32(value ^ (value >> 14))

    def integer(self, maximum_exclusive: int) -> int:
        if maximum_exclusive <= 0:
            raise ValueError("maximum_exclusive must be positive")
        # Engine computes floor((word / 2**32) * maximum_exclusive).
        # Integer arithmetic is exactly equivalent for this bounded domain.
        return (self._word() * maximum_exclusive) // _UINT32_RANGE


def _shuffled_rules(domain: int) -> tuple[str, ...]:
    values = list(ENGINE_RULE_IDS)
    rng = _Mulberry32(_derive_stream_seed(LABEL_SEED, domain, 0))
    for index in range(len(values) - 1, 0, -1):
        other = rng.integer(index + 1)
        values[index], values[other] = values[other], values[index]
    return tuple(values)


def _game_id(
    gameplay_seed: int,
    game_index: int,
    white_parameter_seed: int,
    black_parameter_seed: int,
) -> str:
    return (
        f"player-private-v1-{gameplay_seed:08x}-{game_index:06d}-"
        f"{white_parameter_seed:08x}-{black_parameter_seed:08x}"
    )


def _build_fixed_assignments() -> tuple[FixedScheduleAssignment, ...]:
    white_rules = _shuffled_rules(_WHITE_LABEL_DOMAIN)
    black_rules = _shuffled_rules(_BLACK_LABEL_DOMAIN)
    assignments: list[FixedScheduleAssignment] = []
    for game_index in range(FIXED_GAME_COUNT):
        slot = game_index % len(ENGINE_RULE_IDS)
        round_index = (
            game_index // len(ENGINE_RULE_IDS)
        ) % len(ENGINE_RULE_IDS)
        gameplay_seed = _derive_game_seed(GAMEPLAY_SEED, game_index)
        white_parameter_seed = _derive_stream_seed(
            PARAMETER_SEED,
            _WHITE_PARAMETER_DOMAIN,
            game_index,
        )
        black_parameter_seed = _derive_stream_seed(
            PARAMETER_SEED,
            _BLACK_PARAMETER_DOMAIN,
            game_index,
        )
        assignments.append(
            FixedScheduleAssignment(
                game_index=game_index,
                game_id=_game_id(
                    gameplay_seed,
                    game_index,
                    white_parameter_seed,
                    black_parameter_seed,
                ),
                gameplay_seed=gameplay_seed,
                white_parameter_seed=white_parameter_seed,
                black_parameter_seed=black_parameter_seed,
                white_rule_id=white_rules[slot],
                black_rule_id=black_rules[
                    (slot + round_index) % len(ENGINE_RULE_IDS)
                ],
            )
        )
    return tuple(assignments)


def _schedule_payload(
    assignments: tuple[FixedScheduleAssignment, ...],
) -> bytes:
    values = [
        {
            "blackParameterSeed": assignment.black_parameter_seed,
            "blackRuleId": assignment.black_rule_id,
            "gameId": assignment.game_id,
            "gameIndex": assignment.game_index,
            "gameplaySeed": assignment.gameplay_seed,
            "whiteParameterSeed": assignment.white_parameter_seed,
            "whiteRuleId": assignment.white_rule_id,
        }
        for assignment in assignments
    ]
    return (
        json.dumps(
            values,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


EXPECTED_FIXED_ASSIGNMENTS = _build_fixed_assignments()
EXPECTED_FIXED_ASSIGNMENTS_BY_GAME_ID: Mapping[
    str,
    FixedScheduleAssignment,
] = MappingProxyType(
    {
        assignment.game_id: assignment
        for assignment in EXPECTED_FIXED_ASSIGNMENTS
    }
)

_FIRST_SENTINEL = FixedScheduleAssignment(
    game_index=0,
    game_id="player-private-v1-35c406ed-000000-24debaee-05392f4a",
    gameplay_seed=902_039_277,
    white_parameter_seed=618_576_622,
    black_parameter_seed=87_633_738,
    white_rule_id="forward-march",
    black_rule_id="trophy-wife",
)
_LAST_SENTINEL = FixedScheduleAssignment(
    game_index=624,
    game_id="player-private-v1-88e07315-000624-429fc786-067b4919",
    gameplay_seed=2_296_410_901,
    white_parameter_seed=1_117_767_558,
    black_parameter_seed=108_742_937,
    white_rule_id="even-keeled",
    black_rule_id="barbarian-rage",
)


def _validate_frozen_schedule() -> None:
    if (
        len(ENGINE_RULE_IDS) != 25
        or len(set(ENGINE_RULE_IDS)) != 25
        or _shuffled_rules(_WHITE_LABEL_DOMAIN)
        != EXPECTED_WHITE_RULE_ORDER
        or _shuffled_rules(_BLACK_LABEL_DOMAIN)
        != EXPECTED_BLACK_RULE_ORDER
        or len(EXPECTED_FIXED_ASSIGNMENTS) != FIXED_GAME_COUNT
        or EXPECTED_FIXED_ASSIGNMENTS[0] != _FIRST_SENTINEL
        or EXPECTED_FIXED_ASSIGNMENTS[-1] != _LAST_SENTINEL
        or len(EXPECTED_FIXED_ASSIGNMENTS_BY_GAME_ID)
        != FIXED_GAME_COUNT
        or hashlib.sha256(
            _schedule_payload(EXPECTED_FIXED_ASSIGNMENTS)
        ).hexdigest()
        != SCHEDULE_SHA256
    ):
        raise RuntimeError("frozen Engine confirmation schedule is invalid")
    pairs = Counter(
        (assignment.white_rule_id, assignment.black_rule_id)
        for assignment in EXPECTED_FIXED_ASSIGNMENTS
    )
    if (
        len(pairs) != FIXED_GAME_COUNT
        or any(count != 1 for count in pairs.values())
    ):
        raise RuntimeError(
            "frozen Engine confirmation schedule lost an ordered pair"
        )


_validate_frozen_schedule()
