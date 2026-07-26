"""Independent verifier for the balanced evaluator assignment schedule."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence


MASK_32 = 0xFFFF_FFFF
SPLIT_DOMAIN = {
    "train": 0x243F_6A88,
    "validation": 0x85A3_08D3,
    "test": 0x1319_8A2E,
}
RULE_DOMAIN = 0x9E37_79B9
WHITE_AGENT_DOMAIN = 0xA409_3822
BLACK_AGENT_DOMAIN = 0x299F_31D0


@dataclass(frozen=True)
class ExpectedEvaluatorSlot:
    white_rule_id: str
    black_rule_id: str
    white_agent_id: str
    black_agent_id: str


class _Mulberry32:
    def __init__(self, seed: int) -> None:
        self._state = seed & MASK_32

    def integer(self, maximum: int) -> int:
        self._state = (self._state + 0x6D2B_79F5) & MASK_32
        value = self._state
        value = ((value ^ (value >> 15)) * (value | 1)) & MASK_32
        value ^= (
            value
            + (((value ^ (value >> 7)) * (value | 61)) & MASK_32)
        ) & MASK_32
        value &= MASK_32
        unit = ((value ^ (value >> 14)) & MASK_32) / 4_294_967_296
        return math.floor(unit * maximum)


def _permutation(values: Sequence[str], seed: int) -> tuple[str, ...]:
    shuffled = list(values)
    rng = _Mulberry32(seed)
    for index in range(len(shuffled) - 1, 0, -1):
        selected = rng.integer(index + 1)
        shuffled[index], shuffled[selected] = (
            shuffled[selected],
            shuffled[index],
        )
    return tuple(shuffled)


def expected_balanced_slots(
    *,
    root_seed: int,
    split: str,
    seeds: Sequence[int],
    rule_ids: Sequence[str],
    agent_ids: Sequence[str],
) -> tuple[ExpectedEvaluatorSlot, ...]:
    """Reconstruct TypeScript balanced-symmetric-v1 assignments."""

    if split not in SPLIT_DOMAIN:
        raise ValueError("unknown evaluator schedule split")
    if not rule_ids or len(set(rule_ids)) != len(rule_ids):
        raise ValueError("evaluator schedule rule IDs must be unique")
    if not agent_ids or len(set(agent_ids)) != len(agent_ids):
        raise ValueError("evaluator schedule agent IDs must be unique")
    rule_count = len(rule_ids)
    if not seeds or len(seeds) % rule_count != 0:
        raise ValueError("balanced schedule size must be a rule-count multiple")
    rounds = len(seeds) // rule_count
    if rounds % len(agent_ids) != 0:
        raise ValueError("balanced schedule rounds must divide across agents")
    if rule_count > 1 and (
        rounds % 2 != 0 or rounds > 2 * (rule_count - 1)
    ):
        raise ValueError("balanced schedule rounds violate pair symmetry")
    split_domain = SPLIT_DOMAIN[split]
    rules = _permutation(
        rule_ids, (root_seed ^ split_domain ^ RULE_DOMAIN) & MASK_32
    )
    white_agents = _permutation(
        agent_ids,
        (root_seed ^ split_domain ^ WHITE_AGENT_DOMAIN) & MASK_32,
    )
    black_agents = _permutation(
        agent_ids,
        (root_seed ^ split_domain ^ BLACK_AGENT_DOMAIN) & MASK_32,
    )
    slots: list[ExpectedEvaluatorSlot] = []
    for split_index in range(len(seeds)):
        round_index = split_index // rule_count
        rule_index = split_index % rule_count
        magnitude = round_index // 2 + 1
        offset = magnitude if round_index % 2 == 0 else -magnitude
        black_index = (
            0
            if rule_count == 1
            else (rule_index + offset + rule_count) % rule_count
        )
        slots.append(
            ExpectedEvaluatorSlot(
                white_rule_id=rules[rule_index],
                black_rule_id=rules[black_index],
                white_agent_id=white_agents[
                    (round_index + rule_index) % len(white_agents)
                ],
                black_agent_id=black_agents[
                    (round_index + black_index) % len(black_agents)
                ],
            )
        )
    return tuple(slots)
