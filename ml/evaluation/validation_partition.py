"""Frozen, game-seed validation subpartitions.

Partitioning uses only the complete game seed.  It never inspects a dataset
row, model prediction, or label.
"""

from __future__ import annotations

from enum import Enum
import hashlib
import json
from typing import Iterable


VALIDATION_PARTITION_SALT = "current-catalog-182-v2:validation"
VALIDATION_PARTITION_IDENTITY = (
    "BLAKE2b-64(current-catalog-182-v2:validation:gameSeed)"
)


class ValidationPartition(str, Enum):
    SELECTION = "selection"
    CALIBRATION_FIT = "calibration-fit"
    VALIDATION_GATE = "validation-gate"


def validation_partition_unit(seed: int) -> float:
    """Map a complete game seed deterministically into ``[0, 1)``."""

    _validate_seed(seed)
    digest = hashlib.blake2b(
        f"{VALIDATION_PARTITION_SALT}:{seed}".encode("utf-8"),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, "big") / float(1 << 64)


def assign_validation_partition(seed: int) -> ValidationPartition:
    """Assign a validation game to the frozen selection/calibration/gate cut."""

    unit = validation_partition_unit(seed)
    if unit < 0.70:
        return ValidationPartition.SELECTION
    if unit < 0.85:
        return ValidationPartition.CALIBRATION_FIT
    return ValidationPartition.VALIDATION_GATE


def validation_seed_sha256(seeds: Iterable[int]) -> str:
    """Hash a canonical unique complete-game seed set."""

    materialized = tuple(seeds)
    for seed in materialized:
        _validate_seed(seed)
    if len(set(materialized)) != len(materialized):
        raise ValueError("validation seed set contains duplicates")
    canonical = json.dumps(
        sorted(materialized),
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(canonical).hexdigest()


def _validate_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("game seed must be a non-negative integer")
