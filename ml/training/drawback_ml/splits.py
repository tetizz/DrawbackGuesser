"""Order-independent, seed-isolated dataset splits."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import math


class Split(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


@dataclass(frozen=True)
class SplitConfig:
    train_fraction: float = 0.8
    validation_fraction: float = 0.1
    test_fraction: float = 0.1
    salt: str = "drawbacktrainer-v1"

    def __post_init__(self) -> None:
        fractions = (
            self.train_fraction,
            self.validation_fraction,
            self.test_fraction,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in fractions
        ):
            raise ValueError("split fractions must be finite numbers")
        if any(value < 0.0 or value > 1.0 for value in fractions):
            raise ValueError("split fractions must be between zero and one")
        if abs(sum(fractions) - 1.0) > 1e-12:
            raise ValueError("split fractions must sum to one")
        if not isinstance(self.salt, str) or not self.salt:
            raise ValueError("split salt must be a non-empty string")


def assign_split(seed: int, config: SplitConfig = SplitConfig()) -> Split:
    """Assign all examples from a simulation seed to exactly one split."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    digest = hashlib.blake2b(
        f"{config.salt}:{seed}".encode("utf-8"), digest_size=8
    ).digest()
    unit = int.from_bytes(digest, "big") / float(1 << 64)
    if unit < config.train_fraction:
        return Split.TRAIN
    if unit < config.train_fraction + config.validation_fraction:
        return Split.VALIDATION
    return Split.TEST
