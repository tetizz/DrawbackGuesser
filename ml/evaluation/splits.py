"""Seed split manifests with overlap rejection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping


class SplitOverlapError(ValueError):
    """Raised when a seed appears in more than one dataset split."""


@dataclass(frozen=True)
class SplitManifest:
    train: tuple[int, ...]
    validation: tuple[int, ...]
    test: tuple[int, ...]

    def __post_init__(self) -> None:
        for name, values in (
            ("train", self.train),
            ("validation", self.validation),
            ("test", self.test),
        ):
            if len(values) != len(set(values)):
                raise SplitOverlapError(f"{name} contains duplicate seeds")
            if any(
                not isinstance(seed, int) or isinstance(seed, bool) or seed < 0
                for seed in values
            ):
                raise ValueError(f"{name} seeds must be non-negative integers")
        validate_split_manifest(
            {
                "train": self.train,
                "validation": self.validation,
                "test": self.test,
            }
        )

    @classmethod
    def from_mapping(
        cls,
        manifest: Mapping[str, Iterable[int]],
    ) -> "SplitManifest":
        missing = {"train", "validation", "test"} - manifest.keys()
        if missing:
            raise ValueError(f"missing split(s): {', '.join(sorted(missing))}")
        return cls(
            train=tuple(manifest["train"]),
            validation=tuple(manifest["validation"]),
            test=tuple(manifest["test"]),
        )


def validate_split_manifest(
    manifest: Mapping[str, Iterable[int]],
) -> None:
    """Reject any seed shared by two named splits."""

    materialized = {
        name: set(seeds)
        for name, seeds in manifest.items()
    }
    names = sorted(materialized)
    for index, left_name in enumerate(names):
        for right_name in names[index + 1 :]:
            overlap = materialized[left_name] & materialized[right_name]
            if overlap:
                rendered = ", ".join(str(seed) for seed in sorted(overlap))
                raise SplitOverlapError(
                    f"{left_name} and {right_name} overlap: {rendered}"
                )
