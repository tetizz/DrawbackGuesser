"""Canonical hidden-parameter labels and deterministic vocabularies."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any, Iterable, Mapping


MASKED_PARAMETER_TOKEN = "__MASKED_NO_PARAMETERS__"


class ParameterEncodingError(ValueError):
    """Raised when hidden parameters are not deterministic JSON data."""


def supervised_parameter_label(label: str | None) -> str | None:
    """Remove opaque RNG seeds from a trusted parameter label.

    Replay records retain the canonical label for auditing and reproducibility,
    but an arbitrary seed is not a learnable categorical parameter: held-out
    seeds are necessarily absent from a training-only vocabulary. Other
    parameters in the same object remain valid supervision.
    """

    if label is None:
        return None
    try:
        value = json.loads(label)
    except (json.JSONDecodeError, TypeError):
        return label
    if not isinstance(value, dict) or "seed" not in value:
        return label
    supervised = {key: item for key, item in value.items() if key != "seed"}
    if not supervised:
        return None
    return json.dumps(
        supervised,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _validate_json(value: Any, path: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ParameterEncodingError(f"{path} contains a non-finite number")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json(item, f"{path}[{index}]")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ParameterEncodingError(f"{path} contains a non-string key")
            _validate_json(item, f"{path}.{key}")
        return
    raise ParameterEncodingError(
        f"{path} contains unsupported type {type(value).__name__}"
    )


def canonical_hidden_parameters(value: Any) -> str | None:
    """Encode a parameter object as stable JSON, or return None for no target."""

    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ParameterEncodingError("hiddenParameters must be an object or null")
    if not value:
        return None
    _validate_json(value, "hiddenParameters")
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


@dataclass(frozen=True)
class ParameterVocabulary:
    tokens: tuple[str, ...]

    @classmethod
    def build(cls, labels: Iterable[str | None]) -> "ParameterVocabulary":
        values = sorted(
            {
                supervised
                for label in labels
                if (supervised := supervised_parameter_label(label)) is not None
            }
        )
        return cls(tuple(values) if values else (MASKED_PARAMETER_TOKEN,))

    def encode(self, label: str | None) -> tuple[int, bool]:
        supervised = supervised_parameter_label(label)
        if supervised is None:
            return 0, False
        try:
            return self.tokens.index(supervised), True
        except ValueError as error:
            raise ParameterEncodingError(
                "parameter label is absent from the fitted vocabulary"
            ) from error


@dataclass(frozen=True)
class ParameterTargetBatch:
    white_indices: tuple[int, ...]
    white_mask: tuple[bool, ...]
    black_indices: tuple[int, ...]
    black_mask: tuple[bool, ...]


def encode_parameter_targets(
    vocabulary: ParameterVocabulary,
    white_labels: Iterable[str | None],
    black_labels: Iterable[str | None],
) -> ParameterTargetBatch:
    white = tuple(vocabulary.encode(label) for label in white_labels)
    black = tuple(vocabulary.encode(label) for label in black_labels)
    if len(white) != len(black):
        raise ParameterEncodingError("White and Black parameter batches must align")
    return ParameterTargetBatch(
        white_indices=tuple(index for index, _included in white),
        white_mask=tuple(included for _index, included in white),
        black_indices=tuple(index for index, _included in black),
        black_mask=tuple(included for _index, included in black),
    )
