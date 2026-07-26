"""Validation-only multiclass temperature scaling with hard eliminations.

This module is dependency-free and deliberately separate from inference. A
fitted temperature may soften or sharpen probabilities among surviving
hypotheses, but an eliminated hypothesis always receives exactly zero.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Sequence


CALIBRATION_FORMAT_VERSION = 1
VALIDATION_SPLIT = "validation"
MINIMUM_CALIBRATION_TEMPERATURE = 0.05
MAXIMUM_CALIBRATION_TEMPERATURE = 10.0


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a real number")
    rendered = float(value)
    if not math.isfinite(rendered):
        raise ValueError(f"{name} must be finite")
    return rendered


def _positive_float(value: object, name: str) -> float:
    rendered = _finite_float(value, name)
    if rendered <= 0.0:
        raise ValueError(f"{name} must be positive")
    return rendered


def _validated_inputs(
    logits: Sequence[float],
    eliminated: Sequence[bool],
) -> tuple[tuple[float, ...], tuple[bool, ...], tuple[int, ...]]:
    rendered_logits = tuple(_finite_float(value, "logit") for value in logits)
    rendered_eliminated = tuple(eliminated)
    if not rendered_logits:
        raise ValueError("logits must not be empty")
    if len(rendered_logits) != len(rendered_eliminated):
        raise ValueError("logits and hard-elimination mask must align")
    if any(type(value) is not bool for value in rendered_eliminated):
        raise ValueError("hard-elimination mask must contain booleans")
    surviving = tuple(
        index for index, is_eliminated in enumerate(rendered_eliminated)
        if not is_eliminated
    )
    if not surviving:
        raise ValueError("hard-elimination mask cannot eliminate every class")
    return rendered_logits, rendered_eliminated, surviving


@dataclass(frozen=True)
class CalibrationExample:
    """One labeled validation example used to fit temperature."""

    logits: tuple[float, ...]
    true_index: int
    eliminated: tuple[bool, ...]

    def __post_init__(self) -> None:
        logits, eliminated, _surviving = _validated_inputs(
            self.logits, self.eliminated
        )
        if (
            isinstance(self.true_index, bool)
            or not isinstance(self.true_index, int)
            or self.true_index < 0
            or self.true_index >= len(logits)
        ):
            raise ValueError("true_index must identify a logit")
        if eliminated[self.true_index]:
            raise ValueError(
                "validation label cannot be a hard-eliminated hypothesis"
            )
        object.__setattr__(self, "logits", logits)
        object.__setattr__(self, "eliminated", eliminated)


@dataclass(frozen=True)
class TemperatureCalibration:
    """Serializable output of a validation-only temperature fit."""

    temperature: float
    example_count: int
    nll_before: float
    nll_after: float
    fitted_split: str = VALIDATION_SPLIT
    format_version: int = CALIBRATION_FORMAT_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "temperature", _positive_float(self.temperature, "temperature")
        )
        if (
            isinstance(self.example_count, bool)
            or not isinstance(self.example_count, int)
            or self.example_count <= 0
        ):
            raise ValueError("example_count must be a positive integer")
        object.__setattr__(
            self, "nll_before", _finite_float(self.nll_before, "nll_before")
        )
        object.__setattr__(
            self, "nll_after", _finite_float(self.nll_after, "nll_after")
        )
        if self.fitted_split != VALIDATION_SPLIT:
            raise ValueError("temperature calibration may be fitted on validation only")
        if self.format_version != CALIBRATION_FORMAT_VERSION:
            raise ValueError(
                f"unsupported calibration format_version: {self.format_version}"
            )

    def to_metadata(self) -> dict[str, object]:
        return {
            "format_version": self.format_version,
            "method": "multiclass-temperature-scaling",
            "fitted_split": self.fitted_split,
            "temperature": self.temperature,
            "example_count": self.example_count,
            "nll_before": self.nll_before,
            "nll_after": self.nll_after,
            "preserves_hard_eliminations": True,
        }

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, object]) -> "TemperatureCalibration":
        expected = {
            "format_version",
            "method",
            "fitted_split",
            "temperature",
            "example_count",
            "nll_before",
            "nll_after",
            "preserves_hard_eliminations",
        }
        if set(metadata) != expected:
            raise ValueError("calibration metadata fields do not match format version 1")
        if metadata["method"] != "multiclass-temperature-scaling":
            raise ValueError("unsupported calibration method")
        if metadata["preserves_hard_eliminations"] is not True:
            raise ValueError("calibration metadata must preserve hard eliminations")
        return cls(
            format_version=metadata["format_version"],  # type: ignore[arg-type]
            fitted_split=metadata["fitted_split"],  # type: ignore[arg-type]
            temperature=metadata["temperature"],  # type: ignore[arg-type]
            example_count=metadata["example_count"],  # type: ignore[arg-type]
            nll_before=metadata["nll_before"],  # type: ignore[arg-type]
            nll_after=metadata["nll_after"],  # type: ignore[arg-type]
        )


def masked_temperature_softmax(
    logits: Sequence[float],
    eliminated: Sequence[bool],
    temperature: float,
) -> tuple[float, ...]:
    """Scale surviving logits and return exact zero for eliminated classes."""

    values, mask, surviving = _validated_inputs(logits, eliminated)
    scale = _positive_float(temperature, "temperature")
    scaled = tuple(values[index] / scale for index in surviving)
    maximum = max(scaled)
    exponentials = tuple(math.exp(value - maximum) for value in scaled)
    denominator = math.fsum(exponentials)
    probabilities = [0.0] * len(values)
    for index, exponential in zip(surviving, exponentials, strict=True):
        probabilities[index] = exponential / denominator
    # This assignment is intentional: it is the legality invariant, not an
    # approximation produced by a large negative logit.
    for index, is_eliminated in enumerate(mask):
        if is_eliminated:
            probabilities[index] = 0.0
    return tuple(probabilities)


def _example_nll(example: CalibrationExample, temperature: float) -> float:
    surviving_logits = [
        value / temperature
        for value, is_eliminated in zip(
            example.logits, example.eliminated, strict=True
        )
        if not is_eliminated
    ]
    maximum = max(surviving_logits)
    log_denominator = maximum + math.log(
        math.fsum(math.exp(value - maximum) for value in surviving_logits)
    )
    return log_denominator - example.logits[example.true_index] / temperature


def multiclass_nll(
    examples: Iterable[CalibrationExample],
    temperature: float,
) -> float:
    rows = tuple(examples)
    if not rows:
        raise ValueError("at least one calibration example is required")
    scale = _positive_float(temperature, "temperature")
    return math.fsum(_example_nll(row, scale) for row in rows) / len(rows)


def fit_validation_temperature(
    examples: Iterable[CalibrationExample],
    *,
    split: str,
    minimum_temperature: float = MINIMUM_CALIBRATION_TEMPERATURE,
    maximum_temperature: float = MAXIMUM_CALIBRATION_TEMPERATURE,
    iterations: int = 96,
) -> TemperatureCalibration:
    """Fit one deterministic scalar temperature using validation labels only."""

    if split != VALIDATION_SPLIT:
        raise ValueError(
            "temperature fitting requires validation labels; test labels are forbidden"
        )
    rows = tuple(examples)
    if not rows:
        raise ValueError("at least one validation calibration example is required")
    minimum = _positive_float(minimum_temperature, "minimum_temperature")
    maximum = _positive_float(maximum_temperature, "maximum_temperature")
    if minimum >= maximum:
        raise ValueError("minimum_temperature must be below maximum_temperature")
    if (
        isinstance(iterations, bool)
        or not isinstance(iterations, int)
        or iterations <= 0
    ):
        raise ValueError("iterations must be a positive integer")

    # Golden-section search in log-temperature space is deterministic and keeps
    # the strictly-positive temperature constraint without a third-party solver.
    left = math.log(minimum)
    right = math.log(maximum)
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    inner_right = left + ratio * (right - left)
    inner_left = right - ratio * (right - left)
    left_loss = multiclass_nll(rows, math.exp(inner_left))
    right_loss = multiclass_nll(rows, math.exp(inner_right))
    for _iteration in range(iterations):
        if left_loss <= right_loss:
            right = inner_right
            inner_right = inner_left
            right_loss = left_loss
            inner_left = right - ratio * (right - left)
            left_loss = multiclass_nll(rows, math.exp(inner_left))
        else:
            left = inner_left
            inner_left = inner_right
            left_loss = right_loss
            inner_right = left + ratio * (right - left)
            right_loss = multiclass_nll(rows, math.exp(inner_right))

    candidates = (
        minimum,
        maximum,
        math.exp(left),
        math.exp(inner_left),
        math.exp(inner_right),
        math.exp(right),
    )
    temperature, nll_after = min(
        (
            (candidate, multiclass_nll(rows, candidate))
            for candidate in candidates
        ),
        key=lambda item: (item[1], item[0]),
    )
    return TemperatureCalibration(
        temperature=temperature,
        example_count=len(rows),
        nll_before=multiclass_nll(rows, 1.0),
        nll_after=nll_after,
    )


def apply_temperature_calibration(
    logits: Sequence[float],
    eliminated: Sequence[bool],
    calibration: TemperatureCalibration,
) -> tuple[float, ...]:
    return masked_temperature_softmax(
        logits, eliminated, calibration.temperature
    )
