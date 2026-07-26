"""Immutable prediction and release contract for the capturable blend."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Mapping, Sequence

from .capturable_baseline import (
    _canonical_json,
    _hard_mask_fusion,
    _raw_predictions,
    evaluate_capturable_posteriors,
)
from .capturable_experiment import _training_config_from_json
from .capturable_records import (
    CAPTURABLE_RULE_IDS,
    CapturableDatasetError,
    CapturableDatasetRow,
    active_symbolic,
)
from .capturable_reliability import validation_reliability_checks

CAPTURABLE_BLEND_FORMAT = "drawbackguesser-capturable-convex-blend"
CAPTURABLE_BLEND_VERSION = 1
PROTOCOL_COMMIT = "220ecaf792ff24c9c23ac3b6facdc686953d244d"
PROTOCOL_FILE = "capturable-25-convex-blend-protocol.md"
PROTOCOL_SHA256 = (
    "1ecd2c1f65c567f7d53cd3efcc812f1aa3f26ffff4c499413082e03306a3381b"
)
BLEND_WEIGHTS = tuple(index / 10.0 for index in range(1, 11))
SELECTION_METRIC = (
    "highest game-normalized Top-1, then Top-3, then lowest NLL, "
    "then lower treatment weight"
)

_EXPECTED_INPUTS = {
    "control": {
        "directory": "capturable25-v3-control-s3235776259-t2",
        "selection": "selection.json",
        "selectionSha256": (
            "889e8a22f812f6359b7aa36d66436f077bbbbd80fbe7ef6ac393533eb47fbf06"
        ),
        "checkpoint": "model.pt",
        "checkpointSha256": (
            "b314ae8e0c020490363237a025c8f6291dc24073542e00c17091a87df9c65469"
        ),
    },
    "treatment": {
        "directory": "capturable25-v3-balanced-s3235776257-t1",
        "selection": "selection.json",
        "selectionSha256": (
            "3a6e66e46002d037e4a1e599b7accf7d9ede753b155a8d12fc3cd1ad3086ce57"
        ),
        "checkpoint": "model.pt",
        "checkpointSha256": (
            "8f821461304bdc03786aea354676c0dd8fccadaecbad89496d0322dc095c424f"
        ),
    },
    "validation": {
        "file": "capturable25-v3-balanced-validation-schema8.ndjson",
        "sha256": (
            "09d5d9a4991d76e9fb564ac1fbc64c10212f80b75b0e28fbb9f74b4a93bbaf3d"
        ),
        "rows": 30352,
        "games": 625,
    },
    "priorComparison": {
        "file": "capturable25-v3-balanced-treatment-comparison.json",
        "sha256": (
            "2535b9916f75a5eff075570295911e738108a3a92ddcf4ab2e209fe65a096a3d"
        ),
    },
}


@dataclass(frozen=True)
class ComponentPredictions:
    drawback: tuple[tuple[float, ...], ...]
    trigger: tuple[float, ...]
    forced: tuple[float, ...]
    parameters: tuple[tuple[float, ...], ...]
    hard_masks: tuple[tuple[bool, ...], ...]
    sha256: str

    def __post_init__(self) -> None:
        count = len(self.drawback)
        if (
            count == 0
            or len(self.trigger) != count
            or len(self.forced) != count
            or len(self.parameters) != count
            or len(self.hard_masks) != count
        ):
            raise CapturableDatasetError(
                "component prediction rows must align"
            )
        for index in range(count):
            probabilities = self.drawback[index]
            mask = self.hard_masks[index]
            parameters = self.parameters[index]
            if (
                len(probabilities) != len(CAPTURABLE_RULE_IDS)
                or len(mask) != len(CAPTURABLE_RULE_IDS)
                or len(parameters) != 2
            ):
                raise CapturableDatasetError(
                    "component prediction dimensions are invalid"
                )
            _validate_distribution(
                probabilities,
                mask,
                f"component drawback row {index}",
            )
            _validate_distribution(
                parameters,
                (False, False),
                f"component parameter row {index}",
            )
            for label, value in (
                ("trigger", self.trigger[index]),
                ("forced", self.forced[index]),
            ):
                if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                    raise CapturableDatasetError(
                        f"component {label} row {index} is invalid"
                    )
        if not _is_sha256(self.sha256):
            raise CapturableDatasetError(
                "component prediction digest is invalid"
            )


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(token in "0123456789abcdef" for token in value)
    )


def _validate_distribution(
    probabilities: Sequence[float],
    eliminated: Sequence[bool],
    label: str,
) -> None:
    if (
        len(probabilities) == 0
        or len(probabilities) != len(eliminated)
        or any(type(value) is not bool for value in eliminated)
        or any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0
            for value in probabilities
        )
        or abs(math.fsum(probabilities) - 1.0) > 1e-12
        or any(
            probabilities[index] != 0.0
            for index, value in enumerate(eliminated)
            if value
        )
    ):
        raise CapturableDatasetError(f"{label} is not a valid distribution")


def _prediction_digest(
    rows: Sequence[CapturableDatasetRow],
    drawback: Sequence[Sequence[float]],
    trigger: Sequence[float],
    forced: Sequence[float],
    parameters: Sequence[Sequence[float]],
    hard_masks: Sequence[Sequence[bool]],
) -> str:
    digest = hashlib.sha256()
    for index, row in enumerate(rows):
        digest.update(
            _canonical_json(
                {
                    "drawback": list(drawback[index]),
                    "forced": forced[index],
                    "gameId": row.evaluation.game_id,
                    "hardEliminated": list(hard_masks[index]),
                    "move": row.features.move,
                    "moveNumber": row.features.move_number,
                    "parameters": list(parameters[index]),
                    "playerColor": row.features.player_color,
                    "rowIndex": index,
                    "trigger": trigger[index],
                }
            )
        )
    return digest.hexdigest()


def component_predictions(
    model: Any,
    metadata: Mapping[str, Any],
    rows: Sequence[CapturableDatasetRow],
    tensors: Any,
) -> ComponentPredictions:
    """Run one authenticated component and publish final hard-masked values."""

    import torch

    config = _training_config_from_json(metadata["config"])
    torch.set_num_threads(config.torch_threads)
    residuals, trigger, forced, raw_parameters = _raw_predictions(
        model,
        tensors,
        config.batch_size,
    )
    parameters = []
    for index, probabilities in enumerate(raw_parameters):
        total = math.fsum(probabilities)
        if (
            len(probabilities) != 2
            or not math.isfinite(total)
            or total <= 0.0
        ):
            raise CapturableDatasetError(
                f"component parameter row {index} is invalid"
            )
        parameters.append(
            tuple(probability / total for probability in probabilities)
        )
    drawback: list[tuple[float, ...]] = []
    hard_masks: list[tuple[bool, ...]] = []
    for row, neural in zip(rows, residuals, strict=True):
        prior, eliminated = active_symbolic(row.features)
        drawback.append(
            _hard_mask_fusion(
                neural,
                prior,
                eliminated,
                alpha=float(metadata["selectedFusionAlpha"]),
                prior_smoothing=float(
                    metadata["selectedPriorSmoothing"]
                ),
            )
        )
        hard_masks.append(eliminated)
    return ComponentPredictions(
        drawback=tuple(drawback),
        trigger=tuple(trigger),
        forced=tuple(forced),
        parameters=tuple(parameters),
        hard_masks=tuple(hard_masks),
        sha256=_prediction_digest(
            rows,
            drawback,
            trigger,
            forced,
            parameters,
            hard_masks,
        ),
    )


def blend_components(
    rows: Sequence[CapturableDatasetRow],
    control: ComponentPredictions,
    treatment: ComponentPredictions,
    weight: float,
) -> ComponentPredictions:
    """Mix final probabilities without intersecting or weakening hard masks."""

    if (
        isinstance(weight, bool)
        or not isinstance(weight, (int, float))
        or not math.isfinite(float(weight))
        or not 0.0 <= float(weight) <= 1.0
    ):
        raise CapturableDatasetError("blend weight must be finite in [0, 1]")
    if (
        len(rows) != len(control.drawback)
        or len(rows) != len(treatment.drawback)
        or control.hard_masks != treatment.hard_masks
    ):
        raise CapturableDatasetError(
            "blend components have different rows or hard masks"
        )
    right_weight = float(weight)
    left_weight = 1.0 - right_weight

    def mix(left: Sequence[float], right: Sequence[float]) -> tuple[float, ...]:
        if len(left) != len(right):
            raise CapturableDatasetError(
                "blend component dimensions do not match"
            )
        return tuple(
            left_weight * first + right_weight * second
            for first, second in zip(left, right, strict=True)
        )

    drawback = tuple(
        mix(left, right)
        for left, right in zip(
            control.drawback,
            treatment.drawback,
            strict=True,
        )
    )
    trigger = tuple(
        left_weight * left + right_weight * right
        for left, right in zip(
            control.trigger,
            treatment.trigger,
            strict=True,
        )
    )
    forced = tuple(
        left_weight * left + right_weight * right
        for left, right in zip(
            control.forced,
            treatment.forced,
            strict=True,
        )
    )
    parameters = tuple(
        mix(left, right)
        for left, right in zip(
            control.parameters,
            treatment.parameters,
            strict=True,
        )
    )
    return ComponentPredictions(
        drawback=drawback,
        trigger=trigger,
        forced=forced,
        parameters=parameters,
        hard_masks=control.hard_masks,
        sha256=_prediction_digest(
            rows,
            drawback,
            trigger,
            forced,
            parameters,
            control.hard_masks,
        ),
    )


def evaluate_predictions(
    rows: Sequence[CapturableDatasetRow],
    predictions: ComponentPredictions,
) -> Mapping[str, Any]:
    return evaluate_capturable_posteriors(
        rows,
        predictions.drawback,
        predictions.trigger,
        predictions.forced,
        predictions.parameters,
    )


def _metric(
    section: Mapping[str, Any],
    key: str,
    label: str,
    *,
    maximum: float | None = None,
) -> float:
    value = section.get(key)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
        or (maximum is not None and float(value) > maximum)
    ):
        raise CapturableDatasetError(f"{label} {key} is invalid")
    return float(value)


def _non_regression(
    control: Mapping[str, Any],
    candidate: Mapping[str, Any],
    keys: Sequence[tuple[str, bool, float | None]],
    label: str,
) -> bool:
    for key, higher_is_better, maximum in keys:
        left = _metric(control, key, f"control {label}", maximum=maximum)
        right = _metric(
            candidate,
            key,
            f"candidate {label}",
            maximum=maximum,
        )
        if (higher_is_better and right < left) or (
            not higher_is_better and right > left
        ):
            return False
    return True


def blend_reliability_checks(
    control: Mapping[str, Any],
    candidate: Mapping[str, Any],
    primary_confirmed: bool,
) -> Mapping[str, bool]:
    """Apply the preregistered superset of the common release gate."""

    checks = dict(
        validation_reliability_checks(
            control,
            candidate,
            primary_confirmed,
        )
    )
    control_hybrid = control.get("hybrid")
    candidate_hybrid = candidate.get("hybrid")
    if not isinstance(control_hybrid, Mapping) or not isinstance(
        candidate_hybrid,
        Mapping,
    ):
        raise CapturableDatasetError("blend hybrid metrics are invalid")
    checks["top5NonRegression"] = _non_regression(
        control_hybrid,
        candidate_hybrid,
        (("game_normalized_top_5_accuracy", True, 1.0),),
        "hybrid",
    )

    color_checks = []
    control_colors = control.get("hybridByColor")
    candidate_colors = candidate.get("hybridByColor")
    if not isinstance(control_colors, Mapping) or not isinstance(
        candidate_colors,
        Mapping,
    ):
        raise CapturableDatasetError("blend color metrics are invalid")
    color_keys = (
        ("game_normalized_top_1_accuracy", True, 1.0),
        ("game_normalized_top_3_accuracy", True, 1.0),
        ("game_normalized_negative_log_likelihood", False, None),
    )
    for color in ("white", "black"):
        control_color = control_colors.get(color)
        candidate_color = candidate_colors.get(color)
        if not isinstance(control_color, Mapping) or not isinstance(
            candidate_color,
            Mapping,
        ):
            raise CapturableDatasetError(
                f"blend {color} metrics are invalid"
            )
        color_checks.append(
            _non_regression(
                control_color,
                candidate_color,
                color_keys,
                color,
            )
        )
    checks["bothColorsNonRegression"] = all(color_checks)

    auxiliary_checks = []
    auxiliary_keys = (
        ("negative_log_likelihood", False, None),
        ("brier_score", False, 1.0),
    )
    for head in ("trigger", "forced"):
        control_head = control.get(head)
        candidate_head = candidate.get(head)
        if not isinstance(control_head, Mapping) or not isinstance(
            candidate_head,
            Mapping,
        ):
            raise CapturableDatasetError(
                f"blend {head} metrics are invalid"
            )
        auxiliary_checks.append(
            _non_regression(
                control_head,
                candidate_head,
                auxiliary_keys,
                head,
            )
        )
    checks["auxiliaryCalibrationNonRegression"] = all(auxiliary_checks)
    checks["parameterAccuracyNonRegression"] = _non_regression(
        control_hybrid,
        candidate_hybrid,
        (("hidden_parameter_accuracy", True, 1.0),),
        "parameter",
    )
    return checks


def performance_order(metrics: Mapping[str, Any]) -> tuple[float, ...]:
    hybrid = metrics.get("hybrid")
    if not isinstance(hybrid, Mapping):
        raise CapturableDatasetError("candidate hybrid metrics are invalid")
    return (
        _metric(
            hybrid,
            "game_normalized_top_1_accuracy",
            "candidate",
            maximum=1.0,
        ),
        _metric(
            hybrid,
            "game_normalized_top_3_accuracy",
            "candidate",
            maximum=1.0,
        ),
        -_metric(
            hybrid,
            "game_normalized_negative_log_likelihood",
            "candidate",
        ),
    )


def candidate_order(candidate: Mapping[str, Any]) -> tuple[float, ...]:
    weight = candidate.get("weight")
    if (
        isinstance(weight, bool)
        or not isinstance(weight, (int, float))
        or float(weight) not in BLEND_WEIGHTS
    ):
        raise CapturableDatasetError("candidate blend weight is invalid")
    metrics = candidate.get("metrics")
    if not isinstance(metrics, Mapping):
        raise CapturableDatasetError("candidate metrics are invalid")
    return performance_order(metrics) + (-float(weight),)
