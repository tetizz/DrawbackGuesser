"""Validation-only comparison for frozen capturable model candidates."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .capturable_baseline import _canonical_json, _publish_bytes
from .capturable_experiment import (
    CAPTURABLE_SELECTION_FORMAT,
    CAPTURABLE_SELECTION_VERSION,
    _load_selection_checkpoint,
)
from .capturable_records import (
    CAPTURABLE_FEATURE_DIMENSION,
    CAPTURABLE_RULE_IDS,
    CapturableDatasetError,
)


def _selection_report(path: Path) -> tuple[Mapping[str, Any], str]:
    payload = path.resolve().read_bytes()
    if not payload.endswith(b"\n") or payload.endswith(b"\r\n"):
        raise CapturableDatasetError(
            f"{path.name} must use canonical LF framing"
        )

    def pairs(items: Sequence[tuple[str, Any]]) -> Mapping[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise CapturableDatasetError(
                    f"{path.name} contains duplicate key {key}"
                )
            result[key] = value
        return result

    def reject_constant(token: str) -> Any:
        raise CapturableDatasetError(
            f"{path.name} contains non-finite number {token}"
        )

    try:
        report = json.loads(
            payload,
            object_pairs_hook=pairs,
            parse_constant=reject_constant,
        )
    except (json.JSONDecodeError, UnicodeError) as error:
        raise CapturableDatasetError(
            f"{path.name} is invalid JSON"
        ) from error
    if not isinstance(report, Mapping) or payload != _canonical_json(report):
        raise CapturableDatasetError(
            f"{path.name} is not canonical selection JSON"
        )
    return report, hashlib.sha256(payload).hexdigest()


def _finite_metric(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise CapturableDatasetError(f"{label} must be finite")
    return float(value)


def _bounded_metric(
    value: Any,
    label: str,
    *,
    minimum: float,
    maximum: float | None = None,
) -> float:
    measured = _finite_metric(value, label)
    if measured < minimum or (
        maximum is not None and measured > maximum
    ):
        upper = "unbounded" if maximum is None else str(maximum)
        raise CapturableDatasetError(
            f"{label} must be between {minimum} and {upper}"
        )
    return measured


def run_candidate_selection(
    report_paths: Sequence[Path],
    output_path: Path,
) -> Mapping[str, Any]:
    """Choose among frozen candidates using validation metrics only."""

    if output_path.exists():
        raise FileExistsError("candidate selection output already exists")
    if len(report_paths) < 2:
        raise ValueError("candidate selection requires at least two reports")
    candidates: list[dict[str, Any]] = []
    common_inputs: Any = None
    identities: set[tuple[int, float]] = set()
    for path in report_paths:
        report, report_sha256 = _selection_report(path)
        if (
            report.get("format") != CAPTURABLE_SELECTION_FORMAT
            or report.get("version") != CAPTURABLE_SELECTION_VERSION
            or report.get("freshStart") is not True
            or report.get("sealedTestStatus") != "unopened"
            or report.get("ruleIds") != list(CAPTURABLE_RULE_IDS)
            or report.get("featureDimension")
            != CAPTURABLE_FEATURE_DIMENSION
            or "test" in report
        ):
            raise CapturableDatasetError(
                f"{path.name} is not a compatible unopened selection"
            )
        config = report.get("config")
        validation = report.get("validation")
        checkpoint = report.get("checkpoint")
        inputs = report.get("inputs")
        if (
            not isinstance(config, Mapping)
            or not isinstance(validation, Mapping)
            or not isinstance(checkpoint, Mapping)
            or not isinstance(inputs, Mapping)
        ):
            raise CapturableDatasetError(
                f"{path.name} selection sections are invalid"
            )
        if common_inputs is None:
            common_inputs = inputs
        elif inputs != common_inputs:
            raise CapturableDatasetError(
                "candidate reports use different train or validation inputs"
            )
        seed = config.get("seed")
        trigger_multiplier = config.get("trigger_row_multiplier")
        if (
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or not 0 <= seed <= 0xFFFF_FFFF
        ):
            raise CapturableDatasetError(
                f"{path.name} model seed is invalid"
            )
        trigger = _finite_metric(
            trigger_multiplier,
            f"{path.name} trigger multiplier",
        )
        identity = (seed, trigger)
        if identity in identities:
            raise CapturableDatasetError(
                "candidate seed and trigger multiplier must be unique"
            )
        identities.add(identity)
        hybrid = validation.get("hybrid")
        if not isinstance(hybrid, Mapping):
            raise CapturableDatasetError(
                f"{path.name} validation hybrid metrics are invalid"
            )
        top1 = _bounded_metric(
            hybrid.get("game_normalized_top_1_accuracy"),
            f"{path.name} Top-1",
            minimum=0.0,
            maximum=1.0,
        )
        top3 = _bounded_metric(
            hybrid.get("game_normalized_top_3_accuracy"),
            f"{path.name} Top-3",
            minimum=0.0,
            maximum=1.0,
        )
        nll = _bounded_metric(
            hybrid.get("game_normalized_negative_log_likelihood"),
            f"{path.name} NLL",
            minimum=0.0,
        )
        if top3 < top1:
            raise CapturableDatasetError(
                f"{path.name} Top-3 cannot be below Top-1"
            )
        checkpoint_file = checkpoint.get("file")
        checkpoint_sha256 = checkpoint.get("sha256")
        if (
            not isinstance(checkpoint_file, str)
            or not checkpoint_file
            or Path(checkpoint_file).name != checkpoint_file
            or not isinstance(checkpoint_sha256, str)
            or len(checkpoint_sha256) != 64
        ):
            raise CapturableDatasetError(
                f"{path.name} checkpoint identity is invalid"
            )
        checkpoint_path = path.resolve().parent / checkpoint_file
        _, checkpoint_metadata, measured_checkpoint_sha256 = (
            _load_selection_checkpoint(checkpoint_path)
        )
        if (
            measured_checkpoint_sha256 != checkpoint_sha256
            or checkpoint_metadata.get("validation") != validation
            or checkpoint_metadata.get("inputs") != inputs
            or checkpoint_metadata.get("config") != config
            or checkpoint_metadata.get("selectedEpoch")
            != report.get("selectedEpoch")
            or checkpoint_metadata.get("selectedFusionAlpha")
            != report.get("selectedFusionAlpha")
            or checkpoint_metadata.get("selectedPriorSmoothing")
            != report.get("selectedPriorSmoothing")
        ):
            raise CapturableDatasetError(
                f"{path.name} checkpoint does not bind its selection report"
            )
        candidates.append(
            {
                "selectionDirectory": path.resolve().parent.name,
                "selectionReport": path.resolve().name,
                "selectionReportSha256": report_sha256,
                "checkpointFile": checkpoint_file,
                "checkpointSha256": checkpoint_sha256,
                "seed": seed,
                "triggerRowMultiplier": trigger,
                "validationGameNormalizedTop1": top1,
                "validationGameNormalizedTop3": top3,
                "validationGameNormalizedNll": nll,
            }
        )
    selected = max(
        candidates,
        key=lambda candidate: (
            candidate["validationGameNormalizedTop1"],
            candidate["validationGameNormalizedTop3"],
            -candidate["validationGameNormalizedNll"],
            -candidate["triggerRowMultiplier"],
            -candidate["seed"],
        ),
    )
    artifact = {
        "format": "drawbackguesser-capturable-candidate-selection",
        "version": 1,
        "selectionMetric": (
            "validation game-normalized Top-1, then Top-3, then lowest "
            "NLL, then lower trigger multiplier, then lower seed"
        ),
        "inputs": common_inputs,
        "candidates": candidates,
        "selected": selected,
        "sealedTestStatus": "unopened",
    }
    payload = _canonical_json(artifact)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _publish_bytes(output_path, payload)
    return {
        "artifactPath": str(output_path),
        "artifactSha256": hashlib.sha256(payload).hexdigest(),
        "selectedDirectory": selected["selectionDirectory"],
        "selectedCheckpointSha256": selected["checkpointSha256"],
        "validationHybridTop1": selected[
            "validationGameNormalizedTop1"
        ],
        "validationHybridTop3": selected[
            "validationGameNormalizedTop3"
        ],
    }
