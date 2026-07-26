"""Authenticated validation CLI for the preregistered capturable blend."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

from .capturable_baseline import (
    _canonical_json,
    _publish_bytes,
    tensorize,
)
from .capturable_blend_contract import (
    BLEND_WEIGHTS,
    CAPTURABLE_BLEND_FORMAT,
    CAPTURABLE_BLEND_VERSION,
    PROTOCOL_COMMIT,
    PROTOCOL_FILE,
    PROTOCOL_SHA256,
    SELECTION_METRIC,
    ComponentPredictions,
    _EXPECTED_INPUTS,
    _is_sha256,
    blend_components,
    blend_reliability_checks,
    candidate_order,
    component_predictions,
    evaluate_predictions,
    performance_order,
)
from .capturable_candidate_selection import (
    _selection_report,
    _validated_candidate,
    load_treatment_comparison,
)
from .capturable_experiment import (
    _load_selection_checkpoint,
    _load_stable_capturable_dataset,
)
from .capturable_records import CapturableDatasetError


def _path_identity(
    path: Path,
    *,
    directory: str | None,
    filename: str,
    label: str,
) -> Path:
    resolved = path.resolve()
    if resolved.name != filename or (
        directory is not None and resolved.parent.name != directory
    ):
        raise CapturableDatasetError(f"{label} path is not preregistered")
    return resolved


def _execution_identity() -> Mapping[str, Any]:
    repository = Path(__file__).resolve().parents[3]
    protocol_path = repository / "docs" / "research" / PROTOCOL_FILE
    try:
        protocol_sha256 = hashlib.sha256(
            protocol_path.read_bytes()
        ).hexdigest()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        )
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        ancestor = subprocess.run(
            [
                "git",
                "merge-base",
                "--is-ancestor",
                PROTOCOL_COMMIT,
                revision,
            ],
            cwd=repository,
            check=False,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise CapturableDatasetError(
            "blend execution identity cannot be verified"
        ) from error
    if protocol_sha256 != PROTOCOL_SHA256:
        raise CapturableDatasetError(
            "the committed blend protocol bytes have changed"
        )
    if status.stdout:
        raise CapturableDatasetError(
            "blend validation requires a clean committed worktree"
        )
    if (
        len(revision) != 40
        or any(token not in "0123456789abcdef" for token in revision)
        or ancestor.returncode != 0
    ):
        raise CapturableDatasetError(
            "blend execution revision does not contain the protocol"
        )
    return {
        "cleanWorktree": True,
        "repository": "DrawbackGuesser",
        "revision": revision,
    }


def _selection_input(
    path: Path,
    role: str,
) -> tuple[Any, Mapping[str, Any], Mapping[str, Any]]:
    expected = _EXPECTED_INPUTS[role]
    resolved = _path_identity(
        path,
        directory=str(expected["directory"]),
        filename=str(expected["selection"]),
        label=role,
    )
    candidate, _report = _validated_candidate(resolved)
    if (
        candidate["selectionReportSha256"]
        != expected["selectionSha256"]
        or candidate["checkpointFile"] != expected["checkpoint"]
        or candidate["checkpointSha256"]
        != expected["checkpointSha256"]
    ):
        raise CapturableDatasetError(
            f"{role} selection does not match the frozen digest"
        )
    model, metadata, checkpoint_sha256 = _load_selection_checkpoint(
        resolved.parent / candidate["checkpointFile"]
    )
    if checkpoint_sha256 != expected["checkpointSha256"]:
        raise CapturableDatasetError(
            f"{role} checkpoint does not match the frozen digest"
        )
    return model, metadata, {
        "checkpoint": candidate["checkpointFile"],
        "checkpointSha256": candidate["checkpointSha256"],
        "directory": resolved.parent.name,
        "selection": resolved.name,
        "selectionSha256": candidate["selectionReportSha256"],
    }


def _load_frozen_inputs(
    control_selection_path: Path,
    treatment_selection_path: Path,
    validation_path: Path,
    prior_comparison_path: Path,
) -> tuple[
    Any,
    Mapping[str, Any],
    Mapping[str, Any],
    Any,
    Mapping[str, Any],
    Mapping[str, Any],
    tuple[Any, ...],
    Mapping[str, Any],
    Mapping[str, Any],
]:
    control_model, control_metadata, control_input = _selection_input(
        control_selection_path,
        "control",
    )
    treatment_model, treatment_metadata, treatment_input = _selection_input(
        treatment_selection_path,
        "treatment",
    )
    if (
        control_metadata["ruleIds"] != treatment_metadata["ruleIds"]
        or control_metadata["featureDimension"]
        != treatment_metadata["featureDimension"]
        or control_metadata["inputs"]["validation"]
        != treatment_metadata["inputs"]["validation"]
    ):
        raise CapturableDatasetError(
            "blend checkpoints use incompatible public contracts"
        )

    expected_validation = _EXPECTED_INPUTS["validation"]
    resolved_validation = _path_identity(
        validation_path,
        directory=None,
        filename=str(expected_validation["file"]),
        label="validation",
    )
    rows, validation_sha256 = _load_stable_capturable_dataset(
        resolved_validation
    )
    game_count = len({row.evaluation.game_id for row in rows})
    validation_identity = {
        "file": resolved_validation.name,
        "games": game_count,
        "rows": len(rows),
        "sha256": validation_sha256,
    }
    if validation_identity != expected_validation:
        raise CapturableDatasetError(
            "validation corpus does not match the frozen identity"
        )
    checkpoint_validation = {
        "path": resolved_validation.name,
        "sha256": validation_sha256,
        "rows": len(rows),
        "games": game_count,
    }
    if control_metadata["inputs"]["validation"] != checkpoint_validation:
        raise CapturableDatasetError(
            "checkpoint validation identity does not match the corpus"
        )

    expected_comparison = _EXPECTED_INPUTS["priorComparison"]
    resolved_comparison = _path_identity(
        prior_comparison_path,
        directory=None,
        filename=str(expected_comparison["file"]),
        label="prior comparison",
    )
    comparison, comparison_sha256 = load_treatment_comparison(
        resolved_comparison
    )
    if (
        comparison_sha256 != expected_comparison["sha256"]
        or comparison["releaseDecision"] != "retain-control"
        or comparison["control"]["checkpointSha256"]
        != control_input["checkpointSha256"]
        or comparison["bestTreatment"]["checkpointSha256"]
        != treatment_input["checkpointSha256"]
    ):
        raise CapturableDatasetError(
            "prior comparison does not bind the frozen candidates"
        )
    inputs = {
        "control": control_input,
        "priorComparison": {
            "file": resolved_comparison.name,
            "releaseDecision": comparison["releaseDecision"],
            "sha256": comparison_sha256,
        },
        "treatment": treatment_input,
        "validation": validation_identity,
    }
    return (
        control_model,
        control_metadata,
        control_input,
        treatment_model,
        treatment_metadata,
        treatment_input,
        tuple(rows),
        inputs,
        comparison,
    )


def run_blend_validation(
    control_selection_path: Path,
    treatment_selection_path: Path,
    validation_path: Path,
    prior_comparison_path: Path,
    output_path: Path,
) -> Mapping[str, Any]:
    """Evaluate the frozen grid without accepting or opening a test path."""

    if output_path.exists():
        raise FileExistsError("blend validation output already exists")
    (
        control_model,
        control_metadata,
        _control_input,
        treatment_model,
        treatment_metadata,
        _treatment_input,
        rows,
        inputs,
        _comparison,
    ) = _load_frozen_inputs(
        control_selection_path,
        treatment_selection_path,
        validation_path,
        prior_comparison_path,
    )

    tensors = tensorize(rows)
    control_predictions = component_predictions(
        control_model,
        control_metadata,
        rows,
        tensors,
    )
    treatment_predictions = component_predictions(
        treatment_model,
        treatment_metadata,
        rows,
        tensors,
    )
    control_metrics = evaluate_predictions(rows, control_predictions)
    treatment_metrics = evaluate_predictions(rows, treatment_predictions)
    if (
        control_metrics != control_metadata["validation"]
        or treatment_metrics != treatment_metadata["validation"]
    ):
        raise CapturableDatasetError(
            "checkpoint predictions do not reproduce selected validation"
        )

    candidates = []
    for weight in BLEND_WEIGHTS:
        predictions = blend_components(
            rows,
            control_predictions,
            treatment_predictions,
            weight,
        )
        candidates.append(
            {
                "metrics": evaluate_predictions(rows, predictions),
                "predictionsSha256": predictions.sha256,
                "weight": weight,
            }
        )
    selected = max(candidates, key=candidate_order)
    primary_confirmed = (
        performance_order(selected["metrics"])
        > performance_order(control_metrics)
    )
    reliability_checks = blend_reliability_checks(
        control_metrics,
        selected["metrics"],
        primary_confirmed,
    )
    release_decision = (
        "promote-blend"
        if all(reliability_checks.values())
        else "retain-control"
    )
    artifact = {
        "candidates": candidates,
        "control": {
            "metrics": control_metrics,
            "predictionsSha256": control_predictions.sha256,
            "weight": 0.0,
        },
        "execution": _execution_identity(),
        "format": CAPTURABLE_BLEND_FORMAT,
        "inputs": inputs,
        "primaryDecision": (
            "confirm-blend" if primary_confirmed else "reject-blend"
        ),
        "protocol": {
            "commit": PROTOCOL_COMMIT,
            "file": PROTOCOL_FILE,
            "sha256": PROTOCOL_SHA256,
        },
        "releaseDecision": release_decision,
        "reliabilityChecks": reliability_checks,
        "sealedTestStatus": "unopened",
        "selected": selected,
        "selectionMetric": SELECTION_METRIC,
        "treatment": {
            "metrics": treatment_metrics,
            "predictionsSha256": treatment_predictions.sha256,
            "weight": 1.0,
        },
        "version": CAPTURABLE_BLEND_VERSION,
        "weightGrid": list(BLEND_WEIGHTS),
    }
    payload = _canonical_json(artifact)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _publish_bytes(output_path, payload)
    selected_hybrid = selected["metrics"]["hybrid"]
    control_hybrid = control_metrics["hybrid"]
    return {
        "artifactPath": str(output_path),
        "artifactSha256": hashlib.sha256(payload).hexdigest(),
        "controlGameNormalizedTop1": control_hybrid[
            "game_normalized_top_1_accuracy"
        ],
        "releaseDecision": release_decision,
        "selectedGameNormalizedTop1": selected_hybrid[
            "game_normalized_top_1_accuracy"
        ],
        "selectedWeight": selected["weight"],
    }


def load_blend_validation(path: Path) -> tuple[Mapping[str, Any], str]:
    """Load a canonical blend decision and recompute its selection decision."""

    artifact, sha256 = _selection_report(path)
    expected_keys = {
        "candidates",
        "control",
        "execution",
        "format",
        "inputs",
        "primaryDecision",
        "protocol",
        "releaseDecision",
        "reliabilityChecks",
        "sealedTestStatus",
        "selected",
        "selectionMetric",
        "treatment",
        "version",
        "weightGrid",
    }
    candidates = artifact.get("candidates")
    control = artifact.get("control")
    treatment = artifact.get("treatment")
    execution = artifact.get("execution")
    expected_inputs = {
        "control": dict(_EXPECTED_INPUTS["control"]),
        "priorComparison": {
            **_EXPECTED_INPUTS["priorComparison"],
            "releaseDecision": "retain-control",
        },
        "treatment": dict(_EXPECTED_INPUTS["treatment"]),
        "validation": dict(_EXPECTED_INPUTS["validation"]),
    }
    if (
        set(artifact) != expected_keys
        or artifact.get("format") != CAPTURABLE_BLEND_FORMAT
        or artifact.get("version") != CAPTURABLE_BLEND_VERSION
        or artifact.get("protocol")
        != {
            "commit": PROTOCOL_COMMIT,
            "file": PROTOCOL_FILE,
            "sha256": PROTOCOL_SHA256,
        }
        or artifact.get("weightGrid") != list(BLEND_WEIGHTS)
        or artifact.get("selectionMetric") != SELECTION_METRIC
        or artifact.get("inputs") != expected_inputs
        or artifact.get("sealedTestStatus") != "unopened"
        or not isinstance(execution, Mapping)
        or set(execution)
        != {"cleanWorktree", "repository", "revision"}
        or execution.get("cleanWorktree") is not True
        or execution.get("repository") != "DrawbackGuesser"
        or not isinstance(execution.get("revision"), str)
        or len(execution["revision"]) != 40
        or any(
            token not in "0123456789abcdef"
            for token in execution["revision"]
        )
        or not isinstance(candidates, list)
        or len(candidates) != len(BLEND_WEIGHTS)
        or not isinstance(control, Mapping)
        or not isinstance(treatment, Mapping)
    ):
        raise CapturableDatasetError(
            f"{path.name} is not a compatible blend validation"
        )
    if [candidate.get("weight") for candidate in candidates] != list(
        BLEND_WEIGHTS
    ):
        raise CapturableDatasetError(
            f"{path.name} blend grid is not ordered and complete"
        )
    entries = [control, *candidates, treatment]
    if any(
        set(entry) != {"metrics", "predictionsSha256", "weight"}
        or not isinstance(entry.get("metrics"), Mapping)
        or not _is_sha256(entry.get("predictionsSha256"))
        for entry in entries
    ):
        raise CapturableDatasetError(
            f"{path.name} blend candidate shape is invalid"
        )
    if control.get("weight") != 0.0 or treatment.get("weight") != 1.0:
        raise CapturableDatasetError(
            f"{path.name} component weights are invalid"
        )
    selected = max(candidates, key=candidate_order)
    if artifact.get("selected") != selected:
        raise CapturableDatasetError(
            f"{path.name} selected blend is inconsistent"
        )
    control_metrics = control.get("metrics")
    selected_metrics = selected.get("metrics")
    if not isinstance(control_metrics, Mapping) or not isinstance(
        selected_metrics,
        Mapping,
    ):
        raise CapturableDatasetError(
            f"{path.name} blend metrics are invalid"
        )
    primary_confirmed = (
        performance_order(selected_metrics)
        > performance_order(control_metrics)
    )
    reliability_checks = blend_reliability_checks(
        control_metrics,
        selected_metrics,
        primary_confirmed,
    )
    decision = (
        "promote-blend"
        if all(reliability_checks.values())
        else "retain-control"
    )
    if (
        artifact.get("primaryDecision")
        != ("confirm-blend" if primary_confirmed else "reject-blend")
        or artifact.get("reliabilityChecks") != reliability_checks
        or artifact.get("releaseDecision") != decision
        or treatment != candidates[-1]
    ):
        raise CapturableDatasetError(
            f"{path.name} blend decision is inconsistent"
        )
    return artifact, sha256


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the preregistered capturable 25-label convex blend "
            "on selection validation only."
        )
    )
    parser.add_argument("--control-selection", type=Path, required=True)
    parser.add_argument("--treatment-selection", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--prior-comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    options = _parser().parse_args(arguments)
    result = run_blend_validation(
        options.control_selection,
        options.treatment_selection,
        options.validation,
        options.prior_comparison,
        options.output,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
