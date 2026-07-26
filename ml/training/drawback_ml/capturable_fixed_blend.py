"""One-pass sealed evaluator for the frozen capturable fixed blend."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .capturable_baseline import (
    _canonical_json,
    tensorize,
)
from .capturable_blend import (
    _authenticated_execution_identity,
    _path_identity,
    _selection_input,
    load_blend_validation,
)
from .capturable_blend_contract import (
    _EXPECTED_INPUTS,
    blend_components,
    component_predictions,
    evaluate_predictions,
    performance_order,
)
from .capturable_candidate_selection import _selection_report
from .capturable_fixed_blend_contract import (
    BOOTSTRAP_DRAWS_PER_REPLICATE,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    CONFIRMATION_TEST_FILE,
    CONFIRMATION_TRACE_FILE,
    CONSUMPTION_FILE,
    CORPUS_RECEIPT_FILE,
    FIXED_BLEND_FORMAT,
    FIXED_BLEND_VERSION,
    FIXED_PROTOCOL_COMMIT,
    FIXED_PROTOCOL_FILE,
    FIXED_PROTOCOL_SHA256,
    FIXED_TREATMENT_WEIGHT,
    GRID_EXECUTION_REVISION,
    GRID_FILE,
    GRID_SHA256,
    MINIMUM_OBSERVED_TOP1_DELTA,
    PRIOR_REGISTRY_FILE,
    PRIOR_REGISTRY_GAME_COUNT,
    PRIOR_REGISTRY_SHA256,
    PRIOR_REGISTRY_SOURCE_COUNT,
    REPORT_PREFIX,
    SAMPLER_ID,
    fixed_consumption_artifact,
    fixed_paired_bootstrap,
    fixed_release_checks,
    fixed_validation_candidate,
    paired_game_top1_deltas,
)
from .capturable_fixed_corpus import (
    authenticate_corpus_environment,
    load_fixed_corpus_receipt,
    reauthenticate_fixed_corpus_files,
    require_isolated_python_runtime,
    require_private_regular_file,
    require_private_root,
    verify_fixed_corpus_receipt,
)
from .capturable_prior_registry import load_prior_corpus_registry
from .capturable_records import CapturableDatasetError
from .durable_publish import publish_bytes_durable


def _require_private_layout(
    *,
    output_directory: Path,
    grid_path: Path,
    control_selection_path: Path,
    treatment_selection_path: Path,
    registry_path: Path,
    test_path: Path,
) -> Path:
    root = require_private_root(output_directory)
    if (
        grid_path.resolve().parent != root
        or registry_path.resolve().parent != root
        or test_path.resolve().parent != root
        or control_selection_path.resolve().parent.parent != root
        or treatment_selection_path.resolve().parent.parent != root
    ):
        raise CapturableDatasetError(
            "fixed confirmation inputs must use the one private root"
        )
    require_private_regular_file(
        root,
        grid_path,
        GRID_FILE,
        "fixed confirmation grid",
    )
    require_private_regular_file(
        root,
        registry_path,
        PRIOR_REGISTRY_FILE,
        "prior corpus registry",
    )
    require_private_regular_file(
        root,
        test_path,
        CONFIRMATION_TEST_FILE,
        "fixed confirmation dataset",
    )
    require_private_regular_file(
        root,
        root / CONFIRMATION_TRACE_FILE,
        CONFIRMATION_TRACE_FILE,
        "fixed confirmation trace",
    )
    require_private_regular_file(
        root,
        root / CORPUS_RECEIPT_FILE,
        CORPUS_RECEIPT_FILE,
        "fixed corpus receipt",
    )
    for role, selection_path in (
        ("control", control_selection_path),
        ("treatment", treatment_selection_path),
    ):
        expected = _EXPECTED_INPUTS[role]
        require_private_regular_file(
            root,
            selection_path,
            str(expected["selection"]),
            f"{role} selection",
        )
        require_private_regular_file(
            root,
            selection_path.parent / str(expected["checkpoint"]),
            str(expected["checkpoint"]),
            f"{role} checkpoint",
        )
    marker_path = root / CONSUMPTION_FILE
    if marker_path.exists():
        raise FileExistsError(
            "fixed confirmation test was already consumed"
        )
    if any(root.glob(f"{REPORT_PREFIX}*.json")):
        raise FileExistsError(
            "fixed confirmation report already exists"
        )
    return root


def _authenticate_fixed_inputs(
    *,
    root: Path,
    grid_path: Path,
    control_selection_path: Path,
    treatment_selection_path: Path,
    registry_path: Path,
) -> tuple[
    Any,
    Mapping[str, Any],
    Mapping[str, Any],
    Any,
    Mapping[str, Any],
    Mapping[str, Any],
    Mapping[str, Any],
    Mapping[str, Any],
    Mapping[str, Any],
    str,
]:
    resolved_grid = _path_identity(
        grid_path,
        directory=None,
        filename=GRID_FILE,
        label="fixed confirmation grid",
    )
    grid, grid_sha256 = load_blend_validation(resolved_grid)
    if grid_sha256 != GRID_SHA256:
        raise CapturableDatasetError(
            "fixed confirmation grid digest is not frozen"
        )
    fixed_validation_candidate(grid)

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
            "fixed confirmation checkpoints are incompatible"
        )
    resolved_registry = _path_identity(
        registry_path,
        directory=None,
        filename=PRIOR_REGISTRY_FILE,
        label="prior corpus registry",
    )
    registry, registry_sha256 = load_prior_corpus_registry(
        resolved_registry
    )
    if (
        registry_sha256 != PRIOR_REGISTRY_SHA256
        or registry["sourceCount"] != PRIOR_REGISTRY_SOURCE_COUNT
        or registry["uniqueGameCount"] != PRIOR_REGISTRY_GAME_COUNT
    ):
        raise CapturableDatasetError(
            "prior corpus registry does not match the frozen inventory"
        )
    execution = _authenticated_execution_identity(
        protocol_commit=FIXED_PROTOCOL_COMMIT,
        protocol_file=FIXED_PROTOCOL_FILE,
        protocol_sha256=FIXED_PROTOCOL_SHA256,
        operation="fixed blend confirmation",
    )
    if resolved_grid.parent != root or resolved_registry.parent != root:
        raise CapturableDatasetError(
            "fixed confirmation authenticated input root changed"
        )
    return (
        control_model,
        control_metadata,
        control_input,
        treatment_model,
        treatment_metadata,
        treatment_input,
        grid,
        registry,
        execution,
        registry_sha256,
    )


def run_fixed_blend_confirmation(
    grid_path: Path,
    control_selection_path: Path,
    treatment_selection_path: Path,
    registry_path: Path,
    test_path: Path,
    output_directory: Path,
) -> Mapping[str, Any]:
    """Consume one frozen test and evaluate only control versus weight 0.1."""

    root = _require_private_layout(
        output_directory=output_directory,
        grid_path=grid_path,
        control_selection_path=control_selection_path,
        treatment_selection_path=treatment_selection_path,
        registry_path=registry_path,
        test_path=test_path,
    )
    (
        control_model,
        control_metadata,
        control_input,
        treatment_model,
        treatment_metadata,
        treatment_input,
        _grid,
        registry,
        execution,
        registry_sha256,
    ) = _authenticate_fixed_inputs(
        root=root,
        grid_path=grid_path,
        control_selection_path=control_selection_path,
        treatment_selection_path=treatment_selection_path,
        registry_path=registry_path,
    )

    receipt_path = root / CORPUS_RECEIPT_FILE
    receipt, receipt_sha256 = load_fixed_corpus_receipt(
        receipt_path,
        execution,
    )
    authenticate_corpus_environment(root, execution)
    inputs = {
        "control": control_input,
        "grid": {
            "executionRevision": GRID_EXECUTION_REVISION,
            "file": GRID_FILE,
            "sha256": GRID_SHA256,
        },
        "registry": {
            "file": PRIOR_REGISTRY_FILE,
            "games": PRIOR_REGISTRY_GAME_COUNT,
            "sha256": registry_sha256,
            "sources": PRIOR_REGISTRY_SOURCE_COUNT,
        },
        "treatment": treatment_input,
    }
    marker = fixed_consumption_artifact(
        execution=execution,
        inputs=inputs,
        receipt_sha256=receipt_sha256,
    )
    marker_payload = _canonical_json(marker)
    marker_path = root / CONSUMPTION_FILE
    publish_bytes_durable(marker_path, marker_payload)
    marker_sha256 = hashlib.sha256(marker_payload).hexdigest()

    verified = verify_fixed_corpus_receipt(
        root,
        receipt,
        execution,
        registry,
    )
    rows = verified.rows
    test_sha256 = verified.test_sha256
    corpus = verified.corpus
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
    fixed_predictions = blend_components(
        rows,
        control_predictions,
        treatment_predictions,
        FIXED_TREATMENT_WEIGHT,
    )
    control_metrics = evaluate_predictions(rows, control_predictions)
    fixed_metrics = evaluate_predictions(rows, fixed_predictions)
    paired = paired_game_top1_deltas(
        rows,
        control_predictions,
        fixed_predictions,
    )
    bootstrap = fixed_paired_bootstrap(paired.deltas)
    checks = fixed_release_checks(
        control_metrics,
        fixed_metrics,
        paired,
        bootstrap,
    )
    primary = performance_order(fixed_metrics) > performance_order(
        control_metrics
    )
    release_decision = (
        "promote-fixed-blend"
        if all(checks.values())
        else "retain-control"
    )
    current_receipt, current_receipt_sha256 = load_fixed_corpus_receipt(
        receipt_path,
        execution,
    )
    marker_file = require_private_regular_file(
        root,
        marker_path,
        CONSUMPTION_FILE,
        "fixed consumption marker",
    )
    current_marker, current_marker_sha256 = _selection_report(marker_file)
    reauthenticate_fixed_corpus_files(root, receipt, execution)
    if (
        current_receipt != receipt
        or current_receipt_sha256 != receipt_sha256
        or current_marker != marker
        or current_marker_sha256 != marker_sha256
    ):
        raise CapturableDatasetError(
            "fixed receipt or consumption marker changed during evaluation"
        )
    report = {
        "consumption": {
            "file": marker_path.name,
            "sha256": marker_sha256,
        },
        "control": {
            "metrics": control_metrics,
            "predictionsSha256": control_predictions.sha256,
        },
        "corpusReceipt": {
            "file": receipt_path.name,
            "sha256": receipt_sha256,
        },
        "execution": execution,
        "fixedBlend": {
            "metrics": fixed_metrics,
            "predictionsSha256": fixed_predictions.sha256,
        },
        "fixedTreatmentWeight": FIXED_TREATMENT_WEIGHT,
        "format": FIXED_BLEND_FORMAT,
        "inputs": inputs,
        "pairedTop1": {
            "bootstrapLowerBound": bootstrap.lower_bound,
            "bootstrapReplicates": BOOTSTRAP_REPLICATES,
            "drawsPerReplicate": BOOTSTRAP_DRAWS_PER_REPLICATE,
            "gameDeltas": [
                {"delta": delta, "gameId": game_id}
                for game_id, delta in zip(
                    paired.game_ids,
                    paired.deltas,
                    strict=True,
                )
            ],
            "minimumObservedDelta": MINIMUM_OBSERVED_TOP1_DELTA,
            "observedDelta": paired.observed_delta,
            "rejectedDraws": bootstrap.rejected_draws,
            "sampler": SAMPLER_ID,
            "seed": BOOTSTRAP_SEED,
        },
        "primaryDecision": (
            "confirm-fixed-blend"
            if primary
            else "reject-fixed-blend"
        ),
        "protocol": marker["protocol"],
        "releaseDecision": release_decision,
        "reliabilityChecks": checks,
        "sealedTestStatus": "consumed",
        "test": {
            "file": test_path.resolve().name,
            "games": corpus["games"],
            "rows": corpus["rows"],
            "sha256": test_sha256,
        },
        "trace": {
            "file": CONFIRMATION_TRACE_FILE,
            "sha256": receipt["trace"]["sha256"],
        },
        "version": FIXED_BLEND_VERSION,
    }
    authenticate_corpus_environment(root, execution)
    payload = _canonical_json(report)
    digest = hashlib.sha256(payload).hexdigest()
    report_path = root / f"{REPORT_PREFIX}{digest}.json"
    publish_bytes_durable(report_path, payload)
    authenticate_corpus_environment(root, execution)
    return {
        "controlGameNormalizedTop1": control_metrics["hybrid"][
            "game_normalized_top_1_accuracy"
        ],
        "fixedGameNormalizedTop1": fixed_metrics["hybrid"][
            "game_normalized_top_1_accuracy"
        ],
        "pairedBootstrapLowerBound": bootstrap.lower_bound,
        "pairedObservedTop1Delta": paired.observed_delta,
        "primaryDecision": report["primaryDecision"],
        "releaseDecision": release_decision,
        "reportPath": str(report_path),
        "reportSha256": digest,
        "testSha256": test_sha256,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description=(
            "Evaluate the one preregistered weight-0.1 blend on its single "
            "fresh confirmation corpus."
        ),
    )
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--control-selection", type=Path, required=True)
    parser.add_argument("--treatment-selection", type=Path, required=True)
    parser.add_argument("--prior-registry", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    require_isolated_python_runtime()
    options = _parser().parse_args(arguments)
    result = run_fixed_blend_confirmation(
        options.grid,
        options.control_selection,
        options.treatment_selection,
        options.prior_registry,
        options.test,
        options.output_directory,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
