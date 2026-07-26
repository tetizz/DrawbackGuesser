"""Two-stage capturable-king selection and sealed evaluation CLI."""

from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .capturable_baseline import (
    CapturableTrainingConfig,
    _canonical_json,
    _publish_bytes,
    _publish_checkpoint,
    _sha256_file,
    create_capturable_model,
    evaluate_capturable,
    tensorize,
    train_capturable_baseline,
)
from .capturable_records import (
    CAPTURABLE_FEATURE_DIMENSION,
    CAPTURABLE_RULE_IDS,
    CapturableDatasetError,
    CapturableDatasetRow,
    assert_disjoint_games,
    load_capturable_dataset,
)

CAPTURABLE_SELECTION_FORMAT = "drawbackguesser-capturable-selection"
CAPTURABLE_SELECTION_VERSION = 1


def _load_stable_capturable_dataset(
    path: Path,
) -> tuple[tuple[CapturableDatasetRow, ...], str]:
    resolved = path.resolve()
    before = _sha256_file(resolved)
    rows = load_capturable_dataset(resolved)
    after = _sha256_file(resolved)
    if before != after:
        raise CapturableDatasetError(
            f"{path.name} changed while it was being loaded"
        )
    return rows, before


def _capturable_source_identity(
    paths: Sequence[Path],
    sources: Sequence[Sequence[CapturableDatasetRow]],
    sha256s: Sequence[str],
) -> list[Mapping[str, Any]]:
    return [
        {
            "path": path.resolve().name,
            "sha256": sha256,
            "rows": len(rows),
            "games": len({row.evaluation.game_id for row in rows}),
        }
        for path, rows, sha256 in zip(
            paths,
            sources,
            sha256s,
            strict=True,
        )
    ]


def run_selection(
    train_paths: Sequence[Path],
    validation_path: Path,
    output_directory: Path,
    config: CapturableTrainingConfig,
) -> Mapping[str, Any]:
    """Select and publish a model without opening any test dataset."""

    output_directory.mkdir(parents=True, exist_ok=True)
    report_path = output_directory / "selection.json"
    checkpoint_path = output_directory / "model.pt"
    if report_path.exists() or checkpoint_path.exists():
        raise FileExistsError("capturable selection outputs already exist")
    if not train_paths:
        raise ValueError("at least one train dataset is required")
    loaded_train_sources = [
        _load_stable_capturable_dataset(path) for path in train_paths
    ]
    train_sources = [item[0] for item in loaded_train_sources]
    train_sha256s = [item[1] for item in loaded_train_sources]
    validation_rows, validation_sha256 = (
        _load_stable_capturable_dataset(validation_path)
    )
    assert_disjoint_games(*train_sources, validation_rows)
    train_rows = tuple(
        row
        for source in train_sources
        for row in source
    )
    model, measured = train_capturable_baseline(
        train_rows,
        validation_rows,
        None,
        config,
    )
    input_identity = {
        "train": {
            "sources": _capturable_source_identity(
                train_paths,
                train_sources,
                train_sha256s,
            ),
            "rows": len(train_rows),
            "games": len({row.evaluation.game_id for row in train_rows}),
        },
        "validation": {
            "path": validation_path.resolve().name,
            "sha256": validation_sha256,
            "rows": len(validation_rows),
            "games": len(
                {row.evaluation.game_id for row in validation_rows}
            ),
        },
    }
    source_game_ids = sorted(
        {
            row.evaluation.game_id
            for rows in (*train_sources, validation_rows)
            for row in rows
        }
    )
    checkpoint_sha256 = _publish_checkpoint(
        checkpoint_path,
        model,
        {
            "freshStart": True,
            "selectionOnly": True,
            "selectedEpoch": measured["selectedEpoch"],
            "selectedFusionAlpha": measured["selectedFusionAlpha"],
            "selectedPriorSmoothing": measured[
                "selectedPriorSmoothing"
            ],
            "ruleIds": list(CAPTURABLE_RULE_IDS),
            "featureDimension": CAPTURABLE_FEATURE_DIMENSION,
            "config": _jsonable_config(config),
            "inputs": input_identity,
            "sourceGameIds": source_game_ids,
            "validation": measured["validation"],
        },
        artifact_format=CAPTURABLE_SELECTION_FORMAT,
        artifact_version=CAPTURABLE_SELECTION_VERSION,
    )
    report = {
        **measured,
        "format": CAPTURABLE_SELECTION_FORMAT,
        "version": CAPTURABLE_SELECTION_VERSION,
        "inputs": input_identity,
        "checkpoint": {
            "file": checkpoint_path.name,
            "sha256": checkpoint_sha256,
        },
        "sealedTestStatus": "unopened",
    }
    payload = _canonical_json(report)
    _publish_bytes(report_path, payload)
    return {
        "reportPath": str(report_path),
        "reportSha256": hashlib.sha256(payload).hexdigest(),
        "checkpointPath": str(checkpoint_path),
        "checkpointSha256": checkpoint_sha256,
        "selectedEpoch": report["selectedEpoch"],
        "selectedFusionAlpha": report["selectedFusionAlpha"],
        "selectedPriorSmoothing": report["selectedPriorSmoothing"],
        "validationGameNormalizedTop1": report["validation"]["hybrid"][
            "game_normalized_top_1_accuracy"
        ],
        "validationGameNormalizedTop3": report["validation"]["hybrid"][
            "game_normalized_top_3_accuracy"
        ],
        "validationMoveWeightedTop1": report["validation"]["hybrid"][
            "top_1_accuracy"
        ],
        "validationMoveWeightedTop3": report["validation"]["hybrid"][
            "top_3_accuracy"
        ],
    }


def _jsonable_config(config: CapturableTrainingConfig) -> Mapping[str, Any]:
    result: dict[str, Any] = {}
    for field in fields(CapturableTrainingConfig):
        value = getattr(config, field.name)
        result[field.name] = list(value) if isinstance(value, tuple) else value
    return result


def _training_config_from_json(value: Any) -> CapturableTrainingConfig:
    if not isinstance(value, Mapping):
        raise CapturableDatasetError("checkpoint config must be an object")
    expected = {field.name for field in fields(CapturableTrainingConfig)}
    if set(value) != expected:
        raise CapturableDatasetError(
            "checkpoint config fields do not match the training contract"
        )
    arguments = dict(value)
    for key in ("fusion_alpha_grid", "prior_smoothing_grid"):
        grid = arguments[key]
        if not isinstance(grid, list):
            raise CapturableDatasetError(
                f"checkpoint {key} must be a JSON array"
            )
        arguments[key] = tuple(grid)
    try:
        return CapturableTrainingConfig(**arguments)
    except (TypeError, ValueError) as error:
        raise CapturableDatasetError(
            "checkpoint training config is invalid"
        ) from error


def _load_selection_checkpoint(
    checkpoint_path: Path,
) -> tuple[Any, Mapping[str, Any], str]:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "PyTorch is required; install ml/requirements.txt"
        ) from error
    digest = hashlib.sha256()
    with checkpoint_path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
        source.seek(0)
        checkpoint = torch.load(
            source,
            map_location="cpu",
            weights_only=True,
        )
    if (
        not isinstance(checkpoint, Mapping)
        or set(checkpoint) != {"format", "version", "metadata", "stateDict"}
        or checkpoint.get("format") != CAPTURABLE_SELECTION_FORMAT
        or checkpoint.get("version") != CAPTURABLE_SELECTION_VERSION
    ):
        raise CapturableDatasetError(
            "checkpoint outer contract is invalid"
        )
    metadata = checkpoint.get("metadata")
    expected_metadata = {
        "freshStart",
        "selectionOnly",
        "selectedEpoch",
        "selectedFusionAlpha",
        "selectedPriorSmoothing",
        "ruleIds",
        "featureDimension",
        "config",
        "inputs",
        "sourceGameIds",
        "validation",
    }
    if (
        not isinstance(metadata, Mapping)
        or set(metadata) != expected_metadata
        or metadata.get("freshStart") is not True
        or metadata.get("selectionOnly") is not True
        or metadata.get("ruleIds") != list(CAPTURABLE_RULE_IDS)
        or metadata.get("featureDimension") != CAPTURABLE_FEATURE_DIMENSION
    ):
        raise CapturableDatasetError(
            "checkpoint selection metadata is invalid"
        )
    config = _training_config_from_json(metadata.get("config"))
    selected_epoch = metadata.get("selectedEpoch")
    selected_alpha = metadata.get("selectedFusionAlpha")
    selected_smoothing = metadata.get("selectedPriorSmoothing")
    if (
        isinstance(selected_epoch, bool)
        or not isinstance(selected_epoch, int)
        or not 1 <= selected_epoch <= config.epochs
        or isinstance(selected_alpha, bool)
        or not isinstance(selected_alpha, (int, float))
        or selected_alpha not in config.fusion_alpha_grid
        or isinstance(selected_smoothing, bool)
        or not isinstance(selected_smoothing, (int, float))
        or selected_smoothing not in config.prior_smoothing_grid
    ):
        raise CapturableDatasetError(
            "checkpoint selection values are invalid"
        )
    source_game_ids = metadata.get("sourceGameIds")
    if (
        not isinstance(source_game_ids, list)
        or not source_game_ids
        or any(
            not isinstance(game_id, str) or not game_id
            for game_id in source_game_ids
        )
        or source_game_ids != sorted(set(source_game_ids))
    ):
        raise CapturableDatasetError(
            "checkpoint sourceGameIds are invalid"
        )
    if (
        not isinstance(metadata.get("inputs"), Mapping)
        or not isinstance(metadata.get("validation"), Mapping)
    ):
        raise CapturableDatasetError(
            "checkpoint inputs or validation metrics are invalid"
        )
    model = create_capturable_model(config.hidden_dimension)
    try:
        model.load_state_dict(checkpoint.get("stateDict"), strict=True)
    except (RuntimeError, TypeError) as error:
        raise CapturableDatasetError(
            "checkpoint tensors do not match the selected model"
        ) from error
    return model, metadata, digest.hexdigest()


def run_sealed_evaluation(
    checkpoint_path: Path,
    test_path: Path,
    output_path: Path,
) -> Mapping[str, Any]:
    """Evaluate a frozen selection once against a disjoint test dataset."""

    if output_path.exists():
        raise FileExistsError("sealed evaluation output already exists")
    model, metadata, checkpoint_sha256 = _load_selection_checkpoint(
        checkpoint_path.resolve()
    )
    config = _training_config_from_json(metadata["config"])
    test_rows, test_sha256 = _load_stable_capturable_dataset(test_path)
    source_game_ids = set(metadata["sourceGameIds"])
    overlap = sorted(
        {
            row.evaluation.game_id
            for row in test_rows
            if row.evaluation.game_id in source_game_ids
        }
    )
    if overlap:
        raise CapturableDatasetError(
            "sealed test overlaps selection games: "
            + ", ".join(overlap[:5])
        )
    selected_alpha = float(metadata["selectedFusionAlpha"])
    selected_smoothing = float(metadata["selectedPriorSmoothing"])
    report = {
        "format": "drawbackguesser-capturable-sealed-evaluation",
        "version": 1,
        "checkpoint": {
            "file": checkpoint_path.resolve().name,
            "sha256": checkpoint_sha256,
        },
        "selection": {
            "selectedEpoch": metadata["selectedEpoch"],
            "selectedFusionAlpha": selected_alpha,
            "selectedPriorSmoothing": selected_smoothing,
            "config": metadata["config"],
            "inputs": metadata["inputs"],
        },
        "test": {
            "input": {
                "path": test_path.resolve().name,
                "sha256": test_sha256,
                "rows": len(test_rows),
                "games": len(
                    {row.evaluation.game_id for row in test_rows}
                ),
            },
            "metrics": evaluate_capturable(
                model,
                test_rows,
                tensorize(test_rows),
                config,
                selected_alpha,
                selected_smoothing,
            ),
        },
    }
    payload = _canonical_json(report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _publish_bytes(output_path, payload)
    return {
        "reportPath": str(output_path),
        "reportSha256": hashlib.sha256(payload).hexdigest(),
        "checkpointSha256": checkpoint_sha256,
        "testHybridTop1": report["test"]["metrics"]["hybrid"][
            "top_1_accuracy"
        ],
        "testHybridTop3": report["test"]["metrics"]["hybrid"][
            "top_3_accuracy"
        ],
        "testSymbolicTop1": report["test"]["metrics"]["symbolicOnly"][
            "top_1_accuracy"
        ],
    }


def _evaluate_selection_on_test(
    checkpoint_path: Path,
    test_rows: Sequence[CapturableDatasetRow],
    test_tensors: Any,
) -> Mapping[str, Any]:
    model, metadata, checkpoint_sha256 = _load_selection_checkpoint(
        checkpoint_path.resolve()
    )
    source_game_ids = set(metadata["sourceGameIds"])
    overlap = sorted(
        {
            row.evaluation.game_id
            for row in test_rows
            if row.evaluation.game_id in source_game_ids
        }
    )
    if overlap:
        raise CapturableDatasetError(
            "sealed test overlaps selection games: "
            + ", ".join(overlap[:5])
        )
    config = _training_config_from_json(metadata["config"])
    selected_alpha = float(metadata["selectedFusionAlpha"])
    selected_smoothing = float(metadata["selectedPriorSmoothing"])
    return {
        "checkpoint": {
            "file": checkpoint_path.resolve().name,
            "sha256": checkpoint_sha256,
        },
        "selection": {
            "selectedEpoch": metadata["selectedEpoch"],
            "selectedFusionAlpha": selected_alpha,
            "selectedPriorSmoothing": selected_smoothing,
            "config": metadata["config"],
            "inputs": metadata["inputs"],
        },
        "metrics": evaluate_capturable(
            model,
            test_rows,
            test_tensors,
            config,
            selected_alpha,
            selected_smoothing,
        ),
    }


def _test_performance_order(result: Mapping[str, Any]) -> tuple[float, ...]:
    metrics = result["metrics"]["hybrid"]
    return (
        float(metrics["game_normalized_top_1_accuracy"]),
        float(metrics["game_normalized_top_3_accuracy"]),
        -float(metrics["game_normalized_negative_log_likelihood"]),
    )


def _paired_reliability_checks(
    control: Mapping[str, Any],
    treatment: Mapping[str, Any],
    primary_confirmed: bool,
) -> Mapping[str, bool]:
    control_hybrid = control["metrics"]["hybrid"]
    treatment_hybrid = treatment["metrics"]["hybrid"]
    control_horizons = control_hybrid["accuracy_after_moves"]
    treatment_horizons = treatment_hybrid["accuracy_after_moves"]
    return {
        "primaryRankingConfirmed": primary_confirmed,
        "top1NonRegression": (
            treatment_hybrid["game_normalized_top_1_accuracy"]
            >= control_hybrid["game_normalized_top_1_accuracy"]
        ),
        "top3NonRegression": (
            treatment_hybrid["game_normalized_top_3_accuracy"]
            >= control_hybrid["game_normalized_top_3_accuracy"]
        ),
        "negativeLogLikelihoodNonRegression": (
            treatment_hybrid[
                "game_normalized_negative_log_likelihood"
            ]
            <= control_hybrid[
                "game_normalized_negative_log_likelihood"
            ]
        ),
        "brierNonRegression": (
            treatment_hybrid["game_normalized_brier_score"]
            <= control_hybrid["game_normalized_brier_score"]
        ),
        "calibrationNonRegression": (
            treatment_hybrid["expected_calibration_error"]
            <= control_hybrid["expected_calibration_error"]
        ),
        "allMoveHorizonsNonRegression": all(
            treatment_horizons[horizon] >= control_accuracy
            for horizon, control_accuracy in control_horizons.items()
        ),
        "triggerAccuracyNonRegression": (
            treatment["metrics"]["trigger"]["accuracy"]
            >= control["metrics"]["trigger"]["accuracy"]
        ),
        "forcedAccuracyNonRegression": (
            treatment["metrics"]["forced"]["accuracy"]
            >= control["metrics"]["forced"]["accuracy"]
        ),
    }


def run_paired_sealed_evaluation(
    comparison_path: Path,
    test_path: Path,
    output_path: Path,
) -> Mapping[str, Any]:
    """Evaluate one frozen control/treatment pair in a single sealed pass."""

    if output_path.exists():
        raise FileExistsError(
            "paired sealed evaluation output already exists"
        )
    from .capturable_candidate_selection import (
        load_treatment_comparison,
    )

    comparison, comparison_sha256 = load_treatment_comparison(
        comparison_path
    )
    test_rows, test_sha256 = _load_stable_capturable_dataset(test_path)
    test_tensors = tensorize(test_rows)
    root = comparison_path.resolve().parent

    def checkpoint_path(candidate: Mapping[str, Any]) -> Path:
        return (
            root
            / str(candidate["selectionDirectory"])
            / str(candidate["checkpointFile"])
        )

    control = _evaluate_selection_on_test(
        checkpoint_path(comparison["control"]),
        test_rows,
        test_tensors,
    )
    treatment = _evaluate_selection_on_test(
        checkpoint_path(comparison["bestTreatment"]),
        test_rows,
        test_tensors,
    )
    control_order = _test_performance_order(control)
    treatment_order = _test_performance_order(treatment)
    confirmed = treatment_order > control_order
    reliability_checks = _paired_reliability_checks(
        control,
        treatment,
        confirmed,
    )
    reliable = all(reliability_checks.values())
    control_hybrid = control["metrics"]["hybrid"]
    treatment_hybrid = treatment["metrics"]["hybrid"]
    report = {
        "format": (
            "drawbackguesser-capturable-paired-sealed-evaluation"
        ),
        "version": 2,
        "comparison": {
            "file": comparison_path.resolve().name,
            "sha256": comparison_sha256,
            "validationDecision": comparison["decision"],
        },
        "test": {
            "input": {
                "path": test_path.resolve().name,
                "sha256": test_sha256,
                "rows": len(test_rows),
                "games": len(
                    {row.evaluation.game_id for row in test_rows}
                ),
            },
            "control": control,
            "treatment": treatment,
            "gameNormalizedDeltas": {
                "top1": (
                    treatment_hybrid["game_normalized_top_1_accuracy"]
                    - control_hybrid["game_normalized_top_1_accuracy"]
                ),
                "top3": (
                    treatment_hybrid["game_normalized_top_3_accuracy"]
                    - control_hybrid["game_normalized_top_3_accuracy"]
                ),
                "negativeLogLikelihood": (
                    treatment_hybrid[
                        "game_normalized_negative_log_likelihood"
                    ]
                    - control_hybrid[
                        "game_normalized_negative_log_likelihood"
                    ]
                ),
            },
            "primaryDecision": (
                "confirm-treatment" if confirmed else "reject-treatment"
            ),
            "reliabilityChecks": reliability_checks,
            "releaseDecision": (
                "promote-treatment" if reliable else "retain-control"
            ),
        },
        "sealedTestStatus": "consumed",
    }
    payload = _canonical_json(report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _publish_bytes(output_path, payload)
    return {
        "reportPath": str(output_path),
        "reportSha256": hashlib.sha256(payload).hexdigest(),
        "primaryDecision": report["test"]["primaryDecision"],
        "releaseDecision": report["test"]["releaseDecision"],
        "testSha256": test_sha256,
        "controlGameNormalizedTop1": control_order[0],
        "treatmentGameNormalizedTop1": treatment_order[0],
        "controlGameNormalizedTop3": control_order[1],
        "treatmentGameNormalizedTop3": treatment_order[1],
    }


def run_candidate_selection(
    report_paths: Sequence[Path],
    output_path: Path,
) -> Mapping[str, Any]:
    """Choose among frozen candidates using validation metrics only."""

    from .capturable_candidate_selection import (
        run_candidate_selection as choose_candidate,
    )

    return choose_candidate(report_paths, output_path)


def run_treatment_comparison(
    control_report_path: Path,
    treatment_report_paths: Sequence[Path],
    output_path: Path,
) -> Mapping[str, Any]:
    """Compare a frozen control with train-only interventions."""

    from .capturable_candidate_selection import (
        run_treatment_comparison as compare_treatment,
    )

    return compare_treatment(
        control_report_path,
        treatment_report_paths,
        output_path,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select a fresh capturable model without test access, then "
            "evaluate its frozen checkpoint on a sealed split."
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)

    select = commands.add_parser(
        "select",
        help="Train and select using only train and validation datasets.",
    )
    select.add_argument(
        "--train",
        type=Path,
        action="append",
        required=True,
        help="Repeat for each disjoint training dataset.",
    )
    select.add_argument("--validation", type=Path, required=True)
    select.add_argument("--output", type=Path, required=True)
    select.add_argument("--seed", type=int, default=0xC0DE_0701)
    select.add_argument("--epochs", type=int, default=8)
    select.add_argument("--batch-size", type=int, default=256)
    select.add_argument("--hidden-dimension", type=int, default=128)
    select.add_argument("--torch-threads", type=int, default=14)
    select.add_argument(
        "--trigger-row-multiplier",
        type=float,
        default=1.0,
    )

    evaluate = commands.add_parser(
        "evaluate",
        help="Evaluate one frozen selection checkpoint on a sealed split.",
    )
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--test", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)

    choose = commands.add_parser(
        "choose",
        help="Select one frozen candidate using validation reports only.",
    )
    choose.add_argument(
        "--candidate",
        type=Path,
        action="append",
        required=True,
        help="Repeat for each selection.json candidate.",
    )
    choose.add_argument("--output", type=Path, required=True)
    compare = commands.add_parser(
        "compare-treatment",
        help=(
            "Compare train-only interventions against a frozen control "
            "using the exact same validation corpus."
        ),
    )
    compare.add_argument("--control", type=Path, required=True)
    compare.add_argument(
        "--treatment",
        type=Path,
        action="append",
        required=True,
        help="Repeat for each treatment selection.json.",
    )
    compare.add_argument("--output", type=Path, required=True)
    paired = commands.add_parser(
        "evaluate-treatment",
        help=(
            "Evaluate one authenticated control/treatment comparison "
            "against a fresh sealed test in one pass."
        ),
    )
    paired.add_argument("--comparison", type=Path, required=True)
    paired.add_argument("--test", type=Path, required=True)
    paired.add_argument("--output", type=Path, required=True)
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    options = _parser().parse_args(arguments)
    if options.command == "select":
        config = CapturableTrainingConfig(
            seed=options.seed,
            epochs=options.epochs,
            batch_size=options.batch_size,
            hidden_dimension=options.hidden_dimension,
            torch_threads=options.torch_threads,
            trigger_row_multiplier=options.trigger_row_multiplier,
        )
        result = run_selection(
            options.train,
            options.validation,
            options.output,
            config,
        )
    elif options.command == "evaluate":
        result = run_sealed_evaluation(
            options.checkpoint,
            options.test,
            options.output,
        )
    elif options.command == "choose":
        result = run_candidate_selection(
            options.candidate,
            options.output,
        )
    elif options.command == "compare-treatment":
        result = run_treatment_comparison(
            options.control,
            options.treatment,
            options.output,
        )
    else:
        result = run_paired_sealed_evaluation(
            options.comparison,
            options.test,
            options.output,
        )
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
