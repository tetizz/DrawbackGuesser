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
    SOURCE_WEIGHTING_OBJECTIVE,
    _canonical_json,
    _checked_positive_weight,
    _checked_train_source_weights,
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
from .capturable_reliability import validation_reliability_checks

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
    weights: Sequence[float] | None = None,
) -> list[Mapping[str, Any]]:
    return [
        {
            "path": path.resolve().name,
            "sha256": sha256,
            "rows": len(rows),
            "games": len({row.evaluation.game_id for row in rows}),
            **({} if weights is None else {"weight": weights[index]}),
        }
        for index, (path, rows, sha256) in enumerate(
            zip(
                paths,
                sources,
                sha256s,
                strict=True,
            )
        )
    ]


def run_selection(
    train_paths: Sequence[Path],
    validation_path: Path,
    output_directory: Path,
    config: CapturableTrainingConfig,
    train_source_weights: Sequence[float] | None = None,
) -> Mapping[str, Any]:
    """Select and publish a model without opening any test dataset."""

    report_path = output_directory / "selection.json"
    checkpoint_path = output_directory / "model.pt"
    if report_path.exists() or checkpoint_path.exists():
        raise FileExistsError("capturable selection outputs already exist")
    if not train_paths:
        raise ValueError("at least one train dataset is required")
    checked_source_weights = _checked_train_source_weights(
        train_paths,
        train_source_weights,
    )
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
    game_source_weights = (
        None
        if checked_source_weights is None
        else {
            row.evaluation.game_id: weight
            for source, weight in zip(
                train_sources,
                checked_source_weights,
                strict=True,
            )
            for row in source
        }
    )
    model, measured = train_capturable_baseline(
        train_rows,
        validation_rows,
        None,
        config,
        train_game_source_weights=game_source_weights,
    )
    input_identity = {
        "train": {
            "sources": _capturable_source_identity(
                train_paths,
                train_sources,
                train_sha256s,
                checked_source_weights,
            ),
            "rows": len(train_rows),
            "games": len({row.evaluation.game_id for row in train_rows}),
            **(
                {}
                if checked_source_weights is None
                else {
                    "sourceWeightingObjective": (
                        SOURCE_WEIGHTING_OBJECTIVE
                    )
                }
            ),
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
    output_directory.mkdir(parents=True, exist_ok=True)
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
    _validate_source_weighting_identity(metadata["inputs"])
    model = create_capturable_model(config.hidden_dimension)
    try:
        model.load_state_dict(checkpoint.get("stateDict"), strict=True)
    except (RuntimeError, TypeError) as error:
        raise CapturableDatasetError(
            "checkpoint tensors do not match the selected model"
        ) from error
    return model, metadata, digest.hexdigest()


def _validate_source_weighting_identity(inputs: Mapping[str, Any]) -> None:
    train = inputs.get("train")
    if not isinstance(train, Mapping):
        raise CapturableDatasetError(
            "checkpoint training inputs are invalid"
        )
    sources = train.get("sources")
    if not isinstance(sources, list) or not sources:
        raise CapturableDatasetError(
            "checkpoint training sources are invalid"
        )
    weight_presence = []
    for source in sources:
        if not isinstance(source, Mapping):
            raise CapturableDatasetError(
                "checkpoint training sources are invalid"
            )
        weight_presence.append("weight" in source)
        if "weight" in source:
            try:
                _checked_positive_weight(
                    source["weight"],
                    "checkpoint source weight",
                )
            except ValueError as error:
                raise CapturableDatasetError(
                    "checkpoint source weight is invalid"
                ) from error
    objective = train.get("sourceWeightingObjective")
    if any(weight_presence):
        if (
            not all(weight_presence)
            or objective != SOURCE_WEIGHTING_OBJECTIVE
        ):
            raise CapturableDatasetError(
                "checkpoint source weighting objective is invalid"
            )
    elif "sourceWeightingObjective" in train:
        raise CapturableDatasetError(
            "unweighted checkpoint declares a source weighting objective"
        )


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


def _evaluate_loaded_selection_on_test(
    checkpoint_path: Path,
    model: Any,
    metadata: Mapping[str, Any],
    checkpoint_sha256: str,
    test_rows: Sequence[CapturableDatasetRow],
    test_tensors: Any,
) -> Mapping[str, Any]:
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


def _load_bound_selection_checkpoint(
    checkpoint_path: Path,
    expected_sha256: str,
) -> tuple[Path, Any, Mapping[str, Any], str]:
    model, metadata, measured_sha256 = _load_selection_checkpoint(
        checkpoint_path.resolve()
    )
    if measured_sha256 != expected_sha256:
        raise CapturableDatasetError(
            f"{checkpoint_path.name} changed after comparison authentication"
        )
    return checkpoint_path, model, metadata, measured_sha256


def _test_performance_order(result: Mapping[str, Any]) -> tuple[float, ...]:
    metrics = result["metrics"]["hybrid"]
    return (
        float(metrics["game_normalized_top_1_accuracy"]),
        float(metrics["game_normalized_top_3_accuracy"]),
        -float(metrics["game_normalized_negative_log_likelihood"]),
    )


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
    if comparison["releaseDecision"] != "promote-treatment":
        raise CapturableDatasetError(
            "validation comparison did not authorize sealed test"
        )
    root = comparison_path.resolve().parent

    def checkpoint_path(candidate: Mapping[str, Any]) -> Path:
        return (
            root
            / str(candidate["selectionDirectory"])
            / str(candidate["checkpointFile"])
        )

    control_loaded = _load_bound_selection_checkpoint(
        checkpoint_path(comparison["control"]),
        str(comparison["control"]["checkpointSha256"]),
    )
    treatment_loaded = _load_bound_selection_checkpoint(
        checkpoint_path(comparison["bestTreatment"]),
        str(comparison["bestTreatment"]["checkpointSha256"]),
    )
    test_rows, test_sha256 = _load_stable_capturable_dataset(test_path)
    test_tensors = tensorize(test_rows)
    control = _evaluate_loaded_selection_on_test(
        *control_loaded,
        test_rows,
        test_tensors,
    )
    treatment = _evaluate_loaded_selection_on_test(
        *treatment_loaded,
        test_rows,
        test_tensors,
    )
    control_order = _test_performance_order(control)
    treatment_order = _test_performance_order(treatment)
    confirmed = treatment_order > control_order
    reliability_checks = validation_reliability_checks(
        control["metrics"],
        treatment["metrics"],
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
    select.add_argument(
        "--train-source-weight",
        type=float,
        action="append",
        help=(
            "Optional positive weight for each --train dataset, in the same "
            "order. Supply one value per --train."
        ),
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
            options.train_source_weight,
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
