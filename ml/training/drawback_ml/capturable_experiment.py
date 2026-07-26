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
        "validationHybridTop1": report["validation"]["hybrid"][
            "top_1_accuracy"
        ],
        "validationHybridTop3": report["validation"]["hybrid"][
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


def run_candidate_selection(
    report_paths: Sequence[Path],
    output_path: Path,
) -> Mapping[str, Any]:
    """Choose among frozen candidates using validation metrics only."""

    from .capturable_candidate_selection import (
        run_candidate_selection as choose_candidate,
    )

    return choose_candidate(report_paths, output_path)


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
    else:
        result = run_candidate_selection(
            options.candidate,
            options.output,
        )
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
