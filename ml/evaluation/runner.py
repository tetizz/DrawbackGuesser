"""Held-out checkpoint evaluation with explicit label/inference separation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, BinaryIO, Callable, Iterable, Mapping, Protocol, Sequence

from ml.training.drawback_ml.features import encode_move
from ml.training.drawback_ml.inference import InferenceOutput
from ml.training.drawback_ml.parameters import (
    MASKED_PARAMETER_TOKEN,
    supervised_parameter_label,
)
from ml.training.drawback_ml.records import (
    DatasetSchemaError,
    FeatureRecord,
    TrainingExample,
    group_training_examples,
    parse_dataset_row,
)

from .metrics import (
    BinaryEvaluationReport,
    EvaluationReport,
    LegalMaskEvaluationReport,
    PredictionExample,
    StreamingBinaryEvaluation,
    StreamingEvaluation,
    StreamingLegalMaskEvaluation,
)
from .calibration import CalibrationExample
from .calibration_release import CalibrationObservation
from .splits import SplitManifest


class EvaluationDataError(ValueError):
    """Raised when held-out data cannot be scored without guessing."""


class Predictor(Protocol):
    drawback_vocabulary: Sequence[str]
    parameter_vocabulary: Sequence[str]
    legal_mask_dimension: int

    def predict(self, features: FeatureRecord) -> InferenceOutput: ...


def _batched(
    examples: Iterable[TrainingExample],
    batch_size: int,
) -> Iterable[tuple[TrainingExample, ...]]:
    if batch_size <= 0:
        raise ValueError("evaluation batch_size must be positive")
    batch: list[TrainingExample] = []
    for example in examples:
        batch.append(example)
        if len(batch) == batch_size:
            yield tuple(batch)
            batch.clear()
    if batch:
        yield tuple(batch)


def _predict_batch(
    predictor: Predictor,
    batch: Sequence[TrainingExample],
    legal_indices: Sequence[Sequence[int]],
) -> tuple[tuple[InferenceOutput, ...], Any | None]:
    statistics_method = getattr(
        predictor,
        "predict_batch_with_legal_statistics",
        None,
    )
    if callable(statistics_method):
        outputs, statistics = statistics_method(
            tuple(item.features for item in batch),
            legal_indices,
        )
        outputs = tuple(outputs)
        if len(outputs) != len(batch):
            raise EvaluationDataError(
                "predictor batch output count does not match its input count"
            )
        return outputs, statistics
    batch_method = getattr(predictor, "predict_batch", None)
    if callable(batch_method):
        outputs = tuple(batch_method(tuple(item.features for item in batch)))
    else:
        outputs = tuple(predictor.predict(item.features) for item in batch)
    if len(outputs) != len(batch):
        raise EvaluationDataError(
            "predictor batch output count does not match its input count"
        )
    return outputs, None


@dataclass(frozen=True)
class MeasuredEvaluationReport:
    split: str
    move_examples: int
    white_drawback: EvaluationReport
    black_drawback: EvaluationReport
    trigger: BinaryEvaluationReport
    legal_mask: LegalMaskEvaluationReport
    white_unscorable_parameter_examples: int
    black_unscorable_parameter_examples: int


def load_rule_families(catalog_paths: Iterable[Path]) -> Mapping[str, str]:
    families: dict[str, str] = {}
    for path in catalog_paths:
        if not path.exists():
            raise EvaluationDataError(f"rule-family catalog does not exist: {path}")
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict) and isinstance(value.get("entries"), list):
            entries = value["entries"]
        elif isinstance(value, list):
            entries = value
        else:
            raise EvaluationDataError(
                f"{path} must contain a JSON array or an entries array"
            )
        for entry in entries:
            if not isinstance(entry, dict):
                raise EvaluationDataError(f"{path} contains a non-object entry")
            rule_id = entry.get("id")
            family = entry.get("ruleFamily")
            if isinstance(rule_id, str) and isinstance(family, str):
                prior = families.get(rule_id)
                if prior is not None and prior != family:
                    raise EvaluationDataError(
                        f"conflicting ruleFamily metadata for {rule_id}"
                    )
                families[rule_id] = family
    return families


def read_ndjson(path: Path) -> Iterable[Mapping[str, Any]]:
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise EvaluationDataError(
                    f"{path}:{line_number} is not valid JSON"
                ) from error
            if not isinstance(value, dict):
                raise EvaluationDataError(f"{path}:{line_number} is not an object")
            yield value


def read_ndjson_stream(
    source: BinaryIO,
    *,
    label: str,
) -> Iterable[Mapping[str, Any]]:
    """Read rows from an authenticated, caller-owned binary handle."""

    source.seek(0)
    for line_number, raw_line in enumerate(source, start=1):
        if not raw_line.strip():
            continue
        try:
            line = raw_line.decode("utf-8", errors="strict")
            value = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise EvaluationDataError(
                f"{label}:{line_number} is not valid UTF-8 JSON"
            ) from error
        if not isinstance(value, dict):
            raise EvaluationDataError(
                f"{label}:{line_number} is not an object"
            )
        yield value


def _held_out_seeds(split: str, manifest: SplitManifest) -> frozenset[int]:
    if split == "validation":
        return frozenset(manifest.validation)
    if split == "test":
        return frozenset(manifest.test)
    raise EvaluationDataError("evaluation split must be validation or test")


def decode_predicted_parameter(
    probabilities: Mapping[str, float],
) -> Mapping[str, object] | None:
    """Decode the deterministic maximum parameter token from model output."""

    if not probabilities:
        raise EvaluationDataError("parameter probabilities must not be empty")
    token = min(probabilities, key=lambda item: (-probabilities[item], item))
    if token == MASKED_PARAMETER_TOKEN:
        return None
    decoded = json.loads(token)
    if not isinstance(decoded, dict):
        raise EvaluationDataError("predicted parameter token is not a JSON object")
    return decoded


def decode_true_parameter(
    token: str | None,
    vocabulary: frozenset[str],
) -> tuple[Mapping[str, object] | None, bool]:
    """Decode a supervised parameter label and report vocabulary misses."""

    supervised = supervised_parameter_label(token)
    if supervised is None:
        return None, False
    if supervised not in vocabulary:
        return None, True
    decoded = json.loads(supervised)
    if not isinstance(decoded, dict):
        raise EvaluationDataError("true parameter token is not a JSON object")
    return decoded, False


def _prediction_example(
    *,
    example: TrainingExample,
    color: str,
    probabilities: Mapping[str, float],
    parameter_probabilities: Mapping[str, float],
    parameter_vocabulary: frozenset[str],
    rule_families: Mapping[str, str],
    hard_eliminated: Sequence[bool] | None,
    drawback_vocabulary: Sequence[str],
) -> tuple[PredictionExample, bool]:
    if (
        hard_eliminated is not None
        and len(hard_eliminated) != len(drawback_vocabulary)
    ):
        raise EvaluationDataError("hard-elimination mask dimension mismatch")
    drawback = example.white_drawback if color == "white" else example.black_drawback
    parameter_token = (
        example.white_parameters if color == "white" else example.black_parameters
    )
    parameter_observed = (
        example.white_parameters_observed
        if color == "white"
        else example.black_parameters_observed
    )
    if drawback not in probabilities:
        raise EvaluationDataError(
            f"checkpoint cannot score unseen drawback label {drawback}"
        )
    if parameter_observed:
        true_parameters, unscorable = decode_true_parameter(
            parameter_token, parameter_vocabulary
        )
    else:
        true_parameters, unscorable = None, True
    predicted_parameters = (
        None if true_parameters is None
        else decode_predicted_parameter(parameter_probabilities)
    )
    return (
        PredictionExample(
            game_id=example.game_id,
            move_number=example.features.move_number,
            observed_ply=example.features.ply // 2 + 1,
            player_color=color,
            true_drawback=drawback,
            probabilities=probabilities,
            rule_family=rule_families.get(drawback),
            true_parameters=true_parameters,
            predicted_parameters=predicted_parameters,
            hard_eliminated=(
                None
                if hard_eliminated is None
                else dict(
                    zip(
                        drawback_vocabulary,
                        hard_eliminated,
                        strict=True,
                    )
                )
            ),
        ),
        unscorable,
    )


def _stream_examples(
    rows: Iterable[Mapping[str, Any]],
    assignments: Mapping[str, tuple[str, str]],
    max_rows_per_game: int,
) -> Iterable[TrainingExample]:
    """Convert canonical game-contiguous rows with one-game memory."""

    if max_rows_per_game <= 0:
        raise ValueError("max_rows_per_game must be positive")
    current_game: str | None = None
    current_rows: list[Mapping[str, Any]] = []

    def flush() -> Iterable[TrainingExample]:
        if current_rows:
            yield from group_training_examples(current_rows, assignments)

    for raw in rows:
        try:
            row = parse_dataset_row(raw)
        except DatasetSchemaError as error:
            raise EvaluationDataError(str(error)) from error
        assignment = assignments.get(row.game_id)
        if assignment is None:
            raise EvaluationDataError(
                f"game {row.game_id} has no authenticated assignment"
            )
        expected = (
            assignment[0]
            if row.labels.player_color == "white"
            else assignment[1]
        )
        if row.labels.true_drawback != expected:
            raise EvaluationDataError(
                f"game {row.game_id} labels disagree with its assignment"
            )
        if current_game is not None and row.game_id != current_game:
            yield from flush()
            current_rows.clear()
        current_game = row.game_id
        current_rows.append(raw)
        if len(current_rows) > max_rows_per_game:
            raise EvaluationDataError(
                f"game {row.game_id} exceeds the evaluation row bound"
            )
    yield from flush()
def evaluate_held_out(
    rows: Iterable[Mapping[str, Any]],
    *,
    predictor: Predictor,
    split: str,
    manifest: SplitManifest,
    rule_families: Mapping[str, str] | None = None,
    game_assignments: Mapping[str, tuple[str, str]] | None = None,
    max_rows_per_game: int = 10_000,
    batch_size: int = 256,
    calibration_sink: Callable[[CalibrationObservation], None] | None = None,
    prediction_sink: (
        Callable[[TrainingExample, InferenceOutput], None] | None
    ) = None,
) -> MeasuredEvaluationReport:
    examples: Iterable[TrainingExample] = (
        group_training_examples(rows)
        if game_assignments is None
        else _stream_examples(rows, game_assignments, max_rows_per_game)
    )
    resolved_rule_families = rule_families or {}
    allowed_seeds = _held_out_seeds(split, manifest)
    white_metrics = StreamingEvaluation()
    black_metrics = StreamingEvaluation()
    trigger_metrics = StreamingBinaryEvaluation()
    legal_metrics = StreamingLegalMaskEvaluation(predictor.legal_mask_dimension)
    white_unscorable = black_unscorable = 0
    parameter_vocabulary = frozenset(predictor.parameter_vocabulary)
    move_examples = 0

    for batch in _batched(examples, batch_size):
        for example in batch:
            if example.seed not in allowed_seeds:
                raise EvaluationDataError(
                    f"{split} dataset contains seed outside its manifest: "
                    f"{example.seed}"
                )
        batch_legal_indices: list[list[int]] = []
        for example in batch:
            indices: list[int] = []
            for legal_move in example.drawback_legal_moves:
                index = encode_move(legal_move)
                if index >= predictor.legal_mask_dimension:
                    raise EvaluationDataError(
                        "checkpoint legal-mask vocabulary is too small"
                    )
                indices.append(index)
            batch_legal_indices.append(indices)
        outputs, legal_statistics = _predict_batch(
            predictor,
            batch,
            batch_legal_indices,
        )
        if legal_statistics is not None:
            legal_metrics.add_batch_statistics(
                example_count=legal_statistics.example_count,
                dimension=legal_statistics.dimension,
                exact_matches=legal_statistics.exact_matches,
                true_positives=legal_statistics.true_positives,
                false_positives=legal_statistics.false_positives,
                false_negatives=legal_statistics.false_negatives,
                binary_cross_entropy_sum=(
                    legal_statistics.binary_cross_entropy_sum
                ),
                has_infinite_loss=legal_statistics.has_infinite_loss,
            )
        for example, output in zip(batch, outputs, strict=True):
            if example.features.player_color == "white":
                white, white_unknown = _prediction_example(
                    example=example,
                    color="white",
                    probabilities=output.white_drawback_probabilities,
                    parameter_probabilities=(
                        output.white_parameter_probabilities
                    ),
                    parameter_vocabulary=parameter_vocabulary,
                    rule_families=resolved_rule_families,
                    hard_eliminated=output.white_hard_eliminated,
                    drawback_vocabulary=predictor.drawback_vocabulary,
                )
                white_metrics.add(white)
                white_unscorable += white_unknown
                fused_logits = output.white_fused_logits
                hard_eliminated = output.white_hard_eliminated
                true_drawback = example.white_drawback
            else:
                black, black_unknown = _prediction_example(
                    example=example,
                    color="black",
                    probabilities=output.black_drawback_probabilities,
                    parameter_probabilities=(
                        output.black_parameter_probabilities
                    ),
                    parameter_vocabulary=parameter_vocabulary,
                    rule_families=resolved_rule_families,
                    hard_eliminated=output.black_hard_eliminated,
                    drawback_vocabulary=predictor.drawback_vocabulary,
                )
                black_metrics.add(black)
                black_unscorable += black_unknown
                fused_logits = output.black_fused_logits
                hard_eliminated = output.black_hard_eliminated
                true_drawback = example.black_drawback
            if prediction_sink is not None:
                prediction_sink(example, output)
            if calibration_sink is not None:
                if fused_logits is None or hard_eliminated is None:
                    raise EvaluationDataError(
                        "calibration-fit requires genuine fused logits and hard masks"
                    )
                try:
                    true_index = tuple(predictor.drawback_vocabulary).index(
                        true_drawback
                    )
                except ValueError as error:
                    raise EvaluationDataError(
                        "calibration truth is outside predictor vocabulary"
                    ) from error
                calibration_sink(
                    CalibrationObservation(
                        example.features.player_color,
                        CalibrationExample(
                            fused_logits,
                            true_index,
                            hard_eliminated,
                        ),
                    )
                )
            trigger_metrics.add(example.rule_triggered, output.trigger_probability)
            move_examples += 1
        if legal_statistics is None:
            for indices, output in zip(
                batch_legal_indices,
                outputs,
                strict=True,
            ):
                if (
                    len(output.legal_mask_probabilities)
                    != predictor.legal_mask_dimension
                ):
                    raise EvaluationDataError("legal-mask output dimension mismatch")
                legal_metrics.add(indices, output.legal_mask_probabilities)

    if move_examples == 0:
        raise EvaluationDataError("held-out dataset contains no move examples")

    return MeasuredEvaluationReport(
        split=split,
        move_examples=move_examples,
        white_drawback=white_metrics.report(),
        black_drawback=black_metrics.report(),
        trigger=trigger_metrics.report(),
        legal_mask=legal_metrics.report(),
        white_unscorable_parameter_examples=white_unscorable,
        black_unscorable_parameter_examples=black_unscorable,
    )
