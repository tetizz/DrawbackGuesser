"""Strict verification of a release selection artifact and all local sources."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Mapping

from ml.training.drawback_ml.training_corpus_set import (
    TrainingCorpusSetError,
    verify_training_corpus_set,
)
from ml.training.drawback_ml.checkpoint import (
    FUSION_GRID_DRAWBACK_OBJECTIVE,
    LEGACY_ADDITIVE_DRAWBACK_OBJECTIVE,
    parse_training_run_drawback_objective,
)
from ml.training.drawback_ml.inference import (
    CheckpointPredictor,
    load_checkpoint_predictor,
)

from .selection import (
    ContentAddressedSummary,
    EpochSelectionSummary,
    LoadedSelectionArtifact,
    load_selection_artifact,
    load_selection_summary,
    fusion_grid_selection_objective_metadata,
    validate_fusion_grid_head_counts,
)
from .validation_partition import VALIDATION_PARTITION_IDENTITY


@dataclass(frozen=True)
class ContentAddressedJson:
    path: Path
    sha256: str

    def __post_init__(self) -> None:
        _digest(self.sha256, "content sha256")


@dataclass(frozen=True)
class TrainingRunIdentity:
    path: Path
    sha256: str
    run_id: str
    seed: int
    epochs: int
    training_corpus_set_sha256: str
    drawback_loss_objective: str = LEGACY_ADDITIVE_DRAWBACK_OBJECTIVE


@dataclass(frozen=True)
class VerifiedSelectionBundle:
    artifact: LoadedSelectionArtifact
    training_run: TrainingRunIdentity
    candidate_count: int


def verify_checkpoint_training_identity(
    predictor: CheckpointPredictor,
    training_run: TrainingRunIdentity,
    *,
    expected_seed: int,
    expected_epoch: int,
    expected_objective: str,
) -> None:
    """Bind a loaded v4 checkpoint to its authenticated training claim."""

    if predictor.training_run_id != training_run.run_id:
        raise ValueError("checkpoint training run disagrees with training claim")
    if predictor.checkpoint_seed != expected_seed:
        raise ValueError("checkpoint seed disagrees with selection candidate")
    if predictor.checkpoint_epoch != expected_epoch:
        raise ValueError("checkpoint epoch disagrees with selection candidate")
    if predictor.drawback_loss_objective != expected_objective:
        raise ValueError(
            "checkpoint drawback objective disagrees with selection artifact"
        )
    provenance = predictor.corpus_provenance
    if not isinstance(provenance, Mapping):
        raise ValueError(
            "checkpoint lacks authenticated training-corpus provenance"
        )
    declared_sha256 = provenance.get("training_corpus_set_sha256")
    if declared_sha256 != training_run.training_corpus_set_sha256:
        raise ValueError(
            "checkpoint training corpus set disagrees with training claim"
        )
    try:
        embedded = verify_training_corpus_set(
            provenance.get("training_corpus_set")
        )
    except TrainingCorpusSetError as error:
        raise ValueError(
            "checkpoint embedded training corpus set is invalid"
        ) from error
    if embedded.get("sha256") != declared_sha256:
        raise ValueError(
            "checkpoint embedded training corpus set disagrees with its digest"
        )


def load_training_run(
    reference: ContentAddressedJson,
) -> TrainingRunIdentity:
    payload = _verified_bytes(reference, "training run")
    value = _strict_json(payload, "training run")
    if _canonical_json(value) != payload:
        raise ValueError("training run JSON bytes are not canonical")
    expected_keys = {"run_id", "format", "version", "config", "runtime", "sampling"}
    if set(value) != expected_keys:
        raise ValueError("training run claim fields are invalid")
    if (
        value.get("format") != "drawbacktrainer-streaming-run"
        or value.get("version") != 1
    ):
        raise ValueError("training run claim format is unsupported")
    config = _object(value.get("config"), "training run config")
    corpus_provenance = _object(
        config.get("corpus_provenance"),
        "training run corpus_provenance",
    )
    try:
        training_corpus_set = verify_training_corpus_set(
            corpus_provenance.get("training_corpus_set")
        )
    except TrainingCorpusSetError as error:
        raise ValueError(
            "training run embeds an invalid training corpus set"
        ) from error
    training_corpus_set_sha256 = _digest(
        corpus_provenance.get("training_corpus_set_sha256"),
        "training corpus set sha256",
    )
    if training_corpus_set.get("sha256") != training_corpus_set_sha256:
        raise ValueError(
            "training run corpus-set digest disagrees with its embedded set"
        )
    runtime = _object(value.get("runtime"), "training run runtime")
    sampling = _object(value.get("sampling"), "training run sampling")
    material = {
        "format": value["format"],
        "version": value["version"],
        "config": dict(config),
        "runtime": dict(runtime),
        "sampling": dict(sampling),
    }
    computed_run_id = hashlib.sha256(
        json.dumps(
            material,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    declared_run_id = _digest(value.get("run_id"), "training run_id")
    if declared_run_id != computed_run_id:
        raise ValueError("training run_id does not match its claim material")
    return TrainingRunIdentity(
        path=reference.path,
        sha256=reference.sha256,
        run_id=declared_run_id,
        seed=_nonnegative_int(config.get("seed"), "training seed"),
        epochs=_positive_int(config.get("epochs"), "training epochs"),
        training_corpus_set_sha256=training_corpus_set_sha256,
        drawback_loss_objective=parse_training_run_drawback_objective(config),
    )


def verify_release_selection_bundle(
    artifact_reference: ContentAddressedJson,
    training_run_reference: ContentAddressedJson,
) -> VerifiedSelectionBundle:
    artifact_payload = _verified_bytes(
        artifact_reference, "selection artifact"
    )
    artifact_value = _strict_json(
        artifact_payload, "selection artifact"
    )
    artifact = load_selection_artifact(
        ContentAddressedSummary(
            artifact_reference.path,
            artifact_reference.sha256,
        )
    )
    training_run = load_training_run(training_run_reference)
    if artifact.provenance.model_run_config_sha256 != training_run.sha256:
        raise ValueError("selection artifact binds a different training run")
    if artifact.training_seed != training_run.seed:
        raise ValueError("selection artifact training seed disagrees with run")
    if artifact.provenance.planned_epoch_count != training_run.epochs:
        raise ValueError("selection artifact epoch plan disagrees with run")
    if (
        artifact.provenance.training_corpus_set_sha256
        != training_run.training_corpus_set_sha256
    ):
        raise ValueError(
            "selection artifact binds a different training corpus set"
        )
    if artifact.objective_id != training_run.drawback_loss_objective:
        raise ValueError(
            "selection artifact drawback objective disagrees with training run"
        )
    candidates = artifact_value["candidates"]
    assert isinstance(candidates, list)  # enforced by strict artifact loader
    evidence_directory = artifact_reference.path.parent
    checkpoint_directory = training_run_reference.path.parent
    for candidate in candidates:
        assert isinstance(candidate, Mapping)
        _verify_candidate(
            evidence_directory,
            checkpoint_directory,
            artifact,
            training_run,
            candidate,
        )
    return VerifiedSelectionBundle(
        artifact=artifact,
        training_run=training_run,
        candidate_count=len(candidates),
    )


def _verify_candidate(
    evidence_directory: Path,
    checkpoint_directory: Path,
    artifact: LoadedSelectionArtifact,
    training_run: TrainingRunIdentity,
    candidate: Mapping[str, object],
) -> None:
    epoch = _positive_int(candidate.get("epoch"), "candidate epoch")
    summary_reference = ContentAddressedSummary(
        _local_file(
            evidence_directory,
            candidate.get("summary_file"),
            "summary",
        ),
        _digest(candidate.get("summary_sha256"), "summary sha256"),
    )
    summary = load_selection_summary(summary_reference)
    if summary.epoch != epoch:
        raise ValueError("candidate epoch disagrees with summary")
    _require_summary_provenance(artifact, summary)
    checkpoint_file = _basename(
        candidate.get("checkpoint_file"), "checkpoint file"
    )
    checkpoint_sha256 = _digest(
        candidate.get("checkpoint_sha256"), "checkpoint sha256"
    )
    report_file = _basename(
        candidate.get("evaluation_report_file"), "evaluation report file"
    )
    report_sha256 = _digest(
        candidate.get("evaluation_report_sha256"),
        "evaluation report sha256",
    )
    if (
        summary.checkpoint_file != checkpoint_file
        or summary.checkpoint_sha256 != checkpoint_sha256
        or summary.evaluation_report_file != report_file
        or summary.evaluation_report_sha256 != report_sha256
    ):
        raise ValueError("candidate source bindings disagree with summary")
    white_nll = _finite_nonnegative(
        candidate.get("white_nll"), "candidate White NLL"
    )
    black_nll = _finite_nonnegative(
        candidate.get("black_nll"), "candidate Black NLL"
    )
    mean_nll = _finite_nonnegative(
        candidate.get("mean_nll"), "candidate mean NLL"
    )
    if (
        summary.white_nll != white_nll
        or summary.black_nll != black_nll
        or summary.mean_nll != mean_nll
    ):
        raise ValueError("candidate metrics disagree with summary")
    checkpoint = ContentAddressedJson(
        _local_file(
            checkpoint_directory,
            checkpoint_file,
            "checkpoint",
        ),
        checkpoint_sha256,
    )
    checkpoint_payload = _verified_bytes(checkpoint, "checkpoint")
    # Version 4 authenticates the executable checkpoint contract recursively.
    # Version 3 remains byte-addressed for compatibility with already
    # published legacy evidence whose fixtures predate reconstructable models.
    if artifact.objective_id == FUSION_GRID_DRAWBACK_OBJECTIVE:
        predictor = load_checkpoint_predictor(io.BytesIO(checkpoint_payload))
        verify_checkpoint_training_identity(
            predictor,
            training_run,
            expected_seed=artifact.training_seed,
            expected_epoch=epoch,
            expected_objective=artifact.objective_id,
        )
    report_reference = ContentAddressedJson(
        _local_file(
            evidence_directory,
            report_file,
            "evaluation report",
        ),
        report_sha256,
    )
    report = _strict_json(
        _verified_bytes(report_reference, "evaluation report"),
        "evaluation report",
    )
    _verify_report(report, artifact, training_run, summary)


def _require_summary_provenance(
    artifact: LoadedSelectionArtifact,
    summary: EpochSelectionSummary,
) -> None:
    if summary.provenance != artifact.provenance:
        raise ValueError("summary provenance disagrees with selection artifact")
    if summary.training_seed != artifact.training_seed:
        raise ValueError("summary training seed disagrees with selection artifact")
    if summary.partition_seed_sha256 != artifact.partition_seed_sha256:
        raise ValueError("summary partition disagrees with selection artifact")
    if summary.objective_id != artifact.objective_id:
        raise ValueError("summary drawback objective disagrees with selection artifact")


def _verify_report(
    report: Mapping[str, object],
    artifact: LoadedSelectionArtifact,
    training_run: TrainingRunIdentity,
    summary: EpochSelectionSummary,
) -> None:
    evaluation = _object(report.get("evaluation"), "report evaluation")
    partition = _object(
        evaluation.get("validationPartition"), "report partition"
    )
    expected_partition = {
        "identity": VALIDATION_PARTITION_IDENTITY,
        "name": "selection",
        "seedSha256": artifact.partition_seed_sha256,
    }
    if dict(partition) != expected_partition:
        raise ValueError("evaluation report partition disagrees with artifact")
    provenance = _object(report.get("provenance"), "report provenance")
    expected_provenance = {
        "release_root_sha256": artifact.provenance.release_root_sha256,
        "corpus_run_id": artifact.provenance.corpus_run_id,
        "manifest_sha256": (
            artifact.provenance.private_validation_manifest_sha256
        ),
        "dataset_sha256": artifact.provenance.validation_dataset_sha256,
        "checkpoint_file": summary.checkpoint_file,
        "checkpoint_sha256": summary.checkpoint_sha256,
        "checkpoint_seed": artifact.training_seed,
        "checkpoint_epoch": summary.epoch,
        "training_run_id": training_run.run_id,
    }
    for key, expected in expected_provenance.items():
        if provenance.get(key) != expected:
            raise ValueError(f"evaluation report provenance mismatch: {key}")
    if summary.objective_id == FUSION_GRID_DRAWBACK_OBJECTIVE:
        if report.get("formatVersion") != 2:
            raise ValueError("fusion-grid evaluation report format is invalid")
        if (
            provenance.get("drawback_loss_objective")
            != FUSION_GRID_DRAWBACK_OBJECTIVE
        ):
            raise ValueError(
                "evaluation report drawback objective disagrees with summary"
            )
        selection = _object(
            report.get("epochSelectionObjective"),
            "report epoch selection objective",
        )
        if set(selection) != {"identity", "metrics"}:
            raise ValueError("evaluation report selection objective fields are invalid")
        identity = _object(
            selection.get("identity"),
            "report epoch selection objective identity",
        )
        if dict(identity) != fusion_grid_selection_objective_metadata():
            raise ValueError("evaluation report selection objective is invalid")
        head_metrics = _object(
            selection.get("metrics"),
            "report epoch selection objective metrics",
        )
        if set(head_metrics) != {"white", "black"}:
            raise ValueError("evaluation report selection metric heads are invalid")
        white = _object(head_metrics.get("white"), "White selection metrics")
        black = _object(head_metrics.get("black"), "Black selection metrics")
        expected_metric_keys = {
            "observation_count",
            "player_game_count",
            "player_game_normalized_nll",
        }
        if set(white) != expected_metric_keys or set(black) != expected_metric_keys:
            raise ValueError("evaluation report selection head fields are invalid")
        for color, head in (("White", white), ("Black", black)):
            validate_fusion_grid_head_counts(head, color)
        if (
            white.get("player_game_normalized_nll") != summary.white_nll
            or black.get("player_game_normalized_nll") != summary.black_nll
        ):
            raise ValueError("evaluation report metrics disagree with summary")
    else:
        if report.get("formatVersion") not in {None, 1}:
            raise ValueError("legacy evaluation report format is invalid")
        metrics = _object(report.get("metrics"), "report metrics")
        white = _object(metrics.get("white_drawback"), "White metrics")
        black = _object(metrics.get("black_drawback"), "Black metrics")
        if (
            white.get("negative_log_likelihood") != summary.white_nll
            or black.get("negative_log_likelihood") != summary.black_nll
        ):
            raise ValueError("evaluation report metrics disagree with summary")


def _verified_bytes(reference: ContentAddressedJson, name: str) -> bytes:
    try:
        payload = reference.path.read_bytes()
    except OSError as error:
        raise ValueError(f"cannot read {name}: {reference.path}") from error
    if hashlib.sha256(payload).hexdigest() != reference.sha256:
        raise ValueError(f"{name} sha256 does not match")
    return payload


def _strict_json(payload: bytes, name: str) -> Mapping[str, object]:
    def pairs(values: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in values:
            if key in result:
                raise ValueError(f"{name} contains duplicate key {key!r}")
            result[key] = value
        return result

    def constant(token: str) -> None:
        raise ValueError(f"{name} contains non-finite constant {token}")

    try:
        value = json.loads(
            payload,
            object_pairs_hook=pairs,
            parse_constant=constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is not strict UTF-8 JSON") from error
    _require_finite(value, name)
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} root must be an object")
    return value


def _canonical_json(value: Mapping[str, object]) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _require_finite(value: object, name: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{name} contains a non-finite number")
    if isinstance(value, Mapping):
        for item in value.values():
            _require_finite(item, name)
    elif isinstance(value, list):
        for item in value:
            _require_finite(item, name)


def _local_file(directory: Path, value: object, name: str) -> Path:
    basename = _basename(value, f"{name} file")
    path = directory / basename
    if not path.is_file():
        raise ValueError(f"{name} file is missing: {basename}")
    return path


def _basename(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or Path(value).name != value:
        raise ValueError(f"{name} must be a non-empty basename")
    return value


def _digest(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _positive_int(value: object, name: str) -> int:
    result = _nonnegative_int(value, name)
    if result == 0:
        raise ValueError(f"{name} must be positive")
    return result


def _finite_nonnegative(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _object(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value
