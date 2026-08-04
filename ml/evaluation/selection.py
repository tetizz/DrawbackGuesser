"""Validation-selection-only epoch choice and immutable selection artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from ml.training.drawback_ml.durable_publish import publish_bytes_durable_exact
from ml.training.drawback_ml.path_validation import is_portable_safe_basename

from ml.training.drawback_ml.checkpoint import (
    FUSION_GRID_DRAWBACK_OBJECTIVE,
    LEGACY_ADDITIVE_DRAWBACK_OBJECTIVE,
    fusion_grid_drawback_objective_metadata,
)
from ml.training.drawback_ml.rank_preserving_fusion import (
    RankPreservingFusionError,
    rank_preserving_fusion,
)
from ml.training.drawback_ml.symbolic import FUSION_AWARE_LOSS_ALPHA_GRID

from .validation_partition import (
    VALIDATION_PARTITION_IDENTITY,
    ValidationPartition,
)


LEGACY_SELECTION_FORMAT_VERSION = 3
SELECTION_SUMMARY_FORMAT_VERSION = 4
SELECTION_ARTIFACT_FORMAT_VERSION = 4
NLL_TIE_TOLERANCE = 0.005
FUSION_GRID_SELECTION_OBJECTIVE = {
    "checkpoint_objective": fusion_grid_drawback_objective_metadata(),
    "metric": "player-game-normalized-nll",
    "move_metric": "negative-log-likelihood",
    "alpha_grid": list(FUSION_AWARE_LOSS_ALPHA_GRID),
    "alpha_aggregation": "arithmetic-mean",
    "player_game_aggregation": "mean-moves-then-mean-player-games",
    "color_aggregation": "separate-heads-then-arithmetic-mean",
}


@dataclass(frozen=True)
class FusionGridHeadMetric:
    observation_count: int
    player_game_count: int
    player_game_normalized_nll: float


class FusionGridEpochScorer:
    """Score one checkpoint using the frozen deployment fusion grid."""

    def __init__(self) -> None:
        self._losses: dict[tuple[str, str], dict[int, float]] = {}

    def add(
        self,
        *,
        game_id: str,
        color: str,
        observed_ply: int,
        true_index: int,
        residual_logits: Sequence[float],
        symbolic_prior: Sequence[float],
        hard_eliminated: Sequence[bool],
    ) -> None:
        if not game_id or color not in {"white", "black"}:
            raise ValueError("fusion-grid observation identity is invalid")
        if (
            isinstance(observed_ply, bool)
            or not isinstance(observed_ply, int)
            or observed_ply <= 0
        ):
            raise ValueError("fusion-grid observed ply must be positive")
        if (
            isinstance(true_index, bool)
            or not isinstance(true_index, int)
            or true_index < 0
            or true_index >= len(residual_logits)
        ):
            raise ValueError("fusion-grid true index is invalid")
        if len(symbolic_prior) != len(residual_logits) or len(
            hard_eliminated
        ) != len(residual_logits):
            raise ValueError("fusion-grid dimensions disagree")
        if hard_eliminated[true_index]:
            raise ValueError("fusion-grid symbolic engine eliminated the truth")
        losses: list[float] = []
        for alpha in FUSION_AWARE_LOSS_ALPHA_GRID:
            try:
                fused = rank_preserving_fusion(
                    residual_logits,
                    symbolic_prior,
                    hard_eliminated,
                    alpha=alpha,
                )
            except RankPreservingFusionError as error:
                raise ValueError(
                    "fusion-grid observation cannot satisfy production fusion"
                ) from error
            probability = fused.probabilities[true_index]
            if probability <= 0.0:
                raise ValueError(
                    "fusion-grid truth has zero symbolic posterior mass"
                )
            losses.append(-math.log(probability))
        loss = math.fsum(losses) / len(losses)
        key = (color, game_id)
        plies = self._losses.setdefault(key, {})
        if observed_ply in plies:
            raise ValueError("fusion-grid player-game contains a duplicate ply")
        plies[observed_ply] = loss

    def report(self) -> Mapping[str, FusionGridHeadMetric]:
        result: dict[str, FusionGridHeadMetric] = {}
        for color in ("white", "black"):
            games = tuple(
                values
                for (row_color, _game_id), values in self._losses.items()
                if row_color == color
            )
            if not games:
                raise ValueError(
                    f"fusion-grid selection contains no {color} player-games"
                )
            game_losses = tuple(
                math.fsum(values.values()) / len(values) for values in games
            )
            result[color] = FusionGridHeadMetric(
                observation_count=sum(len(values) for values in games),
                player_game_count=len(games),
                player_game_normalized_nll=(
                    math.fsum(game_losses) / len(game_losses)
                ),
            )
        return result


def validate_fusion_grid_head_counts(
    head: Mapping[str, object],
    color: str,
) -> None:
    """Validate independently reported move and player-game counts."""

    for key in ("observation_count", "player_game_count"):
        count = head.get(key)
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or count <= 0
        ):
            raise ValueError(f"{color} selection {key} must be positive")
    observation_count = head["observation_count"]
    player_game_count = head["player_game_count"]
    assert isinstance(observation_count, int)
    assert isinstance(player_game_count, int)
    if observation_count < player_game_count:
        raise ValueError(
            f"{color} selection observation_count cannot be smaller "
            "than player_game_count"
        )


@dataclass(frozen=True)
class ContentAddressedSummary:
    path: Path
    sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "sha256", _digest(self.sha256, "summary sha256"))


@dataclass(frozen=True)
class SelectionProvenance:
    release_root_sha256: str
    corpus_run_id: str
    training_corpus_set_sha256: str
    private_validation_manifest_sha256: str
    validation_dataset_sha256: str
    model_run_config_sha256: str
    planned_epoch_count: int

    def __post_init__(self) -> None:
        for name in (
            "release_root_sha256",
            "corpus_run_id",
            "training_corpus_set_sha256",
            "private_validation_manifest_sha256",
            "validation_dataset_sha256",
            "model_run_config_sha256",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        _positive_int(self.planned_epoch_count, "planned_epoch_count")

    def metadata(self) -> dict[str, object]:
        return {
            "release_root_sha256": self.release_root_sha256,
            "corpus_run_id": self.corpus_run_id,
            "training_corpus_set_sha256": self.training_corpus_set_sha256,
            "private_validation_manifest_sha256": (
                self.private_validation_manifest_sha256
            ),
            "validation_dataset_sha256": self.validation_dataset_sha256,
            "model_run_config_sha256": self.model_run_config_sha256,
            "planned_epoch_count": self.planned_epoch_count,
        }


@dataclass(frozen=True)
class EpochSelectionSummary:
    provenance: SelectionProvenance
    training_seed: int
    epoch: int
    partition_seed_sha256: str
    checkpoint_file: str
    checkpoint_sha256: str
    evaluation_report_file: str
    evaluation_report_sha256: str
    white_nll: float
    black_nll: float
    source_file: str
    source_sha256: str
    objective_id: str = LEGACY_ADDITIVE_DRAWBACK_OBJECTIVE

    @property
    def mean_nll(self) -> float:
        return (self.white_nll + self.black_nll) / 2.0


def load_selection_summary(
    reference: ContentAddressedSummary,
) -> EpochSelectionSummary:
    """Load one strict, content-addressed selection-only summary."""

    try:
        payload = reference.path.read_bytes()
    except OSError as error:
        raise ValueError(f"cannot read epoch summary: {reference.path}") from error
    actual = hashlib.sha256(payload).hexdigest()
    if actual != reference.sha256:
        raise ValueError("epoch summary sha256 does not match its reference")
    try:
        value = json.loads(
            payload,
            parse_constant=lambda token: _reject_json_constant(token),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("epoch summary is not strict UTF-8 JSON") from error
    if not isinstance(value, Mapping):
        raise ValueError("epoch summary root must be an object")
    format_version = value.get("format_version")
    if format_version not in {
        LEGACY_SELECTION_FORMAT_VERSION,
        SELECTION_SUMMARY_FORMAT_VERSION,
    }:
        raise ValueError("unsupported epoch summary format version")
    expected_keys = {
        "format_version",
        "training_seed",
        "epoch",
        "provenance",
        "partition",
        "checkpoint",
        "evaluation_report",
        "metrics",
    }
    if format_version == SELECTION_SUMMARY_FORMAT_VERSION:
        expected_keys.add("objective")
    _exact_keys(value, expected_keys, "epoch summary")
    objective_id = LEGACY_ADDITIVE_DRAWBACK_OBJECTIVE
    if format_version == SELECTION_SUMMARY_FORMAT_VERSION:
        _require_fusion_grid_selection_objective(
            value["objective"],
            "epoch summary objective",
        )
        objective_id = FUSION_GRID_DRAWBACK_OBJECTIVE
    provenance_value = _object(value["provenance"], "provenance")
    provenance = _selection_provenance_from_mapping(provenance_value)
    partition = _object(value["partition"], "partition")
    _exact_keys(
        partition,
        {"identity", "name", "seed_sha256"},
        "partition",
    )
    if partition["identity"] != VALIDATION_PARTITION_IDENTITY:
        raise ValueError("epoch summary uses the wrong partition identity")
    if partition["name"] != ValidationPartition.SELECTION.value:
        raise ValueError("epoch selection may consume selection summaries only")
    checkpoint = _object(value["checkpoint"], "checkpoint")
    _exact_keys(checkpoint, {"file", "sha256"}, "checkpoint")
    evaluation_report = _object(
        value["evaluation_report"], "evaluation_report"
    )
    _exact_keys(
        evaluation_report,
        {"file", "sha256"},
        "evaluation_report",
    )
    metrics = _object(value["metrics"], "metrics")
    _exact_keys(metrics, {"white_nll", "black_nll"}, "metrics")
    return EpochSelectionSummary(
        provenance=provenance,
        training_seed=_nonnegative_int(value["training_seed"], "training_seed"),
        epoch=_positive_int(value["epoch"], "epoch"),
        partition_seed_sha256=_digest(
            partition["seed_sha256"], "partition seed sha256"
        ),
        checkpoint_file=_basename(checkpoint["file"], "checkpoint file"),
        checkpoint_sha256=_digest(
            checkpoint["sha256"], "checkpoint sha256"
        ),
        evaluation_report_file=_basename(
            evaluation_report["file"], "evaluation report file"
        ),
        evaluation_report_sha256=_digest(
            evaluation_report["sha256"], "evaluation report sha256"
        ),
        white_nll=_finite_nonnegative(metrics["white_nll"], "White NLL"),
        black_nll=_finite_nonnegative(metrics["black_nll"], "Black NLL"),
        source_file=reference.path.name,
        source_sha256=reference.sha256,
        objective_id=objective_id,
    )


def choose_epoch(
    summaries: Sequence[EpochSelectionSummary],
) -> EpochSelectionSummary:
    """Choose minimum mean head NLL, preferring earlier epochs within 0.005."""

    if not summaries:
        raise ValueError("epoch selection requires at least one summary")
    provenances = {summary.provenance for summary in summaries}
    if len(provenances) != 1:
        raise ValueError("epoch summaries use mixed release provenance or config")
    objectives = {summary.objective_id for summary in summaries}
    if len(objectives) != 1:
        raise ValueError("epoch summaries use mixed drawback objectives")
    provenance = summaries[0].provenance
    training_seeds = {summary.training_seed for summary in summaries}
    if len(training_seeds) != 1:
        raise ValueError("epoch summaries must belong to one training seed")
    partition_hashes = {
        summary.partition_seed_sha256 for summary in summaries
    }
    if len(partition_hashes) != 1:
        raise ValueError("epoch summaries use different selection seed sets")
    epochs = [summary.epoch for summary in summaries]
    if len(set(epochs)) != len(epochs):
        raise ValueError("epoch summaries contain duplicate epochs")
    expected_epochs = set(range(1, provenance.planned_epoch_count + 1))
    if set(epochs) != expected_epochs:
        raise ValueError(
            "epoch summaries must contain every planned epoch exactly once"
        )
    checkpoint_hashes = [summary.checkpoint_sha256 for summary in summaries]
    if len(set(checkpoint_hashes)) != len(checkpoint_hashes):
        raise ValueError("different epochs cannot reference one checkpoint")
    minimum = min(summary.mean_nll for summary in summaries)
    eligible = [
        summary
        for summary in summaries
        if summary.mean_nll <= minimum + NLL_TIE_TOLERANCE
    ]
    return min(eligible, key=lambda summary: summary.epoch)


def write_selection_artifact(
    output: Path,
    references: Sequence[ContentAddressedSummary],
) -> Path:
    """Verify summaries, select an epoch, and publish an immutable artifact."""

    summaries = tuple(load_selection_summary(item) for item in references)
    selected = choose_epoch(summaries)
    ordered = sorted(summaries, key=lambda summary: summary.epoch)
    fusion_grid = (
        selected.objective_id == FUSION_GRID_DRAWBACK_OBJECTIVE
    )
    artifact = {
        "format_version": (
            SELECTION_ARTIFACT_FORMAT_VERSION
            if fusion_grid
            else LEGACY_SELECTION_FORMAT_VERSION
        ),
        "method": {
            "metric": (
                "arithmetic-mean-white-black-player-game-normalized-"
                "fusion-grid-nll"
                if fusion_grid
                else "arithmetic-mean-white-black-nll"
            ),
            "tie_tolerance": NLL_TIE_TOLERANCE,
            "tie_break": "earlier-epoch",
        },
        "partition": {
            "identity": VALIDATION_PARTITION_IDENTITY,
            "name": ValidationPartition.SELECTION.value,
            "seed_sha256": selected.partition_seed_sha256,
        },
        "provenance": selected.provenance.metadata(),
        "training_seed": selected.training_seed,
        "candidates": [
            {
                "epoch": summary.epoch,
                "white_nll": summary.white_nll,
                "black_nll": summary.black_nll,
                "mean_nll": summary.mean_nll,
                "checkpoint_file": summary.checkpoint_file,
                "checkpoint_sha256": summary.checkpoint_sha256,
                "evaluation_report_file": summary.evaluation_report_file,
                "evaluation_report_sha256": (
                    summary.evaluation_report_sha256
                ),
                "summary_file": summary.source_file,
                "summary_sha256": summary.source_sha256,
            }
            for summary in ordered
        ],
        "selected": {
            "epoch": selected.epoch,
            "mean_nll": selected.mean_nll,
            "checkpoint_file": selected.checkpoint_file,
            "checkpoint_sha256": selected.checkpoint_sha256,
            "evaluation_report_file": selected.evaluation_report_file,
            "evaluation_report_sha256": (
                selected.evaluation_report_sha256
            ),
            "summary_file": selected.source_file,
            "summary_sha256": selected.source_sha256,
        },
    }
    if fusion_grid:
        artifact["objective"] = _fusion_grid_selection_objective()
    rendered = _canonical_json(artifact)
    _write_atomic_no_clobber(output, rendered)
    return output


@dataclass(frozen=True)
class LoadedSelectionArtifact:
    provenance: SelectionProvenance
    training_seed: int
    partition_seed_sha256: str
    selected_epoch: int
    selected_checkpoint_file: str
    selected_checkpoint_sha256: str
    sha256: str
    objective_id: str = LEGACY_ADDITIVE_DRAWBACK_OBJECTIVE


def load_selection_artifact(
    reference: ContentAddressedSummary,
) -> LoadedSelectionArtifact:
    payload = reference.path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != reference.sha256:
        raise ValueError("selection artifact sha256 does not match")
    value = _load_strict_json(payload, "selection artifact")
    if not isinstance(value, Mapping):
        raise ValueError("selection artifact root must be an object")
    if payload != _canonical_json(value).encode("utf-8"):
        raise ValueError("selection artifact must use canonical JSON bytes")
    format_version = value.get("format_version")
    if format_version not in {
        LEGACY_SELECTION_FORMAT_VERSION,
        SELECTION_ARTIFACT_FORMAT_VERSION,
    }:
        raise ValueError("unsupported selection artifact format version")
    expected_keys = {
        "format_version", "method", "partition", "provenance",
        "training_seed", "candidates", "selected",
    }
    objective_id = LEGACY_ADDITIVE_DRAWBACK_OBJECTIVE
    if format_version == SELECTION_ARTIFACT_FORMAT_VERSION:
        expected_keys.add("objective")
        _require_fusion_grid_selection_objective(
            value.get("objective"),
            "selection artifact objective",
        )
        objective_id = FUSION_GRID_DRAWBACK_OBJECTIVE
    _exact_keys(value, expected_keys, "selection artifact")
    provenance = _selection_provenance_from_mapping(
        _object(value["provenance"], "selection provenance")
    )
    method = _object(value["method"], "selection method")
    _exact_keys(
        method, {"metric", "tie_tolerance", "tie_break"}, "selection method"
    )
    expected_metric = (
        "arithmetic-mean-white-black-player-game-normalized-fusion-grid-nll"
        if objective_id == FUSION_GRID_DRAWBACK_OBJECTIVE
        else "arithmetic-mean-white-black-nll"
    )
    if method != {
        "metric": expected_metric,
        "tie_tolerance": NLL_TIE_TOLERANCE,
        "tie_break": "earlier-epoch",
    }:
        raise ValueError("selection artifact method is invalid")
    partition = _object(value["partition"], "selection partition")
    _exact_keys(partition, {"identity", "name", "seed_sha256"}, "selection partition")
    if (
        partition["identity"] != VALIDATION_PARTITION_IDENTITY
        or partition["name"] != ValidationPartition.SELECTION.value
    ):
        raise ValueError("selection artifact partition is invalid")
    candidates = value["candidates"]
    if not isinstance(candidates, list):
        raise ValueError("selection artifact candidates must be an array")
    candidate_keys = {
        "epoch", "white_nll", "black_nll", "mean_nll",
        "checkpoint_file", "checkpoint_sha256", "evaluation_report_file",
        "evaluation_report_sha256", "summary_file", "summary_sha256",
    }
    parsed_candidates: list[dict[str, object]] = []
    for item in candidates:
        if not isinstance(item, Mapping):
            raise ValueError("selection candidate must be an object")
        _exact_keys(item, candidate_keys, "selection candidate")
        white_nll = _finite_nonnegative(
            item["white_nll"], "candidate White NLL"
        )
        black_nll = _finite_nonnegative(
            item["black_nll"], "candidate Black NLL"
        )
        mean_nll = _finite_nonnegative(
            item["mean_nll"], "candidate mean NLL"
        )
        expected_mean = (white_nll + black_nll) / 2.0
        if mean_nll != expected_mean:
            raise ValueError("selection candidate mean NLL is inconsistent")
        parsed_candidates.append(
            {
                "epoch": _positive_int(item["epoch"], "candidate epoch"),
                "white_nll": white_nll,
                "black_nll": black_nll,
                "mean_nll": mean_nll,
                "checkpoint_file": _basename(
                    item["checkpoint_file"], "candidate checkpoint file"
                ),
                "checkpoint_sha256": _digest(
                    item["checkpoint_sha256"], "candidate checkpoint sha256"
                ),
                "evaluation_report_file": _basename(
                    item["evaluation_report_file"],
                    "candidate evaluation report file",
                ),
                "evaluation_report_sha256": _digest(
                    item["evaluation_report_sha256"],
                    "candidate evaluation report sha256",
                ),
                "summary_file": _basename(
                    item["summary_file"], "candidate summary file"
                ),
                "summary_sha256": _digest(
                    item["summary_sha256"], "candidate summary sha256"
                ),
            }
        )
    epochs = [item["epoch"] for item in parsed_candidates]
    if len(epochs) != len(parsed_candidates) or set(epochs) != set(
        range(1, provenance.planned_epoch_count + 1)
    ):
        raise ValueError("selection artifact candidate epochs are incomplete")
    for key in (
        "checkpoint_sha256",
        "evaluation_report_sha256",
        "summary_sha256",
    ):
        values = [item[key] for item in parsed_candidates]
        if len(set(values)) != len(values):
            raise ValueError(f"selection candidates reuse {key}")
    selected = _object(value["selected"], "selected checkpoint")
    _exact_keys(
        selected,
        {
            "epoch", "mean_nll", "checkpoint_file", "checkpoint_sha256",
            "evaluation_report_file", "evaluation_report_sha256",
            "summary_file", "summary_sha256",
        },
        "selected checkpoint",
    )
    selected_epoch = _positive_int(selected.get("epoch"), "selected epoch")
    if selected_epoch not in epochs:
        raise ValueError("selected epoch is absent from candidates")
    selected_values = {
        "epoch": selected_epoch,
        "mean_nll": _finite_nonnegative(
            selected.get("mean_nll"), "selected mean NLL"
        ),
        "checkpoint_file": _basename(
            selected.get("checkpoint_file"), "selected checkpoint file"
        ),
        "checkpoint_sha256": _digest(
            selected.get("checkpoint_sha256"), "selected checkpoint sha256"
        ),
        "evaluation_report_file": _basename(
            selected.get("evaluation_report_file"),
            "selected evaluation report file",
        ),
        "evaluation_report_sha256": _digest(
            selected.get("evaluation_report_sha256"),
            "selected evaluation report sha256",
        ),
        "summary_file": _basename(
            selected.get("summary_file"), "selected summary file"
        ),
        "summary_sha256": _digest(
            selected.get("summary_sha256"), "selected summary sha256"
        ),
    }
    selected_candidate = next(
        item for item in parsed_candidates if item["epoch"] == selected_epoch
    )
    if any(
        selected_values[key] != selected_candidate[key]
        for key in selected_values
    ):
        raise ValueError("selected checkpoint disagrees with its candidate")
    minimum = min(
        float(item["mean_nll"]) for item in parsed_candidates
    )
    expected_selected = min(
        (
            item
            for item in parsed_candidates
            if float(item["mean_nll"]) <= minimum + NLL_TIE_TOLERANCE
        ),
        key=lambda item: int(item["epoch"]),
    )
    if selected_epoch != expected_selected["epoch"]:
        raise ValueError("selected checkpoint violates the selection method")
    return LoadedSelectionArtifact(
        provenance=provenance,
        training_seed=_nonnegative_int(value["training_seed"], "training_seed"),
        partition_seed_sha256=_digest(partition["seed_sha256"], "partition seed"),
        selected_epoch=selected_epoch,
        selected_checkpoint_file=str(selected_values["checkpoint_file"]),
        selected_checkpoint_sha256=str(selected_values["checkpoint_sha256"]),
        sha256=reference.sha256,
        objective_id=objective_id,
    )


def _selection_provenance_from_mapping(
    value: Mapping[object, object],
) -> SelectionProvenance:
    expected = {
        "release_root_sha256", "corpus_run_id",
        "training_corpus_set_sha256",
        "private_validation_manifest_sha256", "validation_dataset_sha256",
        "model_run_config_sha256", "planned_epoch_count",
    }
    _exact_keys(value, expected, "selection provenance")
    return SelectionProvenance(
        release_root_sha256=value["release_root_sha256"],
        corpus_run_id=value["corpus_run_id"],
        training_corpus_set_sha256=value["training_corpus_set_sha256"],
        private_validation_manifest_sha256=value[
            "private_validation_manifest_sha256"
        ],
        validation_dataset_sha256=value["validation_dataset_sha256"],
        model_run_config_sha256=value["model_run_config_sha256"],
        planned_epoch_count=_positive_int(
            value["planned_epoch_count"], "planned_epoch_count"
        ),
    )


def _fusion_grid_selection_objective() -> dict[str, object]:
    return {
        **FUSION_GRID_SELECTION_OBJECTIVE,
        "checkpoint_objective": fusion_grid_drawback_objective_metadata(),
        "alpha_grid": list(FUSION_AWARE_LOSS_ALPHA_GRID),
    }


def fusion_grid_selection_objective_metadata() -> dict[str, object]:
    """Return a defensive copy of the authenticated epoch metric identity."""

    return _fusion_grid_selection_objective()


def _require_fusion_grid_selection_objective(
    value: object,
    name: str,
) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    expected = _fusion_grid_selection_objective()
    if set(value) != set(expected):
        raise ValueError(f"{name} fields are invalid")
    checkpoint_objective = value["checkpoint_objective"]
    expected_checkpoint = expected["checkpoint_objective"]
    if (
        not isinstance(checkpoint_objective, Mapping)
        or not isinstance(expected_checkpoint, Mapping)
        or set(checkpoint_objective) != set(expected_checkpoint)
        or dict(checkpoint_objective) != dict(expected_checkpoint)
    ):
        raise ValueError(f"{name} checkpoint objective is invalid")
    raw_grid = value["alpha_grid"]
    if (
        not isinstance(raw_grid, list)
        or any(type(item) is not float for item in raw_grid)
        or tuple(raw_grid) != FUSION_AWARE_LOSS_ALPHA_GRID
    ):
        raise ValueError(f"{name} alpha grid is invalid")
    for key in (
        "metric",
        "move_metric",
        "alpha_aggregation",
        "player_game_aggregation",
        "color_aggregation",
    ):
        if type(value[key]) is not str or value[key] != expected[key]:
            raise ValueError(f"{name} is invalid")


def _write_atomic_no_clobber(path: Path, rendered: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        publish_bytes_durable_exact(
            path,
            rendered.encode("utf-8"),
            label="epoch selection artifact",
        )
    except ValueError as error:
        raise ValueError(
            f"refusing to overwrite epoch selection artifact: {path}"
        ) from error


def _reject_json_constant(token: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {token}")


def _canonical_json(value: object) -> str:
    return (
        json.dumps(
            value,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )


def _load_strict_json(payload: bytes, name: str) -> object:
    def reject_duplicates(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"{name} contains duplicate key {key}")
            value[key] = item
        return value

    try:
        return json.loads(
            payload,
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda token: _reject_json_constant(token),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is not valid UTF-8 JSON") from error


def _exact_keys(
    value: Mapping[object, object],
    expected: set[str],
    name: str,
) -> None:
    if set(value) != expected:
        raise ValueError(f"{name} fields do not match its format version")


def _object(value: object, name: str) -> Mapping[object, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
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
    converted = float(value)
    if not math.isfinite(converted) or converted < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return converted


def _digest(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _basename(value: object, name: str) -> str:
    if not is_portable_safe_basename(value):
        raise ValueError(f"{name} must be a safe basename")
    return value
