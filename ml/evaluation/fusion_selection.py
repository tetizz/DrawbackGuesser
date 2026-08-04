"""Authenticated selection of the rank-preserving fusion alpha.

This module is deliberately limited to the frozen validation ``selection``
partition.  It does not authorize use of the sealed test partition.  Every
candidate alpha is scored by first averaging move NLL within a player-game and
then averaging player-games, so long games cannot dominate the selection.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from ml.training.drawback_ml.durable_publish import publish_bytes_durable_exact

from ml.training.drawback_ml.rank_preserving_fusion import (
    RANK_PRESERVING_FUSION_METHOD,
    RankPreservingFusionError,
    apply_rank_preserving_fusion,
    prepare_rank_preserving_fusion,
)

from .release_selection_bundle import ContentAddressedJson
from .validation_partition import VALIDATION_PARTITION_IDENTITY


FUSION_SELECTION_FORMAT = (
    "drawbacktrainer-rank-preserving-fusion-selection"
)
FUSION_SELECTION_VERSION = 1
FUSION_SELECTION_PARTITION = "selection"
FUSION_SELECTION_CLASS_COUNT = 182
FROZEN_ALPHA_GRID = (0.0, 0.125, 0.25, 0.5, 1.0)
_PROBABILITY_SUM_TOLERANCE = 1e-9


class FusionSelectionError(ValueError):
    """Raised when fusion-selection evidence is invalid or unauthenticated."""


@dataclass(frozen=True)
class FusionSelectionIdentity:
    """Immutable release and validation identities bound by the artifact."""

    ensemble_release_sha256: str
    private_validation_manifest_sha256: str
    validation_dataset_sha256: str
    validation_seed_sha256: str
    training_corpus_set_sha256: str
    symbolic_schema_sha256: str
    class_count: int = FUSION_SELECTION_CLASS_COUNT
    partition: str = FUSION_SELECTION_PARTITION
    partition_identity: str = VALIDATION_PARTITION_IDENTITY

    def __post_init__(self) -> None:
        for name in (
            "ensemble_release_sha256",
            "private_validation_manifest_sha256",
            "validation_dataset_sha256",
            "validation_seed_sha256",
            "training_corpus_set_sha256",
            "symbolic_schema_sha256",
        ):
            object.__setattr__(
                self,
                name,
                _digest(getattr(self, name), name.replace("_", " ")),
            )
        if self.class_count != FUSION_SELECTION_CLASS_COUNT:
            raise FusionSelectionError(
                "fusion selection requires exactly 182 classes"
            )
        if self.partition != FUSION_SELECTION_PARTITION:
            raise FusionSelectionError(
                "fusion selection accepts only the selection partition"
            )
        if self.partition_identity != VALIDATION_PARTITION_IDENTITY:
            raise FusionSelectionError(
                "fusion selection partition identity is unsupported"
            )


@dataclass(frozen=True)
class FusionSelectionObservation:
    """One authenticated selection-prefix observation.

    The true index is evaluation metadata and must not be supplied to model
    inference.  ``identity`` is repeated intentionally: it makes accidental
    mixing of observations from another release or validation corpus fail
    closed at the selection boundary.
    """

    identity: FusionSelectionIdentity
    partition: str
    game_id: str
    color: str
    observed_ply: int
    true_index: int
    residual_logits: Sequence[float]
    symbolic_prior: Sequence[float]
    hard_eliminated: Sequence[bool]


@dataclass(frozen=True)
class FusionSelectionSideMetric:
    observation_count: int
    player_game_count: int
    game_normalized_nll: float


@dataclass(frozen=True)
class FusionSelectionCandidate:
    alpha: float
    white: FusionSelectionSideMetric
    black: FusionSelectionSideMetric
    mean_white_black_game_normalized_nll: float


@dataclass(frozen=True)
class LoadedFusionSelection:
    identity: FusionSelectionIdentity
    candidates: tuple[FusionSelectionCandidate, ...]
    selected_alpha: float
    source: ContentAddressedJson


@dataclass(frozen=True)
class _ValidatedObservation:
    game_id: str
    color: str
    observed_ply: int
    true_index: int
    residual_logits: tuple[float, ...]
    symbolic_prior: tuple[float, ...]
    hard_eliminated: tuple[bool, ...]


class FusionSelectionAccumulator:
    """Reduced-memory accumulator for one authenticated selection run."""

    def __init__(self, identity: FusionSelectionIdentity) -> None:
        if not isinstance(identity, FusionSelectionIdentity):
            raise FusionSelectionError(
                "fusion selection accumulator identity is invalid"
            )
        self._identity = identity
        self._scores: dict[
            float,
            dict[tuple[str, str], list[float]],
        ] = {
            alpha: {}
            for alpha in FROZEN_ALPHA_GRID
        }
        self._observation_counts = {"white": 0, "black": 0}
        self._colors_by_game: dict[str, set[str]] = {}
        self._truth_by_player_game: dict[tuple[str, str], int] = {}
        self._seen_plies: set[tuple[str, str, int]] = set()
        self._row_count = 0
        self._finalized = False

    @property
    def identity(self) -> FusionSelectionIdentity:
        """Return the immutable evidence identity for this run."""

        return self._identity

    def add(self, observation: FusionSelectionObservation) -> None:
        if self._finalized:
            raise FusionSelectionError(
                "fusion selection accumulator is already finalized"
            )
        try:
            row = _validate_observation(self._identity, observation)
            _record_player_game(
                row,
                colors_by_game=self._colors_by_game,
                truth_by_player_game=self._truth_by_player_game,
                seen_plies=self._seen_plies,
            )
            preparation = prepare_rank_preserving_fusion(
                row.residual_logits,
                row.symbolic_prior,
                row.hard_eliminated,
            )
            player_game = (row.game_id, row.color)
            for alpha in FROZEN_ALPHA_GRID:
                fusion = apply_rank_preserving_fusion(
                    preparation,
                    alpha=alpha,
                )
                probability = _validated_true_probability(
                    fusion.probabilities,
                    row.true_index,
                )
                self._scores[alpha].setdefault(
                    player_game,
                    [],
                ).append(-math.log(probability))
            self._observation_counts[row.color] += 1
            self._row_count += 1
        except RankPreservingFusionError as error:
            self._finalized = True
            raise FusionSelectionError(
                "rank-preserving fusion rejected selection evidence"
            ) from error
        except Exception:
            self._finalized = True
            raise

    def finalize(self) -> bytes:
        if self._finalized:
            raise FusionSelectionError(
                "fusion selection accumulator is already finalized"
            )
        self._finalized = True
        _validate_player_game_state(
            row_count=self._row_count,
            colors_by_game=self._colors_by_game,
        )
        candidates = tuple(
            _candidate_from_scores(
                alpha,
                self._scores[alpha],
                self._observation_counts,
            )
            for alpha in FROZEN_ALPHA_GRID
        )
        return _artifact_payload(self._identity, candidates)


def build_fusion_selection_artifact(
    identity: FusionSelectionIdentity,
    observations: Iterable[FusionSelectionObservation],
) -> bytes:
    """Build canonical artifact bytes from selection-only observations."""

    accumulator = FusionSelectionAccumulator(identity)
    for observation in observations:
        accumulator.add(observation)
    return accumulator.finalize()


def _artifact_payload(
    identity: FusionSelectionIdentity,
    candidates: Sequence[FusionSelectionCandidate],
) -> bytes:
    selected = min(
        candidates,
        key=lambda candidate: (
            candidate.mean_white_black_game_normalized_nll,
            candidate.alpha,
        ),
    )
    value = {
        "format": FUSION_SELECTION_FORMAT,
        "version": FUSION_SELECTION_VERSION,
        "method": RANK_PRESERVING_FUSION_METHOD,
        "alpha_grid": list(FROZEN_ALPHA_GRID),
        "identity": _identity_value(identity),
        "candidates": [
            _candidate_value(candidate) for candidate in candidates
        ],
        "selected_alpha": selected.alpha,
    }
    return _canonical_json(value)


def write_fusion_selection_artifact(
    output: Path,
    identity: FusionSelectionIdentity,
    observations: Iterable[FusionSelectionObservation],
) -> ContentAddressedJson:
    """Atomically publish a no-clobber, content-addressed selection artifact."""

    payload = build_fusion_selection_artifact(identity, observations)
    return _write_fusion_selection_payload(output, payload)


def write_fusion_selection_accumulator(
    output: Path,
    accumulator: FusionSelectionAccumulator,
) -> ContentAddressedJson:
    """Finalize and atomically publish an incrementally populated selection."""

    if not isinstance(accumulator, FusionSelectionAccumulator):
        raise FusionSelectionError(
            "fusion selection accumulator is invalid"
        )
    return _write_fusion_selection_payload(
        output,
        accumulator.finalize(),
    )


def _write_fusion_selection_payload(
    output: Path,
    payload: bytes,
) -> ContentAddressedJson:
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        publish_bytes_durable_exact(
            output,
            payload,
            label="fusion selection artifact",
        )
    except ValueError as error:
        raise FusionSelectionError(
            f"refusing to overwrite fusion selection artifact: {output}"
        ) from error
    return ContentAddressedJson(
        output,
        hashlib.sha256(payload).hexdigest(),
    )


def load_fusion_selection_artifact(
    reference: ContentAddressedJson,
    *,
    expected_identity: FusionSelectionIdentity,
) -> LoadedFusionSelection:
    """Strictly load canonical bytes and require the expected full identity."""

    payload = _verified_bytes(reference)
    value = _strict_json(payload)
    if _canonical_json(value) != payload:
        raise FusionSelectionError(
            "fusion selection JSON bytes are not canonical"
        )
    _exact_keys(
        value,
        {
            "format",
            "version",
            "method",
            "alpha_grid",
            "identity",
            "candidates",
            "selected_alpha",
        },
        "fusion selection artifact",
    )
    if (
        value["format"] != FUSION_SELECTION_FORMAT
        or value["version"] != FUSION_SELECTION_VERSION
    ):
        raise FusionSelectionError(
            "fusion selection artifact format is unsupported"
        )
    if value["method"] != RANK_PRESERVING_FUSION_METHOD:
        raise FusionSelectionError(
            "fusion selection artifact uses an unsupported fusion method"
        )
    alpha_grid = _alpha_list(value["alpha_grid"], "alpha grid")
    if alpha_grid != FROZEN_ALPHA_GRID:
        raise FusionSelectionError(
            "fusion selection artifact alpha grid is not frozen"
        )
    identity = _load_identity(value["identity"])
    if identity != expected_identity:
        raise FusionSelectionError(
            "fusion selection artifact identity does not match expected identity"
        )

    raw_candidates = value["candidates"]
    if not isinstance(raw_candidates, list):
        raise FusionSelectionError(
            "fusion selection candidates must be an array"
        )
    candidates = tuple(_load_candidate(item) for item in raw_candidates)
    if tuple(candidate.alpha for candidate in candidates) != FROZEN_ALPHA_GRID:
        raise FusionSelectionError(
            "fusion selection candidates do not match the frozen alpha grid"
        )
    _validate_candidate_counts(candidates)
    selected_alpha = _alpha(value["selected_alpha"], "selected alpha")
    selected = min(
        candidates,
        key=lambda candidate: (
            candidate.mean_white_black_game_normalized_nll,
            candidate.alpha,
        ),
    )
    if selected_alpha != selected.alpha:
        raise FusionSelectionError(
            "selected alpha is inconsistent with candidate metrics"
        )
    return LoadedFusionSelection(
        identity=identity,
        candidates=candidates,
        selected_alpha=selected_alpha,
        source=reference,
    )


def _validate_observation(
    identity: FusionSelectionIdentity,
    row: FusionSelectionObservation,
) -> _ValidatedObservation:
    if row.identity != identity:
        raise FusionSelectionError(
            "fusion selection observation identity mismatch"
        )
    if row.partition != FUSION_SELECTION_PARTITION:
        raise FusionSelectionError(
            "fusion selection observations must use the selection partition"
        )
    if not isinstance(row.game_id, str) or not row.game_id:
        raise FusionSelectionError("fusion selection game_id must not be empty")
    if row.color not in {"white", "black"}:
        raise FusionSelectionError(
            "fusion selection color must be white or black"
        )
    observed_ply = _non_negative_int(row.observed_ply, "observed ply")
    true_index = _index(row.true_index, "true index")
    residuals = _finite_vector(row.residual_logits, "residual logits")
    prior = _finite_vector(row.symbolic_prior, "symbolic prior")
    mask = tuple(row.hard_eliminated)
    if (
        len(residuals) != FUSION_SELECTION_CLASS_COUNT
        or len(prior) != FUSION_SELECTION_CLASS_COUNT
        or len(mask) != FUSION_SELECTION_CLASS_COUNT
    ):
        raise FusionSelectionError(
            "fusion selection observations must contain exactly 182 classes"
        )
    if any(type(value) is not bool for value in mask):
        raise FusionSelectionError(
            "fusion selection hard mask must contain booleans"
        )
    if any(value < 0.0 or value > 1.0 for value in prior):
        raise FusionSelectionError(
            "symbolic prior probabilities must be between zero and one"
        )
    if not math.isclose(
        math.fsum(prior),
        1.0,
        rel_tol=0.0,
        abs_tol=_PROBABILITY_SUM_TOLERANCE,
    ):
        raise FusionSelectionError(
            "symbolic prior probabilities must sum to one"
        )
    if mask[true_index]:
        raise FusionSelectionError(
            "true drawback is hard-eliminated in selection evidence"
        )
    return _ValidatedObservation(
        game_id=row.game_id,
        color=row.color,
        observed_ply=observed_ply,
        true_index=true_index,
        residual_logits=residuals,
        symbolic_prior=prior,
        hard_eliminated=mask,
    )


def _record_player_game(
    row: _ValidatedObservation,
    *,
    colors_by_game: dict[str, set[str]],
    truth_by_player_game: dict[tuple[str, str], int],
    seen_plies: set[tuple[str, str, int]],
) -> None:
    colors_by_game.setdefault(row.game_id, set()).add(row.color)
    player_game = (row.game_id, row.color)
    prior_truth = truth_by_player_game.setdefault(
        player_game,
        row.true_index,
    )
    if prior_truth != row.true_index:
        raise FusionSelectionError(
            "fusion selection player-game contains mixed truth labels"
        )
    ply_key = (row.game_id, row.color, row.observed_ply)
    if ply_key in seen_plies:
        raise FusionSelectionError(
            "fusion selection player-game contains duplicate observed plies"
        )
    seen_plies.add(ply_key)


def _validate_player_game_state(
    *,
    row_count: int,
    colors_by_game: Mapping[str, set[str]],
) -> None:
    if row_count == 0:
        raise FusionSelectionError(
            "fusion selection requires selection observations"
        )
    black_only = sorted(
        game_id
        for game_id, colors in colors_by_game.items()
        if colors == {"black"}
    )
    if black_only:
        raise FusionSelectionError(
            "fusion selection contains a game without its opening White "
            "player-game"
        )
    observed_colors = {
        color
        for colors in colors_by_game.values()
        for color in colors
    }
    if observed_colors != {"white", "black"}:
        raise FusionSelectionError(
            "fusion selection requires observations for both colors"
        )


def _validated_true_probability(
    probabilities: Sequence[float],
    true_index: int,
) -> float:
    probability = probabilities[true_index]
    if (
        not math.isfinite(probability)
        or probability <= 0.0
        or probability > 1.0
    ):
        raise FusionSelectionError(
            "fusion produced an invalid true-class probability"
        )
    if any(
        not math.isfinite(value) or value < 0.0 or value > 1.0
        for value in probabilities
    ) or not math.isclose(
        math.fsum(probabilities),
        1.0,
        rel_tol=0.0,
        abs_tol=_PROBABILITY_SUM_TOLERANCE,
    ):
        raise FusionSelectionError(
            "fusion produced an invalid probability distribution"
        )
    return probability


def _candidate_from_scores(
    alpha: float,
    nll_by_player_game: Mapping[
        tuple[str, str],
        Sequence[float],
    ],
    observation_counts: Mapping[str, int],
) -> FusionSelectionCandidate:
    side_metrics: dict[str, FusionSelectionSideMetric] = {}
    for color in ("white", "black"):
        player_game_nlls = [
            math.fsum(values) / len(values)
            for (_game_id, player_color), values
            in sorted(nll_by_player_game.items())
            if player_color == color
        ]
        if not player_game_nlls:
            raise FusionSelectionError(
                f"fusion selection contains no {color} player-games"
            )
        metric = math.fsum(player_game_nlls) / len(player_game_nlls)
        if not math.isfinite(metric) or metric < 0.0:
            raise FusionSelectionError(
                "fusion selection produced an invalid game-normalized NLL"
            )
        side_metrics[color] = FusionSelectionSideMetric(
            observation_count=observation_counts[color],
            player_game_count=len(player_game_nlls),
            game_normalized_nll=metric,
        )
    mean_nll = math.fsum(
        (
            side_metrics["white"].game_normalized_nll,
            side_metrics["black"].game_normalized_nll,
        )
    ) / 2.0
    return FusionSelectionCandidate(
        alpha=alpha,
        white=side_metrics["white"],
        black=side_metrics["black"],
        mean_white_black_game_normalized_nll=mean_nll,
    )


def _identity_value(identity: FusionSelectionIdentity) -> dict[str, object]:
    return {
        "ensemble_release_sha256": identity.ensemble_release_sha256,
        "private_validation_manifest_sha256": (
            identity.private_validation_manifest_sha256
        ),
        "validation_dataset_sha256": identity.validation_dataset_sha256,
        "validation_seed_sha256": identity.validation_seed_sha256,
        "training_corpus_set_sha256": identity.training_corpus_set_sha256,
        "symbolic_schema_sha256": identity.symbolic_schema_sha256,
        "class_count": identity.class_count,
        "partition": identity.partition,
        "partition_identity": identity.partition_identity,
    }


def _candidate_value(
    candidate: FusionSelectionCandidate,
) -> dict[str, object]:
    return {
        "alpha": candidate.alpha,
        "white": _side_value(candidate.white),
        "black": _side_value(candidate.black),
        "mean_white_black_game_normalized_nll": (
            candidate.mean_white_black_game_normalized_nll
        ),
    }


def _side_value(metric: FusionSelectionSideMetric) -> dict[str, object]:
    return {
        "observation_count": metric.observation_count,
        "player_game_count": metric.player_game_count,
        "game_normalized_nll": metric.game_normalized_nll,
    }


def _load_identity(value: object) -> FusionSelectionIdentity:
    identity = _object(value, "fusion selection identity")
    _exact_keys(
        identity,
        {
            "ensemble_release_sha256",
            "private_validation_manifest_sha256",
            "validation_dataset_sha256",
            "validation_seed_sha256",
            "training_corpus_set_sha256",
            "symbolic_schema_sha256",
            "class_count",
            "partition",
            "partition_identity",
        },
        "fusion selection identity",
    )
    return FusionSelectionIdentity(
        ensemble_release_sha256=_digest(
            identity["ensemble_release_sha256"],
            "ensemble release sha256",
        ),
        private_validation_manifest_sha256=_digest(
            identity["private_validation_manifest_sha256"],
            "private validation manifest sha256",
        ),
        validation_dataset_sha256=_digest(
            identity["validation_dataset_sha256"],
            "validation dataset sha256",
        ),
        validation_seed_sha256=_digest(
            identity["validation_seed_sha256"],
            "validation seed sha256",
        ),
        training_corpus_set_sha256=_digest(
            identity["training_corpus_set_sha256"],
            "training corpus set sha256",
        ),
        symbolic_schema_sha256=_digest(
            identity["symbolic_schema_sha256"],
            "symbolic schema sha256",
        ),
        class_count=_positive_int(
            identity["class_count"], "fusion selection class count"
        ),
        partition=_string(identity["partition"], "fusion selection partition"),
        partition_identity=_string(
            identity["partition_identity"],
            "fusion selection partition identity",
        ),
    )


def _load_candidate(value: object) -> FusionSelectionCandidate:
    candidate = _object(value, "fusion selection candidate")
    _exact_keys(
        candidate,
        {
            "alpha",
            "white",
            "black",
            "mean_white_black_game_normalized_nll",
        },
        "fusion selection candidate",
    )
    white = _load_side(candidate["white"], "white")
    black = _load_side(candidate["black"], "black")
    mean_nll = _finite_nonnegative(
        candidate["mean_white_black_game_normalized_nll"],
        "mean White/Black game-normalized NLL",
    )
    expected_mean = math.fsum(
        (white.game_normalized_nll, black.game_normalized_nll)
    ) / 2.0
    if mean_nll != expected_mean:
        raise FusionSelectionError(
            "candidate mean NLL is inconsistent with its side metrics"
        )
    return FusionSelectionCandidate(
        alpha=_alpha(candidate["alpha"], "candidate alpha"),
        white=white,
        black=black,
        mean_white_black_game_normalized_nll=mean_nll,
    )


def _load_side(value: object, color: str) -> FusionSelectionSideMetric:
    side = _object(value, f"{color} fusion selection metric")
    _exact_keys(
        side,
        {
            "observation_count",
            "player_game_count",
            "game_normalized_nll",
        },
        f"{color} fusion selection metric",
    )
    observation_count = _positive_int(
        side["observation_count"], f"{color} observation count"
    )
    player_game_count = _positive_int(
        side["player_game_count"], f"{color} player-game count"
    )
    if observation_count < player_game_count:
        raise FusionSelectionError(
            f"{color} observation count is below player-game count"
        )
    return FusionSelectionSideMetric(
        observation_count=observation_count,
        player_game_count=player_game_count,
        game_normalized_nll=_finite_nonnegative(
            side["game_normalized_nll"],
            f"{color} game-normalized NLL",
        ),
    )


def _validate_candidate_counts(
    candidates: Sequence[FusionSelectionCandidate],
) -> None:
    if not candidates:
        raise FusionSelectionError(
            "fusion selection artifact contains no candidates"
        )
    expected = (
        candidates[0].white.observation_count,
        candidates[0].white.player_game_count,
        candidates[0].black.observation_count,
        candidates[0].black.player_game_count,
    )
    for candidate in candidates[1:]:
        actual = (
            candidate.white.observation_count,
            candidate.white.player_game_count,
            candidate.black.observation_count,
            candidate.black.player_game_count,
        )
        if actual != expected:
            raise FusionSelectionError(
                "fusion selection candidate counts are inconsistent"
            )


def _verified_bytes(reference: ContentAddressedJson) -> bytes:
    try:
        payload = reference.path.read_bytes()
    except OSError as error:
        raise FusionSelectionError(
            f"cannot read fusion selection artifact: {reference.path}"
        ) from error
    if hashlib.sha256(payload).hexdigest() != reference.sha256:
        raise FusionSelectionError(
            "fusion selection artifact sha256 does not match"
        )
    return payload


def _strict_json(payload: bytes) -> Mapping[str, object]:
    def pairs(values: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in values:
            if key in result:
                raise FusionSelectionError(
                    f"fusion selection JSON contains duplicate key {key!r}"
                )
            result[key] = value
        return result

    def constant(token: str) -> None:
        raise FusionSelectionError(
            f"fusion selection JSON contains non-finite constant {token}"
        )

    try:
        value = json.loads(
            payload,
            object_pairs_hook=pairs,
            parse_constant=constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FusionSelectionError(
            "fusion selection artifact is not strict UTF-8 JSON"
        ) from error
    _require_finite(value)
    if not isinstance(value, Mapping):
        raise FusionSelectionError(
            "fusion selection artifact root must be an object"
        )
    return value


def _canonical_json(value: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _require_finite(value: object) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise FusionSelectionError(
            "fusion selection JSON contains a non-finite number"
        )
    if isinstance(value, Mapping):
        for item in value.values():
            _require_finite(item)
    elif isinstance(value, list):
        for item in value:
            _require_finite(item)


def _exact_keys(
    value: Mapping[str, object],
    expected: set[str],
    name: str,
) -> None:
    if set(value) != expected:
        raise FusionSelectionError(f"{name} fields are invalid")


def _object(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise FusionSelectionError(f"{name} must be an object")
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise FusionSelectionError(f"{name} must be a non-empty string")
    return value


def _digest(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise FusionSelectionError(
            f"{name} must be a lowercase SHA-256 digest"
        )
    return value


def _non_negative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise FusionSelectionError(
            f"{name} must be a non-negative integer"
        )
    return value


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise FusionSelectionError(f"{name} must be a positive integer")
    return value


def _index(value: object, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or value >= FUSION_SELECTION_CLASS_COUNT
    ):
        raise FusionSelectionError(
            f"{name} must be a valid 182-class index"
        )
    return value


def _finite_vector(
    values: Sequence[float],
    name: str,
) -> tuple[float, ...]:
    result: list[float] = []
    for value in values:
        if isinstance(value, bool):
            raise FusionSelectionError(f"{name} must be numeric")
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise FusionSelectionError(f"{name} must be numeric") from error
        if not math.isfinite(number):
            raise FusionSelectionError(f"{name} must be finite")
        result.append(number)
    return tuple(result)


def _finite_nonnegative(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FusionSelectionError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise FusionSelectionError(
            f"{name} must be finite and non-negative"
        )
    return result


def _alpha(value: object, name: str) -> float:
    result = _finite_nonnegative(value, name)
    if result > 1.0:
        raise FusionSelectionError(f"{name} must be between zero and one")
    return result


def _alpha_list(value: object, name: str) -> tuple[float, ...]:
    if not isinstance(value, list):
        raise FusionSelectionError(f"{name} must be an array")
    return tuple(_alpha(item, f"{name} item") for item in value)
