"""Pure reference baselines for drawback classification.

These helpers deliberately operate on small, explicit value objects rather
than dataset rows.  In particular, :class:`PredictionInput` contains only the
moving color and public symbolic evidence; labels are retained by the
evaluation harness and are never passed to a predictor.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Iterable, Literal, Mapping, Sequence


Color = Literal["white", "black"]
Distribution = tuple[float, ...]


@dataclass(frozen=True)
class PublicSymbolicEvidence:
    """Public posterior and exact eliminations in vocabulary order."""

    probabilities: tuple[float, ...]
    eliminated: tuple[bool, ...]


@dataclass(frozen=True)
class PredictionInput:
    """The complete, label-free input made available to a baseline."""

    mover: Color
    symbolic: PublicSymbolicEvidence | None = None


@dataclass(frozen=True)
class MoverObservation:
    """One labelled mover observation used only by metric computation."""

    game_id: str
    mover: Color
    true_drawback: str
    symbolic: PublicSymbolicEvidence | None = None


@dataclass(frozen=True)
class GameAssignment:
    """Authenticated game-level labels, including zero/one-sided games."""

    game_id: str
    white_drawback: str
    black_drawback: str


@dataclass(frozen=True)
class ClassificationMetrics:
    observations: int
    top1_accuracy: float
    top3_accuracy: float
    top5_accuracy: float
    negative_log_likelihood: float
    brier_score: float


@dataclass(frozen=True)
class MoverOnlyReport:
    """Metrics with exactly one prediction and label per observed move."""

    overall: ClassificationMetrics
    white: ClassificationMetrics
    black: ClassificationMetrics


Predictor = Callable[[PredictionInput], Sequence[float]]


def _checked_vocabulary(vocabulary: Sequence[str]) -> tuple[str, ...]:
    result = tuple(vocabulary)
    if not result:
        raise ValueError("vocabulary must not be empty")
    if any(not item for item in result):
        raise ValueError("vocabulary ids must not be empty")
    if len(set(result)) != len(result):
        raise ValueError("vocabulary ids must be unique")
    return result


def _normalize(
    weights: Sequence[float],
    expected_size: int,
    *,
    context: str,
) -> Distribution:
    if len(weights) != expected_size:
        raise ValueError(
            f"{context} has {len(weights)} values; expected {expected_size}"
        )
    converted = tuple(float(value) for value in weights)
    if any(not math.isfinite(value) or value < 0.0 for value in converted):
        raise ValueError(f"{context} must contain finite non-negative values")
    total = math.fsum(converted)
    if total <= 0.0:
        raise ValueError(f"{context} must contain positive probability mass")
    return tuple(value / total for value in converted)


def uniform_prior(vocabulary: Sequence[str]) -> Distribution:
    """Return the uniform drawback prior in vocabulary order."""

    checked = _checked_vocabulary(vocabulary)
    probability = 1.0 / len(checked)
    return (probability,) * len(checked)


def empirical_game_prior(
    assignments: Iterable[GameAssignment],
    vocabulary: Sequence[str],
) -> Distribution:
    """Estimate a prior from authenticated game assignments, never move rows.

    Every game contributes exactly one White and one Black label, including
    games with zero plies or no move by one side.
    """

    checked = _checked_vocabulary(vocabulary)
    indices = {drawback_id: index for index, drawback_id in enumerate(checked)}
    by_game: dict[str, tuple[str, str]] = {}
    for assignment in assignments:
        if not assignment.game_id:
            raise ValueError("game_id must not be empty")
        for drawback in (
            assignment.white_drawback,
            assignment.black_drawback,
        ):
            if drawback not in indices:
                raise ValueError(
                    f"true drawback {drawback!r} is not in vocabulary"
                )
        labels = (assignment.white_drawback, assignment.black_drawback)
        previous = by_game.get(assignment.game_id)
        if previous is not None and previous != labels:
            raise ValueError(
                f"inconsistent drawback assignment for {assignment.game_id!r}"
            )
        by_game[assignment.game_id] = labels
    if not by_game:
        raise ValueError("empirical game prior requires at least one assignment")
    counts = [0.0] * len(checked)
    for white_drawback, black_drawback in by_game.values():
        counts[indices[white_drawback]] += 1.0
        counts[indices[black_drawback]] += 1.0
    return _normalize(counts, len(checked), context="empirical game counts")


def symbolic_only_distribution(
    evidence: PublicSymbolicEvidence,
    vocabulary: Sequence[str],
) -> Distribution:
    """Normalize public symbolic probabilities after exact elimination.

    Eliminated classes receive exactly zero probability.  The function fails
    rather than silently falling back to an unrestricted prior when every
    surviving class has zero mass.
    """

    checked = _checked_vocabulary(vocabulary)
    probabilities = _checked_evidence(evidence, len(checked))
    surviving = tuple(
        0.0 if eliminated else probability
        for probability, eliminated in zip(
            probabilities, evidence.eliminated, strict=True
        )
    )
    return _normalize(
        surviving,
        len(checked),
        context="surviving symbolic probabilities",
    )


def make_constant_predictor(distribution: Sequence[float]) -> Predictor:
    """Build a label-free predictor for a fixed prior."""

    captured = tuple(float(value) for value in distribution)

    def predict(_: PredictionInput) -> Distribution:
        return captured

    return predict


def make_symbolic_only_predictor(vocabulary: Sequence[str]) -> Predictor:
    """Build a predictor which consumes only public symbolic evidence."""

    checked = _checked_vocabulary(vocabulary)

    def predict(item: PredictionInput) -> Distribution:
        if item.symbolic is None:
            raise ValueError("symbolic-only baseline requires symbolic evidence")
        return symbolic_only_distribution(item.symbolic, checked)

    return predict


def evaluate_mover_only(
    observations: Iterable[MoverObservation],
    vocabulary: Sequence[str],
    predictor: Predictor,
) -> MoverOnlyReport:
    """Evaluate one head only: the drawback belonging to the current mover."""

    checked = _checked_vocabulary(vocabulary)
    indices = {drawback_id: index for index, drawback_id in enumerate(checked)}
    by_color: dict[Color, list[tuple[int, Distribution]]] = {
        "white": [],
        "black": [],
    }
    for observation in observations:
        _check_observation(observation, indices)
        true_index = indices[observation.true_drawback]
        if observation.symbolic is not None:
            _checked_evidence(observation.symbolic, len(checked))
            if observation.symbolic.eliminated[true_index]:
                raise ValueError(
                    "true drawback is symbolically eliminated for "
                    f"{observation.game_id!r}/{observation.mover}"
                )
        prediction_input = PredictionInput(
            mover=observation.mover,
            symbolic=observation.symbolic,
        )
        distribution = _normalize(
            predictor(prediction_input),
            len(checked),
            context="predicted distribution",
        )
        by_color[observation.mover].append((true_index, distribution))
    combined = [*by_color["white"], *by_color["black"]]
    return MoverOnlyReport(
        overall=_metrics(combined),
        white=_metrics(by_color["white"]),
        black=_metrics(by_color["black"]),
    )


def _check_observation(
    observation: MoverObservation,
    indices: Mapping[str, int],
) -> None:
    if not observation.game_id:
        raise ValueError("game_id must not be empty")
    if observation.mover not in {"white", "black"}:
        raise ValueError("mover must be white or black")
    if observation.true_drawback not in indices:
        raise ValueError(
            f"true drawback {observation.true_drawback!r} is not in vocabulary"
        )


def _checked_evidence(
    evidence: PublicSymbolicEvidence,
    expected_size: int,
) -> Distribution:
    if len(evidence.eliminated) != expected_size:
        raise ValueError(
            "symbolic elimination mask has "
            f"{len(evidence.eliminated)} values; expected {expected_size}"
        )
    if any(not isinstance(value, bool) for value in evidence.eliminated):
        raise ValueError("symbolic elimination mask must contain booleans")
    if all(evidence.eliminated):
        raise ValueError("symbolic engine eliminated every drawback")
    return _normalize(
        evidence.probabilities,
        expected_size,
        context="symbolic probabilities",
    )


def _metrics(
    labelled_distributions: Sequence[tuple[int, Distribution]],
) -> ClassificationMetrics:
    count = len(labelled_distributions)
    if count == 0:
        nan = float("nan")
        return ClassificationMetrics(0, nan, nan, nan, nan, nan)
    hits = {1: 0.0, 3: 0.0, 5: 0.0}
    nll_terms: list[float] = []
    brier_terms: list[float] = []
    for true_index, probabilities in labelled_distributions:
        true_probability = probabilities[true_index]
        greater = sum(
            probability > true_probability for probability in probabilities
        )
        tied = sum(
            probability == true_probability for probability in probabilities
        )
        for k in hits:
            slots = min(k, len(probabilities)) - greater
            if slots > 0:
                hits[k] += min(1.0, slots / tied)
        nll_terms.append(
            -math.log(true_probability)
            if true_probability > 0.0
            else float("inf")
        )
        brier_terms.append(
            math.fsum(
                (probability - (1.0 if index == true_index else 0.0)) ** 2
                for index, probability in enumerate(probabilities)
            )
        )
    return ClassificationMetrics(
        observations=count,
        top1_accuracy=hits[1] / count,
        top3_accuracy=hits[3] / count,
        top5_accuracy=hits[5] / count,
        negative_log_likelihood=math.fsum(nll_terms) / count,
        brier_score=math.fsum(brier_terms) / count,
    )
