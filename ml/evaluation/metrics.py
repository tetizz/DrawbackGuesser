"""Metrics for drawback-classification experiments.

The module uses only the Python standard library and calculates every value
from caller-provided predictions. It contains no benchmark constants or
reported model results.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Callable, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class PredictionExample:
    """One model posterior at a known game move.

    Inputs intentionally contain evaluation labels. They belong in offline
    evaluation only and must never be passed to model inference.
    """

    game_id: str
    move_number: int
    true_drawback: str
    probabilities: Mapping[str, float]
    player_color: str | None = None
    observed_ply: int | None = None
    rule_family: str | None = None
    true_parameters: Mapping[str, object] | None = None
    predicted_parameters: Mapping[str, object] | None = None
    entropy_before: float | None = None
    entropy_after: float | None = None
    diagnostic_information_gain: float | None = None
    hard_eliminated: Mapping[str, bool] | None = None

    def __post_init__(self) -> None:
        if not self.game_id:
            raise ValueError("game_id must not be empty")
        if self.move_number <= 0:
            raise ValueError("move_number must be positive")
        if not self.true_drawback:
            raise ValueError("true_drawback must not be empty")
        if self.player_color not in {None, "white", "black"}:
            raise ValueError("player_color must be white, black, or None")
        if self.observed_ply is not None and self.observed_ply <= 0:
            raise ValueError("observed_ply must be positive when provided")
        _validate_probabilities(self.probabilities, self.true_drawback)
        object.__setattr__(
            self,
            "probabilities",
            MappingProxyType(dict(self.probabilities)),
        )
        if self.hard_eliminated is not None:
            if (
                set(self.hard_eliminated) != set(self.probabilities)
                or any(type(value) is not bool for value in self.hard_eliminated.values())
            ):
                raise ValueError(
                    "hard_eliminated must contain one boolean for every probability"
                )
            object.__setattr__(
                self,
                "hard_eliminated",
                MappingProxyType(dict(self.hard_eliminated)),
            )
        if self.true_parameters is not None:
            object.__setattr__(
                self,
                "true_parameters",
                MappingProxyType(dict(self.true_parameters)),
            )
        if self.predicted_parameters is not None:
            object.__setattr__(
                self,
                "predicted_parameters",
                MappingProxyType(dict(self.predicted_parameters)),
            )
        for name, value in (
            ("entropy_before", self.entropy_before),
            ("entropy_after", self.entropy_after),
            ("diagnostic_information_gain", self.diagnostic_information_gain),
        ):
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{name} must be finite when provided")


@dataclass(frozen=True)
class ClassificationSliceReport:
    """Classification metrics for one measured drawback or rule-family slice."""

    support: int
    top_1_accuracy: float
    top_3_accuracy: float
    top_5_accuracy: float
    negative_log_likelihood: float
    brier_score: float


@dataclass(frozen=True)
class ProbabilityDiagnostics:
    """Exactness diagnostics for accepted posterior rows."""

    checked_count: int
    maximum_absolute_sum_error: float
    hard_mask_checked_count: int
    missing_hard_mask_count: int
    hard_elimination_violation_count: int
    maximum_eliminated_probability: float


@dataclass(frozen=True)
class EvaluationReport:
    count: int
    player_game_count: int
    top_1_accuracy: float
    top_3_accuracy: float
    top_5_accuracy: float
    negative_log_likelihood: float
    brier_score: float
    game_normalized_top_1_accuracy: float
    game_normalized_top_3_accuracy: float
    game_normalized_top_5_accuracy: float
    game_normalized_negative_log_likelihood: float
    game_normalized_brier_score: float
    expected_calibration_error: float
    accuracy_after_moves: Mapping[int, float | None]
    top_1_accuracy_at_observed_plies: Mapping[int, float | None]
    top_3_accuracy_at_observed_plies: Mapping[int, float | None]
    mean_first_rank_one_move: float | None
    median_first_rank_one_move: float | None
    rank_one_player_games: int
    never_rank_one_player_games: int
    accuracy_per_drawback: Mapping[str, float]
    accuracy_per_rule_family: Mapping[str, float]
    metrics_per_drawback: Mapping[str, ClassificationSliceReport]
    metrics_per_rule_family: Mapping[str, ClassificationSliceReport]
    hidden_parameter_accuracy: float | None
    hidden_parameter_accuracy_by_name: Mapping[str, float]
    mean_entropy_reduction: float | None
    mean_diagnostic_information_gain: float | None
    confusion_matrix: Mapping[str, Mapping[str, float]]
    confusion_counts: Mapping[str, Mapping[str, int]]
    probability_diagnostics: ProbabilityDiagnostics


@dataclass(frozen=True)
class BinaryEvaluationReport:
    count: int
    accuracy: float
    negative_log_likelihood: float
    brier_score: float


@dataclass(frozen=True)
class LegalMaskEvaluationReport:
    example_count: int
    dimension: int
    exact_match_accuracy: float
    micro_precision: float
    micro_recall: float
    micro_f1: float
    binary_cross_entropy: float


class StreamingEvaluation:
    """Bounded-memory multiclass metrics.

    Storage is proportional to the label/family/parameter domains and game
    count, never the number of move rows or posterior vectors.
    """

    def __init__(self, calibration_bins: int = 15) -> None:
        if calibration_bins <= 0:
            raise ValueError("calibration_bins must be positive")
        self._calibration_bins = calibration_bins
        self._count = 0
        self._top = [0.0, 0.0, 0.0]
        self._nll = 0.0
        self._nll_infinite = False
        self._brier = 0.0
        self._bins = [[0, 0, 0.0] for _ in range(calibration_bins)]
        self._horizon_by_game: dict[
            tuple[str, str | None],
            dict[int, tuple[int, float, float]],
        ] = {}
        self._first_by_game: dict[tuple[str, str | None], int] = {}
        self._player_games: set[tuple[str, str | None]] = set()
        self._game_metrics: dict[
            tuple[str, str | None],
            list[float],
        ] = {}
        self._drawbacks: dict[str, list[int]] = {}
        self._families: dict[str, list[int]] = {}
        self._drawback_metrics: dict[str, list[float]] = {}
        self._family_metrics: dict[str, list[float]] = {}
        self._parameters: dict[str, list[int]] = {}
        self._parameter_correct = 0
        self._parameter_total = 0
        self._entropy_sum = 0.0
        self._entropy_count = 0
        self._information_sum = 0.0
        self._information_count = 0
        self._confusion: dict[str, dict[str, float]] = {}
        self._confusion_counts: dict[str, dict[str, int]] = {}
        self._maximum_probability_sum_error = 0.0
        self._hard_mask_checked_count = 0
        self._missing_hard_mask_count = 0
        self._hard_elimination_violation_count = 0
        self._maximum_eliminated_probability = 0.0

    def add(self, example: PredictionExample) -> None:
        top_one_credit = _top_k_credit(example, 1)
        tied_maximum = _maximum_labels(example)
        player_game = (example.game_id, example.player_color)
        self._player_games.add(player_game)
        self._count += 1
        for index, k in enumerate((1, 3, 5)):
            self._top[index] += _top_k_credit(example, k)
        probability = example.probabilities[example.true_drawback]
        if probability == 0.0:
            self._nll_infinite = True
        else:
            self._nll += -math.log(probability)
        row_brier = math.fsum(
            (value - (1.0 if label == example.true_drawback else 0.0)) ** 2
            for label, value in example.probabilities.items()
        )
        self._brier += row_brier
        self._maximum_probability_sum_error = max(
            self._maximum_probability_sum_error,
            abs(math.fsum(example.probabilities.values()) - 1.0),
        )
        if example.hard_eliminated is None:
            self._missing_hard_mask_count += 1
        else:
            self._hard_mask_checked_count += 1
            for label, eliminated in example.hard_eliminated.items():
                if not eliminated:
                    continue
                eliminated_probability = example.probabilities[label]
                self._maximum_eliminated_probability = max(
                    self._maximum_eliminated_probability,
                    eliminated_probability,
                )
                if eliminated_probability != 0.0:
                    self._hard_elimination_violation_count += 1
        row_nll = (
            float("inf") if probability == 0.0 else -math.log(probability)
        )
        game_metrics = self._game_metrics.setdefault(
            player_game,
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        )
        game_metrics[0] += 1.0
        game_metrics[1] += _top_k_credit(example, 1)
        game_metrics[2] += _top_k_credit(example, 3)
        game_metrics[3] += _top_k_credit(example, 5)
        game_metrics[4] += row_nll
        game_metrics[5] += row_brier
        confidence = max(example.probabilities.values())
        bin_index = min(
            int(confidence * self._calibration_bins),
            self._calibration_bins - 1,
        )
        bucket = self._bins[bin_index]
        bucket[0] += 1
        bucket[1] += top_one_credit
        bucket[2] += confidence
        horizon = (
            example.observed_ply
            if example.observed_ply is not None
            else example.move_number
        )
        by_horizon = self._horizon_by_game.setdefault(player_game, {})
        for target in (5, 10, 15, 20):
            if horizon <= target:
                prior_prefix = by_horizon.get(target)
                if prior_prefix is None or horizon > prior_prefix[0]:
                    by_horizon[target] = (
                        horizon,
                        _top_k_credit(example, 1),
                        _top_k_credit(example, 3),
                    )
        if _is_unique_rank_one(example):
            prior = self._first_by_game.get(player_game)
            if prior is None or horizon < prior:
                self._first_by_game[player_game] = horizon
        for domain, key in (
            (self._drawbacks, example.true_drawback),
            (self._families, example.rule_family),
        ):
            if key is not None:
                counts = domain.setdefault(key, [0, 0])
                counts[0] += top_one_credit
                counts[1] += 1
        for domain, key in (
            (self._drawback_metrics, example.true_drawback),
            (self._family_metrics, example.rule_family),
        ):
            if key is not None:
                metrics = domain.setdefault(
                    key,
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                )
                metrics[0] += 1.0
                metrics[1] += top_one_credit
                metrics[2] += _top_k_credit(example, 3)
                metrics[3] += _top_k_credit(example, 5)
                metrics[4] += row_nll
                metrics[5] += row_brier
        if example.true_parameters is not None:
            predicted_parameters = example.predicted_parameters or {}
            for name, value in example.true_parameters.items():
                counts = self._parameters.setdefault(name, [0, 0])
                matches = name in predicted_parameters and predicted_parameters[name] == value
                counts[0] += matches
                counts[1] += 1
                self._parameter_correct += matches
                self._parameter_total += 1
        if example.entropy_before is not None and example.entropy_after is not None:
            self._entropy_sum += example.entropy_before - example.entropy_after
            self._entropy_count += 1
        if example.diagnostic_information_gain is not None:
            self._information_sum += example.diagnostic_information_gain
            self._information_count += 1
        confusion_row = self._confusion.setdefault(example.true_drawback, {})
        predicted_credit = 1.0 / len(tied_maximum)
        for predicted in tied_maximum:
            confusion_row[predicted] = (
                confusion_row.get(predicted, 0.0) + predicted_credit
            )
        deterministic_prediction = tied_maximum[0]
        confusion_count_row = self._confusion_counts.setdefault(
            example.true_drawback,
            {},
        )
        confusion_count_row[deterministic_prediction] = (
            confusion_count_row.get(deterministic_prediction, 0) + 1
        )

    def report(self) -> EvaluationReport:
        if self._count == 0:
            raise ValueError("at least one prediction example is required")
        ece = math.fsum(
            (count / self._count)
            * abs((correct / count) - (confidence / count))
            for count, correct, confidence in self._bins
            if count
        )
        horizon_top_1 = _horizon_accuracies(self._horizon_by_game, 1)
        horizon_top_3 = _horizon_accuracies(self._horizon_by_game, 2)
        first_rank_one = sorted(self._first_by_game.values())
        game_averages = [
            (
                top1 / count,
                top3 / count,
                top5 / count,
                nll / count,
                brier / count,
            )
            for count, top1, top3, top5, nll, brier
            in self._game_metrics.values()
        ]
        return EvaluationReport(
            count=self._count,
            player_game_count=len(game_averages),
            top_1_accuracy=self._top[0] / self._count,
            top_3_accuracy=self._top[1] / self._count,
            top_5_accuracy=self._top[2] / self._count,
            negative_log_likelihood=(
                math.inf if self._nll_infinite else self._nll / self._count
            ),
            brier_score=self._brier / self._count,
            game_normalized_top_1_accuracy=math.fsum(
                value[0] for value in game_averages
            ) / len(game_averages),
            game_normalized_top_3_accuracy=math.fsum(
                value[1] for value in game_averages
            ) / len(game_averages),
            game_normalized_top_5_accuracy=math.fsum(
                value[2] for value in game_averages
            ) / len(game_averages),
            game_normalized_negative_log_likelihood=(
                math.fsum(value[3] for value in game_averages)
                / len(game_averages)
            ),
            game_normalized_brier_score=math.fsum(
                value[4] for value in game_averages
            ) / len(game_averages),
            expected_calibration_error=ece,
            accuracy_after_moves=horizon_top_1,
            top_1_accuracy_at_observed_plies=horizon_top_1,
            top_3_accuracy_at_observed_plies=horizon_top_3,
            mean_first_rank_one_move=(
                None
                if not first_rank_one
                else math.fsum(first_rank_one) / len(first_rank_one)
            ),
            median_first_rank_one_move=_median(first_rank_one),
            rank_one_player_games=len(first_rank_one),
            never_rank_one_player_games=(
                len(self._player_games) - len(first_rank_one)
            ),
            accuracy_per_drawback=MappingProxyType({
                key: correct / count
                for key, (correct, count) in sorted(self._drawbacks.items())
            }),
            accuracy_per_rule_family=MappingProxyType({
                key: correct / count
                for key, (correct, count) in sorted(self._families.items())
            }),
            metrics_per_drawback=_slice_reports(self._drawback_metrics),
            metrics_per_rule_family=_slice_reports(self._family_metrics),
            hidden_parameter_accuracy=(
                None
                if self._parameter_total == 0
                else self._parameter_correct / self._parameter_total
            ),
            hidden_parameter_accuracy_by_name=MappingProxyType({
                key: correct / count
                for key, (correct, count) in sorted(self._parameters.items())
            }),
            mean_entropy_reduction=(
                None
                if self._entropy_count == 0
                else self._entropy_sum / self._entropy_count
            ),
            mean_diagnostic_information_gain=(
                None
                if self._information_count == 0
                else self._information_sum / self._information_count
            ),
            confusion_matrix=MappingProxyType({
                key: MappingProxyType(dict(sorted(values.items())))
                for key, values in sorted(self._confusion.items())
            }),
            confusion_counts=MappingProxyType({
                key: MappingProxyType(dict(sorted(values.items())))
                for key, values in sorted(self._confusion_counts.items())
            }),
            probability_diagnostics=ProbabilityDiagnostics(
                checked_count=self._count,
                maximum_absolute_sum_error=self._maximum_probability_sum_error,
                hard_mask_checked_count=self._hard_mask_checked_count,
                missing_hard_mask_count=self._missing_hard_mask_count,
                hard_elimination_violation_count=(
                    self._hard_elimination_violation_count
                ),
                maximum_eliminated_probability=(
                    self._maximum_eliminated_probability
                ),
            ),
        )


class StreamingBinaryEvaluation:
    def __init__(self, threshold: float = 0.5) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        self._threshold = threshold
        self._count = self._correct = 0
        self._loss = self._brier = 0.0
        self._infinite = False

    def add(self, label: bool, probability: float) -> None:
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("binary probabilities must be finite values in [0, 1]")
        self._count += 1
        self._correct += (probability >= self._threshold) == label
        target = probability if label else 1.0 - probability
        if target == 0.0:
            self._infinite = True
        else:
            self._loss += -math.log(target)
        self._brier += (probability - float(label)) ** 2

    def report(self) -> BinaryEvaluationReport:
        if self._count == 0:
            raise ValueError("binary labels and probabilities must be non-empty")
        return BinaryEvaluationReport(
            count=self._count,
            accuracy=self._correct / self._count,
            negative_log_likelihood=(
                math.inf if self._infinite else self._loss / self._count
            ),
            brier_score=self._brier / self._count,
        )


class StreamingLegalMaskEvaluation:
    def __init__(self, dimension: int, threshold: float = 0.5) -> None:
        if dimension <= 0:
            raise ValueError("legal-mask dimension must be positive")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        self._dimension = dimension
        self._threshold = threshold
        self._count = self._exact = 0
        self._true_positive = self._false_positive = self._false_negative = 0
        self._loss = 0.0
        self._infinite = False

    def add(self, true_indices: Iterable[int], probabilities: Sequence[float]) -> None:
        if len(probabilities) != self._dimension:
            raise ValueError("legal-mask output dimension mismatch")
        true = frozenset(true_indices)
        if any(index < 0 or index >= self._dimension for index in true):
            raise ValueError("legal-mask true index is outside the vocabulary")
        exact = True
        for index, probability in enumerate(probabilities):
            if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise ValueError("mask probabilities must be finite values in [0, 1]")
            label = index in true
            predicted = probability >= self._threshold
            exact = exact and predicted == label
            self._true_positive += predicted and label
            self._false_positive += predicted and not label
            self._false_negative += not predicted and label
            target = probability if label else 1.0 - probability
            if target == 0.0:
                self._infinite = True
            else:
                self._loss += -math.log(target)
        self._count += 1
        self._exact += exact

    def add_batch_statistics(
        self,
        *,
        example_count: int,
        dimension: int,
        exact_matches: int,
        true_positives: int,
        false_positives: int,
        false_negatives: int,
        binary_cross_entropy_sum: float,
        has_infinite_loss: bool,
    ) -> None:
        if dimension != self._dimension or example_count < 0:
            raise ValueError("legal-mask batch statistics have invalid dimensions")
        counts = (
            exact_matches,
            true_positives,
            false_positives,
            false_negatives,
        )
        if any(value < 0 for value in counts) or exact_matches > example_count:
            raise ValueError("legal-mask batch statistics contain invalid counts")
        if (
            not math.isfinite(binary_cross_entropy_sum)
            or binary_cross_entropy_sum < 0.0
        ):
            raise ValueError("legal-mask batch loss sum must be finite and non-negative")
        self._count += example_count
        self._exact += exact_matches
        self._true_positive += true_positives
        self._false_positive += false_positives
        self._false_negative += false_negatives
        self._loss += binary_cross_entropy_sum
        self._infinite = self._infinite or has_infinite_loss

    def report(self) -> LegalMaskEvaluationReport:
        if self._count == 0:
            raise ValueError("legal masks must be non-empty")
        precision_denominator = self._true_positive + self._false_positive
        recall_denominator = self._true_positive + self._false_negative
        precision = (
            0.0
            if precision_denominator == 0
            else self._true_positive / precision_denominator
        )
        recall = (
            0.0
            if recall_denominator == 0
            else self._true_positive / recall_denominator
        )
        f1 = (
            0.0
            if precision + recall == 0.0
            else 2 * precision * recall / (precision + recall)
        )
        return LegalMaskEvaluationReport(
            example_count=self._count,
            dimension=self._dimension,
            exact_match_accuracy=self._exact / self._count,
            micro_precision=precision,
            micro_recall=recall,
            micro_f1=f1,
            binary_cross_entropy=(
                math.inf
                if self._infinite
                else self._loss / (self._count * self._dimension)
            ),
        )


def _materialize(examples: Iterable[PredictionExample]) -> tuple[PredictionExample, ...]:
    materialized = tuple(examples)
    if not materialized:
        raise ValueError("at least one prediction example is required")
    return materialized


def _validate_probabilities(
    probabilities: Mapping[str, float],
    true_drawback: str,
) -> None:
    if not probabilities:
        raise ValueError("probabilities must not be empty")
    if true_drawback not in probabilities:
        raise ValueError("probabilities must include the true drawback")
    total = 0.0
    for label, value in probabilities.items():
        if not label:
            raise ValueError("probability labels must not be empty")
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError("probabilities must be finite values in [0, 1]")
        total += value
    if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("probabilities must sum to one")


def _is_unique_rank_one(example: PredictionExample) -> bool:
    true_probability = example.probabilities[example.true_drawback]
    return (
        sum(
            probability > true_probability
            for probability in example.probabilities.values()
        )
        == 0
        and sum(
            probability == true_probability
            for probability in example.probabilities.values()
        )
        == 1
    )


def _maximum_labels(example: PredictionExample) -> tuple[str, ...]:
    maximum = max(example.probabilities.values())
    return tuple(
        sorted(
            label
            for label, probability in example.probabilities.items()
            if probability == maximum
        )
    )


def _top_k_credit(example: PredictionExample, k: int) -> float:
    """Return expected Top-k credit when equal scores straddle the cutoff."""

    true_probability = example.probabilities[example.true_drawback]
    greater = sum(
        probability > true_probability
        for probability in example.probabilities.values()
    )
    tied = sum(
        probability == true_probability
        for probability in example.probabilities.values()
    )
    slots = min(k, len(example.probabilities)) - greater
    if slots <= 0:
        return 0.0
    return min(1.0, slots / tied)


def top_k_accuracy(
    examples: Iterable[PredictionExample],
    k: int,
) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    rows = _materialize(examples)
    correct = math.fsum(_top_k_credit(example, k) for example in rows)
    return correct / len(rows)


def negative_log_likelihood(examples: Iterable[PredictionExample]) -> float:
    rows = _materialize(examples)
    losses = []
    for example in rows:
        probability = example.probabilities[example.true_drawback]
        if probability == 0.0:
            return math.inf
        losses.append(-math.log(probability))
    return math.fsum(losses) / len(losses)


def brier_score(examples: Iterable[PredictionExample]) -> float:
    """Return the multiclass Brier score (sum over classes, then mean)."""

    rows = _materialize(examples)
    scores = []
    for example in rows:
        scores.append(
            math.fsum(
                (probability - (1.0 if label == example.true_drawback else 0.0))
                ** 2
                for label, probability in example.probabilities.items()
            )
        )
    return math.fsum(scores) / len(scores)


def expected_calibration_error(
    examples: Iterable[PredictionExample],
    bin_count: int = 15,
) -> float:
    if bin_count <= 0:
        raise ValueError("bin_count must be positive")
    rows = _materialize(examples)
    bins: list[list[tuple[float, float]]] = [[] for _ in range(bin_count)]
    for example in rows:
        confidence = max(example.probabilities.values())
        index = min(int(confidence * bin_count), bin_count - 1)
        bins[index].append((confidence, _top_k_credit(example, 1)))

    total = len(rows)
    return math.fsum(
        (len(items) / total)
        * abs(
            (sum(correct for _, correct in items) / len(items))
            - (math.fsum(confidence for confidence, _ in items) / len(items))
        )
        for items in bins
        if items
    )


def accuracy_at_moves(
    examples: Iterable[PredictionExample],
    move_numbers: Sequence[int] = (5, 10, 15, 20),
) -> Mapping[int, float | None]:
    rows = tuple(examples)
    result: dict[int, float | None] = {}
    for move_number in move_numbers:
        if move_number <= 0:
            raise ValueError("move numbers must be positive")
        latest_by_player_game: dict[
            tuple[str, str | None],
            tuple[int, PredictionExample],
        ] = {}
        for row in rows:
            observed = (
                row.observed_ply
                if row.observed_ply is not None
                else row.move_number
            )
            if observed > move_number:
                continue
            key = (row.game_id, row.player_color)
            prior = latest_by_player_game.get(key)
            if prior is None or observed > prior[0]:
                latest_by_player_game[key] = (observed, row)
        at_move = [row for _observed, row in latest_by_player_game.values()]
        result[move_number] = (
            None
            if not at_move
            else math.fsum(_top_k_credit(row, 1) for row in at_move)
            / len(at_move)
        )
    return MappingProxyType(result)


def mean_first_rank_one_move(
    examples: Iterable[PredictionExample],
) -> float | None:
    first_by_game: dict[tuple[str, str | None], int] = {}
    for example in examples:
        if _is_unique_rank_one(example):
            key = (example.game_id, example.player_color)
            horizon = (
                example.observed_ply
                if example.observed_ply is not None
                else example.move_number
            )
            prior = first_by_game.get(key)
            if prior is None or horizon < prior:
                first_by_game[key] = horizon
    if not first_by_game:
        return None
    return math.fsum(first_by_game.values()) / len(first_by_game)


def _median(values: Sequence[int]) -> float | None:
    if not values:
        return None
    midpoint = len(values) // 2
    if len(values) % 2:
        return float(values[midpoint])
    return (values[midpoint - 1] + values[midpoint]) / 2.0


def _horizon_accuracies(
    horizon_by_game: Mapping[
        tuple[str, str | None],
        Mapping[int, tuple[int, float, float]],
    ],
    credit_index: int,
) -> Mapping[int, float | None]:
    result: dict[int, float | None] = {}
    for target in (5, 10, 15, 20):
        credits = [
            entry[credit_index]
            for values in horizon_by_game.values()
            if (entry := values.get(target)) is not None
        ]
        result[target] = (
            None if not credits else math.fsum(credits) / len(credits)
        )
    return MappingProxyType(result)


def _slice_reports(
    values_by_group: Mapping[str, Sequence[float]],
) -> Mapping[str, ClassificationSliceReport]:
    reports: dict[str, ClassificationSliceReport] = {}
    for group, values in sorted(values_by_group.items()):
        count, top_1, top_3, top_5, nll, brier = values
        reports[group] = ClassificationSliceReport(
            support=int(count),
            top_1_accuracy=top_1 / count,
            top_3_accuracy=top_3 / count,
            top_5_accuracy=top_5 / count,
            negative_log_likelihood=nll / count,
            brier_score=brier / count,
        )
    return MappingProxyType(reports)


def _group_accuracy(
    examples: Sequence[PredictionExample],
    group_for: Callable[[PredictionExample], str | None],
) -> Mapping[str, float]:
    counts: dict[str, list[float]] = {}
    for example in examples:
        group = group_for(example)
        if group is None:
            continue
        values = counts.setdefault(group, [0, 0])
        values[1] += 1
        values[0] += _top_k_credit(example, 1)
    return MappingProxyType(
        {
            group: correct / count
            for group, (correct, count) in sorted(counts.items())
        }
    )


def _parameter_accuracy(
    examples: Sequence[PredictionExample],
) -> tuple[float | None, Mapping[str, float]]:
    totals: dict[str, list[int]] = {}
    correct_total = 0
    value_total = 0
    for example in examples:
        if example.true_parameters is None:
            continue
        predicted = example.predicted_parameters or {}
        for name, value in example.true_parameters.items():
            counts = totals.setdefault(name, [0, 0])
            counts[1] += 1
            value_total += 1
            if name in predicted and predicted[name] == value:
                counts[0] += 1
                correct_total += 1
    by_name = MappingProxyType(
        {
            name: correct / count
            for name, (correct, count) in sorted(totals.items())
        }
    )
    overall = None if value_total == 0 else correct_total / value_total
    return overall, by_name


def _optional_mean(values: Iterable[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return None if not present else math.fsum(present) / len(present)


def confusion_matrix(
    examples: Iterable[PredictionExample],
) -> Mapping[str, Mapping[str, float]]:
    matrix: dict[str, dict[str, float]] = {}
    for example in examples:
        row = matrix.setdefault(example.true_drawback, {})
        tied_maximum = _maximum_labels(example)
        credit = 1.0 / len(tied_maximum)
        for predicted in tied_maximum:
            row[predicted] = row.get(predicted, 0.0) + credit
    return MappingProxyType(
        {
            true_label: MappingProxyType(dict(sorted(row.items())))
            for true_label, row in sorted(matrix.items())
        }
    )


def evaluate(
    examples: Iterable[PredictionExample],
    calibration_bins: int = 15,
) -> EvaluationReport:
    rows = _materialize(examples)
    streaming = StreamingEvaluation(calibration_bins)
    for row in rows:
        streaming.add(row)
    return streaming.report()


def evaluate_binary(
    labels: Sequence[bool],
    probabilities: Sequence[float],
    threshold: float = 0.5,
) -> BinaryEvaluationReport:
    if len(labels) == 0 or len(labels) != len(probabilities):
        raise ValueError("binary labels and probabilities must be non-empty and aligned")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    losses: list[float] = []
    brier: list[float] = []
    correct = 0
    for label, probability in zip(labels, probabilities, strict=True):
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("binary probabilities must be finite values in [0, 1]")
        predicted = probability >= threshold
        correct += predicted == label
        target_probability = probability if label else 1.0 - probability
        losses.append(math.inf if target_probability == 0.0 else -math.log(target_probability))
        brier.append((probability - float(label)) ** 2)
    return BinaryEvaluationReport(
        count=len(labels),
        accuracy=correct / len(labels),
        negative_log_likelihood=(
            math.inf if any(math.isinf(value) for value in losses)
            else math.fsum(losses) / len(losses)
        ),
        brier_score=math.fsum(brier) / len(brier),
    )


def evaluate_legal_masks(
    true_masks: Sequence[Sequence[bool]],
    probabilities: Sequence[Sequence[float]],
    threshold: float = 0.5,
) -> LegalMaskEvaluationReport:
    if len(true_masks) == 0 or len(true_masks) != len(probabilities):
        raise ValueError("legal masks and probabilities must be non-empty and aligned")
    dimension = len(true_masks[0])
    if dimension == 0:
        raise ValueError("legal-mask dimension must be positive")
    true_positive = false_positive = false_negative = exact = 0
    losses: list[float] = []
    for labels, predicted_probabilities in zip(true_masks, probabilities, strict=True):
        if len(labels) != dimension or len(predicted_probabilities) != dimension:
            raise ValueError("all legal masks must have the same dimension")
        predicted_mask: list[bool] = []
        for label, probability in zip(labels, predicted_probabilities, strict=True):
            if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise ValueError("mask probabilities must be finite values in [0, 1]")
            predicted = probability >= threshold
            predicted_mask.append(predicted)
            true_positive += predicted and label
            false_positive += predicted and not label
            false_negative += not predicted and label
            target_probability = probability if label else 1.0 - probability
            losses.append(
                math.inf if target_probability == 0.0 else -math.log(target_probability)
            )
        exact += tuple(predicted_mask) == tuple(labels)
    precision = (
        0.0 if true_positive + false_positive == 0
        else true_positive / (true_positive + false_positive)
    )
    recall = (
        0.0 if true_positive + false_negative == 0
        else true_positive / (true_positive + false_negative)
    )
    f1 = 0.0 if precision + recall == 0.0 else 2 * precision * recall / (precision + recall)
    return LegalMaskEvaluationReport(
        example_count=len(true_masks),
        dimension=dimension,
        exact_match_accuracy=exact / len(true_masks),
        micro_precision=precision,
        micro_recall=recall,
        micro_f1=f1,
        binary_cross_entropy=(
            math.inf if any(math.isinf(value) for value in losses)
            else math.fsum(losses) / len(losses)
        ),
    )
