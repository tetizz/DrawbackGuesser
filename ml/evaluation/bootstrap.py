"""Deterministic paired bootstrap over complete simulation-seed clusters.

The sampling unit is a simulation seed. Every player-game trajectory associated
with a sampled seed is retained, so White, Black, and all observed plies move
together through every replicate.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Callable, Iterable, Sequence

from .metrics import PredictionExample


DEFAULT_REPLICATES = 10_000


@dataclass(frozen=True)
class PlayerGamePredictions:
    """One color's complete prediction trajectory from a simulated game."""

    simulation_seed: int
    game_id: str
    player_color: str
    examples: tuple[PredictionExample, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.simulation_seed, bool)
            or not isinstance(self.simulation_seed, int)
            or self.simulation_seed < 0
        ):
            raise ValueError("simulation_seed must be a non-negative integer")
        if not self.game_id:
            raise ValueError("game_id must not be empty")
        if self.player_color not in {"white", "black"}:
            raise ValueError("player_color must be white or black")
        if not self.examples:
            raise ValueError("examples must contain a complete non-empty trajectory")
        prior_horizon = -1
        for example in self.examples:
            if example.game_id != self.game_id:
                raise ValueError("every example must match the trajectory game_id")
            if example.player_color != self.player_color:
                raise ValueError("every example must match the trajectory player_color")
            horizon = (
                example.observed_ply
                if example.observed_ply is not None
                else example.move_number
            )
            if horizon <= prior_horizon:
                raise ValueError("trajectory examples must have increasing horizons")
            prior_horizon = horizon


@dataclass(frozen=True)
class BootstrapConfidenceInterval:
    estimate: float
    lower: float
    upper: float
    confidence_level: float
    replicates: int


@dataclass(frozen=True)
class ComparatorDifferenceSummary:
    """Candidate, comparator, and paired candidate-minus-comparator intervals."""

    candidate: BootstrapConfidenceInterval
    comparator: BootstrapConfidenceInterval
    difference: BootstrapConfidenceInterval


PlayerGameStatistic = Callable[[Sequence[PlayerGamePredictions]], float]


def _validate_options(
    *,
    replicates: int,
    confidence_level: float,
    random_seed: int,
) -> None:
    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates <= 0:
        raise ValueError("replicates must be a positive integer")
    if (
        not math.isfinite(confidence_level)
        or confidence_level <= 0.0
        or confidence_level >= 1.0
    ):
        raise ValueError("confidence_level must be strictly between zero and one")
    if (
        isinstance(random_seed, bool)
        or not isinstance(random_seed, int)
        or random_seed < 0
    ):
        raise ValueError("random_seed must be a non-negative integer")


def _identity(trajectory: PlayerGamePredictions) -> tuple[int, str, str]:
    return (
        trajectory.simulation_seed,
        trajectory.game_id,
        trajectory.player_color,
    )


def _normalize(
    trajectories: Iterable[PlayerGamePredictions],
) -> tuple[PlayerGamePredictions, ...]:
    rows = tuple(sorted(trajectories, key=_identity))
    if not rows:
        raise ValueError("at least one player-game trajectory is required")
    identities = [_identity(row) for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("player-game trajectory identities must be unique")

    colors_by_game: dict[tuple[int, str], set[str]] = {}
    for row in rows:
        colors_by_game.setdefault(
            (row.simulation_seed, row.game_id), set()
        ).add(row.player_color)
    incomplete = [
        identity
        for identity, colors in colors_by_game.items()
        if colors != {"white", "black"}
    ]
    if incomplete:
        raise ValueError(
            "every sampled game must include complete White and Black trajectories"
        )
    return rows


def _group_by_seed(
    rows: Sequence[PlayerGamePredictions],
) -> tuple[tuple[PlayerGamePredictions, ...], ...]:
    grouped: dict[int, list[PlayerGamePredictions]] = {}
    for row in rows:
        grouped.setdefault(row.simulation_seed, []).append(row)
    return tuple(tuple(grouped[seed]) for seed in sorted(grouped))


def _metric(
    statistic: PlayerGameStatistic,
    rows: Sequence[PlayerGamePredictions],
) -> float:
    value = statistic(rows)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("bootstrap statistic must return a real number")
    rendered = float(value)
    if not math.isfinite(rendered):
        raise ValueError("bootstrap statistic must return a finite number")
    return rendered


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = probability * (len(sorted_values) - 1)
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return sorted_values[lower_index]
    weight = position - lower_index
    return (
        sorted_values[lower_index] * (1.0 - weight)
        + sorted_values[upper_index] * weight
    )


def _interval(
    estimate: float,
    values: Sequence[float],
    *,
    confidence_level: float,
    replicates: int,
) -> BootstrapConfidenceInterval:
    ordered = sorted(values)
    tail = (1.0 - confidence_level) / 2.0
    return BootstrapConfidenceInterval(
        estimate=estimate,
        lower=_quantile(ordered, tail),
        upper=_quantile(ordered, 1.0 - tail),
        confidence_level=confidence_level,
        replicates=replicates,
    )


def player_game_bootstrap_ci(
    trajectories: Iterable[PlayerGamePredictions],
    statistic: PlayerGameStatistic,
    *,
    replicates: int = DEFAULT_REPLICATES,
    confidence_level: float = 0.95,
    random_seed: int = 0,
) -> BootstrapConfidenceInterval:
    """Return a percentile CI from complete-seed bootstrap replicates."""

    _validate_options(
        replicates=replicates,
        confidence_level=confidence_level,
        random_seed=random_seed,
    )
    rows = _normalize(trajectories)
    groups = _group_by_seed(rows)
    generator = random.Random(random_seed)
    sampled_values: list[float] = []
    for _replicate in range(replicates):
        sample = tuple(
            row
            for _draw in range(len(groups))
            for row in groups[generator.randrange(len(groups))]
        )
        sampled_values.append(_metric(statistic, sample))
    return _interval(
        _metric(statistic, rows),
        sampled_values,
        confidence_level=confidence_level,
        replicates=replicates,
    )


def paired_comparator_bootstrap(
    candidate: Iterable[PlayerGamePredictions],
    comparator: Iterable[PlayerGamePredictions],
    statistic: PlayerGameStatistic,
    *,
    replicates: int = DEFAULT_REPLICATES,
    confidence_level: float = 0.95,
    random_seed: int = 0,
) -> ComparatorDifferenceSummary:
    """Compare aligned systems using identical complete-seed resamples."""

    _validate_options(
        replicates=replicates,
        confidence_level=confidence_level,
        random_seed=random_seed,
    )
    candidate_rows = _normalize(candidate)
    comparator_rows = _normalize(comparator)
    if tuple(map(_identity, candidate_rows)) != tuple(map(_identity, comparator_rows)):
        raise ValueError("candidate and comparator player-game identities must align")
    for left, right in zip(candidate_rows, comparator_rows, strict=True):
        left_targets = tuple(
            (
                example.move_number,
                example.observed_ply,
                example.true_drawback,
                example.rule_family,
            )
            for example in left.examples
        )
        right_targets = tuple(
            (
                example.move_number,
                example.observed_ply,
                example.true_drawback,
                example.rule_family,
            )
            for example in right.examples
        )
        if left_targets != right_targets:
            raise ValueError(
                "candidate and comparator trajectory horizons and targets must align"
            )

    candidate_groups = _group_by_seed(candidate_rows)
    comparator_groups = _group_by_seed(comparator_rows)
    candidate_by_seed = {
        group[0].simulation_seed: group for group in candidate_groups
    }
    comparator_by_seed = {
        group[0].simulation_seed: group for group in comparator_groups
    }
    seeds = tuple(sorted(candidate_by_seed))
    if seeds != tuple(sorted(comparator_by_seed)):
        raise ValueError("candidate and comparator simulation seeds must align")

    generator = random.Random(random_seed)
    candidate_values: list[float] = []
    comparator_values: list[float] = []
    differences: list[float] = []
    for _replicate in range(replicates):
        sampled_seeds = tuple(
            seeds[generator.randrange(len(seeds))] for _draw in range(len(seeds))
        )
        candidate_sample = tuple(
            row for seed in sampled_seeds for row in candidate_by_seed[seed]
        )
        comparator_sample = tuple(
            row for seed in sampled_seeds for row in comparator_by_seed[seed]
        )
        candidate_value = _metric(statistic, candidate_sample)
        comparator_value = _metric(statistic, comparator_sample)
        candidate_values.append(candidate_value)
        comparator_values.append(comparator_value)
        differences.append(candidate_value - comparator_value)

    candidate_estimate = _metric(statistic, candidate_rows)
    comparator_estimate = _metric(statistic, comparator_rows)
    return ComparatorDifferenceSummary(
        candidate=_interval(
            candidate_estimate,
            candidate_values,
            confidence_level=confidence_level,
            replicates=replicates,
        ),
        comparator=_interval(
            comparator_estimate,
            comparator_values,
            confidence_level=confidence_level,
            replicates=replicates,
        ),
        difference=_interval(
            candidate_estimate - comparator_estimate,
            differences,
            confidence_level=confidence_level,
            replicates=replicates,
        ),
    )
