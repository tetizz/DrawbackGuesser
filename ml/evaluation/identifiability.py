"""Hard-legal identifiability and oracle-only mask diagnostics.

This module cannot establish how candidate masks were produced. Release use
requires a separate, label-blind replay receipt binding every input trajectory.
Truth labels and the true parameter-hypothesis ID exist only for offline
scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Callable, Hashable, Iterable, Mapping, TypeVar


LegalSetHistory = tuple[frozenset[str], ...]
GroupKey = TypeVar("GroupKey", bound=Hashable)


@dataclass(frozen=True)
class HypothesisLegality:
    hypothesis_id: str
    drawback_id: str
    eliminated: bool
    permitted_move_history: LegalSetHistory
    posterior_probability: float | None = None

    def __post_init__(self) -> None:
        if not self.hypothesis_id or not self.drawback_id:
            raise ValueError("hypothesis_id and drawback_id must not be empty")
        history = tuple(frozenset(moves) for moves in self.permitted_move_history)
        _validate_move_history(history, "permitted")
        object.__setattr__(self, "permitted_move_history", history)
        probability = self.posterior_probability
        if probability is not None and (
            isinstance(probability, bool)
            or not isinstance(probability, (int, float))
            or not math.isfinite(probability)
            or probability < 0.0
        ):
            raise ValueError("posterior_probability must be finite and non-negative")
        if self.eliminated and probability not in {None, 0.0}:
            raise ValueError("an eliminated hypothesis must have zero posterior")


@dataclass(frozen=True)
class IdentifiabilityObservation:
    """One offline label joined to caller-supplied, label-blind replay output."""

    game_id: str
    color: str
    horizon: int
    turn_indices: tuple[int, ...]
    true_drawback: str
    true_hypothesis_id: str
    ordinary_legal_history: LegalSetHistory
    hypotheses: tuple[HypothesisLegality, ...]

    def __post_init__(self) -> None:
        if not self.game_id:
            raise ValueError("game_id must not be empty")
        if self.color not in {"white", "black"}:
            raise ValueError("color must be white or black")
        if (
            isinstance(self.horizon, bool)
            or not isinstance(self.horizon, int)
            or self.horizon <= 0
        ):
            raise ValueError("horizon must be a positive integer")
        indices = tuple(self.turn_indices)
        if (
            not indices
            or any(
                isinstance(index, bool)
                or not isinstance(index, int)
                or index <= 0
                for index in indices
            )
            or any(left >= right for left, right in zip(indices, indices[1:]))
            or indices[-1] > self.horizon
        ):
            raise ValueError(
                "turn_indices must be positive, strictly increasing, and end at or before horizon"
            )
        if not self.true_drawback or not self.true_hypothesis_id:
            raise ValueError("truth label and hypothesis ID must not be empty")
        ordinary = tuple(frozenset(moves) for moves in self.ordinary_legal_history)
        _validate_move_history(ordinary, "ordinary")
        if len(ordinary) != len(indices):
            raise ValueError("turn_indices must align with legal-set history")
        hypotheses = tuple(self.hypotheses)
        if not hypotheses:
            raise ValueError("hypotheses must not be empty")
        hypothesis_ids = [item.hypothesis_id for item in hypotheses]
        if len(set(hypothesis_ids)) != len(hypothesis_ids):
            raise ValueError("hypothesis IDs must be unique")
        by_id = {item.hypothesis_id: item for item in hypotheses}
        truth = by_id.get(self.true_hypothesis_id)
        if truth is None:
            raise ValueError("true hypothesis is absent from hypotheses")
        if truth.drawback_id != self.true_drawback:
            raise ValueError("true hypothesis drawback does not match true_drawback")
        if truth.eliminated:
            raise ValueError("hard legality eliminated the true hypothesis")
        if any(len(item.permitted_move_history) != len(ordinary) for item in hypotheses):
            raise ValueError("all hypothesis histories must align with ordinary history")
        probabilities = [
            item.posterior_probability
            for item in hypotheses
            if item.posterior_probability is not None
        ]
        if probabilities and len(probabilities) != len(hypotheses):
            raise ValueError("posterior probabilities must be supplied for every hypothesis")
        if probabilities and abs(math.fsum(probabilities) - 1.0) > 1e-6:
            raise ValueError("posterior probabilities must sum to one")
        object.__setattr__(self, "turn_indices", indices)
        object.__setattr__(self, "ordinary_legal_history", ordinary)
        object.__setattr__(self, "hypotheses", hypotheses)


@dataclass(frozen=True)
class PrefixIdentifiability:
    game_id: str
    color: str
    horizon: int
    true_drawback: str
    survivor_hypothesis_count: int
    survivor_label_count: int
    exact_publicly_identifiable: bool
    uniform_hard_legal_top_1_tie_credit: float
    uniform_hard_legal_top_3_tie_credit: float
    uniform_hard_legal_top_5_tie_credit: float
    full_mask_diagnostic_separability: bool
    full_mask_diagnostic_class_count: int
    true_variant_mask_class_size: int
    true_variant_mask_distinct_label_count: int
    variant_mask_partition_entropy: float
    truth_current_opportunity: bool
    cumulative_truth_opportunity_count: int
    cumulative_truth_opportunity_rate: float
    first_truth_opportunity_index: int | None
    truth_restriction_count: int
    truth_addition_count: int
    truth_restriction_fraction: float
    truth_addition_fraction: float
    model_scored: bool
    model_top_max_label_count: int
    model_top_max_tie_credit: float
    model_top_max_excludes_truth: bool
    model_unique_top_correct: bool
    model_unique_top_error: bool
    top_max_labels_disjoint_from_true_variant_mask_labels: bool


@dataclass(frozen=True)
class IdentifiabilitySlice:
    support: int
    mean_survivor_hypothesis_count: float
    mean_survivor_label_count: float
    exact_publicly_identifiable_count: int
    exact_publicly_identifiable_rate: float
    uniform_hard_legal_top_1_tie_credit: float
    uniform_hard_legal_top_3_tie_credit: float
    uniform_hard_legal_top_5_tie_credit: float
    full_mask_diagnostic_separable_count: int
    full_mask_diagnostic_separable_rate: float
    mean_full_mask_diagnostic_class_count: float
    mean_true_variant_mask_class_size: float
    mean_true_variant_mask_distinct_label_count: float
    mean_variant_mask_partition_entropy: float
    current_opportunity_count: int
    current_opportunity_rate: float
    mean_cumulative_truth_opportunity_count: float
    mean_cumulative_truth_opportunity_rate: float
    mean_truth_restriction_count: float
    mean_truth_addition_count: float
    mean_truth_restriction_fraction: float
    mean_truth_addition_fraction: float
    model_scored_count: int
    mean_model_top_max_tie_credit: float | None
    model_top_max_excludes_truth_count: int
    model_top_max_excludes_truth_rate: float | None
    model_unique_top_correct_count: int
    model_unique_top_correct_rate: float | None
    model_unique_top_error_count: int
    model_unique_top_error_rate: float | None
    model_unique_top_error_on_exact_identifiable_count: int
    top_max_labels_disjoint_from_true_variant_mask_labels_count: int


@dataclass(frozen=True)
class IdentifiabilityReport:
    overall: IdentifiabilitySlice
    by_horizon: Mapping[int, IdentifiabilitySlice]
    by_rule: Mapping[str, IdentifiabilitySlice]
    by_color: Mapping[str, IdentifiabilitySlice]
    by_color_horizon_rule: Mapping[
        str, Mapping[int, Mapping[str, IdentifiabilitySlice]]
    ]
    prefixes: tuple[PrefixIdentifiability, ...]
    provenance_verified: bool = False


def score_identifiability(
    observation: IdentifiabilityObservation,
) -> PrefixIdentifiability:
    """Score hard-legal labels and separately describe oracle mask contrast."""

    survivors = tuple(item for item in observation.hypotheses if not item.eliminated)
    survivor_labels = frozenset(item.drawback_id for item in survivors)
    truth = next(
        item
        for item in survivors
        if item.hypothesis_id == observation.true_hypothesis_id
    )
    mask_groups: dict[LegalSetHistory, list[HypothesisLegality]] = {}
    for item in survivors:
        mask_groups.setdefault(item.permitted_move_history, []).append(item)
    truth_mask_variants = tuple(mask_groups[truth.permitted_move_history])
    truth_mask_labels = frozenset(
        item.drawback_id for item in truth_mask_variants
    )
    survivor_count = len(survivors)
    mask_entropy = -math.fsum(
        (len(group) / survivor_count) * math.log(len(group) / survivor_count)
        for group in mask_groups.values()
    )
    restrictions = tuple(
        len(ordinary - permitted)
        for ordinary, permitted in zip(
            observation.ordinary_legal_history,
            truth.permitted_move_history,
            strict=True,
        )
    )
    additions = tuple(
        len(permitted - ordinary)
        for ordinary, permitted in zip(
            observation.ordinary_legal_history,
            truth.permitted_move_history,
            strict=True,
        )
    )
    opportunities = tuple(
        removed > 0 or added > 0
        for removed, added in zip(restrictions, additions, strict=True)
    )
    opportunity_count = sum(opportunities)
    ordinary_move_total = sum(
        len(moves) for moves in observation.ordinary_legal_history
    )
    permitted_move_total = sum(
        len(moves) for moves in truth.permitted_move_history
    )
    probabilities_supplied = all(
        item.posterior_probability is not None for item in observation.hypotheses
    )
    top_labels: frozenset[str] = frozenset()
    top_credit = 0.0
    excludes_truth = False
    unique_correct = False
    unique_error = False
    outside = False
    if probabilities_supplied:
        label_members: dict[str, list[tuple[str, float]]] = {}
        for item in observation.hypotheses:
            label_members.setdefault(item.drawback_id, []).append(
                (item.hypothesis_id, float(item.posterior_probability or 0.0))
            )
        label_probabilities = {
            label: math.fsum(
                probability
                for _identifier, probability in sorted(members)
            )
            for label, members in sorted(label_members.items())
        }
        maximum = max(label_probabilities.values())
        top_labels = frozenset(
            label
            for label, probability in label_probabilities.items()
            if probability == maximum
        )
        top_credit = (
            1.0 / len(top_labels)
            if observation.true_drawback in top_labels
            else 0.0
        )
        excludes_truth = observation.true_drawback not in top_labels
        unique_correct = (
            len(top_labels) == 1 and observation.true_drawback in top_labels
        )
        unique_error = (
            len(top_labels) == 1 and observation.true_drawback not in top_labels
        )
        outside = bool(top_labels) and top_labels.isdisjoint(truth_mask_labels)
    label_count = len(survivor_labels)
    return PrefixIdentifiability(
        game_id=observation.game_id,
        color=observation.color,
        horizon=observation.horizon,
        true_drawback=observation.true_drawback,
        survivor_hypothesis_count=survivor_count,
        survivor_label_count=label_count,
        exact_publicly_identifiable=label_count == 1,
        uniform_hard_legal_top_1_tie_credit=min(1, 1 / label_count),
        uniform_hard_legal_top_3_tie_credit=min(1, 3 / label_count),
        uniform_hard_legal_top_5_tie_credit=min(1, 5 / label_count),
        full_mask_diagnostic_separability=len(mask_groups) > 1,
        full_mask_diagnostic_class_count=len(mask_groups),
        true_variant_mask_class_size=len(truth_mask_variants),
        true_variant_mask_distinct_label_count=len(truth_mask_labels),
        variant_mask_partition_entropy=mask_entropy,
        truth_current_opportunity=opportunities[-1],
        cumulative_truth_opportunity_count=opportunity_count,
        cumulative_truth_opportunity_rate=opportunity_count / len(opportunities),
        first_truth_opportunity_index=next(
            (
                observation.turn_indices[index]
                for index, present in enumerate(opportunities)
                if present
            ),
            None,
        ),
        truth_restriction_count=sum(restrictions),
        truth_addition_count=sum(additions),
        truth_restriction_fraction=(
            sum(restrictions) / ordinary_move_total
            if ordinary_move_total
            else 0.0
        ),
        truth_addition_fraction=(
            sum(additions) / permitted_move_total
            if permitted_move_total
            else 0.0
        ),
        model_scored=probabilities_supplied,
        model_top_max_label_count=len(top_labels),
        model_top_max_tie_credit=top_credit,
        model_top_max_excludes_truth=excludes_truth,
        model_unique_top_correct=unique_correct,
        model_unique_top_error=unique_error,
        top_max_labels_disjoint_from_true_variant_mask_labels=outside,
    )


class IdentifiabilityAccumulator:
    """Aggregate rows only after validating their longitudinal replay contract."""

    def __init__(self) -> None:
        self._observations: list[IdentifiabilityObservation] = []
        self._keys: set[tuple[str, str, int]] = set()

    def add(self, observation: IdentifiabilityObservation) -> None:
        key = (observation.game_id, observation.color, observation.horizon)
        if key in self._keys:
            raise ValueError("duplicate game/color/horizon observation")
        self._keys.add(key)
        self._observations.append(observation)

    def report(self) -> IdentifiabilityReport:
        if not self._observations:
            raise ValueError("at least one identifiability observation is required")
        _validate_longitudinal(self._observations)
        prefixes = tuple(
            sorted(
                (score_identifiability(item) for item in self._observations),
                key=lambda item: (
                    item.game_id, item.color, item.horizon, item.true_drawback
                ),
            )
        )
        by_horizon = _group(prefixes, lambda item: item.horizon)
        by_rule = _group(prefixes, lambda item: item.true_drawback)
        by_color = _group(prefixes, lambda item: item.color)
        nested: dict[str, dict[int, dict[str, IdentifiabilitySlice]]] = {}
        for color in sorted({item.color for item in prefixes}):
            color_rows = tuple(item for item in prefixes if item.color == color)
            nested[color] = {}
            for horizon in sorted({item.horizon for item in color_rows}):
                rows = tuple(item for item in color_rows if item.horizon == horizon)
                nested[color][horizon] = {
                    rule: _slice(
                        item for item in rows if item.true_drawback == rule
                    )
                    for rule in sorted({item.true_drawback for item in rows})
                }
        return IdentifiabilityReport(
            overall=_slice(prefixes),
            by_horizon=MappingProxyType(by_horizon),
            by_rule=MappingProxyType(by_rule),
            by_color=MappingProxyType(by_color),
            by_color_horizon_rule=MappingProxyType({
                color: MappingProxyType({
                    horizon: MappingProxyType(rules)
                    for horizon, rules in horizons.items()
                })
                for color, horizons in nested.items()
            }),
            prefixes=prefixes,
        )


def evaluate_identifiability(
    observations: Iterable[IdentifiabilityObservation],
) -> IdentifiabilityReport:
    accumulator = IdentifiabilityAccumulator()
    for observation in observations:
        accumulator.add(observation)
    return accumulator.report()


def _validate_longitudinal(
    observations: Iterable[IdentifiabilityObservation],
) -> None:
    players: dict[tuple[str, str], list[IdentifiabilityObservation]] = {}
    for item in observations:
        players.setdefault((item.game_id, item.color), []).append(item)
    for rows in players.values():
        ordered = sorted(rows, key=lambda item: item.horizon)
        truth_pairs = {
            (item.true_drawback, item.true_hypothesis_id) for item in ordered
        }
        if len(truth_pairs) != 1:
            raise ValueError("player-game truth label or variant changed")
        first_hypotheses = {
            item.hypothesis_id: item.drawback_id
            for item in ordered[0].hypotheses
        }
        for prior, current in zip(ordered, ordered[1:]):
            if prior.horizon >= current.horizon:
                raise ValueError("player-game horizons must strictly increase")
            if not _is_prefix(prior.turn_indices, current.turn_indices):
                raise ValueError("player-game turn indices are not exact prefixes")
            if not _is_prefix(
                prior.ordinary_legal_history, current.ordinary_legal_history
            ):
                raise ValueError("ordinary legal histories are not exact prefixes")
            current_hypotheses = {
                item.hypothesis_id: item for item in current.hypotheses
            }
            if {
                identifier: item.drawback_id
                for identifier, item in current_hypotheses.items()
            } != first_hypotheses:
                raise ValueError("player-game hypothesis identities changed")
            for prior_hypothesis in prior.hypotheses:
                current_hypothesis = current_hypotheses[prior_hypothesis.hypothesis_id]
                if not _is_prefix(
                    prior_hypothesis.permitted_move_history,
                    current_hypothesis.permitted_move_history,
                ):
                    raise ValueError(
                        "per-hypothesis legal histories are not exact prefixes"
                    )
                if prior_hypothesis.eliminated and not current_hypothesis.eliminated:
                    raise ValueError("a hard-eliminated hypothesis was restored")


def _is_prefix(prior: tuple[object, ...], current: tuple[object, ...]) -> bool:
    return len(prior) < len(current) and current[: len(prior)] == prior


def _validate_move_history(history: LegalSetHistory, label: str) -> None:
    if not history:
        raise ValueError(f"{label} legal-set history must not be empty")
    if any(
        not move or not isinstance(move, str)
        for moves in history
        for move in moves
    ):
        raise ValueError(f"{label} legal moves must be non-empty strings")


def _group(
    rows: tuple[PrefixIdentifiability, ...],
    key: Callable[[PrefixIdentifiability], GroupKey],
) -> dict[GroupKey, IdentifiabilitySlice]:
    values: dict[GroupKey, list[PrefixIdentifiability]] = {}
    for row in rows:
        values.setdefault(key(row), []).append(row)
    return {
        group: _slice(group_rows)
        for group, group_rows in sorted(values.items(), key=lambda item: item[0])
    }


def _slice(rows: Iterable[PrefixIdentifiability]) -> IdentifiabilitySlice:
    values = tuple(rows)
    if not values:
        raise ValueError("identifiability slice must not be empty")
    support = len(values)
    scored = tuple(item for item in values if item.model_scored)
    scored_count = len(scored)
    return IdentifiabilitySlice(
        support=support,
        mean_survivor_hypothesis_count=math.fsum(
            item.survivor_hypothesis_count for item in values
        ) / support,
        mean_survivor_label_count=math.fsum(
            item.survivor_label_count for item in values
        ) / support,
        exact_publicly_identifiable_count=sum(
            item.exact_publicly_identifiable for item in values
        ),
        exact_publicly_identifiable_rate=sum(
            item.exact_publicly_identifiable for item in values
        ) / support,
        uniform_hard_legal_top_1_tie_credit=math.fsum(
            item.uniform_hard_legal_top_1_tie_credit for item in values
        ) / support,
        uniform_hard_legal_top_3_tie_credit=math.fsum(
            item.uniform_hard_legal_top_3_tie_credit for item in values
        ) / support,
        uniform_hard_legal_top_5_tie_credit=math.fsum(
            item.uniform_hard_legal_top_5_tie_credit for item in values
        ) / support,
        full_mask_diagnostic_separable_count=sum(
            item.full_mask_diagnostic_separability for item in values
        ),
        full_mask_diagnostic_separable_rate=sum(
            item.full_mask_diagnostic_separability for item in values
        ) / support,
        mean_full_mask_diagnostic_class_count=math.fsum(
            item.full_mask_diagnostic_class_count for item in values
        ) / support,
        mean_true_variant_mask_class_size=math.fsum(
            item.true_variant_mask_class_size for item in values
        ) / support,
        mean_true_variant_mask_distinct_label_count=math.fsum(
            item.true_variant_mask_distinct_label_count for item in values
        ) / support,
        mean_variant_mask_partition_entropy=math.fsum(
            item.variant_mask_partition_entropy for item in values
        ) / support,
        current_opportunity_count=sum(
            item.truth_current_opportunity for item in values
        ),
        current_opportunity_rate=sum(
            item.truth_current_opportunity for item in values
        ) / support,
        mean_cumulative_truth_opportunity_count=math.fsum(
            item.cumulative_truth_opportunity_count for item in values
        ) / support,
        mean_cumulative_truth_opportunity_rate=math.fsum(
            item.cumulative_truth_opportunity_rate for item in values
        ) / support,
        mean_truth_restriction_count=math.fsum(
            item.truth_restriction_count for item in values
        ) / support,
        mean_truth_addition_count=math.fsum(
            item.truth_addition_count for item in values
        ) / support,
        mean_truth_restriction_fraction=math.fsum(
            item.truth_restriction_fraction for item in values
        ) / support,
        mean_truth_addition_fraction=math.fsum(
            item.truth_addition_fraction for item in values
        ) / support,
        model_scored_count=scored_count,
        mean_model_top_max_tie_credit=(
            math.fsum(item.model_top_max_tie_credit for item in scored)
            / scored_count
            if scored_count
            else None
        ),
        model_top_max_excludes_truth_count=sum(
            item.model_top_max_excludes_truth for item in scored
        ),
        model_top_max_excludes_truth_rate=(
            sum(item.model_top_max_excludes_truth for item in scored)
            / scored_count
            if scored_count
            else None
        ),
        model_unique_top_correct_count=sum(
            item.model_unique_top_correct for item in scored
        ),
        model_unique_top_correct_rate=(
            sum(item.model_unique_top_correct for item in scored) / scored_count
            if scored_count
            else None
        ),
        model_unique_top_error_count=sum(
            item.model_unique_top_error for item in scored
        ),
        model_unique_top_error_rate=(
            sum(item.model_unique_top_error for item in scored) / scored_count
            if scored_count
            else None
        ),
        model_unique_top_error_on_exact_identifiable_count=sum(
            item.model_unique_top_error and item.exact_publicly_identifiable
            for item in scored
        ),
        top_max_labels_disjoint_from_true_variant_mask_labels_count=sum(
            item.top_max_labels_disjoint_from_true_variant_mask_labels
            for item in scored
        ),
    )
