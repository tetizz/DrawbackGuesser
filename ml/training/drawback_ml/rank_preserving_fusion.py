"""Rank-preserving fusion of symbolic evidence and neural residuals.

The symbolic rule engine is authoritative. Neural evidence may distinguish
symbolic ties, but it must never reverse a strict symbolic ordering or restore
an eliminated hypothesis.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence


RANK_PRESERVING_FUSION_METHOD = (
    "rank-preserving-bounded-residual-plus-symbolic-prior-v1"
)
MAXIMUM_NEURAL_SCALE = 1.0
ZERO_PRIOR_SEPARATION = 1_000_000.0


class RankPreservingFusionError(ValueError):
    """Raised when fusion inputs cannot satisfy the symbolic authority contract."""


@dataclass(frozen=True)
class RankPreservingFusionResult:
    """Serializable pre-mask logits plus the exact posterior and diagnostics."""

    logits: tuple[float, ...]
    probabilities: tuple[float, ...]
    bounded_neural_signal: tuple[float, ...]
    neural_scales: tuple[float, ...]
    alpha: float


@dataclass(frozen=True)
class RankPreservingSymbolicPreparation:
    """Public symbolic tier bases and local neural headroom for one row."""

    base_logits: tuple[float, ...]
    neural_scales: tuple[float, ...]
    prior: tuple[float, ...]
    eliminated: tuple[bool, ...]


@dataclass(frozen=True)
class _RankPreservingFusionPreparation:
    """Alpha-independent symbolic and neural preparation for one row."""

    symbolic: RankPreservingSymbolicPreparation
    bounded_neural_signal: tuple[float, ...]


def prepare_rank_preserving_symbolic(
    prior: Sequence[float],
    eliminated: Sequence[bool],
) -> RankPreservingSymbolicPreparation:
    """Validate and prepare the symbolic authority portion of fusion.

    This pure preparation is independent of neural residuals and alpha. Scalar
    production fusion and differentiable Torch training share it so their tier
    bases and local headroom cannot drift.
    """

    prior_values = _finite_values(prior, "symbolic prior")
    mask = tuple(eliminated)
    if not prior_values or len(mask) != len(prior_values):
        raise RankPreservingFusionError(
            "fusion prior and hard-mask dimensions must match"
        )
    if any(type(value) is not bool for value in mask):
        raise RankPreservingFusionError("fusion hard mask must contain booleans")
    if any(value < 0.0 for value in prior_values):
        raise RankPreservingFusionError(
            "symbolic prior must contain finite non-negative values"
        )
    survivors = tuple(index for index, value in enumerate(mask) if not value)
    if not survivors:
        raise RankPreservingFusionError(
            "symbolic engine eliminated every drawback"
        )
    positive_survivors = tuple(
        index for index in survivors if prior_values[index] > 0.0
    )
    if not positive_survivors:
        raise RankPreservingFusionError(
            "surviving symbolic hypotheses must contain positive mass"
        )

    base_by_probability = _monotonic_log_tiers(
        tuple(prior_values[index] for index in positive_survivors)
    )
    zero_base = min(base_by_probability.values()) - ZERO_PRIOR_SEPARATION
    if any(prior_values[index] == 0.0 for index in survivors):
        base_by_probability[0.0] = zero_base
    scale_by_probability = _local_neural_scales(base_by_probability)
    base_logits = tuple(
        0.0 if mask[index] else base_by_probability[prior_values[index]]
        for index in range(len(prior_values))
    )
    neural_scales = tuple(
        0.0 if mask[index] else scale_by_probability[prior_values[index]]
        for index in range(len(prior_values))
    )
    _assert_rank_preserved(base_logits, prior_values, mask)
    return RankPreservingSymbolicPreparation(
        base_logits=base_logits,
        neural_scales=neural_scales,
        prior=prior_values,
        eliminated=mask,
    )


def prepare_rank_preserving_fusion(
    residuals: Sequence[float],
    prior: Sequence[float],
    eliminated: Sequence[bool],
) -> _RankPreservingFusionPreparation:
    """Validate one row and prepare everything shared by all alpha values."""

    residual_values = _finite_values(residuals, "neural residual")
    symbolic = prepare_rank_preserving_symbolic(prior, eliminated)
    if (
        not residual_values
        or len(symbolic.prior) != len(residual_values)
    ):
        raise RankPreservingFusionError(
            "fusion residual, prior, and hard-mask dimensions must match"
        )
    survivors = tuple(
        index
        for index, value in enumerate(symbolic.eliminated)
        if not value
    )
    return _RankPreservingFusionPreparation(
        symbolic=symbolic,
        bounded_neural_signal=_bounded_signal(
            residual_values,
            survivors,
        ),
    )


def apply_rank_preserving_fusion(
    preparation: _RankPreservingFusionPreparation,
    *,
    alpha: float = 1.0,
) -> RankPreservingFusionResult:
    """Apply one alpha to a validated, alpha-independent preparation."""

    if not isinstance(preparation, _RankPreservingFusionPreparation):
        raise RankPreservingFusionError(
            "fusion preparation has an invalid type"
        )
    symbolic = preparation.symbolic
    alpha_value = _alpha(alpha)
    logits = tuple(
        0.0
        if symbolic.eliminated[index]
        else (
            symbolic.base_logits[index]
            + alpha_value
            * symbolic.neural_scales[index]
            * preparation.bounded_neural_signal[index]
        )
        for index in range(len(symbolic.prior))
    )
    _assert_rank_preserved(
        logits,
        symbolic.prior,
        symbolic.eliminated,
    )
    probabilities = _masked_softmax(logits, symbolic.eliminated)
    return RankPreservingFusionResult(
        logits=logits,
        probabilities=probabilities,
        bounded_neural_signal=preparation.bounded_neural_signal,
        neural_scales=symbolic.neural_scales,
        alpha=alpha_value,
    )


def rank_preserving_fusion(
    residuals: Sequence[float],
    prior: Sequence[float],
    eliminated: Sequence[bool],
    *,
    alpha: float = 1.0,
) -> RankPreservingFusionResult:
    """Fuse one neural residual row without weakening symbolic authority.

    Positive symbolic probabilities define ordered log-probability tiers.
    A centered softmax transforms arbitrary residual logits into a bounded,
    shift-invariant signal in ``[-1, 1]``. Equal residuals therefore produce
    exactly zero neural signal. Each tier receives local neural
    headroom of at most one quarter of either adjacent symbolic log gap, so
    even the largest opposing neural swing leaves at least half of every
    strict gap intact. An unrelated near-tie therefore cannot suppress useful
    neural evidence in a distant tier.

    Non-eliminated zero-prior hypotheses form a dedicated bottom tier. Their
    finite score is separated far enough to retain exact zero posterior mass,
    so neural evidence cannot resurrect a Bayesian zero.
    Eliminated hypotheses have canonical finite pre-mask logits for strict JSON
    serialization and exactly zero posterior probability.
    """

    return apply_rank_preserving_fusion(
        prepare_rank_preserving_fusion(
            residuals,
            prior,
            eliminated,
        ),
        alpha=alpha,
    )


def validate_fusion_alpha(value: float) -> float:
    """Return a canonical fusion alpha or fail closed."""

    return _alpha(value)


def _finite_values(values: Sequence[float], name: str) -> tuple[float, ...]:
    rendered: list[float] = []
    for value in values:
        if isinstance(value, bool):
            raise RankPreservingFusionError(f"{name} values must be numeric")
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise RankPreservingFusionError(
                f"{name} values must be numeric"
            ) from error
        if not math.isfinite(number):
            raise RankPreservingFusionError(f"{name} values must be finite")
        rendered.append(number)
    return tuple(rendered)


def _alpha(value: float) -> float:
    if isinstance(value, bool):
        raise RankPreservingFusionError("fusion alpha must be a finite number")
    try:
        rendered = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise RankPreservingFusionError(
            "fusion alpha must be a finite number"
        ) from error
    if not math.isfinite(rendered) or not 0.0 <= rendered <= 1.0:
        raise RankPreservingFusionError(
            "fusion alpha must be between zero and one"
        )
    return rendered


def _bounded_signal(
    residuals: tuple[float, ...],
    survivors: tuple[int, ...],
) -> tuple[float, ...]:
    maximum = max(residuals[index] for index in survivors)
    weights = tuple(
        math.exp(residuals[index] - maximum) for index in survivors
    )
    total = math.fsum(weights)
    if not math.isfinite(total) or total <= 0.0:
        raise RankPreservingFusionError(
            "neural residuals produced an invalid survivor distribution"
        )
    uniform = 1.0 / len(survivors)
    survivor_signal = {
        index: weight / total - uniform
        for index, weight in zip(survivors, weights, strict=True)
    }
    return tuple(
        survivor_signal.get(index, 0.0) for index in range(len(residuals))
    )


def _monotonic_log_tiers(
    positive_probabilities: tuple[float, ...],
) -> dict[float, float]:
    tiers: dict[float, float] = {}
    previous: float | None = None
    for probability in sorted(set(positive_probabilities)):
        score = math.log(probability)
        if previous is not None and score <= previous:
            score = math.nextafter(previous, math.inf)
        if not math.isfinite(score) or (
            previous is not None and score <= previous
        ):
            raise RankPreservingFusionError(
                "symbolic prior tiers do not have a representable strict gap"
            )
        tiers[probability] = score
        previous = score
    return tiers


def _local_neural_scales(
    base_by_probability: dict[float, float],
) -> dict[float, float]:
    ordered = sorted(
        base_by_probability.items(),
        key=lambda item: item[1],
    )
    scales: dict[float, float] = {}
    for index, (probability, base) in enumerate(ordered):
        headroom = [MAXIMUM_NEURAL_SCALE]
        if index > 0:
            headroom.append((base - ordered[index - 1][1]) / 4.0)
        if index + 1 < len(ordered):
            headroom.append((ordered[index + 1][1] - base) / 4.0)
        scale = min(headroom)
        if not math.isfinite(scale) or scale < 0.0:
            raise RankPreservingFusionError(
                "symbolic prior tiers have invalid neural headroom"
            )
        scales[probability] = scale
    return scales


def _assert_rank_preserved(
    logits: tuple[float, ...],
    prior: tuple[float, ...],
    eliminated: tuple[bool, ...],
) -> None:
    tiers: dict[float, list[float]] = {}
    for index, is_eliminated in enumerate(eliminated):
        if not is_eliminated:
            tiers.setdefault(prior[index], []).append(logits[index])
    previous_maximum: float | None = None
    for probability in sorted(tiers):
        scores = tiers[probability]
        current_minimum = min(scores)
        if (
            previous_maximum is not None
            and current_minimum <= previous_maximum
        ):
            raise RankPreservingFusionError(
                "fusion failed to preserve symbolic survivor ordering"
            )
        previous_maximum = max(scores)


def _masked_softmax(
    logits: tuple[float, ...],
    eliminated: tuple[bool, ...],
) -> tuple[float, ...]:
    maximum = max(
        value
        for value, is_eliminated in zip(logits, eliminated, strict=True)
        if not is_eliminated
    )
    masses = tuple(
        0.0 if is_eliminated else math.exp(value - maximum)
        for value, is_eliminated in zip(logits, eliminated, strict=True)
    )
    total = math.fsum(masses)
    if not math.isfinite(total) or total <= 0.0:
        raise RankPreservingFusionError(
            "fusion produced an invalid probability distribution"
        )
    return tuple(mass / total for mass in masses)
