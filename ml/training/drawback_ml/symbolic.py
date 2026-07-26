"""Versioned public symbolic-predictor features for hybrid models."""

from __future__ import annotations

from typing import Any, Sequence

from .rank_preserving_fusion import (
    RANK_PRESERVING_FUSION_METHOD,
    prepare_rank_preserving_symbolic,
    validate_fusion_alpha,
)
from .records import FeatureRecord
from .symbolic_schema import (
    SYMBOLIC_FEATURE_DIMENSION,
    SYMBOLIC_FEATURE_VERSION,
    SYMBOLIC_RULE_COUNT,
    SYMBOLIC_RULE_IDS,
)

FUSION_AWARE_LOSS_METHOD = "rank-preserving-fusion-grid-nll-v1"
FUSION_AWARE_LOSS_VERSION = 1
FUSION_AWARE_LOSS_ALPHA_GRID = (0.125, 0.25, 0.5, 1.0)


def build_symbolic_feature_vector(record: FeatureRecord) -> tuple[float, ...]:
    """Return public posterior/elimination features or fail closed.

    Hybrid training must never silently substitute an unrestricted or empty
    symbolic signal. Older corpora remain valid for v1/v2, but cannot train the
    hybrid model.
    """

    if record.symbolic_feature_version != SYMBOLIC_FEATURE_VERSION:
        raise ValueError(
            f"hybrid model requires symbolic feature version "
            f"{SYMBOLIC_FEATURE_VERSION}"
        )
    vector = (
        *record.symbolic_white_rule_probabilities,
        *record.symbolic_black_rule_probabilities,
        *(1.0 if value else 0.0 for value in record.symbolic_white_eliminated),
        *(1.0 if value else 0.0 for value in record.symbolic_black_eliminated),
    )
    if len(vector) != SYMBOLIC_FEATURE_DIMENSION:
        raise RuntimeError("symbolic feature dimension invariant violated")
    return tuple(vector)


def combine_with_symbolic_prior(
    torch: Any,
    neural_logits: Any,
    records: Sequence[FeatureRecord],
    vocabulary: Sequence[str],
    color: str,
    *,
    true_drawbacks: Sequence[str] | None = None,
) -> Any:
    """Apply exact elimination and a symbolic log-prior to neural logits.

    The neural model is a residual ranker. It cannot restore a hypothesis that
    the executable rule engine eliminated.
    """

    combined, eliminated_tensor = fused_logits_with_symbolic_prior(
        torch,
        neural_logits,
        records,
        vocabulary,
        color,
        true_drawbacks=true_drawbacks,
    )
    return combined.masked_fill(eliminated_tensor, float("-inf"))


def fused_logits_with_symbolic_prior(
    torch: Any,
    neural_logits: Any,
    records: Sequence[FeatureRecord],
    vocabulary: Sequence[str],
    color: str,
    *,
    true_drawbacks: Sequence[str] | None = None,
) -> tuple[Any, Any]:
    """Return genuine pre-mask fused logits and the exact hard mask."""

    if color not in {"white", "black"}:
        raise ValueError("color must be white or black")
    if set(vocabulary) != set(SYMBOLIC_RULE_IDS):
        raise ValueError(
            "hybrid drawback vocabulary must exactly match symbolic rule ids"
        )
    indices = [SYMBOLIC_RULE_IDS.index(drawback_id) for drawback_id in vocabulary]
    if true_drawbacks is not None and len(true_drawbacks) != len(records):
        raise ValueError("true drawbacks must align with symbolic records")
    priors: list[list[float]] = []
    eliminated: list[list[bool]] = []
    for row_index, record in enumerate(records):
        build_symbolic_feature_vector(record)
        probabilities = (
            record.symbolic_white_rule_probabilities
            if color == "white"
            else record.symbolic_black_rule_probabilities
        )
        hard_eliminated = (
            record.symbolic_white_eliminated
            if color == "white"
            else record.symbolic_black_eliminated
        )
        ordered_probabilities = [probabilities[index] for index in indices]
        ordered_eliminated = [hard_eliminated[index] for index in indices]
        if all(ordered_eliminated):
            raise ValueError("symbolic engine eliminated every drawback")
        if true_drawbacks is not None:
            true_drawback = true_drawbacks[row_index]
            if true_drawback not in vocabulary:
                raise ValueError(
                    f"true drawback is absent from vocabulary: {true_drawback}"
                )
            true_index = vocabulary.index(true_drawback)
            if ordered_eliminated[true_index]:
                raise ValueError(
                    f"symbolic engine eliminated true {color} drawback "
                    f"{true_drawback}"
                )
        priors.append(ordered_probabilities)
        eliminated.append(ordered_eliminated)
    prior_tensor = torch.tensor(
        priors, dtype=neural_logits.dtype, device=neural_logits.device
    )
    eliminated_tensor = torch.tensor(
        eliminated, dtype=torch.bool, device=neural_logits.device
    )
    combined = neural_logits + torch.log(prior_tensor.clamp(min=1e-12))
    return combined, eliminated_tensor


def fusion_aware_loss_metadata(model_variant: str) -> dict[str, object] | None:
    """Describe the frozen hybrid objective without relabeling old runs."""

    if model_variant not in {"v21-hybrid", "v22-hybrid"}:
        return None
    return {
        "method": FUSION_AWARE_LOSS_METHOD,
        "version": FUSION_AWARE_LOSS_VERSION,
        "production_fusion_method": RANK_PRESERVING_FUSION_METHOD,
        "alpha_grid": list(FUSION_AWARE_LOSS_ALPHA_GRID),
        "aggregation": "mean-cross-entropy-across-alpha-grid-v1",
    }


def torch_rank_preserving_fusion(
    torch: Any,
    neural_logits: Any,
    records: Sequence[FeatureRecord],
    vocabulary: Sequence[str],
    color: str,
    *,
    alpha: float,
) -> tuple[Any, Any]:
    """Differentiably reproduce production rank-preserving fusion.

    Symbolic bases, tier headroom, and hard masks are public constants for a
    training row. Only the centered survivor softmax depends on neural output,
    preserving gradients inside symbolic tiers without permitting a cross-tier
    inversion.
    """

    priors, eliminated = _ordered_symbolic_rows(records, vocabulary, color)
    base, perturbation, hard_mask = _torch_rank_preserving_components(
        torch,
        neural_logits,
        priors,
        eliminated,
    )
    alpha_value = validate_fusion_alpha(alpha)
    fused = base + alpha_value * perturbation
    return fused.masked_fill(hard_mask, 0.0), hard_mask


def fusion_aware_drawback_loss(
    torch: Any,
    neural_logits: Any,
    records: Sequence[FeatureRecord],
    vocabulary: Sequence[str],
    color: str,
    targets: Any,
    supervision_mask: Any,
) -> Any:
    """Average supervised NLL over the frozen nonzero production alpha grid."""

    priors, eliminated = _ordered_symbolic_rows(records, vocabulary, color)
    _validate_fusion_training_targets(
        targets,
        supervision_mask,
        priors,
        eliminated,
        vocabulary,
        color,
    )
    if not bool(supervision_mask.any()):
        return neural_logits.sum() * 0.0

    base, perturbation, hard_mask = _torch_rank_preserving_components(
        torch,
        neural_logits,
        priors,
        eliminated,
    )
    supervised_hard_mask = hard_mask[supervision_mask]
    losses = []
    for alpha in FUSION_AWARE_LOSS_ALPHA_GRID:
        supervised_logits = (
            base + alpha * perturbation
        )[supervision_mask].masked_fill(
            supervised_hard_mask,
            float("-inf"),
        )
        losses.append(
            torch.nn.functional.cross_entropy(
                supervised_logits,
                targets[supervision_mask],
            )
        )
    return torch.stack(losses).mean()


def _ordered_symbolic_rows(
    records: Sequence[FeatureRecord],
    vocabulary: Sequence[str],
    color: str,
) -> tuple[tuple[tuple[float, ...], ...], tuple[tuple[bool, ...], ...]]:
    if color not in {"white", "black"}:
        raise ValueError("color must be white or black")
    if set(vocabulary) != set(SYMBOLIC_RULE_IDS):
        raise ValueError(
            "hybrid drawback vocabulary must exactly match symbolic rule ids"
        )
    if len(vocabulary) != len(SYMBOLIC_RULE_IDS):
        raise ValueError("hybrid drawback vocabulary must not contain duplicates")
    indices = tuple(
        SYMBOLIC_RULE_IDS.index(drawback_id) for drawback_id in vocabulary
    )
    priors: list[tuple[float, ...]] = []
    eliminated: list[tuple[bool, ...]] = []
    for record in records:
        build_symbolic_feature_vector(record)
        probabilities = (
            record.symbolic_white_rule_probabilities
            if color == "white"
            else record.symbolic_black_rule_probabilities
        )
        hard_eliminated = (
            record.symbolic_white_eliminated
            if color == "white"
            else record.symbolic_black_eliminated
        )
        priors.append(tuple(probabilities[index] for index in indices))
        eliminated.append(tuple(hard_eliminated[index] for index in indices))
    return tuple(priors), tuple(eliminated)


def _torch_rank_preserving_components(
    torch: Any,
    neural_logits: Any,
    priors: tuple[tuple[float, ...], ...],
    eliminated: tuple[tuple[bool, ...], ...],
) -> tuple[Any, Any, Any]:
    if getattr(neural_logits, "ndim", None) != 2:
        raise ValueError("neural logits must be a rank-two batch")
    expected_shape = (
        len(priors),
        0 if not priors else len(priors[0]),
    )
    if tuple(neural_logits.shape) != expected_shape:
        raise ValueError(
            "neural logits must align with symbolic records and vocabulary"
        )
    if not bool(torch.isfinite(neural_logits).all()):
        raise ValueError("neural logits must be finite")

    bases: list[tuple[float, ...]] = []
    scales: list[tuple[float, ...]] = []
    for prior_row, eliminated_row in zip(priors, eliminated, strict=True):
        symbolic = prepare_rank_preserving_symbolic(
            prior_row,
            eliminated_row,
        )
        bases.append(symbolic.base_logits)
        scales.append(symbolic.neural_scales)

    hard_mask = torch.tensor(
        eliminated,
        dtype=torch.bool,
        device=neural_logits.device,
    )
    # Keep tier bases and their bounded perturbations in binary64. A binary32
    # cast can collapse two distinct public priors into the same log tier and
    # would violate the strict production ordering contract.
    working_logits = neural_logits.to(dtype=torch.float64)
    survivor_count = (
        (~hard_mask)
        .sum(dim=-1, keepdim=True)
        .to(dtype=working_logits.dtype)
    )
    survivor_probabilities = torch.softmax(
        working_logits.masked_fill(hard_mask, float("-inf")),
        dim=-1,
    )
    bounded_signal = survivor_probabilities - survivor_count.reciprocal()
    bounded_signal = bounded_signal.masked_fill(hard_mask, 0.0)
    base_tensor = torch.tensor(
        bases,
        dtype=torch.float64,
        device=neural_logits.device,
    )
    scale_tensor = torch.tensor(
        scales,
        dtype=torch.float64,
        device=neural_logits.device,
    )
    perturbation = scale_tensor * bounded_signal
    return base_tensor, perturbation, hard_mask


def _validate_fusion_training_targets(
    targets: Any,
    supervision_mask: Any,
    priors: tuple[tuple[float, ...], ...],
    eliminated: tuple[tuple[bool, ...], ...],
    vocabulary: Sequence[str],
    color: str,
) -> None:
    if getattr(targets, "ndim", None) != 1 or tuple(targets.shape) != (
        len(priors),
    ):
        raise ValueError("fusion targets must align with symbolic records")
    if getattr(supervision_mask, "ndim", None) != 1 or tuple(
        supervision_mask.shape
    ) != (len(priors),):
        raise ValueError("fusion supervision mask must align with symbolic records")
    if supervision_mask.dtype != supervision_mask.new_empty(()).bool().dtype:
        raise ValueError("fusion supervision mask must be boolean")

    target_values = targets.detach().cpu().tolist()
    for row_index, target in enumerate(target_values):
        if isinstance(target, bool) or not isinstance(target, int):
            raise ValueError("fusion targets must contain class indices")
        if not 0 <= target < len(vocabulary):
            raise ValueError("fusion target class index is out of range")
        drawback_id = vocabulary[target]
        if eliminated[row_index][target]:
            raise ValueError(
                f"symbolic engine eliminated true {color} drawback "
                f"{drawback_id}"
            )
        if priors[row_index][target] <= 0.0:
            raise ValueError(
                f"symbolic prior assigns zero mass to true {color} drawback "
                f"{drawback_id}"
            )
