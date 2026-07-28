"""Fresh-start hybrid baseline for capturable-king drawback inference."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, fields, is_dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import secrets
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from ml.evaluation.metrics import (
    PredictionExample,
    StreamingBinaryEvaluation,
    StreamingEvaluation,
)

from .capturable_records import (
    CAPTURABLE_FEATURE_DIMENSION,
    CAPTURABLE_OPPORTUNITY_FEATURE_VERSION,
    CAPTURABLE_OPPORTUNITY_FIELDS,
    CAPTURABLE_OPPORTUNITY_SHAPE,
    CAPTURABLE_OPPORTUNITY_SYMBOLIC_FEATURE_VERSION,
    CAPTURABLE_RULE_IDS,
    CAPTURABLE_RULE_INDEX,
    CapturableDatasetRow,
    active_symbolic,
    assert_disjoint_games,
    capturable_feature_vector,
    load_capturable_dataset,
    load_capturable_opportunity_dataset,
)
CAPTURABLE_BASELINE_FORMAT = "drawbackguesser-capturable-baseline"
CAPTURABLE_BASELINE_VERSION = 2
CAPTURABLE_OPPORTUNITY_BASELINE_VERSION = 3
CAPTURABLE_OPPORTUNITY_MODES = ("public-exact", "zero-ablation")
SOURCE_WEIGHTING_OBJECTIVE = "global-source-mean-player-game/v1"
_PRIOR_FLOOR = 1e-12


@dataclass(frozen=True)
class CapturableTrainingConfig:
    seed: int = 0xC0DE_0701
    epochs: int = 8
    batch_size: int = 256
    hidden_dimension: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    trigger_loss_weight: float = 0.15
    forced_loss_weight: float = 0.05
    parameter_loss_weight: float = 0.10
    fusion_alpha_grid: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0, 2.0)
    prior_smoothing_grid: tuple[float, ...] = (0.0, 0.01, 0.05, 0.10, 0.20)
    training_prior_smoothing: float = 0.10
    trigger_row_multiplier: float = 1.0
    torch_threads: int = 14

    def __post_init__(self) -> None:
        if not 0 <= self.seed <= 0xFFFF_FFFF:
            raise ValueError("seed must be uint32")
        for name in ("epochs", "batch_size", "hidden_dimension", "torch_threads"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "learning_rate",
            "trigger_loss_weight",
            "forced_loss_weight",
            "parameter_loss_weight",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise ValueError(f"{name} must be finite and positive")
        if (
            isinstance(self.weight_decay, bool)
            or not isinstance(self.weight_decay, (int, float))
            or not math.isfinite(float(self.weight_decay))
            or float(self.weight_decay) < 0.0
        ):
            raise ValueError("weight_decay must be finite and non-negative")
        if (
            not self.fusion_alpha_grid
            or len(set(self.fusion_alpha_grid)) != len(self.fusion_alpha_grid)
            or tuple(sorted(self.fusion_alpha_grid)) != self.fusion_alpha_grid
            or self.fusion_alpha_grid[0] != 0.0
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0.0
                for value in self.fusion_alpha_grid
            )
            or not any(value > 0.0 for value in self.fusion_alpha_grid)
        ):
            raise ValueError(
                "fusion_alpha_grid must be unique, sorted, begin at zero, "
                "and contain a positive finite value"
            )
        if (
            not self.prior_smoothing_grid
            or len(set(self.prior_smoothing_grid))
            != len(self.prior_smoothing_grid)
            or tuple(sorted(self.prior_smoothing_grid))
            != self.prior_smoothing_grid
            or self.prior_smoothing_grid[0] != 0.0
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) < 1.0
                for value in self.prior_smoothing_grid
            )
        ):
            raise ValueError(
                "prior_smoothing_grid must be unique, sorted, begin at zero, "
                "and contain finite values below one"
            )
        if (
            isinstance(self.training_prior_smoothing, bool)
            or not isinstance(self.training_prior_smoothing, (int, float))
            or not math.isfinite(float(self.training_prior_smoothing))
            or not 0.0 <= float(self.training_prior_smoothing) < 1.0
        ):
            raise ValueError(
                "training_prior_smoothing must be finite from zero to one"
            )
        if (
            isinstance(self.trigger_row_multiplier, bool)
            or not isinstance(self.trigger_row_multiplier, (int, float))
            or not math.isfinite(float(self.trigger_row_multiplier))
            or not 1.0 <= float(self.trigger_row_multiplier) <= 100.0
        ):
            raise ValueError(
                "trigger_row_multiplier must be finite from one to 100"
            )


@dataclass(frozen=True)
class TensorRows:
    inputs: Any
    rule_opportunities: Any | None
    colors: Any
    drawbacks: Any
    triggered: Any
    forced: Any
    triple_play_parameters: Any
    player_game_weights: Any
    player_game_normalization_weights: Any
    symbolic_priors: Any
    symbolic_eliminated: Any


def create_capturable_model(hidden_dimension: int) -> Any:
    try:
        import torch.nn as nn
    except ImportError as error:
        raise RuntimeError(
            "PyTorch is required; install ml/requirements.txt"
        ) from error

    class CapturableBaseline(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(CAPTURABLE_FEATURE_DIMENSION, hidden_dimension),
                nn.ReLU(),
                nn.Linear(hidden_dimension, hidden_dimension),
                nn.ReLU(),
            )
            self.white_drawback = nn.Linear(
                hidden_dimension, len(CAPTURABLE_RULE_IDS)
            )
            self.black_drawback = nn.Linear(
                hidden_dimension, len(CAPTURABLE_RULE_IDS)
            )
            self.trigger = nn.Linear(hidden_dimension, 1)
            self.forced = nn.Linear(hidden_dimension, 1)
            self.triple_play_parameter = nn.Linear(hidden_dimension, 2)

        def forward(self, inputs: Any) -> dict[str, Any]:
            encoded = self.encoder(inputs)
            return {
                "white_drawback": self.white_drawback(encoded),
                "black_drawback": self.black_drawback(encoded),
                "trigger": self.trigger(encoded).squeeze(-1),
                "forced": self.forced(encoded).squeeze(-1),
                "triple_play_parameter": self.triple_play_parameter(encoded),
            }

    return CapturableBaseline()


def create_capturable_opportunity_model(hidden_dimension: int) -> Any:
    """Create the schema-9 model with a zero-initialized rule residual."""

    try:
        import torch
        import torch.nn as nn
    except ImportError as error:
        raise RuntimeError(
            "PyTorch is required; install ml/requirements.txt"
        ) from error

    class CapturableOpportunityBaseline(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(CAPTURABLE_FEATURE_DIMENSION, hidden_dimension),
                nn.ReLU(),
                nn.Linear(hidden_dimension, hidden_dimension),
                nn.ReLU(),
            )
            self.white_drawback = nn.Linear(
                hidden_dimension, len(CAPTURABLE_RULE_IDS)
            )
            self.black_drawback = nn.Linear(
                hidden_dimension, len(CAPTURABLE_RULE_IDS)
            )
            self.trigger = nn.Linear(hidden_dimension, 1)
            self.forced = nn.Linear(hidden_dimension, 1)
            self.triple_play_parameter = nn.Linear(hidden_dimension, 2)
            self.opportunity_weights = nn.Parameter(
                torch.zeros(CAPTURABLE_OPPORTUNITY_SHAPE)
            )

        def forward(
            self,
            inputs: Any,
            rule_opportunities: Any,
        ) -> dict[str, Any]:
            if (
                rule_opportunities.ndim != 3
                or rule_opportunities.shape[0] != inputs.shape[0]
                or tuple(rule_opportunities.shape[1:])
                != CAPTURABLE_OPPORTUNITY_SHAPE
            ):
                raise ValueError(
                    "rule opportunities must have shape [N, 25, 4]"
                )
            encoded = self.encoder(inputs)
            opportunity_residual = (
                rule_opportunities * self.opportunity_weights.unsqueeze(0)
            ).sum(dim=-1)
            return {
                "white_drawback": (
                    self.white_drawback(encoded) + opportunity_residual
                ),
                "black_drawback": (
                    self.black_drawback(encoded) + opportunity_residual
                ),
                "trigger": self.trigger(encoded).squeeze(-1),
                "forced": self.forced(encoded).squeeze(-1),
                "triple_play_parameter": self.triple_play_parameter(encoded),
            }

    return CapturableOpportunityBaseline()


def tensorize(
    rows: Sequence[CapturableDatasetRow],
    trigger_row_multiplier: float = 1.0,
    game_source_weights: Mapping[str, float] | None = None,
    opportunity_mode: str | None = None,
) -> TensorRows:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "PyTorch is required; install ml/requirements.txt"
        ) from error
    if not rows:
        raise ValueError("tensorization requires rows")
    _validate_opportunity_rows(rows, opportunity_mode)
    if (
        isinstance(trigger_row_multiplier, bool)
        or not isinstance(trigger_row_multiplier, (int, float))
        or not math.isfinite(float(trigger_row_multiplier))
        or not 1.0 <= float(trigger_row_multiplier) <= 100.0
    ):
        raise ValueError(
            "trigger_row_multiplier must be finite from one to 100"
        )
    source_weights = _validated_game_source_weights(
        rows,
        game_source_weights,
    )
    player_game_weight_totals: dict[tuple[str, str], float] = {}
    for row in rows:
        key = (row.evaluation.game_id, row.features.player_color)
        raw_weight = (
            float(trigger_row_multiplier)
            if row.labels.rule_triggered
            else 1.0
        )
        player_game_weight_totals[key] = (
            player_game_weight_totals.get(key, 0.0) + raw_weight
        )
    parameter_targets = []
    symbolic_priors = []
    symbolic_eliminated = []
    for row in rows:
        required = row.labels.hidden_parameters.get("requiredType")
        parameter_targets.append(
            0 if required == "bishop" else 1 if required == "knight" else -1
        )
        prior, eliminated = active_symbolic(row.features)
        symbolic_priors.append(list(prior))
        symbolic_eliminated.append(list(eliminated))
    base_player_game_weights = [
        (
            float(trigger_row_multiplier)
            if row.labels.rule_triggered
            else 1.0
        )
        / player_game_weight_totals[
            (row.evaluation.game_id, row.features.player_color)
        ]
        for row in rows
    ]
    effective_player_game_weights = list(base_player_game_weights)
    if game_source_weights is not None:
        maximum_source_weight = max(source_weights.values())
        scaled_source_weights = {
            game_id: weight / maximum_source_weight
            for game_id, weight in source_weights.items()
        }
        if any(weight == 0.0 for weight in scaled_source_weights.values()):
            raise ValueError(
                "game_source_weights relative ratios exceed the supported "
                "training range"
            )
        mean_scaled_source_weight = (
            math.fsum(
                base_weight
                * scaled_source_weights[row.evaluation.game_id]
                for row, base_weight in zip(
                    rows,
                    base_player_game_weights,
                    strict=True,
                )
            )
            / math.fsum(base_player_game_weights)
        )
        effective_player_game_weights = [
            base_weight
            * (
                scaled_source_weights[row.evaluation.game_id]
                / mean_scaled_source_weight
            )
            for row, base_weight in zip(
                rows,
                base_player_game_weights,
                strict=True,
            )
        ]
    player_game_weights = torch.tensor(
        effective_player_game_weights,
        dtype=torch.float32,
    )
    if not bool(torch.all(torch.isfinite(player_game_weights))) or bool(
        torch.any(player_game_weights <= 0.0)
    ):
        raise ValueError(
            "game_source_weights relative ratios exceed the supported "
            "float32 training range"
        )
    player_game_normalization_weights = (
        player_game_weights
        if game_source_weights is None
        else torch.tensor(
            base_player_game_weights,
            dtype=torch.float32,
        )
    )
    return TensorRows(
        inputs=torch.tensor(
            [capturable_feature_vector(row.features) for row in rows],
            dtype=torch.float32,
        ),
        rule_opportunities=(
            None
            if opportunity_mode is None
            else (
                torch.zeros(
                    (
                        len(rows),
                        *CAPTURABLE_OPPORTUNITY_SHAPE,
                    ),
                    dtype=torch.float32,
                )
                if opportunity_mode == "zero-ablation"
                else torch.tensor(
                    [row.rule_opportunities for row in rows],
                    dtype=torch.float32,
                )
            )
        ),
        colors=torch.tensor(
            [0 if row.features.player_color == "white" else 1 for row in rows],
            dtype=torch.long,
        ),
        drawbacks=torch.tensor(
            [CAPTURABLE_RULE_INDEX[row.labels.true_drawback] for row in rows],
            dtype=torch.long,
        ),
        triggered=torch.tensor(
            [1.0 if row.labels.rule_triggered else 0.0 for row in rows],
            dtype=torch.float32,
        ),
        forced=torch.tensor(
            [1.0 if row.labels.forced else 0.0 for row in rows],
            dtype=torch.float32,
        ),
        triple_play_parameters=torch.tensor(
            parameter_targets,
            dtype=torch.long,
        ),
        player_game_weights=player_game_weights,
        player_game_normalization_weights=(
            player_game_normalization_weights
        ),
        symbolic_priors=torch.tensor(
            symbolic_priors,
            dtype=torch.float32,
        ),
        symbolic_eliminated=torch.tensor(
            symbolic_eliminated,
            dtype=torch.bool,
        ),
    )


def _validate_opportunity_rows(
    rows: Sequence[CapturableDatasetRow],
    opportunity_mode: str | None,
) -> None:
    if opportunity_mode is None:
        if any(row.rule_opportunities is not None for row in rows):
            raise ValueError(
                "schema-9 rows require an explicit opportunity mode"
            )
        return
    if opportunity_mode not in CAPTURABLE_OPPORTUNITY_MODES:
        raise ValueError(
            "opportunity_mode must be public-exact or zero-ablation"
        )
    if any(row.rule_opportunities is None for row in rows):
        raise ValueError(
            "opportunity-aware training requires schema-9 rows"
        )


def _validated_game_source_weights(
    rows: Sequence[CapturableDatasetRow],
    weights: Mapping[str, float] | None,
) -> Mapping[str, float]:
    game_ids = {row.evaluation.game_id for row in rows}
    if weights is None:
        return MappingProxyType({game_id: 1.0 for game_id in game_ids})
    if set(weights) != game_ids:
        raise ValueError(
            "game_source_weights must contain every training game exactly once"
        )
    checked: dict[str, float] = {}
    for game_id, value in weights.items():
        checked[game_id] = _checked_positive_weight(
            value,
            "game_source_weights",
        )
    return MappingProxyType(checked)


def _checked_positive_weight(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be finite and positive")
    try:
        checked = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            f"{field_name} must be finite and positive"
        ) from error
    if not math.isfinite(checked) or checked <= 0.0:
        raise ValueError(f"{field_name} must be finite and positive")
    return checked


def _selected_drawback_logits(outputs: Mapping[str, Any], colors: Any) -> Any:
    import torch

    return torch.where(
        colors.unsqueeze(1) == 0,
        outputs["white_drawback"],
        outputs["black_drawback"],
    )


def _model_outputs(model: Any, batch: TensorRows) -> Mapping[str, Any]:
    if batch.rule_opportunities is None:
        return model(batch.inputs)
    return model(batch.inputs, batch.rule_opportunities)


def _batch_loss(
    outputs: Mapping[str, Any],
    batch: TensorRows,
    config: CapturableTrainingConfig,
) -> Any:
    import torch
    import torch.nn.functional as functional

    logits = _selected_drawback_logits(outputs, batch.colors)
    symbolic_log_priors = _smoothed_log_priors(
        batch.symbolic_priors,
        batch.symbolic_eliminated,
        config.training_prior_smoothing,
    )
    classification_terms = []
    for alpha in config.fusion_alpha_grid:
        if alpha == 0.0:
            continue
        fused_logits = (
            symbolic_log_priors + float(alpha) * logits
        ).masked_fill(batch.symbolic_eliminated, -1e9)
        classification_terms.append(
            _weighted_mean(
                functional.cross_entropy(
                    fused_logits,
                    batch.drawbacks,
                    reduction="none",
                ),
                batch.player_game_weights,
                batch.player_game_normalization_weights,
            )
        )
    classification = torch.stack(classification_terms).mean()
    trigger = _weighted_mean(
        functional.binary_cross_entropy_with_logits(
            outputs["trigger"],
            batch.triggered,
            reduction="none",
        ),
        batch.player_game_weights,
        batch.player_game_normalization_weights,
    )
    forced = _weighted_mean(
        functional.binary_cross_entropy_with_logits(
            outputs["forced"],
            batch.forced,
            reduction="none",
        ),
        batch.player_game_weights,
        batch.player_game_normalization_weights,
    )
    parameter_mask = batch.triple_play_parameters >= 0
    parameter = (
        _weighted_mean(
            functional.cross_entropy(
                outputs["triple_play_parameter"][parameter_mask],
                batch.triple_play_parameters[parameter_mask],
                reduction="none",
            ),
            batch.player_game_weights[parameter_mask],
            batch.player_game_normalization_weights[parameter_mask],
        )
        if bool(torch.any(parameter_mask))
        else classification.new_zeros(())
    )
    return (
        classification
        + config.trigger_loss_weight * trigger
        + config.forced_loss_weight * forced
        + config.parameter_loss_weight * parameter
    )


def _weighted_mean(
    values: Any,
    weights: Any,
    normalization_weights: Any,
) -> Any:
    total = normalization_weights.sum()
    if float(total.detach()) <= 0.0:
        raise RuntimeError(
            "player-game normalization weights must sum to a positive value"
        )
    return (values * weights).sum() / total


def _smoothed_log_priors(
    priors: Any,
    eliminated: Any,
    smoothing: float,
) -> Any:
    import torch

    survivors = (~eliminated).to(priors.dtype)
    survivor_count = survivors.sum(dim=1, keepdim=True)
    if bool(torch.any(survivor_count <= 0.0)):
        raise RuntimeError("symbolic training rows require a survivor")
    surviving_prior = priors * survivors
    prior_total = surviving_prior.sum(dim=1, keepdim=True)
    normalized = torch.where(
        prior_total > 0.0,
        surviving_prior / prior_total.clamp_min(_PRIOR_FLOOR),
        survivors / survivor_count,
    )
    smoothed = (
        (1.0 - float(smoothing)) * normalized
        + float(smoothing) * survivors / survivor_count
    )
    return smoothed.clamp_min(_PRIOR_FLOOR).log()


def _slice_tensors(batch: TensorRows, indices: Any) -> TensorRows:
    return TensorRows(
        inputs=batch.inputs[indices],
        rule_opportunities=(
            None
            if batch.rule_opportunities is None
            else batch.rule_opportunities[indices]
        ),
        colors=batch.colors[indices],
        drawbacks=batch.drawbacks[indices],
        triggered=batch.triggered[indices],
        forced=batch.forced[indices],
        triple_play_parameters=batch.triple_play_parameters[indices],
        player_game_weights=batch.player_game_weights[indices],
        player_game_normalization_weights=(
            batch.player_game_normalization_weights[indices]
        ),
        symbolic_priors=batch.symbolic_priors[indices],
        symbolic_eliminated=batch.symbolic_eliminated[indices],
    )


def _raw_predictions(
    model: Any,
    tensors: TensorRows,
    batch_size: int,
) -> tuple[list[list[float]], list[float], list[float], list[list[float]]]:
    import torch

    residuals: list[list[float]] = []
    trigger_probabilities: list[float] = []
    forced_probabilities: list[float] = []
    parameter_probabilities: list[list[float]] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(tensors.inputs), batch_size):
            stop = min(start + batch_size, len(tensors.inputs))
            indices = slice(start, stop)
            outputs = _model_outputs(
                model,
                _slice_tensors(tensors, indices),
            )
            selected = _selected_drawback_logits(
                outputs, tensors.colors[indices]
            )
            residuals.extend(selected.tolist())
            trigger_probabilities.extend(
                torch.sigmoid(outputs["trigger"]).tolist()
            )
            forced_probabilities.extend(
                torch.sigmoid(outputs["forced"]).tolist()
            )
            parameter_probabilities.extend(
                torch.softmax(
                    outputs["triple_play_parameter"], dim=1
                ).tolist()
            )
    return (
        residuals,
        trigger_probabilities,
        forced_probabilities,
        parameter_probabilities,
    )


def _rule_family(rule_id: str) -> str:
    if rule_id in {"vegan", "lame-duck"}:
        return "forbidden mover or capture"
    if rule_id in {"checkers", "irresistible"}:
        return "forced move"
    if rule_id in {"truant", "spice-of-life", "you-best-not-miss"}:
        return "history restriction"
    return "king-capture restriction"


def evaluate_capturable(
    model: Any,
    rows: Sequence[CapturableDatasetRow],
    tensors: TensorRows,
    config: CapturableTrainingConfig,
    fusion_alpha: float,
    prior_smoothing: float,
    raw_predictions: tuple[
        list[list[float]],
        list[float],
        list[float],
        list[list[float]],
    ] | None = None,
) -> Mapping[str, Any]:
    (
        residuals,
        trigger_probabilities,
        forced_probabilities,
        parameter_probabilities,
    ) = (
        _raw_predictions(model, tensors, config.batch_size)
        if raw_predictions is None
        else raw_predictions
    )
    hybrid_probabilities = [
        _hard_mask_fusion(
            neural,
            *active_symbolic(row.features),
            alpha=fusion_alpha,
            prior_smoothing=prior_smoothing,
        )
        for row, neural in zip(rows, residuals, strict=True)
    ]
    return evaluate_capturable_posteriors(
        rows,
        hybrid_probabilities,
        trigger_probabilities,
        forced_probabilities,
        parameter_probabilities,
    )


def evaluate_capturable_posteriors(
    rows: Sequence[CapturableDatasetRow],
    hybrid_probabilities: Sequence[Sequence[float]],
    trigger_probabilities: Sequence[float],
    forced_probabilities: Sequence[float],
    parameter_probabilities: Sequence[Sequence[float]],
) -> Mapping[str, Any]:
    """Evaluate already-fused public posteriors with the common metric path."""

    row_count = len(rows)
    if (
        row_count == 0
        or len(hybrid_probabilities) != row_count
        or len(trigger_probabilities) != row_count
        or len(forced_probabilities) != row_count
        or len(parameter_probabilities) != row_count
    ):
        raise ValueError("capturable posterior rows must align")
    hybrid = StreamingEvaluation()
    symbolic = StreamingEvaluation()
    hybrid_by_color = {
        "white": StreamingEvaluation(),
        "black": StreamingEvaluation(),
    }
    color_counts = {"white": 0, "black": 0}
    trigger_metrics = StreamingBinaryEvaluation()
    forced_metrics = StreamingBinaryEvaluation()
    observed: dict[tuple[str, str], int] = {}

    for index, row in enumerate(rows):
        key = (row.evaluation.game_id, row.features.player_color)
        observed[key] = observed.get(key, 0) + 1
        prior, eliminated = active_symbolic(row.features)
        probabilities = hybrid_probabilities[index]
        symbolic_probabilities = _hard_mask_fusion(
            [0.0] * len(CAPTURABLE_RULE_IDS),
            prior,
            eliminated,
            alpha=0.0,
            prior_smoothing=0.0,
        )
        hard_mask = dict(zip(CAPTURABLE_RULE_IDS, eliminated, strict=True))
        predicted_parameter = None
        if row.labels.true_drawback == "triple-play":
            parameter_posterior = parameter_probabilities[index]
            predicted_parameter = {
                "requiredType": (
                    "bishop"
                    if parameter_posterior[0] >= parameter_posterior[1]
                    else "knight"
                )
            }
        common = {
            "game_id": row.evaluation.game_id,
            "move_number": row.features.move_number,
            "true_drawback": row.labels.true_drawback,
            "player_color": row.features.player_color,
            "observed_ply": observed[key],
            "rule_family": _rule_family(row.labels.true_drawback),
            "true_parameters": dict(row.labels.hidden_parameters) or None,
            "hard_eliminated": hard_mask,
        }
        hybrid_example = PredictionExample(
            **common,
            probabilities=dict(
                zip(
                    CAPTURABLE_RULE_IDS,
                    probabilities,
                    strict=True,
                )
            ),
            predicted_parameters=predicted_parameter,
        )
        symbolic_example = PredictionExample(
            **common,
            probabilities=dict(
                zip(
                    CAPTURABLE_RULE_IDS,
                    symbolic_probabilities,
                    strict=True,
                )
            ),
        )
        hybrid.add(hybrid_example)
        hybrid_by_color[row.features.player_color].add(hybrid_example)
        color_counts[row.features.player_color] += 1
        symbolic.add(symbolic_example)
        trigger_metrics.add(
            row.labels.rule_triggered,
            trigger_probabilities[index],
        )
        forced_metrics.add(
            row.labels.forced,
            forced_probabilities[index],
        )
    return {
        "hybrid": _jsonable(hybrid.report()),
        "symbolicOnly": _jsonable(symbolic.report()),
        "hybridByColor": {
            color: (
                _jsonable(metric.report())
                if color_counts[color] > 0
                else None
            )
            for color, metric in hybrid_by_color.items()
        },
        "trigger": _jsonable(trigger_metrics.report()),
        "forced": _jsonable(forced_metrics.report()),
    }


def _validation_selection_metrics(
    rows: Sequence[CapturableDatasetRow],
    residuals: Sequence[Sequence[float]],
    fusion_alpha: float,
    prior_smoothing: float,
) -> tuple[float, float, float]:
    if len(rows) != len(residuals):
        raise ValueError("validation rows and residuals must align")
    by_player_game: dict[tuple[str, str], list[float]] = {}
    for row, neural in zip(rows, residuals, strict=True):
        prior, eliminated = active_symbolic(row.features)
        probabilities = _hard_mask_fusion(
            neural,
            prior,
            eliminated,
            alpha=fusion_alpha,
            prior_smoothing=prior_smoothing,
        )
        true_index = CAPTURABLE_RULE_INDEX[row.labels.true_drawback]
        key = (row.evaluation.game_id, row.features.player_color)
        totals = by_player_game.setdefault(key, [0.0, 0.0, 0.0, 0.0])
        totals[0] += 1.0
        totals[1] += _top_k_vector_credit(probabilities, true_index, 1)
        totals[2] += _top_k_vector_credit(probabilities, true_index, 3)
        probability = probabilities[true_index]
        totals[3] += (
            math.inf if probability == 0.0 else -math.log(probability)
        )
    count = len(by_player_game)
    return (
        math.fsum(values[1] / values[0] for values in by_player_game.values())
        / count,
        math.fsum(values[2] / values[0] for values in by_player_game.values())
        / count,
        math.fsum(values[3] / values[0] for values in by_player_game.values())
        / count,
    )


def _top_k_vector_credit(
    probabilities: Sequence[float],
    true_index: int,
    k: int,
) -> float:
    true_probability = probabilities[true_index]
    greater = sum(value > true_probability for value in probabilities)
    tied = sum(value == true_probability for value in probabilities)
    slots = min(k, len(probabilities)) - greater
    return 0.0 if slots <= 0 else min(1.0, slots / tied)


def _hard_mask_fusion(
    residuals: Sequence[float],
    prior: Sequence[float],
    eliminated: Sequence[bool],
    *,
    alpha: float,
    prior_smoothing: float,
) -> tuple[float, ...]:
    if (
        len(residuals) == 0
        or len(residuals) != len(prior)
        or len(prior) != len(eliminated)
        or not math.isfinite(alpha)
        or alpha < 0.0
        or not math.isfinite(prior_smoothing)
        or not 0.0 <= prior_smoothing < 1.0
    ):
        raise ValueError("hard-mask fusion inputs are invalid")
    survivors = [index for index, value in enumerate(eliminated) if not value]
    if not survivors:
        raise ValueError("hard-mask fusion requires a surviving hypothesis")
    survivor_prior_total = math.fsum(prior[index] for index in survivors)
    normalized_prior = [
        (
            prior[index] / survivor_prior_total
            if survivor_prior_total > 0.0
            else 1.0 / len(survivors)
        )
        for index in survivors
    ]
    smoothed_prior = [
        (1.0 - prior_smoothing) * probability
        + prior_smoothing / len(survivors)
        for probability in normalized_prior
    ]
    if alpha == 0.0:
        probabilities = [0.0] * len(prior)
        for index, probability in zip(
            survivors,
            smoothed_prior,
            strict=True,
        ):
            probabilities[index] = probability
        return tuple(probabilities)
    logits = [
        math.log(max(probability, _PRIOR_FLOOR))
        + alpha * residuals[index]
        for index, probability in zip(
            survivors,
            smoothed_prior,
            strict=True,
        )
    ]
    maximum = max(logits)
    exponentials = [math.exp(value - maximum) for value in logits]
    total = math.fsum(exponentials)
    probabilities = [0.0] * len(prior)
    for index, probability in zip(
        survivors,
        exponentials,
        strict=True,
    ):
        probabilities[index] = probability / total
    return tuple(probabilities)


def train_capturable_baseline(
    train_rows: Sequence[CapturableDatasetRow],
    validation_rows: Sequence[CapturableDatasetRow],
    test_rows: Sequence[CapturableDatasetRow] | None,
    config: CapturableTrainingConfig,
    *,
    train_game_source_weights: Mapping[str, float] | None = None,
    opportunity_mode: str | None = None,
) -> tuple[Any, Mapping[str, Any]]:
    """Train from random initialization and select epochs on validation only."""

    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "PyTorch is required; install ml/requirements.txt"
        ) from error
    if not train_rows or not validation_rows:
        raise ValueError("train and validation rows must be non-empty")
    if test_rows is None:
        assert_disjoint_games(train_rows, validation_rows)
    else:
        if not test_rows:
            raise ValueError("test rows must be non-empty when supplied")
        assert_disjoint_games(train_rows, validation_rows, test_rows)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_num_threads(config.torch_threads)
    torch.use_deterministic_algorithms(True)

    train_tensors = tensorize(
        train_rows,
        config.trigger_row_multiplier,
        train_game_source_weights,
        opportunity_mode,
    )
    validation_tensors = tensorize(
        validation_rows,
        opportunity_mode=opportunity_mode,
    )
    test_tensors = (
        None
        if test_rows is None
        else tensorize(test_rows, opportunity_mode=opportunity_mode)
    )
    model = (
        create_capturable_model(config.hidden_dimension)
        if opportunity_mode is None
        else create_capturable_opportunity_model(config.hidden_dimension)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    generator = torch.Generator().manual_seed(config.seed)
    history = []
    best_epoch = 0
    best_fusion_alpha = 0.0
    best_prior_smoothing = 0.0
    best_selection: tuple[float, float, float, float, float] | None = None
    best_state = None

    for epoch in range(1, config.epochs + 1):
        model.train()
        order = torch.randperm(len(train_rows), generator=generator)
        total_loss = 0.0
        batches = 0
        for start in range(0, len(order), config.batch_size):
            indices = order[start : start + config.batch_size]
            batch = _slice_tensors(train_tensors, indices)
            optimizer.zero_grad(set_to_none=True)
            outputs = _model_outputs(model, batch)
            loss = _batch_loss(outputs, batch, config)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach())
            batches += 1
        validation_raw = _raw_predictions(
            model,
            validation_tensors,
            config.batch_size,
        )
        validation_grid = [
            (
                alpha,
                smoothing,
                _validation_selection_metrics(
                    validation_rows,
                    validation_raw[0],
                    alpha,
                    smoothing,
                ),
            )
            for alpha in config.fusion_alpha_grid
            for smoothing in config.prior_smoothing_grid
        ]
        selected_alpha, selected_smoothing, validation = max(
            validation_grid,
            key=lambda item: (
                item[2][0],
                item[2][1],
                -item[2][2],
                -item[0],
                -item[1],
            ),
        )
        selection = (
            validation[0],
            validation[1],
            -validation[2],
            -selected_alpha,
            -selected_smoothing,
        )
        validation_report = evaluate_capturable(
            model,
            validation_rows,
            validation_tensors,
            config,
            selected_alpha,
            selected_smoothing,
            validation_raw,
        )
        history.append(
            {
                "epoch": epoch,
                "trainingLoss": total_loss / batches,
                "validationSelectedFusionAlpha": selected_alpha,
                "validationSelectedPriorSmoothing": selected_smoothing,
                "validationHybridTop1": validation_report["hybrid"][
                    "top_1_accuracy"
                ],
                "validationHybridTop3": validation_report["hybrid"][
                    "top_3_accuracy"
                ],
                "validationHybridNll": validation_report["hybrid"][
                    "negative_log_likelihood"
                ],
            }
        )
        if best_selection is None or selection > best_selection:
            best_epoch = epoch
            best_fusion_alpha = selected_alpha
            best_prior_smoothing = selected_smoothing
            best_selection = selection
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError("training did not produce a selected epoch")
    model.load_state_dict(best_state)
    opportunity_contract = _opportunity_contract(opportunity_mode)
    report: dict[str, Any] = {
        "format": CAPTURABLE_BASELINE_FORMAT,
        "version": (
            CAPTURABLE_BASELINE_VERSION
            if opportunity_mode is None
            else CAPTURABLE_OPPORTUNITY_BASELINE_VERSION
        ),
        "freshStart": True,
        "ruleIds": list(CAPTURABLE_RULE_IDS),
        "featureDimension": CAPTURABLE_FEATURE_DIMENSION,
        **opportunity_contract,
        "config": _jsonable(config),
        "selectedEpoch": best_epoch,
        "selectedFusionAlpha": best_fusion_alpha,
        "selectedPriorSmoothing": best_prior_smoothing,
        "selectionMetric": (
            "validation game_normalized_top_1_accuracy, then "
            "game_normalized_top_3_accuracy, then "
            "game_normalized_negative_log_likelihood"
        ),
        "history": history,
        "validation": evaluate_capturable(
            model,
            validation_rows,
            validation_tensors,
            config,
            best_fusion_alpha,
            best_prior_smoothing,
        ),
    }
    if test_rows is not None and test_tensors is not None:
        report["test"] = evaluate_capturable(
            model,
            test_rows,
            test_tensors,
            config,
            best_fusion_alpha,
            best_prior_smoothing,
        )
    return model, report


def _opportunity_contract(
    opportunity_mode: str | None,
) -> Mapping[str, Any]:
    if opportunity_mode is None:
        return {}
    if opportunity_mode not in CAPTURABLE_OPPORTUNITY_MODES:
        raise ValueError(
            "opportunity_mode must be public-exact or zero-ablation"
        )
    return {
        "symbolicFeatureVersion": (
            CAPTURABLE_OPPORTUNITY_SYMBOLIC_FEATURE_VERSION
        ),
        "opportunityFeatureVersion": CAPTURABLE_OPPORTUNITY_FEATURE_VERSION,
        "opportunityRuleIds": list(CAPTURABLE_RULE_IDS),
        "opportunityFields": list(CAPTURABLE_OPPORTUNITY_FIELDS),
        "opportunityShape": list(CAPTURABLE_OPPORTUNITY_SHAPE),
        "opportunityMode": opportunity_mode,
    }


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {
            field.name: _jsonable(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, (dict, MappingProxyType)):
        return {
            str(key): _jsonable(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(
            _jsonable(value),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _publish_bytes(path: Path, payload: bytes) -> None:
    temporary = path.with_name(
        f"{path.name}.tmp-{os.getpid()}-{secrets.token_hex(8)}"
    )
    try:
        with temporary.open("xb") as destination:
            os.chmod(temporary, 0o600)
            destination.write(payload)
            destination.flush()
            os.fsync(destination.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _publish_checkpoint(
    path: Path,
    model: Any,
    metadata: Mapping[str, Any],
    *,
    artifact_format: str = CAPTURABLE_BASELINE_FORMAT,
    artifact_version: int = CAPTURABLE_BASELINE_VERSION,
) -> str:
    import torch

    temporary = path.with_name(
        f"{path.name}.tmp-{os.getpid()}-{secrets.token_hex(8)}"
    )
    try:
        with temporary.open("xb") as destination:
            os.chmod(temporary, 0o600)
            torch.save(
                {
                    "format": artifact_format,
                    "version": artifact_version,
                    "metadata": _jsonable(metadata),
                    "stateDict": model.state_dict(),
                },
                destination,
            )
            destination.flush()
            os.fsync(destination.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return _sha256_file(path)


def run_training(
    train_paths: Sequence[Path],
    validation_path: Path,
    test_path: Path,
    output_directory: Path,
    config: CapturableTrainingConfig,
    train_source_weights: Sequence[float] | None = None,
    opportunity_mode: str | None = None,
) -> Mapping[str, Any]:
    report_path = output_directory / "evaluation.json"
    checkpoint_path = output_directory / "model.pt"
    if report_path.exists() or checkpoint_path.exists():
        raise FileExistsError("capturable training outputs already exist")
    if not train_paths:
        raise ValueError("at least one train dataset is required")
    checked_source_weights = _checked_train_source_weights(
        train_paths,
        train_source_weights,
    )
    dataset_loader = (
        load_capturable_dataset
        if opportunity_mode is None
        else load_capturable_opportunity_dataset
    )
    train_sources = [dataset_loader(path) for path in train_paths]
    validation_rows = dataset_loader(validation_path)
    test_rows = dataset_loader(test_path)
    assert_disjoint_games(
        *train_sources,
        validation_rows,
        test_rows,
    )
    train_rows = tuple(
        row
        for source in train_sources
        for row in source
    )
    game_source_weights = (
        None
        if checked_source_weights is None
        else {
            row.evaluation.game_id: weight
            for source, weight in zip(
                train_sources,
                checked_source_weights,
                strict=True,
            )
            for row in source
        }
    )
    model, measured = train_capturable_baseline(
        train_rows,
        validation_rows,
        test_rows,
        config,
        train_game_source_weights=game_source_weights,
        opportunity_mode=opportunity_mode,
    )
    input_identity = {
        "train": {
            "sources": [
                {
                    "path": path.resolve().name,
                    "sha256": _sha256_file(path.resolve()),
                    "rows": len(rows),
                    "games": len(
                        {row.evaluation.game_id for row in rows}
                    ),
                    **(
                        {}
                        if checked_source_weights is None
                        else {"weight": checked_source_weights[index]}
                    ),
                }
                for index, (path, rows) in enumerate(
                    zip(
                        train_paths,
                        train_sources,
                        strict=True,
                    )
                )
            ],
            "rows": len(train_rows),
            "games": len({row.evaluation.game_id for row in train_rows}),
            **(
                {}
                if checked_source_weights is None
                else {
                    "sourceWeightingObjective": (
                        SOURCE_WEIGHTING_OBJECTIVE
                    )
                }
            ),
        },
        "validation": {
            "path": validation_path.resolve().name,
            "sha256": _sha256_file(validation_path.resolve()),
            "rows": len(validation_rows),
            "games": len(
                {row.evaluation.game_id for row in validation_rows}
            ),
        },
        "test": {
            "path": test_path.resolve().name,
            "sha256": _sha256_file(test_path.resolve()),
            "rows": len(test_rows),
            "games": len({row.evaluation.game_id for row in test_rows}),
        },
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_sha256 = _publish_checkpoint(
        checkpoint_path,
        model,
        {
            "freshStart": True,
            "selectedEpoch": measured["selectedEpoch"],
            "selectedFusionAlpha": measured["selectedFusionAlpha"],
            "selectedPriorSmoothing": measured["selectedPriorSmoothing"],
            "ruleIds": list(CAPTURABLE_RULE_IDS),
            "featureDimension": CAPTURABLE_FEATURE_DIMENSION,
            **_opportunity_contract(opportunity_mode),
            "config": _jsonable(config),
            "inputs": input_identity,
        },
        artifact_version=(
            CAPTURABLE_BASELINE_VERSION
            if opportunity_mode is None
            else CAPTURABLE_OPPORTUNITY_BASELINE_VERSION
        ),
    )
    report = {
        **measured,
        "inputs": input_identity,
        "checkpoint": {
            "file": checkpoint_path.name,
            "sha256": checkpoint_sha256,
        },
    }
    payload = _canonical_json(report)
    _publish_bytes(report_path, payload)
    return {
        "reportPath": str(report_path),
        "reportSha256": hashlib.sha256(payload).hexdigest(),
        "checkpointPath": str(checkpoint_path),
        "checkpointSha256": checkpoint_sha256,
        "selectedEpoch": report["selectedEpoch"],
        "selectedFusionAlpha": report["selectedFusionAlpha"],
        "selectedPriorSmoothing": report["selectedPriorSmoothing"],
        "testHybridTop1": report["test"]["hybrid"]["top_1_accuracy"],
        "testHybridTop3": report["test"]["hybrid"]["top_3_accuracy"],
        "testSymbolicTop1": report["test"]["symbolicOnly"]["top_1_accuracy"],
    }


def _checked_train_source_weights(
    train_paths: Sequence[Path],
    weights: Sequence[float] | None,
) -> tuple[float, ...] | None:
    if weights is None:
        return None
    if len(weights) != len(train_paths):
        raise ValueError(
            "train_source_weights must contain one value per train dataset"
        )
    checked = []
    for value in weights:
        checked.append(
            _checked_positive_weight(value, "train_source_weights")
        )
    return tuple(checked)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a fresh capturable-king hybrid baseline."
    )
    parser.add_argument(
        "--train",
        type=Path,
        action="append",
        required=True,
        help="Repeat for each disjoint training dataset.",
    )
    parser.add_argument(
        "--train-source-weight",
        type=float,
        action="append",
        help=(
            "Optional positive weight for each --train dataset, in the same "
            "order. Supply one value per --train."
        ),
    )
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0xC0DE_0701)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dimension", type=int, default=128)
    parser.add_argument("--torch-threads", type=int, default=14)
    parser.add_argument(
        "--opportunity-mode",
        choices=CAPTURABLE_OPPORTUNITY_MODES,
        help=(
            "Enable strict schema-9 opportunities or its zero-input ablation."
        ),
    )
    parser.add_argument(
        "--trigger-row-multiplier",
        type=float,
        default=1.0,
    )
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    options = _parser().parse_args(arguments)
    config = CapturableTrainingConfig(
        seed=options.seed,
        epochs=options.epochs,
        batch_size=options.batch_size,
        hidden_dimension=options.hidden_dimension,
        torch_threads=options.torch_threads,
        trigger_row_multiplier=options.trigger_row_multiplier,
    )
    result = run_training(
        options.train,
        options.validation,
        options.test,
        options.output,
        config,
        options.train_source_weight,
        options.opportunity_mode,
    )
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
