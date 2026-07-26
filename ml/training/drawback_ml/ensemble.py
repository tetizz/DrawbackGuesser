"""Exact protocol-v2 three-member hybrid checkpoint ensemble."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import BinaryIO, Mapping, Sequence

from .checkpoint import LEGACY_ADDITIVE_DRAWBACK_OBJECTIVE
from .inference import (
    CheckpointError,
    CheckpointPredictor,
    InferenceOutput,
    load_checkpoint_predictor,
)
from .records import FeatureRecord
from .rank_preserving_fusion import (
    RankPreservingFusionError,
    RankPreservingFusionResult,
    rank_preserving_fusion,
    validate_fusion_alpha,
)
from .symbolic import build_symbolic_feature_vector
from .symbolic_schema import SYMBOLIC_RULE_IDS


PROTOCOL_V2_ENSEMBLE_SEEDS = (20260811, 20260812, 20260813)


class EnsembleError(CheckpointError):
    """Raised when an ensemble is incomplete or its members are incompatible."""


@dataclass(frozen=True)
class HybridEnsemblePredictor:
    """Average neural residuals before applying the symbolic contract once.

    Drawback outputs average the three members' raw residual logits, apply the
    bounded rank-preserving symbolic fusion contract once, and retain exact
    hard eliminations. Parameter, trigger, and legal-mask probabilities are
    arithmetic means of the three independently transformed member outputs.
    """

    members: tuple[CheckpointPredictor, CheckpointPredictor, CheckpointPredictor]
    fusion_alpha: float

    def __post_init__(self) -> None:
        _validate_members(self.members)
        object.__setattr__(
            self,
            "fusion_alpha",
            validate_fusion_alpha(self.fusion_alpha),
        )

    @property
    def drawback_vocabulary(self) -> tuple[str, ...]:
        return self.members[0].drawback_vocabulary

    @property
    def parameter_vocabulary(self) -> tuple[str, ...]:
        return self.members[0].parameter_vocabulary

    @property
    def legal_mask_dimension(self) -> int:
        return self.members[0].legal_mask_dimension

    @property
    def checkpoint_seeds(self) -> tuple[int, int, int]:
        return (
            self.members[0].checkpoint_seed,
            self.members[1].checkpoint_seed,
            self.members[2].checkpoint_seed,
        )

    @property
    def drawback_loss_objective(self) -> str:
        return getattr(
            self.members[0],
            "drawback_loss_objective",
            LEGACY_ADDITIVE_DRAWBACK_OBJECTIVE,
        )

    def predict(self, features: FeatureRecord) -> InferenceOutput:
        return self.predict_batch((features,))[0]

    def predict_batch(
        self, features: Sequence[FeatureRecord]
    ) -> tuple[InferenceOutput, ...]:
        if not features:
            return ()
        member_batches = tuple(
            member.predict_batch(features) for member in self.members
        )
        if any(len(batch) != len(features) for batch in member_batches):
            raise EnsembleError("ensemble member returned an invalid batch length")
        return tuple(
            self._aggregate_row(
                features[index],
                tuple(batch[index] for batch in member_batches),
            )
            for index in range(len(features))
        )

    def _aggregate_row(
        self,
        features: FeatureRecord,
        outputs: tuple[InferenceOutput, InferenceOutput, InferenceOutput],
    ) -> InferenceOutput:
        white_residuals = _required_rows(
            outputs, "white_neural_residual_logits", len(self.drawback_vocabulary)
        )
        black_residuals = _required_rows(
            outputs, "black_neural_residual_logits", len(self.drawback_vocabulary)
        )
        white_mask = _consistent_mask(outputs, "white_hard_eliminated")
        black_mask = _consistent_mask(outputs, "black_hard_eliminated")
        expected_white_mask, white_prior = _symbolic_contract(features, "white")
        expected_black_mask, black_prior = _symbolic_contract(features, "black")
        if white_mask != expected_white_mask or black_mask != expected_black_mask:
            raise EnsembleError("member hard mask disagrees with public symbolic input")

        white_neural = _mean_rows(white_residuals)
        black_neural = _mean_rows(black_residuals)
        white_fusion = _fuse_once(
            white_neural,
            white_prior,
            white_mask,
            self.fusion_alpha,
        )
        black_fusion = _fuse_once(
            black_neural,
            black_prior,
            black_mask,
            self.fusion_alpha,
        )

        white_parameters = _mean_mappings(
            tuple(output.white_parameter_probabilities for output in outputs),
            self.parameter_vocabulary,
        )
        black_parameters = _mean_mappings(
            tuple(output.black_parameter_probabilities for output in outputs),
            self.parameter_vocabulary,
        )
        legal_rows = tuple(output.legal_mask_probabilities for output in outputs)
        if any(len(row) != self.legal_mask_dimension for row in legal_rows):
            raise EnsembleError("member legal-mask output dimension is incompatible")
        trigger_probability = math.fsum(
            _probability(output.trigger_probability, "trigger")
            for output in outputs
        ) / 3.0
        for row in legal_rows:
            for value in row:
                _probability(value, "legal mask")

        return InferenceOutput(
            white_drawback_probabilities=dict(
                zip(
                    self.drawback_vocabulary,
                    white_fusion.probabilities,
                    strict=True,
                )
            ),
            black_drawback_probabilities=dict(
                zip(
                    self.drawback_vocabulary,
                    black_fusion.probabilities,
                    strict=True,
                )
            ),
            white_parameter_probabilities=white_parameters,
            black_parameter_probabilities=black_parameters,
            trigger_probability=trigger_probability,
            legal_mask_probabilities=_mean_rows(legal_rows),
            white_neural_residual_logits=white_neural,
            black_neural_residual_logits=black_neural,
            white_fused_logits=white_fusion.logits,
            black_fused_logits=black_fusion.logits,
            white_hard_eliminated=white_mask,
            black_hard_eliminated=black_mask,
        )


def load_hybrid_ensemble(
    checkpoints: Sequence[Path | BinaryIO],
    *,
    device: str = "cpu",
    required_corpus_provenance: Mapping[str, object] | None = None,
    fusion_alpha: float,
) -> HybridEnsemblePredictor:
    """Load all three members, failing closed if any member cannot be loaded."""

    if len(checkpoints) != 3:
        raise EnsembleError("protocol-v2 ensemble requires exactly three checkpoints")
    members = (
        load_checkpoint_predictor(
            checkpoints[0],
            device=device,
            required_corpus_provenance=required_corpus_provenance,
        ),
        load_checkpoint_predictor(
            checkpoints[1],
            device=device,
            required_corpus_provenance=required_corpus_provenance,
        ),
        load_checkpoint_predictor(
            checkpoints[2],
            device=device,
            required_corpus_provenance=required_corpus_provenance,
        ),
    )
    return HybridEnsemblePredictor(members, fusion_alpha=fusion_alpha)


def _validate_members(members: Sequence[CheckpointPredictor]) -> None:
    if len(members) != 3:
        raise EnsembleError("protocol-v2 ensemble requires exactly three members")
    seeds = tuple(member.checkpoint_seed for member in members)
    if seeds != PROTOCOL_V2_ENSEMBLE_SEEDS:
        raise EnsembleError(
            "ensemble members must be ordered by fixed seeds "
            "20260811, 20260812, 20260813"
        )
    first = members[0]
    if (
        not first.symbolic_enabled
        or first.model_config is None
        or first.model_config.model_variant != "v21-hybrid"
    ):
        raise EnsembleError("ensemble members must be v21-hybrid checkpoints")
    if first.san_tokenizer is None:
        raise EnsembleError("ensemble members require SAN tokenizer metadata")
    first_objective = getattr(
        first,
        "drawback_loss_objective",
        LEGACY_ADDITIVE_DRAWBACK_OBJECTIVE,
    )
    for member in members[1:]:
        if (
            member.drawback_vocabulary != first.drawback_vocabulary
            or member.parameter_vocabulary != first.parameter_vocabulary
            or member.legal_mask_dimension != first.legal_mask_dimension
            or member.model_config != first.model_config
            or member.san_tokenizer != first.san_tokenizer
            or member.symbolic_enabled != first.symbolic_enabled
            or getattr(
                member,
                "drawback_loss_objective",
                LEGACY_ADDITIVE_DRAWBACK_OBJECTIVE,
            )
            != first_objective
        ):
            raise EnsembleError("ensemble checkpoint contracts are incompatible")
    if tuple(first.drawback_vocabulary) != tuple(SYMBOLIC_RULE_IDS):
        raise EnsembleError(
            "ensemble drawback vocabulary must exactly match symbolic rule order"
        )


def _required_rows(
    outputs: Sequence[InferenceOutput],
    attribute: str,
    dimension: int,
) -> tuple[tuple[float, ...], ...]:
    rows = tuple(getattr(output, attribute) for output in outputs)
    if any(row is None or len(row) != dimension for row in rows):
        raise EnsembleError("member did not expose compatible raw neural residuals")
    typed = tuple(tuple(row) for row in rows if row is not None)
    if any(not all(math.isfinite(value) for value in row) for row in typed):
        raise EnsembleError("member neural residual logits must be finite")
    return typed


def _consistent_mask(
    outputs: Sequence[InferenceOutput],
    attribute: str,
) -> tuple[bool, ...]:
    masks = tuple(getattr(output, attribute) for output in outputs)
    if masks[0] is None or any(mask != masks[0] for mask in masks[1:]):
        raise EnsembleError("ensemble members returned inconsistent hard masks")
    return tuple(masks[0])


def _symbolic_contract(
    features: FeatureRecord, color: str
) -> tuple[tuple[bool, ...], tuple[float, ...]]:
    try:
        build_symbolic_feature_vector(features)
    except (RuntimeError, ValueError) as error:
        raise EnsembleError("public symbolic input is incompatible") from error
    if color == "white":
        eliminated = features.symbolic_white_eliminated
        prior = features.symbolic_white_rule_probabilities
    else:
        eliminated = features.symbolic_black_eliminated
        prior = features.symbolic_black_rule_probabilities
    if len(eliminated) != len(SYMBOLIC_RULE_IDS) or len(prior) != len(
        SYMBOLIC_RULE_IDS
    ):
        raise EnsembleError("public symbolic input has an incompatible dimension")
    if all(eliminated):
        raise EnsembleError("symbolic engine eliminated every drawback")
    return tuple(eliminated), tuple(prior)


def _mean_rows(rows: Sequence[Sequence[float]]) -> tuple[float, ...]:
    if not rows or any(len(row) != len(rows[0]) for row in rows):
        raise EnsembleError("ensemble output dimensions are incompatible")
    result = tuple(
        math.fsum(row[index] for row in rows) / len(rows)
        for index in range(len(rows[0]))
    )
    if not all(math.isfinite(value) for value in result):
        raise EnsembleError("ensemble arithmetic mean is not finite")
    return result


def _mean_mappings(
    mappings: Sequence[Mapping[str, float]], labels: Sequence[str]
) -> Mapping[str, float]:
    if any(tuple(mapping) != tuple(labels) for mapping in mappings):
        raise EnsembleError("member parameter output vocabulary is incompatible")
    return {
        label: math.fsum(
            _probability(mapping[label], "parameter") for mapping in mappings
        )
        / len(mappings)
        for label in labels
    }


def _fuse_once(
    residuals: Sequence[float],
    prior: Sequence[float],
    eliminated: Sequence[bool],
    alpha: float,
) -> RankPreservingFusionResult:
    try:
        return rank_preserving_fusion(
            residuals,
            prior,
            eliminated,
            alpha=alpha,
        )
    except RankPreservingFusionError as error:
        raise EnsembleError(
            "public symbolic input cannot satisfy rank-preserving fusion"
        ) from error


def _probability(value: float, label: str) -> float:
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise EnsembleError(f"member {label} probability is invalid")
    return value
