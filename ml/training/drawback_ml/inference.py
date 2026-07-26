"""Deterministic, label-free inference from a saved baseline checkpoint."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, BinaryIO, Mapping, Sequence

from .checkpoint import (
    FUSION_GRID_DRAWBACK_OBJECTIVE,
    LEGACY_ADDITIVE_DRAWBACK_OBJECTIVE,
    parse_checkpoint_drawback_objective,
)
from .features import FEATURE_SCHEMA_VERSION, build_feature_vector
from .model import (
    HYBRID_MODEL_VARIANTS,
    SEQUENCE_MODEL_VARIANTS,
    ModelConfig,
    create_model,
    create_sequence_model,
)
from .rank_preserving_fusion import (
    RankPreservingFusionError,
    rank_preserving_fusion,
)
from .records import FeatureRecord
from .sequence import (
    ObservationTokenizerV2,
    SanTokenizer,
    SequenceTokenizer,
    encode_public_sequence,
)
from .symbolic import (
    SYMBOLIC_FEATURE_DIMENSION,
    SYMBOLIC_FEATURE_VERSION,
    SYMBOLIC_RULE_IDS,
    build_symbolic_feature_vector,
    fused_logits_with_symbolic_prior,
)


class CheckpointError(ValueError):
    """Raised when checkpoint metadata or tensor shapes are incompatible."""


@dataclass(frozen=True)
class InferenceOutput:
    white_drawback_probabilities: Mapping[str, float]
    black_drawback_probabilities: Mapping[str, float]
    white_parameter_probabilities: Mapping[str, float]
    black_parameter_probabilities: Mapping[str, float]
    trigger_probability: float
    legal_mask_probabilities: tuple[float, ...]
    white_neural_residual_logits: tuple[float, ...] | None = None
    black_neural_residual_logits: tuple[float, ...] | None = None
    white_fused_logits: tuple[float, ...] | None = None
    black_fused_logits: tuple[float, ...] | None = None
    white_hard_eliminated: tuple[bool, ...] | None = None
    black_hard_eliminated: tuple[bool, ...] | None = None


@dataclass(frozen=True)
class LegalMaskBatchStatistics:
    example_count: int
    dimension: int
    exact_matches: int
    true_positives: int
    false_positives: int
    false_negatives: int
    binary_cross_entropy_sum: float
    has_infinite_loss: bool


def _shape(state: Mapping[str, Any], key: str) -> tuple[int, ...]:
    value = state.get(key)
    shape = getattr(value, "shape", None)
    if shape is None:
        raise CheckpointError(f"checkpoint is missing tensor {key}")
    return tuple(int(dimension) for dimension in shape)


def _vocabulary(payload: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = payload.get(key)
    if (
        not isinstance(value, (list, tuple))
        or not value
        or any(not isinstance(item, str) or not item for item in value)
        or len(set(value)) != len(value)
    ):
        raise CheckpointError(f"{key} must be a non-empty unique string list")
    return tuple(value)


def _rank_preserving_tensors(
    torch: Any,
    neural_logits: Any,
    records: Sequence[FeatureRecord],
    vocabulary: Sequence[str],
    color: str,
) -> tuple[Any, Any, Any]:
    """Apply the scalar production fusion contract to a checkpoint batch."""

    if color not in {"white", "black"}:
        raise CheckpointError("rank-preserving fusion color is invalid")
    if (
        len(vocabulary) != len(SYMBOLIC_RULE_IDS)
        or set(vocabulary) != set(SYMBOLIC_RULE_IDS)
    ):
        raise CheckpointError(
            "rank-preserving checkpoint vocabulary is incompatible"
        )
    indices = tuple(
        SYMBOLIC_RULE_IDS.index(drawback_id) for drawback_id in vocabulary
    )
    probabilities: list[tuple[float, ...]] = []
    logits: list[tuple[float, ...]] = []
    masks: list[tuple[bool, ...]] = []
    for record, residuals in zip(
        records,
        _matrix_values(neural_logits),
        strict=True,
    ):
        build_symbolic_feature_vector(record)
        raw_prior = (
            record.symbolic_white_rule_probabilities
            if color == "white"
            else record.symbolic_black_rule_probabilities
        )
        raw_mask = (
            record.symbolic_white_eliminated
            if color == "white"
            else record.symbolic_black_eliminated
        )
        prior = tuple(raw_prior[index] for index in indices)
        mask = tuple(raw_mask[index] for index in indices)
        try:
            result = rank_preserving_fusion(
                residuals,
                prior,
                mask,
                alpha=1.0,
            )
        except RankPreservingFusionError as error:
            raise CheckpointError(
                "checkpoint public symbolic input cannot satisfy "
                "rank-preserving fusion"
            ) from error
        probabilities.append(result.probabilities)
        logits.append(result.logits)
        masks.append(mask)
    return (
        torch.tensor(
            probabilities,
            # Production fusion deliberately computes symbolic tiers in
            # binary64. Casting back to a float32 model dtype can collapse
            # adjacent representable priors into a tie.
            dtype=torch.float64,
            device=neural_logits.device,
        ),
        torch.tensor(
            logits,
            dtype=torch.float64,
            device=neural_logits.device,
        ),
        torch.tensor(
            masks,
            dtype=torch.bool,
            device=neural_logits.device,
        ),
    )


class CheckpointPredictor:
    """A loaded predictor whose public API accepts FeatureRecord only."""

    def __init__(
        self,
        *,
        torch_module: Any,
        model: Any,
        drawback_vocabulary: tuple[str, ...],
        parameter_vocabulary: tuple[str, ...],
        legal_mask_dimension: int,
        device: str,
        checkpoint_seed: int,
        checkpoint_epoch: int,
        training_run_id: str,
        corpus_provenance: Mapping[str, Any] | None = None,
        san_tokenizer: SequenceTokenizer | None = None,
        symbolic_enabled: bool = False,
        model_config: ModelConfig | None = None,
        drawback_loss_objective: str = LEGACY_ADDITIVE_DRAWBACK_OBJECTIVE,
    ) -> None:
        self._torch = torch_module
        self._model = model
        self.drawback_vocabulary = drawback_vocabulary
        self.parameter_vocabulary = parameter_vocabulary
        self.legal_mask_dimension = legal_mask_dimension
        self.device = device
        self.checkpoint_seed = checkpoint_seed
        self.checkpoint_epoch = checkpoint_epoch
        self.training_run_id = training_run_id
        self.corpus_provenance = (
            None if corpus_provenance is None else dict(corpus_provenance)
        )
        self.san_tokenizer = san_tokenizer
        self.symbolic_enabled = symbolic_enabled
        self.model_config = model_config
        self.drawback_loss_objective = drawback_loss_objective

    def predict(self, features: FeatureRecord) -> InferenceOutput:
        return self.predict_batch((features,))[0]

    def predict_batch(
        self,
        features: Sequence[FeatureRecord],
    ) -> tuple[InferenceOutput, ...]:
        outputs, _statistics = self._predict_batch(
            features,
            legal_true_indices=None,
        )
        return outputs

    def predict_batch_with_legal_statistics(
        self,
        features: Sequence[FeatureRecord],
        legal_true_indices: Sequence[Sequence[int]],
    ) -> tuple[tuple[InferenceOutput, ...], LegalMaskBatchStatistics]:
        outputs, statistics = self._predict_batch(
            features,
            legal_true_indices=legal_true_indices,
        )
        if statistics is None:
            raise RuntimeError("legal-mask statistics were not produced")
        return outputs, statistics

    def _predict_batch(
        self,
        features: Sequence[FeatureRecord],
        *,
        legal_true_indices: Sequence[Sequence[int]] | None,
    ) -> tuple[
        tuple[InferenceOutput, ...],
        LegalMaskBatchStatistics | None,
    ]:
        if not features:
            if legal_true_indices is not None and legal_true_indices:
                raise ValueError("empty features require empty legal targets")
            return (), (
                None
                if legal_true_indices is None
                else LegalMaskBatchStatistics(
                    example_count=0,
                    dimension=self.legal_mask_dimension,
                    exact_matches=0,
                    true_positives=0,
                    false_positives=0,
                    false_negatives=0,
                    binary_cross_entropy_sum=0.0,
                    has_infinite_loss=False,
                )
            )
        if (
            legal_true_indices is not None
            and len(legal_true_indices) != len(features)
        ):
            raise ValueError("legal targets must align with inference features")
        inputs = self._torch.tensor(
            [build_feature_vector(item) for item in features],
            dtype=self._torch.float32,
            device=self.device,
        )
        with self._torch.inference_mode():
            if self.san_tokenizer is None:
                output = self._model(inputs)
            else:
                encoded = [
                    encode_public_sequence(
                        self.san_tokenizer,
                        item,
                        (
                            None
                            if self.model_config is None
                            else self.model_config.sequence_observation_mode
                        ),
                    )
                    for item in features
                ]
                history_tokens = self._torch.tensor(
                    [tokens for tokens, _length in encoded],
                    dtype=self._torch.long,
                    device=self.device,
                )
                history_lengths = self._torch.tensor(
                    [length for _tokens, length in encoded],
                    dtype=self._torch.long,
                    device=self.device,
                )
                if self.symbolic_enabled:
                    symbolic_inputs = self._torch.tensor(
                        [
                            build_symbolic_feature_vector(item)
                            for item in features
                        ],
                        dtype=self._torch.float32,
                        device=self.device,
                    )
                    output = self._model(
                        inputs,
                        history_tokens,
                        history_lengths,
                        symbolic_inputs,
                    )
                else:
                    output = self._model(inputs, history_tokens, history_lengths)
            white_neural_logits = output["white_drawback"]
            black_neural_logits = output["black_drawback"]
            white_logits = white_neural_logits
            black_logits = black_neural_logits
            if self.symbolic_enabled:
                if (
                    self.drawback_loss_objective
                    == FUSION_GRID_DRAWBACK_OBJECTIVE
                ):
                    (
                        white,
                        white_fused_logits,
                        white_hard_mask,
                    ) = _rank_preserving_tensors(
                        self._torch,
                        white_logits,
                        features,
                        self.drawback_vocabulary,
                        "white",
                    )
                    (
                        black,
                        black_fused_logits,
                        black_hard_mask,
                    ) = _rank_preserving_tensors(
                        self._torch,
                        black_logits,
                        features,
                        self.drawback_vocabulary,
                        "black",
                    )
                else:
                    (
                        white_fused_logits,
                        white_hard_mask,
                    ) = fused_logits_with_symbolic_prior(
                        self._torch,
                        white_logits,
                        features,
                        self.drawback_vocabulary,
                        "white",
                    )
                    (
                        black_fused_logits,
                        black_hard_mask,
                    ) = fused_logits_with_symbolic_prior(
                        self._torch,
                        black_logits,
                        features,
                        self.drawback_vocabulary,
                        "black",
                    )
                    white_logits = white_fused_logits.masked_fill(
                        white_hard_mask, float("-inf")
                    )
                    black_logits = black_fused_logits.masked_fill(
                        black_hard_mask, float("-inf")
                    )
                    white = self._torch.softmax(white_logits, dim=-1)
                    black = self._torch.softmax(black_logits, dim=-1)
            else:
                white_fused_logits = black_fused_logits = None
                white_hard_mask = black_hard_mask = None
                white = self._torch.softmax(white_logits, dim=-1)
                black = self._torch.softmax(black_logits, dim=-1)
            white_parameters = self._torch.softmax(
                output["white_parameters"], dim=-1
            )
            black_parameters = self._torch.softmax(
                output["black_parameters"], dim=-1
            )
            trigger = self._torch.sigmoid(output["trigger"])
            legal_mask = self._torch.sigmoid(output["legal_mask"])
            legal_statistics = (
                None
                if legal_true_indices is None
                else _legal_mask_batch_statistics(
                    self._torch,
                    legal_mask,
                    legal_true_indices,
                    self.legal_mask_dimension,
                )
            )
        white_rows = _matrix_values(white)
        black_rows = _matrix_values(black)
        white_parameter_rows = _matrix_values(white_parameters)
        black_parameter_rows = _matrix_values(black_parameters)
        trigger_rows = _float_values(trigger)
        legal_rows = (
            _matrix_values(legal_mask)
            if legal_statistics is None
            else [[] for _item in features]
        )
        white_fused_rows = (
            None if white_fused_logits is None else _matrix_values(white_fused_logits)
        )
        black_fused_rows = (
            None if black_fused_logits is None else _matrix_values(black_fused_logits)
        )
        white_mask_rows = (
            None if white_hard_mask is None else white_hard_mask.detach().cpu().tolist()
        )
        black_mask_rows = (
            None if black_hard_mask is None else black_hard_mask.detach().cpu().tolist()
        )
        white_neural_rows = _matrix_values(white_neural_logits)
        black_neural_rows = _matrix_values(black_neural_logits)
        return tuple(
            InferenceOutput(
                white_drawback_probabilities=_normalized_values(
                    self.drawback_vocabulary, white_rows[index]
                ),
                black_drawback_probabilities=_normalized_values(
                    self.drawback_vocabulary, black_rows[index]
                ),
                white_parameter_probabilities=dict(
                    zip(
                        self.parameter_vocabulary,
                        white_parameter_rows[index],
                        strict=True,
                    )
                ),
                black_parameter_probabilities=dict(
                    zip(
                        self.parameter_vocabulary,
                        black_parameter_rows[index],
                        strict=True,
                    )
                ),
                trigger_probability=trigger_rows[index],
                legal_mask_probabilities=tuple(legal_rows[index]),
                white_neural_residual_logits=tuple(white_neural_rows[index]),
                black_neural_residual_logits=tuple(black_neural_rows[index]),
                white_fused_logits=(
                    None
                    if white_fused_rows is None
                    else tuple(white_fused_rows[index])
                ),
                black_fused_logits=(
                    None
                    if black_fused_rows is None
                    else tuple(black_fused_rows[index])
                ),
                white_hard_eliminated=(
                    None
                    if white_mask_rows is None
                    else tuple(bool(value) for value in white_mask_rows[index])
                ),
                black_hard_eliminated=(
                    None
                    if black_mask_rows is None
                    else tuple(bool(value) for value in black_mask_rows[index])
                ),
            )
            for index in range(len(features))
        ), legal_statistics


def _float_values(tensor: Any) -> Sequence[float]:
    return [float(value) for value in tensor.detach().cpu().tolist()]


def _matrix_values(tensor: Any) -> list[list[float]]:
    return [
        [float(value) for value in row]
        for row in tensor.detach().cpu().tolist()
    ]


def _legal_mask_batch_statistics(
    torch: Any,
    probabilities: Any,
    true_indices: Sequence[Sequence[int]],
    dimension: int,
    threshold: float = 0.5,
) -> LegalMaskBatchStatistics:
    if probabilities.ndim != 2 or tuple(probabilities.shape) != (
        len(true_indices),
        dimension,
    ):
        raise ValueError("legal-mask probabilities have an invalid batch shape")
    targets = torch.zeros_like(probabilities, dtype=torch.bool)
    for row, indices in enumerate(true_indices):
        unique = set(indices)
        if any(index < 0 or index >= dimension for index in unique):
            raise ValueError("legal-mask true index is outside the vocabulary")
        if unique:
            targets[row, list(unique)] = True
    predicted = probabilities >= threshold
    matches = predicted == targets
    probabilities64 = probabilities.to(dtype=torch.float64)
    target_probabilities = torch.where(
        targets,
        probabilities64,
        1.0 - probabilities64,
    )
    has_infinite = bool((target_probabilities == 0.0).any().item())
    finite_targets = target_probabilities[target_probabilities > 0.0]
    loss_sum = float(
        (-torch.log(finite_targets)).sum().item()
    )
    return LegalMaskBatchStatistics(
        example_count=len(true_indices),
        dimension=dimension,
        exact_matches=int(matches.all(dim=1).sum().item()),
        true_positives=int((predicted & targets).sum().item()),
        false_positives=int((predicted & ~targets).sum().item()),
        false_negatives=int((~predicted & targets).sum().item()),
        binary_cross_entropy_sum=loss_sum,
        has_infinite_loss=has_infinite,
    )


def _normalized_values(
    labels: Sequence[str],
    values: Sequence[float],
) -> Mapping[str, float]:
    total = math.fsum(values)
    if not math.isfinite(total) or total <= 0.0:
        raise CheckpointError("model produced an invalid probability distribution")
    return dict(
        zip(labels, (value / total for value in values), strict=True)
    )


def _normalized_probabilities(
    labels: Sequence[str],
    tensor: Any,
) -> Mapping[str, float]:
    values = list(_float_values(tensor))
    return _normalized_values(labels, values)


def load_checkpoint_predictor(
    checkpoint: Path | BinaryIO,
    *,
    device: str = "cpu",
    required_corpus_provenance: Mapping[str, Any] | None = None,
) -> CheckpointPredictor:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "PyTorch is required for checkpoint inference; install ml/requirements.txt"
        ) from error
    try:
        payload = torch.load(
            checkpoint,
            map_location=device,
            weights_only=True,
        )
    except Exception as error:
        raise CheckpointError(
            "checkpoint bytes could not be loaded safely"
        ) from error
    if not isinstance(payload, Mapping):
        raise CheckpointError("checkpoint root must be a mapping")
    if payload.get("format_version") != 3:
        raise CheckpointError(
            "checkpoint format 3 with reconstructable model metadata is required"
        )
    state = payload.get("model_state")
    if not isinstance(state, Mapping):
        raise CheckpointError("checkpoint model_state must be a mapping")
    drawback_vocabulary = _vocabulary(payload, "drawback_vocabulary")
    parameter_vocabulary = _vocabulary(payload, "parameter_vocabulary")
    raw_config = payload.get("model_config")
    training_metadata = payload.get("training_metadata")
    if not isinstance(raw_config, Mapping) or not isinstance(
        training_metadata, Mapping
    ):
        raise CheckpointError("checkpoint lacks model or training metadata")
    checkpoint_seed = payload.get("seed")
    checkpoint_epoch = payload.get("epoch")
    training_run_id = training_metadata.get("run_id")
    if (
        isinstance(checkpoint_seed, bool)
        or not isinstance(checkpoint_seed, int)
        or checkpoint_seed < 0
        or isinstance(checkpoint_epoch, bool)
        or not isinstance(checkpoint_epoch, int)
        or checkpoint_epoch <= 0
        or not isinstance(training_run_id, str)
        or len(training_run_id) != 64
        or any(
            character not in "0123456789abcdef"
            for character in training_run_id
        )
    ):
        raise CheckpointError("checkpoint run identity metadata is invalid")
    if training_metadata.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
        raise CheckpointError("checkpoint feature schema is incompatible")
    raw_corpus_provenance = training_metadata.get("corpus_provenance")
    if raw_corpus_provenance is not None and not isinstance(
        raw_corpus_provenance, Mapping
    ):
        raise CheckpointError("checkpoint corpus provenance is invalid")
    if required_corpus_provenance is not None:
        if raw_corpus_provenance is None:
            raise CheckpointError(
                "checkpoint lacks authenticated training-corpus provenance"
            )
        for key, expected in required_corpus_provenance.items():
            if raw_corpus_provenance.get(key) != expected:
                raise CheckpointError(
                    f"checkpoint corpus provenance does not match: {key}"
                )
    try:
        model_variant = raw_config.get("model_variant", "v1")
        config = ModelConfig(
            input_dimension=int(raw_config["input_dimension"]),
            drawback_classes=int(raw_config["drawback_classes"]),
            parameter_classes=int(raw_config["parameter_classes"]),
            legal_mask_dimension=int(raw_config["legal_mask_dimension"]),
            hidden_dimension=int(raw_config["hidden_dimension"]),
            model_variant=str(model_variant),
            sequence_observation_mode=raw_config.get(
                "sequence_observation_mode"
            ),
            san_vocabulary_size=(
                None
                if raw_config.get("san_vocabulary_size") is None
                else int(raw_config["san_vocabulary_size"])
            ),
            san_embedding_dimension=int(
                raw_config.get("san_embedding_dimension", 32)
            ),
            sequence_hidden_dimension=int(
                raw_config.get("sequence_hidden_dimension", 64)
            ),
            symbolic_dimension=(
                None
                if raw_config.get("symbolic_dimension") is None
                else int(raw_config["symbolic_dimension"])
            ),
            symbolic_hidden_dimension=int(
                raw_config.get("symbolic_hidden_dimension", 64)
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise CheckpointError("checkpoint model_config is invalid") from error
    try:
        drawback_loss_objective = parse_checkpoint_drawback_objective(
            config.model_variant,
            training_metadata,
        )
    except ValueError as error:
        raise CheckpointError(
            "checkpoint drawback objective metadata is invalid"
        ) from error
    san_tokenizer: SequenceTokenizer | None = None
    encoder_prefix = "encoder" if config.model_variant == "v1" else "board_encoder"
    hidden_dimension, input_dimension = _shape(
        state, f"{encoder_prefix}.0.weight"
    )
    drawback_classes, drawback_hidden = _shape(state, "white_drawback.weight")
    parameter_classes, parameter_hidden = _shape(state, "white_parameters.weight")
    legal_mask_dimension, legal_hidden = _shape(state, "legal_mask.weight")
    expected_head_hidden = hidden_dimension
    if config.model_variant in SEQUENCE_MODEL_VARIANTS:
        expected_head_hidden += config.sequence_hidden_dimension
    if config.model_variant in HYBRID_MODEL_VARIANTS:
        expected_head_hidden += config.symbolic_hidden_dimension
    if (
        _shape(state, "black_drawback.weight")
        != (drawback_classes, drawback_hidden)
        or _shape(state, "black_parameters.weight")
        != (parameter_classes, parameter_hidden)
        or _shape(state, "trigger.weight") != (1, expected_head_hidden)
        or _shape(state, f"{encoder_prefix}.2.weight")
        != (hidden_dimension, hidden_dimension)
        or drawback_hidden != expected_head_hidden
        or parameter_hidden != expected_head_hidden
        or legal_hidden != expected_head_hidden
        or drawback_classes != len(drawback_vocabulary)
        or parameter_classes != len(parameter_vocabulary)
        or config.input_dimension != input_dimension
        or config.hidden_dimension != hidden_dimension
        or config.drawback_classes != drawback_classes
        or config.parameter_classes != parameter_classes
        or config.legal_mask_dimension != legal_mask_dimension
    ):
        raise CheckpointError("checkpoint tensor shapes disagree with metadata")
    if config.model_variant in SEQUENCE_MODEL_VARIANTS:
        raw_tokenizer = training_metadata.get("san_tokenizer")
        if not isinstance(raw_tokenizer, Mapping):
            raise CheckpointError("sequence checkpoint lacks SAN tokenizer metadata")
        try:
            san_tokenizer = (
                ObservationTokenizerV2.from_metadata(raw_tokenizer)
                if config.model_variant == "v22-hybrid"
                else SanTokenizer.from_metadata(raw_tokenizer)
            )
        except ValueError as error:
            raise CheckpointError("sequence checkpoint tokenizer is invalid") from error
        metadata_mode = training_metadata.get("sequence_observation_mode")
        if config.model_variant == "v22-hybrid":
            if metadata_mode != config.sequence_observation_mode:
                raise CheckpointError(
                    "v22 sequence observation mode metadata disagrees"
                )
        elif metadata_mode is not None:
            raise CheckpointError(
                "legacy sequence checkpoint cannot declare an observation mode"
            )
        expected_embedding = (
            len(san_tokenizer.vocabulary),
            config.san_embedding_dimension,
        )
        if (
            config.san_vocabulary_size != len(san_tokenizer.vocabulary)
            or _shape(state, "san_embedding.weight") != expected_embedding
            or _shape(state, "history_encoder.weight_ih_l0")
            != (
                3 * config.sequence_hidden_dimension,
                config.san_embedding_dimension,
            )
        ):
            raise CheckpointError(
                "sequence checkpoint tensor shapes disagree with metadata"
            )
    if config.model_variant in HYBRID_MODEL_VARIANTS:
        if (
            training_metadata.get("symbolic_feature_version")
            != SYMBOLIC_FEATURE_VERSION
            or training_metadata.get("symbolic_rule_ids")
            != list(SYMBOLIC_RULE_IDS)
            or config.symbolic_dimension != SYMBOLIC_FEATURE_DIMENSION
            or _shape(state, "symbolic_encoder.0.weight")
            != (config.symbolic_hidden_dimension, SYMBOLIC_FEATURE_DIMENSION)
            or _shape(state, "symbolic_encoder.2.weight")
            != (
                config.symbolic_hidden_dimension,
                config.symbolic_hidden_dimension,
            )
        ):
            raise CheckpointError(
                "hybrid checkpoint symbolic metadata or tensor shapes disagree"
            )
    torch.use_deterministic_algorithms(True)
    model = (
        create_sequence_model(config)
        if config.model_variant in SEQUENCE_MODEL_VARIANTS
        else create_model(config)
    )
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return CheckpointPredictor(
        torch_module=torch,
        model=model,
        drawback_vocabulary=drawback_vocabulary,
        parameter_vocabulary=parameter_vocabulary,
        legal_mask_dimension=legal_mask_dimension,
        device=device,
        checkpoint_seed=checkpoint_seed,
        checkpoint_epoch=checkpoint_epoch,
        training_run_id=training_run_id,
        corpus_provenance=raw_corpus_provenance,
        san_tokenizer=san_tokenizer,
        symbolic_enabled=config.model_variant in HYBRID_MODEL_VARIANTS,
        model_config=config,
        drawback_loss_objective=drawback_loss_objective,
    )
