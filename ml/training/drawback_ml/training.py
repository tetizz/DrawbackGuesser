"""Deterministic baseline training. Importing this module does not import torch."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re
from pathlib import Path
import random
from typing import Any, Literal, Mapping, Sequence

from .checkpoint import save_checkpoint, write_run_metadata
from .features import (
    FEATURE_DIMENSION,
    FEATURE_SCHEMA_VERSION,
    MOVE_VOCABULARY_SIZE,
    build_feature_vector,
    encode_move,
)
from .model import (
    HYBRID_MODEL_VARIANTS,
    SEQUENCE_MODEL_VARIANTS,
    ModelConfig,
    create_model,
    create_sequence_model,
)
from .parameters import ParameterVocabulary, encode_parameter_targets
from .records import TrainingExample
from .splits import Split, SplitConfig, assign_split
from .sequence import (
    SEQUENCE_OBSERVATION_MODES,
    ObservationTokenizerV2,
    SanTokenizer,
    SequenceObservationMode,
    encode_public_sequence,
    public_sequence_observation,
)
from .symbolic import (
    FUSION_AWARE_LOSS_ALPHA_GRID,
    FUSION_AWARE_LOSS_METHOD,
    FUSION_AWARE_LOSS_VERSION,
    SYMBOLIC_FEATURE_DIMENSION,
    SYMBOLIC_FEATURE_VERSION,
    SYMBOLIC_RULE_IDS,
    build_symbolic_feature_vector,
    fusion_aware_drawback_loss,
    fusion_aware_loss_metadata,
)


@dataclass(frozen=True)
class TrainingConfig:
    seed: int
    epochs: int = 1
    batch_size: int = 32
    hidden_dimension: int = 128
    learning_rate: float = 1e-3
    model_variant: str = "v1"
    max_history: int = 128
    san_embedding_dimension: int = 32
    sequence_hidden_dimension: int = 64
    symbolic_hidden_dimension: int = 64
    split: SplitConfig = SplitConfig()
    required_drawback_vocabulary: tuple[str, ...] | None = None
    corpus_provenance: Mapping[str, Any] | None = None
    device: str = "cpu"
    shuffle_buffer_size: int = 4096
    game_examples_per_epoch: int = 16
    player_game_examples_per_epoch: int | None = None
    trigger_loss_weight: float = 0.1
    parameter_loss_weight: float = 0.1
    legal_mask_loss_weight: float = 0.05
    execution_source_revision: str | None = None
    fusion_aware_loss_method: str = FUSION_AWARE_LOSS_METHOD
    fusion_aware_loss_version: int = FUSION_AWARE_LOSS_VERSION
    fusion_aware_loss_alpha_grid: tuple[float, ...] = (
        FUSION_AWARE_LOSS_ALPHA_GRID
    )
    sequence_observation_mode: SequenceObservationMode | None = None

    def __post_init__(self) -> None:
        if (
            self.execution_source_revision is not None
            and re.fullmatch(
                r"[0-9a-f]{40}",
                self.execution_source_revision,
            ) is None
        ):
            raise ValueError(
                "execution_source_revision must be a full lowercase Git SHA"
            )
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.epochs <= 0 or self.batch_size <= 0 or self.hidden_dimension <= 0:
            raise ValueError("epochs, batch_size, and hidden_dimension must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")
        if self.shuffle_buffer_size <= 0:
            raise ValueError("shuffle_buffer_size must be positive")
        if self.game_examples_per_epoch <= 0:
            raise ValueError("game_examples_per_epoch must be positive")
        if (
            self.player_game_examples_per_epoch is not None
            and self.player_game_examples_per_epoch <= 0
        ):
            raise ValueError(
                "player_game_examples_per_epoch must be positive"
            )
        if (
            self.player_game_examples_per_epoch is None
            and self.game_examples_per_epoch % 2
        ):
            raise ValueError(
                "legacy game_examples_per_epoch must be even for equal "
                "observed-player balancing"
            )
        if (
            not math.isfinite(self.trigger_loss_weight)
            or not math.isfinite(self.parameter_loss_weight)
            or not math.isfinite(self.legal_mask_loss_weight)
            or self.trigger_loss_weight < 0
            or self.parameter_loss_weight < 0
            or self.legal_mask_loss_weight < 0
        ):
            raise ValueError(
                "auxiliary loss weights must be finite and non-negative"
            )
        if self.model_variant not in {
            "v1",
            "v2-gru",
            "v21-hybrid",
            "v22-hybrid",
        }:
            raise ValueError(
                "model_variant must be v1, v2-gru, v21-hybrid, or v22-hybrid"
            )
        if self.model_variant == "v22-hybrid":
            if self.sequence_observation_mode not in SEQUENCE_OBSERVATION_MODES:
                raise ValueError(
                    "v22-hybrid requires an explicit sequence_observation_mode"
                )
        elif self.sequence_observation_mode is not None:
            raise ValueError(
                "sequence_observation_mode is exclusive to v22-hybrid"
            )
        if (
            self.max_history <= 0
            or self.san_embedding_dimension <= 0
            or self.sequence_hidden_dimension <= 0
            or self.symbolic_hidden_dimension <= 0
        ):
            raise ValueError("sequence dimensions must be positive")
        if self.required_drawback_vocabulary is not None and (
            not self.required_drawback_vocabulary
            or len(set(self.required_drawback_vocabulary))
            != len(self.required_drawback_vocabulary)
        ):
            raise ValueError(
                "required_drawback_vocabulary must contain unique classes"
            )
        if (
            self.fusion_aware_loss_method != FUSION_AWARE_LOSS_METHOD
            or self.fusion_aware_loss_version != FUSION_AWARE_LOSS_VERSION
            or self.fusion_aware_loss_alpha_grid
            != FUSION_AWARE_LOSS_ALPHA_GRID
        ):
            raise ValueError(
                "fusion-aware loss method, version, and alpha grid are frozen"
            )


DrawbackSupervisionPolicy = Literal[
    "available-history-v1",
    "moving-color-only-v1",
]


def drawback_supervision_masks(
    policy: DrawbackSupervisionPolicy,
    player_colors: Sequence[str],
    history_colors: Sequence[Sequence[str]],
) -> tuple[tuple[bool, ...], tuple[bool, ...]]:
    """Select supervised heads using only public move-color history."""

    if policy not in {"available-history-v1", "moving-color-only-v1"}:
        raise ValueError("unsupported drawback supervision policy")
    if len(player_colors) != len(history_colors):
        raise ValueError("player colors and history colors must align")
    white: list[bool] = []
    black: list[bool] = []
    for color, prior_colors in zip(player_colors, history_colors, strict=True):
        if color not in {"white", "black"}:
            raise ValueError("player color must be white or black")
        if any(prior not in {"white", "black"} for prior in prior_colors):
            raise ValueError("history colors must be white or black")
        observed = frozenset((*prior_colors, color))
        white.append(
            color == "white"
            if policy == "moving-color-only-v1"
            else "white" in observed
        )
        black.append(
            color == "black"
            if policy == "moving-color-only-v1"
            else "black" in observed
        )
    return tuple(white), tuple(black)


def normalized_drawback_head_weights(
    white_mask: Sequence[bool],
    black_mask: Sequence[bool],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Give each row unit total drawback-loss weight across enabled heads."""

    if len(white_mask) != len(black_mask):
        raise ValueError("white and black drawback masks must align")
    white_weights: list[float] = []
    black_weights: list[float] = []
    for white, black in zip(white_mask, black_mask, strict=True):
        if type(white) is not bool or type(black) is not bool:
            raise ValueError("drawback masks must contain booleans")
        enabled = int(white) + int(black)
        if enabled == 0:
            raise ValueError("every row must supervise at least one drawback head")
        white_weights.append(1.0 / enabled if white else 0.0)
        black_weights.append(1.0 / enabled if black else 0.0)
    return tuple(white_weights), tuple(black_weights)


def drawback_observation_masks(
    player_colors: tuple[str, ...],
    history_lengths: tuple[int, ...],
) -> tuple[tuple[bool, ...], tuple[bool, ...]]:
    """Compatibility wrapper for White-start, alternating public histories."""

    if len(player_colors) != len(history_lengths):
        raise ValueError("player colors and history lengths must align")
    histories: list[tuple[str, ...]] = []
    for history_length in history_lengths:
        if history_length < 0:
            raise ValueError("history length must be non-negative")
        histories.append(
            tuple(
                "white" if ply % 2 == 0 else "black"
                for ply in range(history_length)
            )
        )
    return drawback_supervision_masks(
        "available-history-v1",
        player_colors,
        histories,
    )


@dataclass(frozen=True)
class PreparedTrainingLabels:
    examples: tuple[TrainingExample, ...]
    drawback_vocabulary: tuple[str, ...]
    parameter_vocabulary: ParameterVocabulary


def prepare_training_labels(
    examples: Sequence[TrainingExample],
    split: SplitConfig,
    required_drawback_vocabulary: Sequence[str] | None = None,
) -> PreparedTrainingLabels:
    train_examples = tuple(
        example
        for example in examples
        if assign_split(example.seed, split) is Split.TRAIN
    )
    if not train_examples:
        raise ValueError("no training examples were assigned to the training split")
    if required_drawback_vocabulary is None:
        drawback_vocabulary = tuple(
            sorted(
                {
                    drawback
                    for example in train_examples
                    for drawback in (
                        example.white_drawback,
                        example.black_drawback,
                    )
                }
            )
        )
    else:
        drawback_vocabulary = tuple(required_drawback_vocabulary)
        required = set(drawback_vocabulary)
        observed_white = {example.white_drawback for example in train_examples}
        observed_black = {example.black_drawback for example in train_examples}
        unexpected = (observed_white | observed_black).difference(required)
        if unexpected:
            raise ValueError(
                "training labels contain drawbacks outside the required vocabulary: "
                + ", ".join(sorted(unexpected))
            )
        missing_white = required.difference(observed_white)
        missing_black = required.difference(observed_black)
        if missing_white or missing_black:
            details: list[str] = []
            if missing_white:
                details.append(
                    "White missing " + ", ".join(sorted(missing_white))
                )
            if missing_black:
                details.append(
                    "Black missing " + ", ".join(sorted(missing_black))
                )
            raise ValueError(
                "required drawback coverage is incomplete: " + "; ".join(details)
            )
    parameter_vocabulary = ParameterVocabulary.build(
        parameter
        for example in train_examples
        for parameter in (example.white_parameters, example.black_parameters)
    )
    return PreparedTrainingLabels(
        examples=train_examples,
        drawback_vocabulary=drawback_vocabulary,
        parameter_vocabulary=parameter_vocabulary,
    )


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "PyTorch is required for training; install ml/requirements.txt"
        ) from error
    return torch


def _batches(
    examples: Sequence[TrainingExample], batch_size: int
) -> list[Sequence[TrainingExample]]:
    return [
        examples[index : index + batch_size]
        for index in range(0, len(examples), batch_size)
    ]


def _in_memory_v22_run_id(
    config: TrainingConfig,
    examples: Sequence[TrainingExample],
) -> str:
    """Bind an opt-in v22 checkpoint to its exact config and training rows."""

    digest = hashlib.sha256()
    header = {
        "format": "drawbacktrainer-in-memory-v22-run",
        "version": 1,
        "config": asdict(config),
    }
    digest.update(
        json.dumps(
            header,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    for example in sorted(
        examples,
        key=lambda item: (item.seed, item.game_id, item.features.ply),
    ):
        digest.update(b"\n")
        digest.update(
            json.dumps(
                asdict(example),
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
    return digest.hexdigest()


def balanced_legal_mask_loss(torch: Any, logits: Any, targets: Any) -> Any:
    """Weight legal and illegal move cells equally despite extreme sparsity."""

    element_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )
    positive = targets > 0.5
    negative = ~positive
    if not positive.any() or not negative.any():
        raise ValueError("balanced legal-mask loss requires both classes")
    return 0.5 * (
        element_loss[positive].mean() + element_loss[negative].mean()
    )


def train_baseline(
    examples: Sequence[TrainingExample],
    output_directory: Path,
    config: TrainingConfig,
) -> Path:
    """Train and checkpoint without evaluating or reporting invented metrics."""

    prepared = prepare_training_labels(
        examples,
        config.split,
        config.required_drawback_vocabulary,
    )
    train_examples = prepared.examples
    vocabulary = list(prepared.drawback_vocabulary)
    class_index = {drawback: index for index, drawback in enumerate(vocabulary)}
    parameter_vocabulary = prepared.parameter_vocabulary
    if config.model_variant == "v22-hybrid":
        san_tokenizer = ObservationTokenizerV2.fit(
            (
                public_sequence_observation(example.features)
                for example in train_examples
            ),
            max_sequence=config.max_history + 1,
        )
    elif config.model_variant in SEQUENCE_MODEL_VARIANTS:
        san_tokenizer = SanTokenizer.fit(
            (example.features.history_san for example in train_examples),
            max_history=config.max_history,
        )
    else:
        san_tokenizer = None
    training_run_id = (
        _in_memory_v22_run_id(config, train_examples)
        if config.model_variant == "v22-hybrid"
        else None
    )
    torch = _require_torch()
    from .streaming_training import validate_training_device
    device = validate_training_device(torch, config.device)

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True)
    model_config = ModelConfig(
        input_dimension=FEATURE_DIMENSION,
        drawback_classes=len(vocabulary),
        legal_mask_dimension=MOVE_VOCABULARY_SIZE,
        parameter_classes=len(parameter_vocabulary.tokens),
        hidden_dimension=config.hidden_dimension,
        model_variant=config.model_variant,
        sequence_observation_mode=config.sequence_observation_mode,
        san_vocabulary_size=(
            len(san_tokenizer.vocabulary) if san_tokenizer is not None else None
        ),
        san_embedding_dimension=config.san_embedding_dimension,
        sequence_hidden_dimension=config.sequence_hidden_dimension,
        symbolic_dimension=(
            SYMBOLIC_FEATURE_DIMENSION
            if config.model_variant in HYBRID_MODEL_VARIANTS
            else None
        ),
        symbolic_hidden_dimension=config.symbolic_hidden_dimension,
    )
    model = (
        create_sequence_model(model_config)
        if config.model_variant in SEQUENCE_MODEL_VARIANTS
        else create_model(model_config)
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    classification_loss = torch.nn.CrossEntropyLoss()
    binary_loss = torch.nn.BCEWithLogitsLoss()
    ordered = sorted(
        train_examples,
        key=lambda item: (item.seed, item.game_id, item.features.ply),
    )

    for epoch in range(1, config.epochs + 1):
        generator = random.Random(config.seed + epoch)
        generator.shuffle(ordered)
        model.train()
        for batch in _batches(ordered, config.batch_size):
            inputs = torch.tensor(
                [build_feature_vector(example.features) for example in batch],
                dtype=torch.float32,
                device=device,
            )
            white = torch.tensor(
                [class_index[example.white_drawback] for example in batch],
                dtype=torch.long,
                device=device,
            )
            black = torch.tensor(
                [class_index[example.black_drawback] for example in batch],
                dtype=torch.long,
                device=device,
            )
            parameter_targets = encode_parameter_targets(
                parameter_vocabulary,
                (example.white_parameters for example in batch),
                (example.black_parameters for example in batch),
            )
            white_parameters = torch.tensor(
                parameter_targets.white_indices,
                dtype=torch.long,
                device=device,
            )
            black_parameters = torch.tensor(
                parameter_targets.black_indices,
                dtype=torch.long,
                device=device,
            )
            white_parameter_mask = torch.tensor(
                parameter_targets.white_mask,
                dtype=torch.bool,
                device=device,
            )
            black_parameter_mask = torch.tensor(
                parameter_targets.black_mask,
                dtype=torch.bool,
                device=device,
            )
            white_observed, black_observed = drawback_observation_masks(
                tuple(example.features.player_color for example in batch),
                tuple(len(example.features.history_san) for example in batch),
            )
            white_drawback_mask = torch.tensor(
                white_observed, dtype=torch.bool, device=device
            )
            black_drawback_mask = torch.tensor(
                black_observed, dtype=torch.bool, device=device
            )
            white_parameter_mask &= white_drawback_mask
            black_parameter_mask &= black_drawback_mask
            trigger = torch.tensor(
                [float(example.rule_triggered) for example in batch],
                dtype=torch.float32,
                device=device,
            )
            legal_mask = torch.zeros(
                (len(batch), MOVE_VOCABULARY_SIZE),
                dtype=torch.float32,
                device=device,
            )
            for row_index, example in enumerate(batch):
                for legal_move in example.drawback_legal_moves:
                    legal_mask[row_index, encode_move(legal_move)] = 1.0
            if san_tokenizer is None:
                outputs = model(inputs)
            else:
                encoded_histories = [
                    encode_public_sequence(
                        san_tokenizer,
                        example.features,
                        config.sequence_observation_mode,
                    )
                    for example in batch
                ]
                history_tokens = torch.tensor(
                    [tokens for tokens, _ in encoded_histories],
                    dtype=torch.long,
                    device=device,
                )
                history_lengths = torch.tensor(
                    [length for _, length in encoded_histories],
                    dtype=torch.long,
                    device=device,
                )
                if config.model_variant in HYBRID_MODEL_VARIANTS:
                    symbolic_inputs = torch.tensor(
                        [
                            build_symbolic_feature_vector(example.features)
                            for example in batch
                        ],
                        dtype=torch.float32,
                        device=device,
                    )
                    outputs = model(
                        inputs,
                        history_tokens,
                        history_lengths,
                        symbolic_inputs,
                    )
                else:
                    outputs = model(inputs, history_tokens, history_lengths)
            parameter_loss = outputs["white_parameters"].sum() * 0.0
            if white_parameter_mask.any():
                parameter_loss = parameter_loss + classification_loss(
                    outputs["white_parameters"][white_parameter_mask],
                    white_parameters[white_parameter_mask],
                )
            if black_parameter_mask.any():
                parameter_loss = parameter_loss + classification_loss(
                    outputs["black_parameters"][black_parameter_mask],
                    black_parameters[black_parameter_mask],
                )
            legal_mask_loss = (
                balanced_legal_mask_loss(
                    torch, outputs["legal_mask"], legal_mask
                )
                if config.model_variant in SEQUENCE_MODEL_VARIANTS
                else binary_loss(outputs["legal_mask"], legal_mask)
            )
            drawback_loss = outputs["white_drawback"].sum() * 0.0
            if config.model_variant in HYBRID_MODEL_VARIANTS:
                drawback_loss += fusion_aware_drawback_loss(
                    torch,
                    outputs["white_drawback"],
                    [example.features for example in batch],
                    vocabulary,
                    "white",
                    white,
                    white_drawback_mask,
                )
                drawback_loss += fusion_aware_drawback_loss(
                    torch,
                    outputs["black_drawback"],
                    [example.features for example in batch],
                    vocabulary,
                    "black",
                    black,
                    black_drawback_mask,
                )
            else:
                if white_drawback_mask.any():
                    drawback_loss += classification_loss(
                        outputs["white_drawback"][white_drawback_mask],
                        white[white_drawback_mask],
                    )
                if black_drawback_mask.any():
                    drawback_loss += classification_loss(
                        outputs["black_drawback"][black_drawback_mask],
                        black[black_drawback_mask],
                    )
            loss = (
                drawback_loss
                + config.trigger_loss_weight
                * binary_loss(outputs["trigger"], trigger)
                + config.legal_mask_loss_weight * legal_mask_loss
                + config.parameter_loss_weight * parameter_loss
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        save_checkpoint(
            output_directory,
            model=model,
            optimizer=optimizer,
            seed=config.seed,
            epoch=epoch,
            drawback_vocabulary=vocabulary,
            parameter_vocabulary=list(parameter_vocabulary.tokens),
            model_config={
                "input_dimension": model_config.input_dimension,
                "drawback_classes": model_config.drawback_classes,
                "parameter_classes": model_config.parameter_classes,
                "legal_mask_dimension": model_config.legal_mask_dimension,
                "hidden_dimension": model_config.hidden_dimension,
                "model_variant": model_config.model_variant,
                **(
                    {
                        "sequence_observation_mode": (
                            model_config.sequence_observation_mode
                        )
                    }
                    if model_config.sequence_observation_mode is not None
                    else {}
                ),
                "san_vocabulary_size": model_config.san_vocabulary_size,
                "san_embedding_dimension": model_config.san_embedding_dimension,
                "sequence_hidden_dimension": model_config.sequence_hidden_dimension,
                "symbolic_dimension": model_config.symbolic_dimension,
                "symbolic_hidden_dimension": model_config.symbolic_hidden_dimension,
            },
            training_metadata={
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "loss_weights": {
                    "white_drawback": 1.0,
                    "black_drawback": 1.0,
                    "white_parameters": 1.0,
                    "black_parameters": 1.0,
                    "trigger": 1.0,
                    "legal_mask": 1.0,
                },
                "split": {
                    "salt": config.split.salt,
                    "train_fraction": config.split.train_fraction,
                    "validation_fraction": config.split.validation_fraction,
                    "test_fraction": config.split.test_fraction,
                },
                "optimizer": "Adam",
                "learning_rate": config.learning_rate,
                "batch_size": config.batch_size,
                "model_variant": config.model_variant,
                **(
                    {
                        "sequence_observation_mode": (
                            config.sequence_observation_mode
                        )
                    }
                    if config.sequence_observation_mode is not None
                    else {}
                ),
                **(
                    {"run_id": training_run_id}
                    if training_run_id is not None
                    else {}
                ),
                "symbolic_feature_version": (
                    SYMBOLIC_FEATURE_VERSION
                    if config.model_variant in HYBRID_MODEL_VARIANTS
                    else None
                ),
                "symbolic_rule_ids": (
                    list(SYMBOLIC_RULE_IDS)
                    if config.model_variant in HYBRID_MODEL_VARIANTS
                    else None
                ),
                "drawback_loss_objective": fusion_aware_loss_metadata(
                    config.model_variant
                ),
                "san_tokenizer": (
                    san_tokenizer.metadata()
                    if san_tokenizer is not None
                    else None
                ),
                "legal_mask_objective": (
                    "balanced-positive-negative-bce"
                    if config.model_variant in SEQUENCE_MODEL_VARIANTS
                    else "elementwise-bce"
                ),
                "corpus_provenance": (
                    dict(config.corpus_provenance)
                    if config.corpus_provenance is not None
                    else None
                ),
            },
        )

    write_run_metadata(
        output_directory,
        {
            "format_version": 3,
            "seed": config.seed,
            "epochs": config.epochs,
            "split_salt": config.split.salt,
            "drawback_vocabulary": vocabulary,
            "parameter_vocabulary": list(parameter_vocabulary.tokens),
            "feature_dimension": FEATURE_DIMENSION,
            "legal_mask_dimension": MOVE_VOCABULARY_SIZE,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "model_variant": config.model_variant,
            **(
                {
                    "sequence_observation_mode": (
                        config.sequence_observation_mode
                    )
                }
                if config.sequence_observation_mode is not None
                else {}
            ),
            **(
                {"run_id": training_run_id}
                if training_run_id is not None
                else {}
            ),
            "symbolic_feature_version": (
                SYMBOLIC_FEATURE_VERSION
                if config.model_variant in HYBRID_MODEL_VARIANTS
                else None
            ),
            "symbolic_rule_ids": (
                list(SYMBOLIC_RULE_IDS)
                if config.model_variant in HYBRID_MODEL_VARIANTS
                else None
            ),
            "drawback_loss_objective": fusion_aware_loss_metadata(
                config.model_variant
            ),
            "san_tokenizer": (
                san_tokenizer.metadata() if san_tokenizer is not None else None
            ),
            "legal_mask_objective": (
                "balanced-positive-negative-bce"
                if config.model_variant in SEQUENCE_MODEL_VARIANTS
                else "elementwise-bce"
            ),
            "loss_weights": {
                "white_drawback": 1.0,
                "black_drawback": 1.0,
                "white_parameters": 1.0,
                "black_parameters": 1.0,
                "trigger": 1.0,
                "legal_mask": 1.0,
            },
            "corpus_provenance": (
                dict(config.corpus_provenance)
                if config.corpus_provenance is not None
                else None
            ),
        },
    )
    return output_directory
