"""Lazy PyTorch model construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .sequence import SEQUENCE_OBSERVATION_MODES, SequenceObservationMode


SEQUENCE_MODEL_VARIANTS = frozenset(
    {"v2-gru", "v21-hybrid", "v22-hybrid"}
)
HYBRID_MODEL_VARIANTS = frozenset({"v21-hybrid", "v22-hybrid"})


@dataclass(frozen=True)
class ModelConfig:
    input_dimension: int
    drawback_classes: int
    legal_mask_dimension: int
    parameter_classes: int = 1
    hidden_dimension: int = 128
    model_variant: str = "v1"
    san_vocabulary_size: int | None = None
    san_embedding_dimension: int = 32
    sequence_hidden_dimension: int = 64
    symbolic_dimension: int | None = None
    symbolic_hidden_dimension: int = 64
    sequence_observation_mode: SequenceObservationMode | None = None

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if name in {
                "model_variant",
                "san_vocabulary_size",
                "symbolic_dimension",
                "sequence_observation_mode",
            }:
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.model_variant not in {
            "v1",
            "v2-gru",
            "v21-hybrid",
            "v22-hybrid",
        }:
            raise ValueError(
                "model_variant must be v1, v2-gru, v21-hybrid, or v22-hybrid"
            )
        if self.model_variant in SEQUENCE_MODEL_VARIANTS and (
            self.san_vocabulary_size is None
            or isinstance(self.san_vocabulary_size, bool)
            or self.san_vocabulary_size < 2
        ):
            raise ValueError("sequence models require a SAN vocabulary")
        if self.model_variant in HYBRID_MODEL_VARIANTS and (
            self.symbolic_dimension is None
            or isinstance(self.symbolic_dimension, bool)
            or self.symbolic_dimension <= 0
        ):
            raise ValueError("hybrid models require symbolic features")
        if self.model_variant == "v22-hybrid":
            if (
                self.san_vocabulary_size is None
                or self.san_vocabulary_size < 4
            ):
                raise ValueError(
                    "v22-hybrid requires the four observation tokenizer "
                    "reserved tokens"
                )
            if self.sequence_observation_mode not in SEQUENCE_OBSERVATION_MODES:
                raise ValueError(
                    "v22-hybrid requires an explicit sequence_observation_mode"
                )
        elif self.sequence_observation_mode is not None:
            raise ValueError(
                "sequence_observation_mode is exclusive to v22-hybrid"
            )


def create_model(config: ModelConfig) -> Any:
    """Create the baseline without importing torch at module import time."""

    try:
        import torch.nn as nn
    except ImportError as error:
        raise RuntimeError(
            "PyTorch is required to construct the model; install ml/requirements.txt"
        ) from error

    class DrawbackBaseline(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(config.input_dimension, config.hidden_dimension),
                nn.ReLU(),
                nn.Linear(config.hidden_dimension, config.hidden_dimension),
                nn.ReLU(),
            )
            self.white_drawback = nn.Linear(
                config.hidden_dimension, config.drawback_classes
            )
            self.black_drawback = nn.Linear(
                config.hidden_dimension, config.drawback_classes
            )
            self.white_parameters = nn.Linear(
                config.hidden_dimension, config.parameter_classes
            )
            self.black_parameters = nn.Linear(
                config.hidden_dimension, config.parameter_classes
            )
            self.trigger = nn.Linear(config.hidden_dimension, 1)
            self.legal_mask = nn.Linear(
                config.hidden_dimension, config.legal_mask_dimension
            )

        def forward(self, inputs: Any) -> dict[str, Any]:
            encoded = self.encoder(inputs)
            return {
                "white_drawback": self.white_drawback(encoded),
                "black_drawback": self.black_drawback(encoded),
                "white_parameters": self.white_parameters(encoded),
                "black_parameters": self.black_parameters(encoded),
                "trigger": self.trigger(encoded).squeeze(-1),
                "legal_mask": self.legal_mask(encoded),
            }

    return DrawbackBaseline()


def create_sequence_model(config: ModelConfig) -> Any:
    """Create the opt-in GRU encoder over public SAN history."""

    if (
        config.model_variant not in SEQUENCE_MODEL_VARIANTS
        or config.san_vocabulary_size is None
    ):
        raise ValueError(
            "sequence model requires v2-gru, v21-hybrid, or v22-hybrid "
            "configuration"
        )
    try:
        import torch
        import torch.nn as nn
        from torch.nn.utils.rnn import pack_padded_sequence
    except ImportError as error:
        raise RuntimeError(
            "PyTorch is required to construct the model; install ml/requirements.txt"
        ) from error

    class DrawbackSequenceModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.board_encoder = nn.Sequential(
                nn.Linear(config.input_dimension, config.hidden_dimension),
                nn.ReLU(),
                nn.Linear(config.hidden_dimension, config.hidden_dimension),
                nn.ReLU(),
            )
            self.san_embedding = nn.Embedding(
                config.san_vocabulary_size,
                config.san_embedding_dimension,
                padding_idx=0,
            )
            self.history_encoder = nn.GRU(
                config.san_embedding_dimension,
                config.sequence_hidden_dimension,
                batch_first=True,
            )
            if config.model_variant in HYBRID_MODEL_VARIANTS:
                assert config.symbolic_dimension is not None
                self.symbolic_encoder = nn.Sequential(
                    nn.Linear(
                        config.symbolic_dimension,
                        config.symbolic_hidden_dimension,
                    ),
                    nn.ReLU(),
                    nn.Linear(
                        config.symbolic_hidden_dimension,
                        config.symbolic_hidden_dimension,
                    ),
                    nn.ReLU(),
                )
            else:
                self.symbolic_encoder = None
            combined = config.hidden_dimension + config.sequence_hidden_dimension
            if config.model_variant in HYBRID_MODEL_VARIANTS:
                combined += config.symbolic_hidden_dimension
            self.white_drawback = nn.Linear(combined, config.drawback_classes)
            self.black_drawback = nn.Linear(combined, config.drawback_classes)
            self.white_parameters = nn.Linear(combined, config.parameter_classes)
            self.black_parameters = nn.Linear(combined, config.parameter_classes)
            self.trigger = nn.Linear(combined, 1)
            self.legal_mask = nn.Linear(combined, config.legal_mask_dimension)

        def forward(
            self,
            board_inputs: Any,
            history_tokens: Any,
            history_lengths: Any,
            symbolic_inputs: Any | None = None,
        ) -> dict[str, Any]:
            board = self.board_encoder(board_inputs)
            embedded = self.san_embedding(history_tokens)
            packed = pack_padded_sequence(
                embedded,
                history_lengths.clamp(min=1).cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            _, hidden = self.history_encoder(packed)
            # Packing prevents padding steps from changing the recurrent state.
            # Empty histories use one padding step and are then explicitly zeroed.
            history = hidden[-1]
            history = history * (history_lengths > 0).unsqueeze(-1)
            parts = [board, history]
            if self.symbolic_encoder is not None:
                if symbolic_inputs is None:
                    raise ValueError("hybrid models require symbolic inputs")
                parts.append(self.symbolic_encoder(symbolic_inputs))
            elif symbolic_inputs is not None:
                raise ValueError("v2-gru does not accept symbolic inputs")
            encoded = torch.cat(parts, dim=-1)
            return {
                "white_drawback": self.white_drawback(encoded),
                "black_drawback": self.black_drawback(encoded),
                "white_parameters": self.white_parameters(encoded),
                "black_parameters": self.black_parameters(encoded),
                "trigger": self.trigger(encoded).squeeze(-1),
                "legal_mask": self.legal_mask(encoded),
            }

    return DrawbackSequenceModel()
