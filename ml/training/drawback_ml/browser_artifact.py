"""Deterministic, versioned browser export for feed-forward v1 checkpoints."""

from __future__ import annotations

from array import array
import base64
from io import BytesIO
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from .checkpoint import (
    FUSION_GRID_DRAWBACK_OBJECTIVE,
    parse_checkpoint_drawback_objective,
)
from .durable_publish import publish_bytes_durable_exact
from .features import FEATURE_DIMENSION, FEATURE_SCHEMA_VERSION, MOVE_VOCABULARY_SIZE
from .sequence import (
    SEQUENCE_OBSERVATION_MODES,
    ObservationTokenizerV2,
    SanTokenizer,
    SequenceTokenizer,
)
from .symbolic_schema import (
    SYMBOLIC_FEATURE_DIMENSION,
    SYMBOLIC_FEATURE_VERSION,
    SYMBOLIC_RULE_IDS,
)


BROWSER_ARTIFACT_FORMAT = "drawbacktrainer-browser-model"
BROWSER_ARTIFACT_VERSION = 1
BROWSER_HYBRID_ARTIFACT_VERSION = 2
BROWSER_OBSERVATION_ARTIFACT_VERSION = 3
BROWSER_TENSOR_ENCODING = "float32-le-base64"
MAX_BROWSER_ARTIFACT_BYTES = 8 * 1024 * 1024
MAX_BROWSER_TENSOR_BYTES = 8 * 1024 * 1024
MAX_BROWSER_HIDDEN_DIMENSION = 256
MAX_BROWSER_SAN_VOCABULARY_SIZE = 65_536
MAX_BROWSER_HISTORY_LENGTH = 600
EXPORTED_TENSORS = frozenset(
    {
        "encoder.0.weight",
        "encoder.0.bias",
        "encoder.2.weight",
        "encoder.2.bias",
        "white_drawback.weight",
        "white_drawback.bias",
        "black_drawback.weight",
        "black_drawback.bias",
    }
)
EXPORTED_HYBRID_TENSORS = frozenset(
    {
        "board_encoder.0.weight",
        "board_encoder.0.bias",
        "board_encoder.2.weight",
        "board_encoder.2.bias",
        "san_embedding.weight",
        "history_encoder.weight_ih_l0",
        "history_encoder.weight_hh_l0",
        "history_encoder.bias_ih_l0",
        "history_encoder.bias_hh_l0",
        "symbolic_encoder.0.weight",
        "symbolic_encoder.0.bias",
        "symbolic_encoder.2.weight",
        "symbolic_encoder.2.bias",
        "white_drawback.weight",
        "white_drawback.bias",
        "black_drawback.weight",
        "black_drawback.bias",
    }
)


class BrowserArtifactError(ValueError):
    """Raised when a checkpoint cannot be safely exported for browser use."""


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise BrowserArtifactError(f"{name} must be a positive integer")
    return value


def _non_negative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BrowserArtifactError(f"{name} must be a non-negative integer")
    return value


def _vocabulary(payload: Mapping[str, Any], key: str) -> list[str]:
    value = payload.get(key)
    if (
        not isinstance(value, (list, tuple))
        or not value
        or any(not isinstance(item, str) or not item for item in value)
        or len(set(value)) != len(value)
    ):
        raise BrowserArtifactError(f"{key} must be a non-empty unique string list")
    return list(value)


def _expected_v1_shapes(
    *,
    input_dimension: int,
    hidden_dimension: int,
    drawback_classes: int,
    parameter_classes: int,
    legal_mask_dimension: int,
) -> dict[str, tuple[int, ...]]:
    return {
        "encoder.0.weight": (hidden_dimension, input_dimension),
        "encoder.0.bias": (hidden_dimension,),
        "encoder.2.weight": (hidden_dimension, hidden_dimension),
        "encoder.2.bias": (hidden_dimension,),
        "white_drawback.weight": (drawback_classes, hidden_dimension),
        "white_drawback.bias": (drawback_classes,),
        "black_drawback.weight": (drawback_classes, hidden_dimension),
        "black_drawback.bias": (drawback_classes,),
        "white_parameters.weight": (parameter_classes, hidden_dimension),
        "white_parameters.bias": (parameter_classes,),
        "black_parameters.weight": (parameter_classes, hidden_dimension),
        "black_parameters.bias": (parameter_classes,),
        "trigger.weight": (1, hidden_dimension),
        "trigger.bias": (1,),
        "legal_mask.weight": (legal_mask_dimension, hidden_dimension),
        "legal_mask.bias": (legal_mask_dimension,),
    }


def _expected_hybrid_shapes(
    *,
    input_dimension: int,
    hidden_dimension: int,
    drawback_classes: int,
    parameter_classes: int,
    legal_mask_dimension: int,
    san_vocabulary_size: int,
    san_embedding_dimension: int,
    sequence_hidden_dimension: int,
    symbolic_dimension: int,
    symbolic_hidden_dimension: int,
) -> dict[str, tuple[int, ...]]:
    combined_dimension = (
        hidden_dimension
        + sequence_hidden_dimension
        + symbolic_hidden_dimension
    )
    return {
        "board_encoder.0.weight": (hidden_dimension, input_dimension),
        "board_encoder.0.bias": (hidden_dimension,),
        "board_encoder.2.weight": (hidden_dimension, hidden_dimension),
        "board_encoder.2.bias": (hidden_dimension,),
        "san_embedding.weight": (
            san_vocabulary_size,
            san_embedding_dimension,
        ),
        "history_encoder.weight_ih_l0": (
            sequence_hidden_dimension * 3,
            san_embedding_dimension,
        ),
        "history_encoder.weight_hh_l0": (
            sequence_hidden_dimension * 3,
            sequence_hidden_dimension,
        ),
        "history_encoder.bias_ih_l0": (sequence_hidden_dimension * 3,),
        "history_encoder.bias_hh_l0": (sequence_hidden_dimension * 3,),
        "symbolic_encoder.0.weight": (
            symbolic_hidden_dimension,
            symbolic_dimension,
        ),
        "symbolic_encoder.0.bias": (symbolic_hidden_dimension,),
        "symbolic_encoder.2.weight": (
            symbolic_hidden_dimension,
            symbolic_hidden_dimension,
        ),
        "symbolic_encoder.2.bias": (symbolic_hidden_dimension,),
        "white_drawback.weight": (drawback_classes, combined_dimension),
        "white_drawback.bias": (drawback_classes,),
        "black_drawback.weight": (drawback_classes, combined_dimension),
        "black_drawback.bias": (drawback_classes,),
        "white_parameters.weight": (parameter_classes, combined_dimension),
        "white_parameters.bias": (parameter_classes,),
        "black_parameters.weight": (parameter_classes, combined_dimension),
        "black_parameters.bias": (parameter_classes,),
        "trigger.weight": (1, combined_dimension),
        "trigger.bias": (1,),
        "legal_mask.weight": (legal_mask_dimension, combined_dimension),
        "legal_mask.bias": (legal_mask_dimension,),
    }


def _validate_tensor_metadata(
    torch: Any,
    value: object,
    expected_shape: tuple[int, ...],
    name: str,
) -> None:
    if not isinstance(value, torch.Tensor):
        raise BrowserArtifactError(f"model_state tensor {name} is missing")
    if value.layout != torch.strided or value.dtype != torch.float32:
        raise BrowserArtifactError(f"tensor {name} must be a dense float32 tensor")
    shape = tuple(int(dimension) for dimension in value.shape)
    if shape != expected_shape:
        raise BrowserArtifactError(
            f"tensor {name} has shape {shape}, expected {expected_shape}"
        )
    if not bool(torch.isfinite(value.detach()).all().item()):
        raise BrowserArtifactError(f"tensor {name} contains a non-finite value")


def _validated_tensor(
    torch: Any,
    value: object,
    expected_shape: tuple[int, ...],
    name: str,
) -> dict[str, object]:
    _validate_tensor_metadata(torch, value, expected_shape, name)
    assert isinstance(value, torch.Tensor)
    shape = tuple(int(dimension) for dimension in value.shape)
    values = [float(item) for item in value.detach().cpu().reshape(-1).tolist()]
    return {"shape": list(shape), "values": values}


def _validated_binary_tensor(
    torch: Any,
    value: object,
    expected_shape: tuple[int, ...],
    name: str,
) -> dict[str, object]:
    _validate_tensor_metadata(torch, value, expected_shape, name)
    assert isinstance(value, torch.Tensor)
    shape = tuple(int(dimension) for dimension in value.shape)
    values = [float(item) for item in value.detach().cpu().reshape(-1).tolist()]
    packed = array("f", values)
    if packed.itemsize != 4:
        raise BrowserArtifactError("platform float representation is incompatible")
    if sys.byteorder != "little":
        packed.byteswap()
    if len(packed) * packed.itemsize > MAX_BROWSER_TENSOR_BYTES:
        raise BrowserArtifactError(
            f"tensor {name} exceeds {MAX_BROWSER_TENSOR_BYTES} raw bytes"
        )
    return {
        "shape": list(shape),
        "data": base64.b64encode(packed.tobytes()).decode("ascii"),
    }


def _bounded_dimension(value: object, name: str) -> int:
    dimension = _positive_integer(value, name)
    if dimension > MAX_BROWSER_HIDDEN_DIMENSION:
        raise BrowserArtifactError(
            f"{name} exceeds browser limit {MAX_BROWSER_HIDDEN_DIMENSION}"
        )
    return dimension


def _hybrid_tokenizer(
    training: Mapping[str, Any],
    model_variant: str,
) -> SequenceTokenizer:
    metadata = training.get("san_tokenizer")
    if not isinstance(metadata, Mapping):
        raise BrowserArtifactError(
            "hybrid checkpoint requires SAN tokenizer metadata"
        )
    try:
        tokenizer = (
            ObservationTokenizerV2.from_metadata(metadata)
            if model_variant == "v22-hybrid"
            else SanTokenizer.from_metadata(metadata)
        )
    except ValueError as error:
        raise BrowserArtifactError(
            "hybrid checkpoint SAN tokenizer is incompatible"
        ) from error
    if len(tokenizer.vocabulary) > MAX_BROWSER_SAN_VOCABULARY_SIZE:
        raise BrowserArtifactError(
            "SAN vocabulary exceeds browser limit "
            f"{MAX_BROWSER_SAN_VOCABULARY_SIZE}"
        )
    if isinstance(tokenizer, ObservationTokenizerV2):
        if tokenizer.max_sequence > MAX_BROWSER_HISTORY_LENGTH + 1:
            raise BrowserArtifactError(
                "sequence max_sequence exceeds browser limit "
                f"{MAX_BROWSER_HISTORY_LENGTH + 1}"
            )
    elif tokenizer.max_history > MAX_BROWSER_HISTORY_LENGTH:
        raise BrowserArtifactError(
            f"SAN max_history exceeds browser limit "
            f"{MAX_BROWSER_HISTORY_LENGTH}"
        )
    return tokenizer


def _validate_state_keys(
    state: Mapping[str, Any],
    expected: Mapping[str, tuple[int, ...]],
) -> None:
    actual_keys = set(state)
    if actual_keys != set(expected):
        missing = sorted(set(expected) - actual_keys)
        unexpected = sorted(actual_keys - set(expected))
        raise BrowserArtifactError(
            f"model_state keys differ: missing={missing}, unexpected={unexpected}"
        )


def build_browser_artifact(checkpoint_bytes: bytes) -> dict[str, object]:
    """Validate one checkpoint snapshot and return its JSON-compatible artifact."""

    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "PyTorch is required to export a browser artifact; "
            "install ml/requirements.txt"
        ) from error
    checkpoint_sha256 = hashlib.sha256(checkpoint_bytes).hexdigest()
    try:
        payload = torch.load(
            BytesIO(checkpoint_bytes),
            map_location="cpu",
            weights_only=True,
        )
    except Exception as error:
        raise BrowserArtifactError("checkpoint could not be safely loaded") from error
    if not isinstance(payload, Mapping):
        raise BrowserArtifactError("checkpoint root must be a mapping")
    if payload.get("format_version") != 3:
        raise BrowserArtifactError("checkpoint format version 3 is required")
    config = payload.get("model_config")
    training = payload.get("training_metadata")
    state = payload.get("model_state")
    if not isinstance(config, Mapping):
        raise BrowserArtifactError("checkpoint model_config must be a mapping")
    if not isinstance(training, Mapping):
        raise BrowserArtifactError("checkpoint training_metadata must be a mapping")
    if not isinstance(state, Mapping):
        raise BrowserArtifactError("checkpoint model_state must be a mapping")
    model_variant = config.get("model_variant", "v1")
    if model_variant not in {"v1", "v21-hybrid", "v22-hybrid"}:
        raise BrowserArtifactError(
            "browser artifact export supports only v1, v21-hybrid, and "
            "v22-hybrid checkpoints"
        )
    _non_negative_integer(payload.get("seed"), "seed")
    _non_negative_integer(payload.get("epoch"), "epoch")

    input_dimension = _positive_integer(
        config.get("input_dimension"), "input_dimension"
    )
    hidden_dimension = _positive_integer(
        config.get("hidden_dimension"), "hidden_dimension"
    )
    if hidden_dimension > MAX_BROWSER_HIDDEN_DIMENSION:
        raise BrowserArtifactError(
            f"hidden_dimension exceeds browser limit "
            f"{MAX_BROWSER_HIDDEN_DIMENSION}"
        )
    drawback_classes = _positive_integer(
        config.get("drawback_classes"), "drawback_classes"
    )
    parameter_classes = _positive_integer(
        config.get("parameter_classes"), "parameter_classes"
    )
    legal_mask_dimension = _positive_integer(
        config.get("legal_mask_dimension"), "legal_mask_dimension"
    )
    feature_schema_version = _positive_integer(
        training.get("feature_schema_version"), "feature_schema_version"
    )
    if feature_schema_version != FEATURE_SCHEMA_VERSION:
        raise BrowserArtifactError("checkpoint feature schema is incompatible")
    if input_dimension != FEATURE_DIMENSION:
        raise BrowserArtifactError("checkpoint input dimension is incompatible")
    if legal_mask_dimension != MOVE_VOCABULARY_SIZE:
        raise BrowserArtifactError("checkpoint legal-mask dimension is incompatible")

    drawback_vocabulary = _vocabulary(payload, "drawback_vocabulary")
    parameter_vocabulary = _vocabulary(payload, "parameter_vocabulary")
    if len(drawback_vocabulary) != drawback_classes:
        raise BrowserArtifactError(
            "drawback vocabulary does not match drawback_classes"
        )
    unknown_drawbacks = sorted(set(drawback_vocabulary) - set(SYMBOLIC_RULE_IDS))
    if unknown_drawbacks:
        raise BrowserArtifactError(
            "drawback vocabulary contains unknown IDs: "
            + ", ".join(unknown_drawbacks)
        )
    if len(parameter_vocabulary) != parameter_classes:
        raise BrowserArtifactError(
            "parameter vocabulary does not match parameter_classes"
        )

    if model_variant in {"v21-hybrid", "v22-hybrid"}:
        if model_variant == "v22-hybrid":
            try:
                drawback_objective = parse_checkpoint_drawback_objective(
                    str(model_variant),
                    training,
                )
            except ValueError as error:
                raise BrowserArtifactError(
                    "hybrid checkpoint drawback objective is incompatible"
                ) from error
            if drawback_objective != FUSION_GRID_DRAWBACK_OBJECTIVE:
                raise BrowserArtifactError(
                    "v22 checkpoint requires the fusion-grid objective"
                )
        if (
            training.get("symbolic_feature_version")
            != SYMBOLIC_FEATURE_VERSION
            or training.get("symbolic_rule_ids") != list(SYMBOLIC_RULE_IDS)
        ):
            raise BrowserArtifactError(
                "hybrid checkpoint symbolic schema is incompatible"
            )
        if (
            len(drawback_vocabulary) != len(SYMBOLIC_RULE_IDS)
            or set(drawback_vocabulary) != set(SYMBOLIC_RULE_IDS)
        ):
            raise BrowserArtifactError(
                "hybrid drawback vocabulary must exactly match symbolic rule IDs"
            )
        tokenizer = _hybrid_tokenizer(training, str(model_variant))
        sequence_observation_mode = config.get(
            "sequence_observation_mode"
        )
        training_observation_mode = training.get(
            "sequence_observation_mode"
        )
        if model_variant == "v22-hybrid":
            if (
                sequence_observation_mode not in SEQUENCE_OBSERVATION_MODES
                or training_observation_mode != sequence_observation_mode
            ):
                raise BrowserArtifactError(
                    "v22 sequence observation mode metadata is incompatible"
                )
        elif (
            sequence_observation_mode is not None
            or training_observation_mode is not None
        ):
            raise BrowserArtifactError(
                "v21 checkpoint cannot declare a sequence observation mode"
            )
        san_vocabulary_size = _positive_integer(
            config.get("san_vocabulary_size"), "san_vocabulary_size"
        )
        if san_vocabulary_size != len(tokenizer.vocabulary):
            raise BrowserArtifactError(
                "SAN tokenizer vocabulary does not match model configuration"
            )
        san_embedding_dimension = _bounded_dimension(
            config.get("san_embedding_dimension"),
            "san_embedding_dimension",
        )
        sequence_hidden_dimension = _bounded_dimension(
            config.get("sequence_hidden_dimension"),
            "sequence_hidden_dimension",
        )
        symbolic_dimension = _positive_integer(
            config.get("symbolic_dimension"), "symbolic_dimension"
        )
        if symbolic_dimension != SYMBOLIC_FEATURE_DIMENSION:
            raise BrowserArtifactError(
                "hybrid symbolic input dimension is incompatible"
            )
        symbolic_hidden_dimension = _bounded_dimension(
            config.get("symbolic_hidden_dimension"),
            "symbolic_hidden_dimension",
        )
        expected = _expected_hybrid_shapes(
            input_dimension=input_dimension,
            hidden_dimension=hidden_dimension,
            drawback_classes=drawback_classes,
            parameter_classes=parameter_classes,
            legal_mask_dimension=legal_mask_dimension,
            san_vocabulary_size=san_vocabulary_size,
            san_embedding_dimension=san_embedding_dimension,
            sequence_hidden_dimension=sequence_hidden_dimension,
            symbolic_dimension=symbolic_dimension,
            symbolic_hidden_dimension=symbolic_hidden_dimension,
        )
        _validate_state_keys(state, expected)
        for name, shape in expected.items():
            _validate_tensor_metadata(torch, state[name], shape, name)
        validated_tensors = {
            name: _validated_binary_tensor(torch, state[name], shape, name)
            for name, shape in sorted(expected.items())
            if name in EXPORTED_HYBRID_TENSORS
        }
        artifact: dict[str, object] = {
            "format": BROWSER_ARTIFACT_FORMAT,
            "formatVersion": (
                BROWSER_OBSERVATION_ARTIFACT_VERSION
                if model_variant == "v22-hybrid"
                else BROWSER_HYBRID_ARTIFACT_VERSION
            ),
            "modelVariant": str(model_variant),
            "featureSchemaVersion": feature_schema_version,
            "symbolicFeatureVersion": SYMBOLIC_FEATURE_VERSION,
            "sourceCheckpointSha256": checkpoint_sha256,
            "drawbackVocabulary": drawback_vocabulary,
            "symbolicRuleIds": list(SYMBOLIC_RULE_IDS),
            "tokenizer": tokenizer.metadata(),
            "tensorEncoding": BROWSER_TENSOR_ENCODING,
            "dimensions": {
                "input": input_dimension,
                "boardHidden": hidden_dimension,
                "sanVocabulary": san_vocabulary_size,
                "sanEmbedding": san_embedding_dimension,
                "sequenceHidden": sequence_hidden_dimension,
                "symbolicInput": symbolic_dimension,
                "symbolicHidden": symbolic_hidden_dimension,
                "drawbackClasses": drawback_classes,
            },
            "tensors": validated_tensors,
        }
        if model_variant == "v22-hybrid":
            artifact["sequenceObservationMode"] = sequence_observation_mode
        return artifact

    expected = _expected_v1_shapes(
        input_dimension=input_dimension,
        hidden_dimension=hidden_dimension,
        drawback_classes=drawback_classes,
        parameter_classes=parameter_classes,
        legal_mask_dimension=legal_mask_dimension,
    )
    _validate_state_keys(state, expected)
    validated_tensors = {
        name: _validated_tensor(torch, state[name], shape, name)
        for name, shape in sorted(expected.items())
    }

    return {
        "format": BROWSER_ARTIFACT_FORMAT,
        "formatVersion": BROWSER_ARTIFACT_VERSION,
        "modelVariant": "v1",
        "featureSchemaVersion": feature_schema_version,
        "sourceCheckpointSha256": checkpoint_sha256,
        "drawbackVocabulary": drawback_vocabulary,
        "dimensions": {
            "input": input_dimension,
            "hidden": hidden_dimension,
            "drawbackClasses": drawback_classes,
        },
        "tensors": {
            name: tensor
            for name, tensor in validated_tensors.items()
            if name in EXPORTED_TENSORS
        },
    }


def canonical_artifact_bytes(artifact: Mapping[str, object]) -> bytes:
    try:
        text = json.dumps(
            artifact,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise BrowserArtifactError("artifact is not canonical JSON data") from error
    return (text + "\n").encode("utf-8")


def export_browser_artifact(checkpoint: Path, output: Path) -> Path:
    """Export atomically from one immutable checkpoint byte snapshot."""

    try:
        checkpoint_bytes = checkpoint.read_bytes()
    except OSError as error:
        raise BrowserArtifactError(f"cannot read checkpoint: {checkpoint}") from error
    artifact = build_browser_artifact(checkpoint_bytes)
    rendered = canonical_artifact_bytes(artifact)
    if len(rendered) > MAX_BROWSER_ARTIFACT_BYTES:
        raise BrowserArtifactError(
            f"browser artifact exceeds {MAX_BROWSER_ARTIFACT_BYTES} bytes"
        )
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        publish_bytes_durable_exact(
            output,
            rendered,
            label="browser artifact",
        )
    except (OSError, ValueError) as error:
        raise BrowserArtifactError(f"cannot write browser artifact: {output}") from error
    return output
