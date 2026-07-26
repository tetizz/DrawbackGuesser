"""Small, genuine checkpoint fixtures for selection-boundary tests."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping

from ml.training.drawback_ml.features import FEATURE_DIMENSION
from ml.training.drawback_ml.model import (
    ModelConfig,
    create_model,
    create_sequence_model,
)
from ml.training.drawback_ml.checkpoint import (
    fusion_grid_drawback_objective_metadata,
)
from ml.training.drawback_ml.sequence import SanTokenizer
from ml.training.drawback_ml.symbolic_schema import (
    SYMBOLIC_FEATURE_DIMENSION,
    SYMBOLIC_FEATURE_VERSION,
    SYMBOLIC_RULE_IDS,
)


def write_legacy_checkpoint(
    path: Path,
    *,
    seed: int,
    epoch: int,
    run_id: str,
    training_corpus_set: Mapping[str, object],
) -> str:
    """Write a loadable legacy v1 checkpoint bound to one training claim."""

    import torch

    torch.manual_seed(seed + epoch)
    config = ModelConfig(
        input_dimension=FEATURE_DIMENSION,
        drawback_classes=2,
        parameter_classes=1,
        legal_mask_dimension=3,
        hidden_dimension=4,
        model_variant="v1",
    )
    model = create_model(config)
    payload = {
        "format_version": 3,
        "seed": seed,
        "epoch": epoch,
        "drawback_vocabulary": ["rule-a", "rule-b"],
        "parameter_vocabulary": ["none"],
        "model_config": {
            "input_dimension": config.input_dimension,
            "drawback_classes": config.drawback_classes,
            "parameter_classes": config.parameter_classes,
            "legal_mask_dimension": config.legal_mask_dimension,
            "hidden_dimension": config.hidden_dimension,
            "model_variant": config.model_variant,
        },
        "training_metadata": {
            "feature_schema_version": 1,
            "run_id": run_id,
            "corpus_provenance": {
                "training_corpus_set": dict(training_corpus_set),
                "training_corpus_set_sha256": training_corpus_set["sha256"],
            },
        },
        "model_state": model.state_dict(),
    }
    torch.save(payload, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_fusion_checkpoint(
    path: Path,
    *,
    seed: int,
    epoch: int,
    run_id: str,
    training_corpus_set: Mapping[str, object],
) -> str:
    """Write a tiny, loadable v21 checkpoint with the production objective."""

    import torch

    torch.manual_seed(seed + epoch)
    tokenizer = SanTokenizer.fit((("e4",),), max_history=8)
    config = ModelConfig(
        input_dimension=FEATURE_DIMENSION,
        drawback_classes=len(SYMBOLIC_RULE_IDS),
        parameter_classes=1,
        legal_mask_dimension=3,
        hidden_dimension=4,
        model_variant="v21-hybrid",
        san_vocabulary_size=len(tokenizer.vocabulary),
        san_embedding_dimension=2,
        sequence_hidden_dimension=2,
        symbolic_dimension=SYMBOLIC_FEATURE_DIMENSION,
        symbolic_hidden_dimension=2,
    )
    model = create_sequence_model(config)
    payload = {
        "format_version": 3,
        "seed": seed,
        "epoch": epoch,
        "drawback_vocabulary": list(SYMBOLIC_RULE_IDS),
        "parameter_vocabulary": ["none"],
        "model_config": {
            "input_dimension": config.input_dimension,
            "drawback_classes": config.drawback_classes,
            "parameter_classes": config.parameter_classes,
            "legal_mask_dimension": config.legal_mask_dimension,
            "hidden_dimension": config.hidden_dimension,
            "model_variant": config.model_variant,
            "san_vocabulary_size": config.san_vocabulary_size,
            "san_embedding_dimension": config.san_embedding_dimension,
            "sequence_hidden_dimension": config.sequence_hidden_dimension,
            "symbolic_dimension": config.symbolic_dimension,
            "symbolic_hidden_dimension": config.symbolic_hidden_dimension,
        },
        "training_metadata": {
            "feature_schema_version": 1,
            "run_id": run_id,
            "san_tokenizer": tokenizer.metadata(),
            "symbolic_feature_version": SYMBOLIC_FEATURE_VERSION,
            "symbolic_rule_ids": list(SYMBOLIC_RULE_IDS),
            "drawback_loss_objective": (
                fusion_grid_drawback_objective_metadata()
            ),
            "corpus_provenance": {
                "training_corpus_set": dict(training_corpus_set),
                "training_corpus_set_sha256": training_corpus_set["sha256"],
            },
        },
        "model_state": model.state_dict(),
    }
    torch.save(payload, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()
