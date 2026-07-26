"""Leakage-audited baseline training utilities."""

from .browser_artifact import (
    BROWSER_ARTIFACT_FORMAT,
    BROWSER_ARTIFACT_VERSION,
    BrowserArtifactError,
    build_browser_artifact,
    canonical_artifact_bytes,
    export_browser_artifact,
)
from .corpus_contract import (
    AuditedCorpusSplit,
    CorpusContractError,
    audit_private_corpus_split,
    audit_corpus_split,
)
from .features import FEATURE_DIMENSION, build_feature_vector, encode_move
from .inference import (
    CheckpointError,
    CheckpointPredictor,
    InferenceOutput,
    load_checkpoint_predictor,
)
from .parameters import (
    ParameterTargetBatch,
    ParameterVocabulary,
    canonical_hidden_parameters,
    encode_parameter_targets,
)
from .records import (
    FeatureRecord,
    LabelRecord,
    TrainingExample,
    group_training_examples,
    parse_dataset_row,
)
from .sequence import SanTokenizer
from .symbolic import (
    SYMBOLIC_FEATURE_DIMENSION,
    SYMBOLIC_FEATURE_VERSION,
    SYMBOLIC_RULE_IDS,
    build_symbolic_feature_vector,
    combine_with_symbolic_prior,
)
from .splits import Split, SplitConfig, assign_split

__all__ = [
    "FEATURE_DIMENSION",
    "BROWSER_ARTIFACT_FORMAT",
    "BROWSER_ARTIFACT_VERSION",
    "BrowserArtifactError",
    "CorpusContractError",
    "AuditedCorpusSplit",
    "CheckpointError",
    "CheckpointPredictor",
    "FeatureRecord",
    "LabelRecord",
    "InferenceOutput",
    "Split",
    "SplitConfig",
    "SanTokenizer",
    "SYMBOLIC_FEATURE_DIMENSION",
    "SYMBOLIC_FEATURE_VERSION",
    "SYMBOLIC_RULE_IDS",
    "TrainingExample",
    "ParameterVocabulary",
    "ParameterTargetBatch",
    "assign_split",
    "audit_corpus_split",
    "audit_private_corpus_split",
    "build_feature_vector",
    "build_browser_artifact",
    "build_symbolic_feature_vector",
    "combine_with_symbolic_prior",
    "canonical_hidden_parameters",
    "encode_parameter_targets",
    "encode_move",
    "export_browser_artifact",
    "canonical_artifact_bytes",
    "group_training_examples",
    "load_checkpoint_predictor",
    "parse_dataset_row",
]
