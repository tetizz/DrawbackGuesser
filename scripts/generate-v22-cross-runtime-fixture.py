"""Generate the compact Python-to-browser v22 integration golden.

The committed fixture contains browser artifacts, not a training checkpoint.
Both observation modes are exported through the production Python exporter,
then exercised through the production Python checkpoint predictor.  The two
artifacts share tensors, so the fixture stores the exact artifact once and the
two fields that differ in the masked artifact.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
import sys
import tempfile
from typing import Mapping
import zipfile


ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = ROOT / "ml" / "training"
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

import torch  # noqa: E402

from drawback_ml.browser_artifact import (  # noqa: E402
    canonical_artifact_bytes,
    export_browser_artifact,
)
from drawback_ml.checkpoint import (  # noqa: E402
    fusion_grid_drawback_objective_metadata,
)
from drawback_ml.features import (  # noqa: E402
    FEATURE_DIMENSION,
    FEATURE_SCHEMA_VERSION,
    MOVE_VOCABULARY_SIZE,
)
from drawback_ml.inference import load_checkpoint_predictor  # noqa: E402
from drawback_ml.model import ModelConfig, create_sequence_model  # noqa: E402
from drawback_ml.records import FeatureRecord  # noqa: E402
from drawback_ml.sequence import (  # noqa: E402
    ObservationTokenizerV2,
    PublicSequenceObservation,
)
from drawback_ml.symbolic_schema import (  # noqa: E402
    SYMBOLIC_FEATURE_DIMENSION,
    SYMBOLIC_FEATURE_VERSION,
    SYMBOLIC_RULE_IDS,
)


FIXTURE_PATH = (
    ROOT / "apps" / "web" / "src" / "fixtures" / "v22-cross-runtime-golden.json"
)
GENERATOR_FORMAT = "drawbacktrainer-v22-cross-runtime-golden"
GENERATOR_FORMAT_VERSION = 1
PORTABLE_CHECKPOINT_ENCODING = "portable-torch-checkpoint-zip-v1"
IGNORED_TORCH_RUNTIME_RECORDS = (
    ".data/serialization_id",
    ".format_version",
    ".storage_alignment",
)
CHECKPOINT_SEED = 2203
CHECKPOINT_EPOCH = 1
EXPECTED_PROBABILITY_DECIMAL_PLACES = 12
RUN_ID = hashlib.sha256(b"drawbacktrainer-v22-cross-runtime-golden-v1").hexdigest()
MODES = ("exact-current-v2", "masked-current-v2")
SOURCE_DEPENDENCIES = (
    "scripts/generate-v22-cross-runtime-fixture.py",
    "ml/training/drawback_ml/browser_artifact.py",
    "ml/training/drawback_ml/features.py",
    "ml/training/drawback_ml/inference.py",
    "ml/training/drawback_ml/model.py",
    "ml/training/drawback_ml/rank_preserving_fusion.py",
    "ml/training/drawback_ml/sequence.py",
    "ml/training/drawback_ml/symbolic.py",
)
TOKENIZER_OBSERVATIONS = (
    PublicSequenceObservation(("e4", "e5", "Nf3"), "b8c6"),
    PublicSequenceObservation(("e4",), "e7e5"),
    PublicSequenceObservation((), "a7a8q"),
    PublicSequenceObservation((), "a7a8r"),
)
MAX_SEQUENCE = 3
ORDINARY_LEGAL_MOVES = (
    "a7a5",
    "a7a6",
    "b7b5",
    "b7b6",
    "b8a6",
    "b8c6",
    "c7c5",
    "c7c6",
    "d7d5",
    "d7d6",
    "d8e7",
    "d8f6",
    "d8g5",
    "d8h4",
    "e8e7",
    "f7f5",
    "f7f6",
    "f8a3",
    "f8b4",
    "f8c5",
    "f8d6",
    "f8e7",
    "g7g5",
    "g7g6",
    "g8e7",
    "g8f6",
    "g8h6",
    "h7h5",
    "h7h6",
)


def _normalized_source_bytes(path: Path) -> bytes:
    text = path.read_text(encoding="utf-8")
    return text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _portable_expected_probability(value: float) -> float:
    """Project runtime-dependent float tails onto one portable JSON value."""

    return float(f"{value:.{EXPECTED_PROBABILITY_DECIMAL_PLACES}f}")


def _tokenizer() -> ObservationTokenizerV2:
    return ObservationTokenizerV2.fit(
        TOKENIZER_OBSERVATIONS,
        max_sequence=MAX_SEQUENCE,
    )


def _input_spec(tokenizer: ObservationTokenizerV2) -> dict[str, object]:
    return {
        "formatVersion": GENERATOR_FORMAT_VERSION,
        "checkpoint": {
            "epoch": CHECKPOINT_EPOCH,
            "runId": RUN_ID,
            "seed": CHECKPOINT_SEED,
            "serialization": {
                "encoding": PORTABLE_CHECKPOINT_ENCODING,
                "ignoredRuntimeRecords": list(
                    IGNORED_TORCH_RUNTIME_RECORDS
                ),
                "pickleProtocol": 2,
                "portableRecords": [
                    "data.pkl",
                    "byteorder",
                    "data/<non-negative-integer>",
                    "version",
                ],
                "producer": "torch.save-new-zip",
                "rawTorchSaveBytesArePortable": False,
            },
        },
        "dimensions": {
            "boardHidden": 1,
            "drawbackClasses": len(SYMBOLIC_RULE_IDS),
            "featureInput": FEATURE_DIMENSION,
            "legalMask": MOVE_VOCABULARY_SIZE,
            "parameterClasses": 1,
            "sanEmbedding": 1,
            "sanVocabulary": len(tokenizer.vocabulary),
            "sequenceHidden": 1,
            "symbolicHidden": 1,
            "symbolicInput": SYMBOLIC_FEATURE_DIMENSION,
        },
        "modelVariant": "v22-hybrid",
        "expectedProbabilityDecimalPlaces": (
            EXPECTED_PROBABILITY_DECIMAL_PLACES
        ),
        "observation": {
            "fenBefore": (
                "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/"
                "PPPP1PPP/RNBQKB1R b KQkq - 1 2"
            ),
            "historySan": ["e4", "e5", "Nf3"],
            "move": "b8c6",
            "moveNumber": 2,
            "ordinaryLegalMoveCount": len(ORDINARY_LEGAL_MOVES),
            "playerColor": "black",
            "ply": 3,
        },
        "symbolicPriorRecipe": {
            "blackEliminatedWhenIndexModulo19IsZero": True,
            "blackSurvivorWeight": "1 + ((ruleCount - index) modulo 4)",
            "whiteEliminatedWhenIndexModulo17IsZero": True,
            "whiteSurvivorWeight": "1 + (index modulo 3)",
        },
        "symbolicRuleIds": list(SYMBOLIC_RULE_IDS),
        "tensorRecipe": {
            "allUnspecifiedValues": 0,
            "blackHeadBias": "((classIndex modulo 7) - 3) / 16",
            "blackHistoryWeight": (
                "(((classCount - classIndex - 1) modulo 11) - 5) / 8"
            ),
            "candidateHiddenWeight": "1/4",
            "candidateInputWeight": 1,
            "embeddings": {
                "<current-move-masked>": "-1/2",
                "<move:b8c6>": "3/4",
                "<unk-current-move>": "3/8",
                "Nf3": "1/4",
                "e4": "1/8",
                "e5": "-1/8",
            },
            "updateGateInputBias": "-1/2",
            "whiteHeadBias": "((classIndex modulo 5) - 2) / 16",
            "whiteHistoryWeight": "((classIndex modulo 9) - 4) / 8",
        },
        "tokenizer": tokenizer.metadata(),
    }


def _model_config(
    tokenizer: ObservationTokenizerV2,
    mode: str,
) -> ModelConfig:
    return ModelConfig(
        input_dimension=FEATURE_DIMENSION,
        drawback_classes=len(SYMBOLIC_RULE_IDS),
        parameter_classes=1,
        legal_mask_dimension=MOVE_VOCABULARY_SIZE,
        hidden_dimension=1,
        model_variant="v22-hybrid",
        sequence_observation_mode=mode,  # type: ignore[arg-type]
        san_vocabulary_size=len(tokenizer.vocabulary),
        san_embedding_dimension=1,
        sequence_hidden_dimension=1,
        symbolic_dimension=SYMBOLIC_FEATURE_DIMENSION,
        symbolic_hidden_dimension=1,
    )


def _intentional_state(
    tokenizer: ObservationTokenizerV2,
) -> Mapping[str, torch.Tensor]:
    torch.manual_seed(CHECKPOINT_SEED)
    model = create_sequence_model(
        _model_config(tokenizer, "exact-current-v2")
    )
    state = model.state_dict()
    with torch.no_grad():
        for tensor in state.values():
            tensor.zero_()

        token_index = {
            token: index for index, token in enumerate(tokenizer.vocabulary)
        }
        embeddings = {
            "<unk-current-move>": 3 / 8,
            "<current-move-masked>": -1 / 2,
            "e4": 1 / 8,
            "e5": -1 / 8,
            "Nf3": 1 / 4,
            "<move:b8c6>": 3 / 4,
        }
        embedding = state["san_embedding.weight"]
        for token, value in embeddings.items():
            embedding[token_index[token], 0] = value

        state["history_encoder.weight_ih_l0"][2, 0] = 1
        state["history_encoder.weight_hh_l0"][2, 0] = 1 / 4
        state["history_encoder.bias_ih_l0"][1] = -1 / 2

        class_count = len(SYMBOLIC_RULE_IDS)
        for index in range(class_count):
            state["white_drawback.weight"][index, 1] = (
                (index % 9) - 4
            ) / 8
            state["white_drawback.bias"][index] = ((index % 5) - 2) / 16
            state["black_drawback.weight"][index, 1] = (
                ((class_count - index - 1) % 11) - 5
            ) / 8
            state["black_drawback.bias"][index] = ((index % 7) - 3) / 16
    return {
        name: tensor.detach().clone()
        for name, tensor in state.items()
    }


def _checkpoint_payload(
    tokenizer: ObservationTokenizerV2,
    state: Mapping[str, torch.Tensor],
    mode: str,
) -> dict[str, object]:
    config = _model_config(tokenizer, mode)
    return {
        "format_version": 3,
        "seed": CHECKPOINT_SEED,
        "epoch": CHECKPOINT_EPOCH,
        "drawback_vocabulary": list(reversed(SYMBOLIC_RULE_IDS)),
        "parameter_vocabulary": ["{}"],
        "model_config": {
            "input_dimension": config.input_dimension,
            "drawback_classes": config.drawback_classes,
            "parameter_classes": config.parameter_classes,
            "legal_mask_dimension": config.legal_mask_dimension,
            "hidden_dimension": config.hidden_dimension,
            "model_variant": config.model_variant,
            "sequence_observation_mode": config.sequence_observation_mode,
            "san_vocabulary_size": config.san_vocabulary_size,
            "san_embedding_dimension": config.san_embedding_dimension,
            "sequence_hidden_dimension": config.sequence_hidden_dimension,
            "symbolic_dimension": config.symbolic_dimension,
            "symbolic_hidden_dimension": config.symbolic_hidden_dimension,
        },
        "training_metadata": {
            "run_id": RUN_ID,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "symbolic_feature_version": SYMBOLIC_FEATURE_VERSION,
            "symbolic_rule_ids": list(SYMBOLIC_RULE_IDS),
            "sequence_observation_mode": mode,
            "san_tokenizer": tokenizer.metadata(),
            "drawback_loss_objective": (
                fusion_grid_drawback_objective_metadata()
            ),
        },
        "model_state": {
            name: tensor.detach().clone()
            for name, tensor in state.items()
        },
        "optimizer_state": {},
    }


def _torch_checkpoint_bytes(payload: Mapping[str, object]) -> bytes:
    output = BytesIO()
    torch.save(
        payload,
        output,
        pickle_protocol=2,
        _use_new_zipfile_serialization=True,
    )
    return output.getvalue()


def _portable_checkpoint_bytes(serialized: bytes) -> bytes:
    """Project a Torch ZIP onto records stable across supported Torch builds.

    ``torch.save`` includes runtime metadata such as ``serialization_id`` and,
    in newer releases, alignment/version hints.  They do not participate in
    tensor reconstruction, but they make the raw archive hash build-specific.
    The production loader accepts the portable records alone.  Repacking those
    records with fixed ZIP metadata gives the exported checkpoint a truthful,
    byte-exact provenance identity across supported builds that preserve this
    reconstruction-record contract. Unknown records fail closed.
    """

    try:
        source = zipfile.ZipFile(BytesIO(serialized), "r")
    except (OSError, zipfile.BadZipFile) as error:
        raise RuntimeError("torch.save did not produce a readable ZIP") from error
    with source:
        members = source.infolist()
        roots = {
            member.filename.split("/", 1)[0]
            for member in members
            if "/" in member.filename
        }
        if len(roots) != 1:
            raise RuntimeError("torch.save ZIP must contain exactly one root")
        root = roots.pop()
        by_relative: dict[str, zipfile.ZipInfo] = {}
        for member in members:
            prefix = f"{root}/"
            if not member.filename.startswith(prefix):
                raise RuntimeError("torch.save ZIP contains an out-of-root record")
            relative = member.filename[len(prefix) :]
            if not relative or relative in by_relative:
                raise RuntimeError("torch.save ZIP contains an invalid record")
            by_relative[relative] = member
        required = ("data.pkl", "byteorder", "version")
        if any(name not in by_relative for name in required):
            raise RuntimeError("torch.save ZIP lacks a portable metadata record")
        data_records = sorted(
            (
                name
                for name in by_relative
                if re.fullmatch(r"data/[0-9]+", name) is not None
            ),
            key=lambda name: int(name.split("/", 1)[1]),
        )
        if not data_records:
            raise RuntimeError("torch.save ZIP lacks tensor storage records")
        selected = (*required, *data_records)
        unexpected = sorted(
            set(by_relative)
            - set(selected)
            - set(IGNORED_TORCH_RUNTIME_RECORDS)
        )
        if unexpected:
            raise RuntimeError(
                "torch.save ZIP contains unsupported runtime records: "
                + ", ".join(unexpected)
            )
        output = BytesIO()
        with zipfile.ZipFile(
            output,
            "w",
            compression=zipfile.ZIP_STORED,
            allowZip64=True,
        ) as target:
            target.comment = b""
            for relative in selected:
                info = zipfile.ZipInfo(
                    f"archive/{relative}",
                    date_time=(1980, 1, 1, 0, 0, 0),
                )
                info.compress_type = zipfile.ZIP_STORED
                info.create_system = 0
                info.create_version = 20
                info.extract_version = 20
                info.flag_bits = 0
                info.internal_attr = 0
                info.external_attr = 0
                info.extra = b""
                info.comment = b""
                target.writestr(info, source.read(by_relative[relative]))
        return output.getvalue()


def _checkpoint_bytes(payload: Mapping[str, object]) -> bytes:
    return _portable_checkpoint_bytes(_torch_checkpoint_bytes(payload))


def _symbolic_probabilities(
    eliminated: tuple[bool, ...],
    *,
    color: str,
) -> tuple[float, ...]:
    rule_count = len(eliminated)
    weights = tuple(
        0
        if eliminated[index]
        else (
            1 + (index % 3)
            if color == "white"
            else 1 + ((rule_count - index) % 4)
        )
        for index in range(rule_count)
    )
    total = sum(weights)
    return tuple(weight / total for weight in weights)


def _feature_record() -> FeatureRecord:
    rule_count = len(SYMBOLIC_RULE_IDS)
    white_eliminated = tuple(index % 17 == 0 for index in range(rule_count))
    black_eliminated = tuple(index % 19 == 0 for index in range(rule_count))
    return FeatureRecord(
        fen_before=(
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/"
            "PPPP1PPP/RNBQKB1R b KQkq - 1 2"
        ),
        move="b8c6",
        move_number=2,
        ply=3,
        player_color="black",
        history_san=("e4", "e5", "Nf3"),
        ordinary_legal_moves=ORDINARY_LEGAL_MOVES,
        clock_ms=None,
        symbolic_feature_version=SYMBOLIC_FEATURE_VERSION,
        symbolic_white_rule_probabilities=_symbolic_probabilities(
            white_eliminated,
            color="white",
        ),
        symbolic_black_rule_probabilities=_symbolic_probabilities(
            black_eliminated,
            color="black",
        ),
        symbolic_white_eliminated=white_eliminated,
        symbolic_black_eliminated=black_eliminated,
        public_evaluator_constraint=None,
    )


def _browser_observation(record: FeatureRecord) -> dict[str, object]:
    return {
        "fenBefore": record.fen_before,
        "move": record.move,
        "moveNumber": record.move_number,
        "ply": record.ply,
        "playerColor": record.player_color,
        "historySan": list(record.history_san),
        "ordinaryLegalMoveCount": len(record.ordinary_legal_moves),
        "symbolic": {
            "ruleIds": list(SYMBOLIC_RULE_IDS),
            "whiteProbabilities": list(
                record.symbolic_white_rule_probabilities
            ),
            "blackProbabilities": list(
                record.symbolic_black_rule_probabilities
            ),
            "whiteEliminated": list(record.symbolic_white_eliminated),
            "blackEliminated": list(record.symbolic_black_eliminated),
        },
    }


def _mode_result(
    *,
    directory: Path,
    mode: str,
    tokenizer: ObservationTokenizerV2,
    state: Mapping[str, torch.Tensor],
    record: FeatureRecord,
) -> tuple[dict[str, object], dict[str, object], list[int]]:
    payload = _checkpoint_payload(tokenizer, state, mode)
    checkpoint = directory / f"{mode}.pt"
    checkpoint_bytes = _checkpoint_bytes(payload)
    checkpoint.write_bytes(checkpoint_bytes)
    artifact_path = directory / f"{mode}.json"
    export_browser_artifact(checkpoint, artifact_path)
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    if not isinstance(artifact, dict):
        raise RuntimeError("browser exporter produced a non-object artifact")
    if artifact.get("sourceCheckpointSha256") != _sha256(checkpoint_bytes):
        raise RuntimeError(
            "browser exporter did not bind the exact portable checkpoint bytes"
        )

    predictor = load_checkpoint_predictor(checkpoint, device="cpu")
    prediction = predictor.predict(record)
    vocabulary = artifact["drawbackVocabulary"]
    if not isinstance(vocabulary, list) or any(
        not isinstance(item, str) for item in vocabulary
    ):
        raise RuntimeError("browser exporter produced an invalid vocabulary")
    expected = {
        "white": [
            _portable_expected_probability(
                prediction.white_drawback_probabilities[drawback_id]
            )
            for drawback_id in vocabulary
        ],
        "black": [
            _portable_expected_probability(
                prediction.black_drawback_probabilities[drawback_id]
            )
            for drawback_id in vocabulary
        ],
    }
    encoded, length = tokenizer.encode(
        PublicSequenceObservation(record.history_san, record.move),
        mask_current=mode == "masked-current-v2",
    )
    return artifact, expected, list(encoded[:length])


def build_fixture() -> dict[str, object]:
    """Build one deterministic fixture from the production Python paths."""

    torch.use_deterministic_algorithms(True)
    tokenizer = _tokenizer()
    state = _intentional_state(tokenizer)
    record = _feature_record()
    artifacts: dict[str, dict[str, object]] = {}
    expected: dict[str, dict[str, object]] = {}
    token_indices: dict[str, list[int]] = {}
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        for mode in MODES:
            artifact, prediction, encoded = _mode_result(
                directory=directory,
                mode=mode,
                tokenizer=tokenizer,
                state=state,
                record=record,
            )
            artifacts[mode] = artifact
            expected[mode] = prediction
            token_indices[mode] = encoded

    exact = artifacts["exact-current-v2"]
    masked = artifacts["masked-current-v2"]
    exact_common = deepcopy(exact)
    masked_common = deepcopy(masked)
    exact_common.pop("sourceCheckpointSha256")
    exact_common.pop("sequenceObservationMode")
    masked_common.pop("sourceCheckpointSha256")
    masked_common.pop("sequenceObservationMode")
    if exact_common != masked_common:
        raise RuntimeError(
            "observation-mode browser artifacts differ beyond their bindings"
        )
    input_spec = _input_spec(tokenizer)
    source_sha256 = {
        relative: _sha256(_normalized_source_bytes(ROOT / relative))
        for relative in SOURCE_DEPENDENCIES
    }
    return {
        "format": GENERATOR_FORMAT,
        "formatVersion": GENERATOR_FORMAT_VERSION,
        "bindings": {
            "checkpointProvenance": {
                "algorithm": "sha256",
                "encoding": PORTABLE_CHECKPOINT_ENCODING,
                "exactSha256": exact["sourceCheckpointSha256"],
                "maskedSha256": masked["sourceCheckpointSha256"],
            },
            "exactArtifactSha256": _sha256(
                canonical_artifact_bytes(exact)
            ),
            "inputSpecSha256": _sha256(
                _canonical_json_bytes(input_spec)
            ),
            "maskedArtifactSha256": _sha256(
                canonical_artifact_bytes(masked)
            ),
            "sourceSha256": source_sha256,
        },
        "inputSpec": input_spec,
        "artifact": exact,
        "maskedArtifactDelta": {
            "sequenceObservationMode": masked[
                "sequenceObservationMode"
            ],
            "sourceCheckpointSha256": masked[
                "sourceCheckpointSha256"
            ],
        },
        "cases": [
            {
                "observation": _browser_observation(record),
                "expectedTokenIndices": {
                    "exact": token_indices["exact-current-v2"],
                    "masked": token_indices["masked-current-v2"],
                },
                "expected": {
                    "exact": expected["exact-current-v2"],
                    "masked": expected["masked-current-v2"],
                },
            }
        ],
    }


def generate_fixture_bytes() -> bytes:
    return _canonical_json_bytes(build_fixture())


def _normalized_fixture_bytes(path: Path) -> bytes:
    return (
        path.read_text(encoding="utf-8")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .encode("utf-8")
    )


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate or verify the v22 Python/browser golden fixture."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if the committed fixture differs from current generation",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=FIXTURE_PATH,
        help="fixture path (defaults to the committed web fixture)",
    )
    parsed = parser.parse_args(arguments)
    rendered = generate_fixture_bytes()
    output = parsed.output.resolve()
    if parsed.check:
        try:
            current = _normalized_fixture_bytes(output)
        except OSError as error:
            parser.error(f"cannot read fixture: {error}")
        if current != rendered:
            print(
                "v22 cross-runtime fixture is stale: "
                f"expected={_sha256(rendered)} actual={_sha256(current)}",
                file=sys.stderr,
            )
            return 1
        print(output)
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(rendered)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
