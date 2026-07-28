"""Preregistered two-fold validation for schema-9 opportunity ablations.

This module is deliberately separate from generic candidate selection.  It
freezes one experiment protocol, authenticates every referenced artifact, and
models the only allowed transitions:

``validation A -> validation B -> one sealed test``.

The sealed-test path is not resolved, opened, hashed, or stat'ed until a
successful validation-B authorization has been replayed from its inputs.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
from typing import Any, Mapping, Sequence

from .capturable_baseline import (
    _canonical_json,
    evaluate_capturable,
    tensorize,
)
from .capturable_blend_contract import blend_reliability_checks
from .capturable_candidate_selection import (
    _selection_report,
    _validated_candidate,
    load_treatment_comparison,
)
from .capturable_experiment import (
    _load_bound_selection_checkpoint,
    _load_stable_capturable_dataset,
    _training_config_from_json,
)
from .capturable_records import (
    CAPTURABLE_OPPORTUNITY_FIELDS,
    CAPTURABLE_OPPORTUNITY_SHAPE,
    CAPTURABLE_RULE_IDS,
    CapturableDatasetError,
    CapturableDatasetRow,
)
from .durable_publish import publish_bytes_durable


OPPORTUNITY_WORKFLOW_VERSION = 2
STAGE_A_FORMAT = "drawbackguesser-schema9-opportunity-stage-a"
STAGE_B_FORMAT = "drawbackguesser-schema9-opportunity-stage-b"
SEALED_TEST_FORMAT = "drawbackguesser-schema9-opportunity-sealed-test"
CONSUMPTION_FORMAT = (
    "drawbackguesser-schema9-opportunity-sealed-test-consumption"
)
CORPUS_LEDGER_FORMAT = "drawbackguesser-schema9-corpus-ledger"
CORPUS_LEDGER_VERSION = 1
CORPUS_LEDGER_SPLITS = (
    "train",
    "validation-a",
    "validation-b",
    "test",
)

EXPERIMENT_DOMAIN = "capturable25-schema9-opportunity-v1"
VALIDATION_GAME_COUNT = 2_500
MODEL_SEEDS = (3_685_459_371, 480_184_104, 3_192_956_725)
SEED_STREAMS = ("label", "gameplay", "parameters")
SPLIT_SEED_ROOTS = {
    "train": (1_261_462_769, 242_269_024, 1_837_697_911),
    "validation-a": (2_069_246_597, 1_391_196_133, 2_739_675_947),
    "validation-b": (3_786_384_219, 3_547_865_132, 2_689_552_677),
    "sealed-test": (2_033_321_041, 1_354_035_545, 4_189_758_462),
}
FROZEN_CONFIG = {
    "epochs": 8,
    "batch_size": 256,
    "hidden_dimension": 128,
    "trigger_row_multiplier": 1.0,
}
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_FULL_GIT_COMMIT = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_SCHEDULE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:+/-]{0,199}\Z")
_REFERENCE_KEYS = frozenset(
    {
        "artifactpath",
        "checkpointfile",
        "consumptionmarker",
        "file",
        "path",
        "selectiondirectory",
        "selectionreport",
    }
)
_WINDOWS_RESERVED_BASENAME = re.compile(
    r"(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?\Z",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class _Pair:
    comparison_path: Path
    comparison: Mapping[str, Any]
    comparison_sha256: str
    control_report: Mapping[str, Any]
    treatment_report: Mapping[str, Any]

    @property
    def seed(self) -> int:
        return int(self.comparison["control"]["seed"])

    @property
    def control_metrics(self) -> Mapping[str, Any]:
        return self.control_report["validation"]

    @property
    def treatment_metrics(self) -> Mapping[str, Any]:
        return self.treatment_report["validation"]


@dataclass(frozen=True)
class _LoadedPair:
    control: tuple[Path, Any, Mapping[str, Any], str]
    treatment: tuple[Path, Any, Mapping[str, Any], str]
    source_game_ids: frozenset[str]


@dataclass(frozen=True)
class _CorpusLedger:
    path: Path
    artifact: Mapping[str, Any]
    sha256: str
    splits: Mapping[str, Mapping[str, Any]]


def _protocol() -> Mapping[str, Any]:
    return {
        "domain": EXPERIMENT_DOMAIN,
        "schemaVersion": 9,
        "opportunityFeatureVersion": 1,
        "validationGamesPerSplit": VALIDATION_GAME_COUNT,
        "modelSeeds": list(MODEL_SEEDS),
        "seedStreams": list(SEED_STREAMS),
        "splitSeedRoots": {
            split: list(roots)
            for split, roots in SPLIT_SEED_ROOTS.items()
        },
        "trainingConfig": dict(FROZEN_CONFIG),
        "medianPolicy": (
            "lower median of promoted pairs by Top-1 delta; ties use "
            "lower NLL delta, then lower model seed"
        ),
    }


def _assert_seed_derivation() -> None:
    for split, expected in SPLIT_SEED_ROOTS.items():
        measured = tuple(
            int.from_bytes(
                hashlib.sha256(
                    f"{EXPERIMENT_DOMAIN}/{split}/{stream}".encode("utf-8")
                ).digest()[:4],
                "big",
            )
            for stream in SEED_STREAMS
        )
        if measured != expected:
            raise RuntimeError(f"{split} seed roots do not match the protocol")


_assert_seed_derivation()


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise CapturableDatasetError(f"{label} must be lowercase SHA-256")
    return value


def _metric(
    value: Any,
    label: str,
    *,
    maximum: float | None = None,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
        or (maximum is not None and float(value) > maximum)
    ):
        raise CapturableDatasetError(f"{label} is invalid")
    return float(value)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CapturableDatasetError(f"{label} must be an object")
    return value


def _exact_keys(
    value: Mapping[str, Any],
    expected: set[str] | frozenset[str],
    label: str,
) -> None:
    observed = set(value)
    if observed != set(expected):
        missing = sorted(set(expected) - observed)
        unexpected = sorted(observed - set(expected))
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unexpected:
            details.append("unexpected " + ", ".join(unexpected))
        raise CapturableDatasetError(
            f"{label} fields are invalid: {'; '.join(details)}"
        )


def _json_equal(left: Any, right: Any) -> bool:
    return _canonical_json(left) == _canonical_json(right)


def _assert_path_free_artifact(value: Any, label: str = "artifact") -> None:
    if isinstance(value, str):
        if (
            PureWindowsPath(value).is_absolute()
            or PurePosixPath(value).is_absolute()
        ):
            raise CapturableDatasetError(f"{label} contains an absolute path")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_path_free_artifact(key, f"{label} key")
            if (
                isinstance(key, str)
                and (
                    key.casefold() in _REFERENCE_KEYS
                    or key.casefold().endswith(
                        (
                            "path",
                            "file",
                            "directory",
                            "report",
                            "marker",
                        )
                    )
                )
            ):
                _safe_basename(item, f"{label}.{key}")
            _assert_path_free_artifact(item, f"{label}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_path_free_artifact(item, f"{label}[{index}]")


def _safe_basename(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or PureWindowsPath(value).name != value
        or PurePosixPath(value).name != value
        or PureWindowsPath(value).drive
        or ":" in value
        or value[-1] in {" ", "."}
        or any(ord(character) < 32 for character in value)
        or _WINDOWS_RESERVED_BASENAME.fullmatch(value) is not None
        or value in {".", ".."}
    ):
        raise CapturableDatasetError(f"{label} must be a path-free basename")
    return value


def _nonnegative_count(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CapturableDatasetError(
            f"{label} must be a non-negative integer"
        )
    return value


def _positive_count(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CapturableDatasetError(f"{label} must be a positive integer")
    return value


def _validate_input_identity(
    identity: Mapping[str, Any],
    label: str,
    *,
    expected_games: int | None = None,
) -> None:
    _positive_count(identity.get("rows"), f"{label} rows")
    games = _positive_count(identity.get("games"), f"{label} games")
    if expected_games is not None and games != expected_games:
        raise CapturableDatasetError(
            f"{label} must contain exactly {expected_games} games"
        )
    if "path" in identity:
        _safe_basename(identity.get("path"), f"{label} path")
        _sha256(identity.get("sha256"), f"{label} SHA-256")
        return
    sources = identity.get("sources")
    if not isinstance(sources, list) or not sources:
        raise CapturableDatasetError(f"{label} sources are invalid")
    for index, source_value in enumerate(sources):
        source = _mapping(source_value, f"{label} source {index}")
        _safe_basename(source.get("path"), f"{label} source {index} path")
        _sha256(
            source.get("sha256"),
            f"{label} source {index} SHA-256",
        )
        _positive_count(source.get("rows"), f"{label} source {index} rows")
        _positive_count(source.get("games"), f"{label} source {index} games")


def _canonical_set_sha256(
    values: Sequence[str] | Sequence[int],
) -> str:
    return hashlib.sha256(_canonical_json(list(values))).hexdigest()


def _ledger_string_set(value: Any, label: str) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
        or len(set(value)) != len(value)
        or value != sorted(value)
    ):
        raise CapturableDatasetError(
            f"{label} must be a sorted unique non-empty string set"
        )
    return tuple(value)


def _ledger_seed_set(value: Any, label: str) -> tuple[int, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(
            isinstance(item, bool)
            or not isinstance(item, int)
            or item < 0
            or item > 0xFFFFFFFF
            for item in value
        )
        or len(set(value)) != len(value)
        or value != sorted(value)
    ):
        raise CapturableDatasetError(
            f"{label} must be a sorted unique unsigned 32-bit seed set"
        )
    return tuple(value)


def _ledger_receipt(
    value: Any,
    label: str,
) -> Mapping[str, Any]:
    receipt = _mapping(value, label)
    _exact_keys(receipt, {"sha256", "bytes"}, label)
    _sha256(receipt.get("sha256"), f"{label} SHA-256")
    _positive_count(receipt.get("bytes"), f"{label} bytes")
    return receipt


def _ledger_split(
    value: Any,
    expected_split: str,
) -> Mapping[str, Any]:
    label = f"corpus ledger {expected_split}"
    split = _mapping(value, label)
    _exact_keys(
        split,
        {
            "split",
            "scheduleId",
            "seedRoots",
            "producerEngineCommit",
            "generatorReceipts",
            "sourceTrace",
            "converted",
        },
        label,
    )
    if split.get("split") != expected_split:
        raise CapturableDatasetError(
            f"{label} is outside the exact split order"
        )
    schedule_id = split.get("scheduleId")
    if (
        not isinstance(schedule_id, str)
        or _SCHEDULE_ID.fullmatch(schedule_id) is None
        or ".." in schedule_id
        or "\\" in schedule_id
        or PurePosixPath(schedule_id).is_absolute()
        or PureWindowsPath(schedule_id).is_absolute()
    ):
        raise CapturableDatasetError(
            f"{label} scheduleId is not canonical"
        )
    workflow_split = (
        "sealed-test" if expected_split == "test" else expected_split
    )
    roots = split.get("seedRoots")
    if (
        not isinstance(roots, list)
        or any(
            isinstance(root, bool)
            or not isinstance(root, int)
            or root < 0
            or root > 0xFFFFFFFF
            for root in roots
        )
        or roots != list(SPLIT_SEED_ROOTS[workflow_split])
    ):
        raise CapturableDatasetError(
            f"{label} seedRoots do not match the frozen schedule"
        )
    producer = split.get("producerEngineCommit")
    if (
        not isinstance(producer, str)
        or _FULL_GIT_COMMIT.fullmatch(producer) is None
    ):
        raise CapturableDatasetError(
            f"{label} producerEngineCommit is invalid"
        )

    receipts = _mapping(
        split.get("generatorReceipts"),
        f"{label} generatorReceipts",
    )
    _exact_keys(
        receipts,
        {"launch", "completion"},
        f"{label} generatorReceipts",
    )
    _ledger_receipt(receipts.get("launch"), f"{label} launch receipt")
    _ledger_receipt(
        receipts.get("completion"),
        f"{label} completion receipt",
    )

    source = _mapping(split.get("sourceTrace"), f"{label} sourceTrace")
    _exact_keys(
        source,
        {
            "sha256",
            "bytes",
            "games",
            "zeroPlyGames",
            "gameIds",
            "simulationSeeds",
            "gameIdSetSha256",
            "simulationSeedSetSha256",
            "labelCountsByColor",
        },
        f"{label} sourceTrace",
    )
    _sha256(source.get("sha256"), f"{label} source SHA-256")
    _positive_count(source.get("bytes"), f"{label} source bytes")
    source_games = _positive_count(
        source.get("games"),
        f"{label} source games",
    )
    zero_ply_games = _nonnegative_count(
        source.get("zeroPlyGames"),
        f"{label} zero-ply games",
    )
    source_game_ids = _ledger_string_set(
        source.get("gameIds"),
        f"{label} source game IDs",
    )
    source_seeds = _ledger_seed_set(
        source.get("simulationSeeds"),
        f"{label} source simulation seeds",
    )
    if (
        len(source_game_ids) != source_games
        or len(source_seeds) != source_games
        or zero_ply_games > source_games
        or source.get("gameIdSetSha256")
        != _canonical_set_sha256(source_game_ids)
        or source.get("simulationSeedSetSha256")
        != _canonical_set_sha256(source_seeds)
    ):
        raise CapturableDatasetError(
            f"{label} source set commitments are inconsistent"
        )
    _sha256(
        source.get("gameIdSetSha256"),
        f"{label} source game-ID set SHA-256",
    )
    _sha256(
        source.get("simulationSeedSetSha256"),
        f"{label} source seed set SHA-256",
    )
    counts = _mapping(
        source.get("labelCountsByColor"),
        f"{label} label counts",
    )
    _exact_keys(counts, {"white", "black"}, f"{label} label counts")
    if source_games % len(CAPTURABLE_RULE_IDS) != 0:
        raise CapturableDatasetError(
            f"{label} cannot be exactly label-balanced"
        )
    expected_count = source_games // len(CAPTURABLE_RULE_IDS)
    for color in ("white", "black"):
        color_counts = _mapping(
            counts.get(color),
            f"{label} {color} label counts",
        )
        _exact_keys(
            color_counts,
            set(CAPTURABLE_RULE_IDS),
            f"{label} {color} label counts",
        )
        if any(
            isinstance(count, bool)
            or not isinstance(count, int)
            or count != expected_count
            for count in color_counts.values()
        ):
            raise CapturableDatasetError(
                f"{label} is not exactly label-balanced for {color}"
            )

    converted = _mapping(split.get("converted"), f"{label} converted")
    _exact_keys(
        converted,
        {
            "sha256",
            "bytes",
            "rows",
            "games",
            "gameIds",
            "simulationSeeds",
            "gameIdSetSha256",
            "simulationSeedSetSha256",
        },
        f"{label} converted",
    )
    _sha256(converted.get("sha256"), f"{label} converted SHA-256")
    _positive_count(converted.get("bytes"), f"{label} converted bytes")
    _positive_count(converted.get("rows"), f"{label} converted rows")
    converted_games = _positive_count(
        converted.get("games"),
        f"{label} converted games",
    )
    if (
        expected_split != "train"
        and converted_games != VALIDATION_GAME_COUNT
    ):
        raise CapturableDatasetError(
            f"{label} must contain exactly {VALIDATION_GAME_COUNT} "
            "converted games"
        )
    converted_game_ids = _ledger_string_set(
        converted.get("gameIds"),
        f"{label} converted game IDs",
    )
    converted_seeds = _ledger_seed_set(
        converted.get("simulationSeeds"),
        f"{label} converted simulation seeds",
    )
    if (
        converted_games != source_games - zero_ply_games
        or len(converted_game_ids) != converted_games
        or len(converted_seeds) != converted_games
        or not set(converted_game_ids).issubset(source_game_ids)
        or not set(converted_seeds).issubset(source_seeds)
        or converted.get("gameIdSetSha256")
        != _canonical_set_sha256(converted_game_ids)
        or converted.get("simulationSeedSetSha256")
        != _canonical_set_sha256(converted_seeds)
    ):
        raise CapturableDatasetError(
            f"{label} converted set commitments are inconsistent"
        )
    _sha256(
        converted.get("gameIdSetSha256"),
        f"{label} converted game-ID set SHA-256",
    )
    _sha256(
        converted.get("simulationSeedSetSha256"),
        f"{label} converted seed set SHA-256",
    )
    return split


def _partition_assignment_sha256(
    splits: Mapping[str, Mapping[str, Any]],
    field: str,
) -> str:
    return hashlib.sha256(
        _canonical_json(
            [
                {
                    "split": split,
                    "values": list(
                        _mapping(
                            splits[split].get("sourceTrace"),
                            f"{split} sourceTrace",
                        )[field]
                    ),
                }
                for split in CORPUS_LEDGER_SPLITS
            ]
        )
    ).hexdigest()


def _load_corpus_ledger(
    path: Path,
    expected_sha256: str,
) -> _CorpusLedger:
    expected_digest = _sha256(
        expected_sha256,
        "expected corpus ledger SHA-256",
    )
    if path.is_symlink():
        raise CapturableDatasetError(
            f"{path.name} corpus ledger must not be a symbolic link"
        )
    artifact, artifact_sha256 = _selection_report(path)
    if artifact_sha256 != expected_digest:
        raise CapturableDatasetError(
            f"{path.name} corpus ledger SHA-256 is inconsistent"
        )
    _assert_path_free_artifact(artifact, f"{path.name} corpus ledger")
    _exact_keys(
        artifact,
        {
            "format",
            "version",
            "identity",
            "scheduleContract",
            "opportunityContract",
            "splits",
            "partition",
            "contentSha256",
        },
        f"{path.name} corpus ledger",
    )
    if (
        artifact.get("format") != CORPUS_LEDGER_FORMAT
        or type(artifact.get("version")) is not int
        or artifact.get("version") != CORPUS_LEDGER_VERSION
    ):
        raise CapturableDatasetError(
            f"{path.name} corpus ledger format or version is invalid"
        )
    declared_content_sha256 = _sha256(
        artifact.get("contentSha256"),
        f"{path.name} corpus ledger content SHA-256",
    )
    content = {
        key: value
        for key, value in artifact.items()
        if key != "contentSha256"
    }
    if hashlib.sha256(_canonical_json(content)).hexdigest() != (
        declared_content_sha256
    ):
        raise CapturableDatasetError(
            f"{path.name} corpus ledger self-hash is inconsistent"
        )

    identity = _mapping(
        artifact.get("identity"),
        f"{path.name} corpus ledger identity",
    )
    _exact_keys(
        identity,
        {
            "guesserCommit",
            "converterEngineCommit",
            "producerConverterPolicy",
        },
        f"{path.name} corpus ledger identity",
    )
    guesser_commit = identity.get("guesserCommit")
    converter_commit = identity.get("converterEngineCommit")
    if (
        not isinstance(guesser_commit, str)
        or _FULL_GIT_COMMIT.fullmatch(guesser_commit) is None
        or not isinstance(converter_commit, str)
        or _FULL_GIT_COMMIT.fullmatch(converter_commit) is None
        or identity.get("producerConverterPolicy") != "exact/v1"
    ):
        raise CapturableDatasetError(
            f"{path.name} corpus ledger identity is not independently "
            "verifiable under exact/v1"
        )

    schedule_contract = _mapping(
        artifact.get("scheduleContract"),
        f"{path.name} schedule contract",
    )
    _exact_keys(
        schedule_contract,
        {"authorityId", "seedStreams"},
        f"{path.name} schedule contract",
    )
    if schedule_contract != {
        "authorityId": "capturable25-schema9-opportunity/v1",
        "seedStreams": list(SEED_STREAMS),
    } or (
        not isinstance(schedule_contract.get("seedStreams"), list)
        or any(
            not isinstance(stream, str)
            for stream in schedule_contract.get("seedStreams", ())
        )
    ):
        raise CapturableDatasetError(
            f"{path.name} schedule contract is not the frozen schema-9 "
            "opportunity schedule"
        )

    contract = _mapping(
        artifact.get("opportunityContract"),
        f"{path.name} opportunity contract",
    )
    _exact_keys(
        contract,
        {
            "authorityId",
            "symbolicFeatureVersion",
            "opportunityFeatureVersion",
            "ruleIds",
            "fields",
            "shape",
        },
        f"{path.name} opportunity contract",
    )
    if (
        contract.get("authorityId") != "capturable-king/v1"
        or type(contract.get("symbolicFeatureVersion")) is not int
        or contract.get("symbolicFeatureVersion") != 9
        or type(contract.get("opportunityFeatureVersion")) is not int
        or contract.get("opportunityFeatureVersion") != 1
        or not isinstance(contract.get("ruleIds"), list)
        or contract.get("ruleIds") != list(CAPTURABLE_RULE_IDS)
        or any(
            not isinstance(rule_id, str)
            for rule_id in contract.get("ruleIds", ())
        )
        or not isinstance(contract.get("fields"), list)
        or contract.get("fields")
        != list(CAPTURABLE_OPPORTUNITY_FIELDS)
        or any(
            not isinstance(field, str)
            for field in contract.get("fields", ())
        )
        or not isinstance(contract.get("shape"), list)
        or any(
            type(dimension) is not int
            for dimension in contract.get("shape", ())
        )
        or contract.get("shape") != list(CAPTURABLE_OPPORTUNITY_SHAPE)
    ):
        raise CapturableDatasetError(
            f"{path.name} opportunity contract is not the frozen schema-9 "
            "contract"
        )

    raw_splits = artifact.get("splits")
    if not isinstance(raw_splits, list) or len(raw_splits) != len(
        CORPUS_LEDGER_SPLITS
    ):
        raise CapturableDatasetError(
            f"{path.name} corpus ledger split list is invalid"
        )
    splits = {
        split: _ledger_split(raw_splits[index], split)
        for index, split in enumerate(CORPUS_LEDGER_SPLITS)
    }
    if len({split["scheduleId"] for split in splits.values()}) != len(splits):
        raise CapturableDatasetError(
            f"{path.name} corpus ledger schedule IDs must be unique"
        )
    if any(
        split["producerEngineCommit"] != converter_commit
        for split in splits.values()
    ):
        raise CapturableDatasetError(
            f"{path.name} exact/v1 producer identity is inconsistent"
        )
    for field, description in (
        ("gameIds", "game ID"),
        ("simulationSeeds", "simulation seed"),
    ):
        seen: set[Any] = set()
        for split_name in CORPUS_LEDGER_SPLITS:
            values = set(
                _mapping(
                    splits[split_name]["sourceTrace"],
                    f"{split_name} sourceTrace",
                )[field]
            )
            if seen & values:
                raise CapturableDatasetError(
                    f"{path.name} corpus ledger {description} sets overlap "
                    "across splits"
                )
            seen.update(values)

    partition = _mapping(
        artifact.get("partition"),
        f"{path.name} corpus partition",
    )
    _exact_keys(
        partition,
        {
            "games",
            "gameIdAssignmentsSha256",
            "simulationSeedAssignmentsSha256",
        },
        f"{path.name} corpus partition",
    )
    total_games = sum(
        int(_mapping(split["sourceTrace"], "sourceTrace")["games"])
        for split in splits.values()
    )
    partition_games = _positive_count(
        partition.get("games"),
        "corpus partition games",
    )
    if (
        partition_games != total_games
        or partition.get("gameIdAssignmentsSha256")
        != _partition_assignment_sha256(splits, "gameIds")
        or partition.get("simulationSeedAssignmentsSha256")
        != _partition_assignment_sha256(splits, "simulationSeeds")
    ):
        raise CapturableDatasetError(
            f"{path.name} corpus partition commitments are inconsistent"
        )
    _sha256(
        partition.get("gameIdAssignmentsSha256"),
        "corpus game-ID assignment SHA-256",
    )
    _sha256(
        partition.get("simulationSeedAssignmentsSha256"),
        "corpus seed assignment SHA-256",
    )
    return _CorpusLedger(
        path=path.resolve(),
        artifact=artifact,
        sha256=artifact_sha256,
        splits=splits,
    )


def _corpus_ledger_reference(
    ledger: _CorpusLedger,
) -> Mapping[str, Any]:
    identity = _mapping(
        ledger.artifact.get("identity"),
        "corpus ledger identity",
    )
    return {
        "file": _safe_basename(ledger.path.name, "corpus ledger file"),
        "sha256": ledger.sha256,
        "contentSha256": ledger.artifact["contentSha256"],
        "guesserCommit": identity["guesserCommit"],
        "converterEngineCommit": identity["converterEngineCommit"],
        "producerConverterPolicy": identity["producerConverterPolicy"],
    }


def _single_input_identity(
    identity: Mapping[str, Any],
    label: str,
) -> Mapping[str, Any]:
    _validate_input_identity(identity, label)
    if "path" in identity:
        return identity
    sources = identity.get("sources")
    if not isinstance(sources, list) or len(sources) != 1:
        raise CapturableDatasetError(
            f"{label} must bind exactly one corpus-ledger dataset"
        )
    return _mapping(sources[0], f"{label} source")


def _assert_input_matches_ledger(
    identity: Mapping[str, Any],
    split: Mapping[str, Any],
    label: str,
) -> Mapping[str, Any]:
    declared = _single_input_identity(identity, label)
    converted = _mapping(split.get("converted"), f"{label} ledger split")
    if (
        declared.get("sha256") != converted.get("sha256")
        or declared.get("rows") != converted.get("rows")
        or declared.get("games") != converted.get("games")
    ):
        raise CapturableDatasetError(
            f"{label} does not match the authenticated corpus ledger"
        )
    file_name = _safe_basename(declared.get("path"), f"{label} file")
    return {
        "file": file_name,
        "sha256": converted["sha256"],
        "rows": converted["rows"],
        "games": converted["games"],
        "gameIdSetSha256": converted["gameIdSetSha256"],
        "simulationSeedSetSha256": converted["simulationSeedSetSha256"],
        "scheduleId": split["scheduleId"],
        "seedRoots": split["seedRoots"],
    }


def _hybrid(metrics: Mapping[str, Any], label: str) -> Mapping[str, Any]:
    return _mapping(metrics.get("hybrid"), f"{label} hybrid metrics")


def _candidate_report(
    comparison_path: Path,
    entry: Mapping[str, Any],
) -> Mapping[str, Any]:
    directory = _safe_basename(
        entry.get("selectionDirectory"),
        "comparison candidate directory",
    )
    report_name = _safe_basename(
        entry.get("selectionReport"),
        "comparison candidate report",
    )
    candidate, report = _validated_candidate(
        comparison_path.parent / directory / report_name
    )
    candidate["trainInput"] = report["inputs"]["train"]
    if candidate != entry:
        raise CapturableDatasetError(
            "comparison candidate changed after authentication"
        )
    return report


def _same_parent_path(path: Path, parent: Path, label: str) -> Path:
    if path.is_symlink():
        raise CapturableDatasetError(f"{label} must not be a symbolic link")
    resolved = path.resolve()
    if resolved.parent != parent.resolve() or resolved.name != path.name:
        raise CapturableDatasetError(
            f"{label} must be a direct sibling of its workflow artifact"
        )
    return resolved


def _authenticate_pair(path: Path) -> _Pair:
    comparison, comparison_sha256 = load_treatment_comparison(path)
    _assert_path_free_artifact(comparison, f"{path.name} comparison")
    treatments = comparison.get("treatments")
    if not isinstance(treatments, list) or len(treatments) != 1:
        raise CapturableDatasetError(
            f"{path.name} must contain exactly one matched treatment"
        )
    control = _mapping(comparison.get("control"), "comparison control")
    treatment = _mapping(treatments[0], "comparison treatment")
    if comparison.get("bestTreatment") != treatment:
        raise CapturableDatasetError(
            f"{path.name} does not bind its sole treatment"
        )
    control_contract = _mapping(
        control.get("opportunityContract"),
        "control opportunity contract",
    )
    treatment_contract = _mapping(
        treatment.get("opportunityContract"),
        "treatment opportunity contract",
    )
    if (
        control_contract.get("opportunityMode") != "zero-ablation"
        or treatment_contract.get("opportunityMode") != "public-exact"
        or control.get("seed") != treatment.get("seed")
    ):
        raise CapturableDatasetError(
            f"{path.name} is not a same-seed schema-9 opportunity pair"
        )
    control_report = _candidate_report(path, control)
    treatment_report = _candidate_report(path, treatment)
    _assert_path_free_artifact(
        control_report,
        f"{path.name} control report",
    )
    _assert_path_free_artifact(
        treatment_report,
        f"{path.name} treatment report",
    )
    control_config = _mapping(control_report.get("config"), "control config")
    treatment_config = _mapping(
        treatment_report.get("config"),
        "treatment config",
    )
    if control_config != treatment_config:
        raise CapturableDatasetError(
            f"{path.name} pair configurations are not identical"
        )
    seed = control_config.get("seed")
    if seed != control.get("seed") or seed not in MODEL_SEEDS:
        raise CapturableDatasetError(f"{path.name} model seed is not frozen")
    for key, expected in FROZEN_CONFIG.items():
        if control_config.get(key) != expected:
            raise CapturableDatasetError(
                f"{path.name} {key} does not match the frozen configuration"
            )
    validation_input = _mapping(
        comparison.get("validationInput"),
        "comparison validation input",
    )
    _validate_input_identity(
        validation_input,
        f"{path.name} validation A",
        expected_games=VALIDATION_GAME_COUNT,
    )
    _validate_input_identity(
        _mapping(control.get("trainInput"), "control training input"),
        f"{path.name} training input",
    )
    control_metrics = _mapping(
        control_report.get("validation"),
        "control validation metrics",
    )
    treatment_metrics = _mapping(
        treatment_report.get("validation"),
        "treatment validation metrics",
    )
    # A self-comparison validates every existing metric gate and makes exact
    # hard-elimination authority a mandatory precondition independent of the
    # treatment's direction.
    for label, metrics in (
        ("control", control_metrics),
        ("treatment", treatment_metrics),
    ):
        checks = blend_reliability_checks(metrics, metrics, True)
        if not all(checks.values()):
            raise CapturableDatasetError(
                f"{path.name} {label} fails exact reliability authority"
            )
    return _Pair(
        comparison_path=path.resolve(),
        comparison=comparison,
        comparison_sha256=comparison_sha256,
        control_report=control_report,
        treatment_report=treatment_report,
    )


def _delta(pair: _Pair) -> Mapping[str, Any]:
    control = pair.control_metrics
    treatment = pair.treatment_metrics
    control_hybrid = _hybrid(control, "control")
    treatment_hybrid = _hybrid(treatment, "treatment")

    def difference(
        key: str,
        *,
        maximum: float | None = None,
    ) -> float:
        return _metric(
            treatment_hybrid.get(key),
            f"treatment {key}",
            maximum=maximum,
        ) - _metric(
            control_hybrid.get(key),
            f"control {key}",
            maximum=maximum,
        )

    control_horizons = _mapping(
        control_hybrid.get("accuracy_after_moves"),
        "control move horizons",
    )
    treatment_horizons = _mapping(
        treatment_hybrid.get("accuracy_after_moves"),
        "treatment move horizons",
    )
    if set(control_horizons) != {"5", "10", "15", "20"} or set(
        treatment_horizons
    ) != set(control_horizons):
        raise CapturableDatasetError("pair move horizons are incompatible")
    horizons = {
        horizon: _metric(
            treatment_horizons[horizon],
            f"treatment horizon {horizon}",
            maximum=1.0,
        )
        - _metric(
            control_horizons[horizon],
            f"control horizon {horizon}",
            maximum=1.0,
        )
        for horizon in ("5", "10", "15", "20")
    }
    control_colors = _mapping(
        control.get("hybridByColor"),
        "control color metrics",
    )
    treatment_colors = _mapping(
        treatment.get("hybridByColor"),
        "treatment color metrics",
    )
    color_deltas: dict[str, Mapping[str, float]] = {}
    for color in ("white", "black"):
        control_color = _mapping(
            control_colors.get(color),
            f"control {color} metrics",
        )
        treatment_color = _mapping(
            treatment_colors.get(color),
            f"treatment {color} metrics",
        )
        color_deltas[color] = {
            "top1": _metric(
                treatment_color.get("game_normalized_top_1_accuracy"),
                f"treatment {color} Top-1",
                maximum=1.0,
            )
            - _metric(
                control_color.get("game_normalized_top_1_accuracy"),
                f"control {color} Top-1",
                maximum=1.0,
            ),
            "top3": _metric(
                treatment_color.get("game_normalized_top_3_accuracy"),
                f"treatment {color} Top-3",
                maximum=1.0,
            )
            - _metric(
                control_color.get("game_normalized_top_3_accuracy"),
                f"control {color} Top-3",
                maximum=1.0,
            ),
            "negativeLogLikelihood": _metric(
                treatment_color.get(
                    "game_normalized_negative_log_likelihood"
                ),
                f"treatment {color} NLL",
            )
            - _metric(
                control_color.get("game_normalized_negative_log_likelihood"),
                f"control {color} NLL",
            ),
        }
    artifact = {
        "top1": difference(
            "game_normalized_top_1_accuracy",
            maximum=1.0,
        ),
        "top3": difference(
            "game_normalized_top_3_accuracy",
            maximum=1.0,
        ),
        "top5": difference(
            "game_normalized_top_5_accuracy",
            maximum=1.0,
        ),
        "negativeLogLikelihood": difference(
            "game_normalized_negative_log_likelihood"
        ),
        "brier": difference("game_normalized_brier_score"),
        "calibration": difference(
            "expected_calibration_error",
            maximum=1.0,
        ),
        "moveHorizons": horizons,
        "colors": color_deltas,
    }
    return artifact


def _mean(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values)


def _aggregate(entries: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    deltas = [_mapping(entry.get("deltas"), "pair deltas") for entry in entries]
    horizons = {
        horizon: _mean(
            [
                float(
                    _mapping(delta["moveHorizons"], "move horizons")[horizon]
                )
                for delta in deltas
            ]
        )
        for horizon in ("5", "10", "15", "20")
    }
    colors = {
        color: {
            metric: _mean(
                [
                    float(
                        _mapping(
                            _mapping(delta["colors"], "colors")[color],
                            f"{color} deltas",
                        )[metric]
                    )
                    for delta in deltas
                ]
            )
            for metric in ("top1", "top3", "negativeLogLikelihood")
        }
        for color in ("white", "black")
    }
    mean_deltas = {
        key: _mean([float(delta[key]) for delta in deltas])
        for key in (
            "top1",
            "top3",
            "top5",
            "negativeLogLikelihood",
            "brier",
            "calibration",
        )
    }
    mean_deltas["moveHorizons"] = horizons
    mean_deltas["colors"] = colors
    promotion_count = sum(
        entry.get("comparisonDecision") == "promote-treatment"
        for entry in entries
    )
    checks = {
        "atLeastTwoPairsPromote": promotion_count >= 2,
        "positiveMeanTop1Delta": mean_deltas["top1"] > 0.0,
        "top3NonRegression": mean_deltas["top3"] >= 0.0,
        "top5NonRegression": mean_deltas["top5"] >= 0.0,
        "negativeLogLikelihoodNonRegression": (
            mean_deltas["negativeLogLikelihood"] <= 0.0
        ),
        "brierNonRegression": mean_deltas["brier"] <= 0.0,
        "calibrationNonRegression": mean_deltas["calibration"] <= 0.0,
        "allMoveHorizonsNonRegression": all(
            value >= 0.0 for value in horizons.values()
        ),
        "bothColorsTop1Top3NllNonRegression": all(
            values["top1"] >= 0.0
            and values["top3"] >= 0.0
            and values["negativeLogLikelihood"] <= 0.0
            for values in colors.values()
        ),
    }
    return {
        "promotedPairCount": promotion_count,
        "meanDeltas": mean_deltas,
        "checks": checks,
    }


def _pair_entry(pair: _Pair) -> Mapping[str, Any]:
    deltas = _delta(pair)
    extended_checks = blend_reliability_checks(
        pair.control_metrics,
        pair.treatment_metrics,
        float(deltas["top1"]) > 0.0,
    )
    comparison_promoted = (
        pair.comparison["releaseDecision"] == "promote-treatment"
    )
    return {
        "file": pair.comparison_path.name,
        "sha256": pair.comparison_sha256,
        "modelSeed": pair.seed,
        "comparisonDecision": pair.comparison["releaseDecision"],
        "extendedReliabilityChecks": extended_checks,
        "eligible": comparison_promoted and all(extended_checks.values()),
        "control": pair.comparison["control"],
        "treatment": pair.comparison["bestTreatment"],
        "deltas": deltas,
    }


def _selected_pair(
    entries: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    eligible = [
        entry
        for entry in entries
        if entry.get("eligible") is True
    ]
    if len(eligible) < 2:
        return None
    ordered_deltas = sorted(
        float(_mapping(entry["deltas"], "pair deltas")["top1"])
        for entry in eligible
    )
    median_delta = ordered_deltas[(len(ordered_deltas) - 1) // 2]
    median_pairs = [
        entry
        for entry in eligible
        if float(_mapping(entry["deltas"], "pair deltas")["top1"])
        == median_delta
    ]
    return min(
        median_pairs,
        key=lambda entry: (
            float(
                _mapping(entry["deltas"], "pair deltas")[
                    "negativeLogLikelihood"
                ]
            ),
            int(entry["modelSeed"]),
        ),
    )


def _build_stage_a(
    comparison_paths: Sequence[Path],
    corpus_ledger: _CorpusLedger,
    output_parent: Path,
) -> Mapping[str, Any]:
    if len(comparison_paths) != len(MODEL_SEEDS):
        raise ValueError("Stage A requires exactly three comparison artifacts")
    resolved = [
        _same_parent_path(path, output_parent, "comparison")
        for path in comparison_paths
    ]
    if len(set(resolved)) != len(resolved):
        raise CapturableDatasetError("Stage A comparison files must be unique")
    pairs = [_authenticate_pair(path) for path in resolved]
    if {pair.seed for pair in pairs} != set(MODEL_SEEDS):
        raise CapturableDatasetError(
            "Stage A comparisons do not use the three frozen model seeds"
        )
    pairs.sort(key=lambda pair: pair.seed)
    first_train = pairs[0].comparison["control"]["trainInput"]
    first_validation = pairs[0].comparison["validationInput"]
    for pair in pairs:
        control_train = _mapping(
            pair.comparison["control"].get("trainInput"),
            f"{pair.comparison_path.name} control training input",
        )
        treatment_train = _mapping(
            pair.comparison["bestTreatment"].get("trainInput"),
            f"{pair.comparison_path.name} treatment training input",
        )
        if control_train != treatment_train:
            raise CapturableDatasetError(
                f"{pair.comparison_path.name} control and treatment must "
                "use the same training input"
            )
        for candidate_label, candidate_train in (
            ("control", control_train),
            ("treatment", treatment_train),
        ):
            _assert_input_matches_ledger(
                candidate_train,
                corpus_ledger.splits["train"],
                f"{pair.comparison_path.name} {candidate_label} "
                "training input",
            )
    validation_a_identity = _assert_input_matches_ledger(
        _mapping(first_validation, "Stage A validation-A input"),
        corpus_ledger.splits["validation-a"],
        "Stage A validation-A input",
    )
    base_config = dict(pairs[0].control_report["config"])
    base_config.pop("seed")
    for pair in pairs[1:]:
        config = dict(pair.control_report["config"])
        config.pop("seed")
        if (
            pair.comparison["control"]["trainInput"] != first_train
            or pair.comparison["validationInput"] != first_validation
            or config != base_config
        ):
            raise CapturableDatasetError(
                "Stage A pairs must share train bytes, validation-A bytes, "
                "and every non-seed configuration value"
            )
    entries = [_pair_entry(pair) for pair in pairs]
    aggregate = _aggregate(entries)
    eligible_count = sum(entry["eligible"] is True for entry in entries)
    aggregate = {
        **aggregate,
        "eligiblePairCount": eligible_count,
        "checks": {
            **aggregate["checks"],
            "atLeastTwoPairsEligible": eligible_count >= 2,
        },
    }
    promoted = all(aggregate["checks"].values())
    selected = _selected_pair(entries) if promoted else None
    if promoted and selected is None:
        raise RuntimeError("promoted Stage A has no eligible median pair")
    artifact = {
        "format": STAGE_A_FORMAT,
        "version": OPPORTUNITY_WORKFLOW_VERSION,
        "protocol": _protocol(),
        "corpusLedger": _corpus_ledger_reference(corpus_ledger),
        "validationAInput": validation_a_identity,
        "comparisons": entries,
        "aggregate": aggregate,
        "decision": (
            "promote-treatment" if promoted else "retain-control"
        ),
        "selectedPair": selected,
        "nextStage": (
            "validation-b-authorized" if promoted else "blocked"
        ),
        "sealedTestStatus": "unopened",
    }
    _assert_path_free_artifact(artifact)
    return artifact


def run_stage_a(
    comparison_paths: Sequence[Path],
    output_path: Path,
    *,
    corpus_ledger_path: Path,
    corpus_ledger_sha256: str,
) -> Mapping[str, Any]:
    """Authenticate three validation-A comparisons and freeze one pair."""

    if output_path.exists():
        raise FileExistsError("Stage A output already exists")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    corpus_ledger_path = _same_parent_path(
        corpus_ledger_path,
        output_path.parent,
        "corpus ledger",
    )
    corpus_ledger = _load_corpus_ledger(
        corpus_ledger_path,
        corpus_ledger_sha256,
    )
    artifact = _build_stage_a(
        comparison_paths,
        corpus_ledger,
        output_path.parent,
    )
    payload = _canonical_json(artifact)
    publish_bytes_durable(output_path, payload)
    return {
        "artifactPath": output_path.name,
        "artifactSha256": hashlib.sha256(payload).hexdigest(),
        "decision": artifact["decision"],
        "selectedModelSeed": (
            None
            if artifact["selectedPair"] is None
            else artifact["selectedPair"]["modelSeed"]
        ),
    }


def load_stage_a(
    path: Path,
    *,
    corpus_ledger_path: Path,
    corpus_ledger_sha256: str,
) -> tuple[Mapping[str, Any], str]:
    """Replay-authenticate Stage A and all three comparison artifacts."""

    artifact, artifact_sha256 = _selection_report(path)
    _assert_path_free_artifact(artifact, f"{path.name} Stage A")
    if (
        set(artifact)
        != {
            "format",
            "version",
            "protocol",
            "corpusLedger",
            "validationAInput",
            "comparisons",
            "aggregate",
            "decision",
            "selectedPair",
            "nextStage",
            "sealedTestStatus",
        }
        or artifact.get("format") != STAGE_A_FORMAT
        or type(artifact.get("version")) is not int
        or artifact.get("version") != OPPORTUNITY_WORKFLOW_VERSION
        or artifact.get("protocol") != _protocol()
        or artifact.get("sealedTestStatus") != "unopened"
    ):
        raise CapturableDatasetError(
            f"{path.name} is not a compatible Stage A artifact"
        )
    entries = artifact.get("comparisons")
    if not isinstance(entries, list) or len(entries) != 3:
        raise CapturableDatasetError(
            f"{path.name} does not bind exactly three comparisons"
        )
    names: list[str] = []
    for entry in entries:
        mapping = _mapping(entry, "Stage A comparison")
        name = mapping.get("file")
        if (
            not isinstance(name, str)
            or not name
        ):
            raise CapturableDatasetError(
                f"{path.name} contains an unsafe comparison filename"
            )
        names.append(_safe_basename(name, "Stage A comparison filename"))
    corpus_ledger_path = _same_parent_path(
        corpus_ledger_path,
        path.resolve().parent,
        "corpus ledger",
    )
    corpus_ledger = _load_corpus_ledger(
        corpus_ledger_path,
        corpus_ledger_sha256,
    )
    if not _json_equal(
        artifact.get("corpusLedger"),
        _corpus_ledger_reference(corpus_ledger),
    ):
        raise CapturableDatasetError(
            f"{path.name} corpus ledger binding is inconsistent"
        )
    expected = _build_stage_a(
        [path.resolve().parent / name for name in names],
        corpus_ledger,
        path.resolve().parent,
    )
    if not _json_equal(artifact, expected):
        raise CapturableDatasetError(
            f"{path.name} Stage A decision is inconsistent"
        )
    return artifact, artifact_sha256


def _load_frozen_pair(
    stage_a_path: Path,
    stage_a: Mapping[str, Any],
    train_split: Mapping[str, Any],
) -> _LoadedPair:
    selected = _mapping(stage_a.get("selectedPair"), "selected pair")
    comparison_name = _safe_basename(
        selected.get("file"),
        "selected comparison filename",
    )
    comparison_path = stage_a_path.resolve().parent / comparison_name
    comparison, comparison_sha256 = load_treatment_comparison(
        comparison_path
    )
    if (
        comparison_sha256 != selected.get("sha256")
        or comparison.get("control") != selected.get("control")
        or comparison.get("bestTreatment") != selected.get("treatment")
    ):
        raise CapturableDatasetError("Stage A frozen pair changed")

    def load(entry: Mapping[str, Any]):
        selection_directory = _safe_basename(
            entry.get("selectionDirectory"),
            "selection directory",
        )
        checkpoint_file = _safe_basename(
            entry.get("checkpointFile"),
            "checkpoint file",
        )
        checkpoint = (
            comparison_path.parent
            / selection_directory
            / checkpoint_file
        )
        return _load_bound_selection_checkpoint(
            checkpoint,
            str(entry["checkpointSha256"]),
        )

    control = load(comparison["control"])
    treatment = load(comparison["bestTreatment"])
    if (
        control[2].get("opportunityMode") != "zero-ablation"
        or treatment[2].get("opportunityMode") != "public-exact"
        or control[2].get("sourceGameIds")
        != treatment[2].get("sourceGameIds")
    ):
        raise CapturableDatasetError(
            "frozen checkpoints are not the authenticated paired experiment"
        )
    source_game_ids = control[2].get("sourceGameIds")
    train_converted = _mapping(
        train_split.get("converted"),
        "corpus ledger training split",
    )
    if (
        not isinstance(source_game_ids, list)
        or any(
            not isinstance(game_id, str) or not game_id
            for game_id in source_game_ids
        )
        or len(set(source_game_ids)) != len(source_game_ids)
        or set(source_game_ids) != set(train_converted["gameIds"])
    ):
        raise CapturableDatasetError(
            "frozen checkpoints do not bind the corpus-ledger training "
            "game-ID set"
        )
    return _LoadedPair(
        control=control,
        treatment=treatment,
        source_game_ids=frozenset(source_game_ids),
    )


def _dataset_identity(
    path: Path,
    rows: Sequence[CapturableDatasetRow],
    sha256: str,
    split: str,
    ledger_split: Mapping[str, Any],
) -> Mapping[str, Any]:
    if split not in SPLIT_SEED_ROOTS:
        raise ValueError("dataset split is not preregistered")
    game_seeds: dict[str, int] = {}
    for row in rows:
        game_id = row.evaluation.game_id
        seed = row.evaluation.seed
        if (
            not isinstance(game_id, str)
            or not game_id
            or isinstance(seed, bool)
            or not isinstance(seed, int)
            or seed < 0
            or seed > 0xFFFFFFFF
        ):
            raise CapturableDatasetError(
                f"{split} contains an invalid game ID or simulation seed"
            )
        previous = game_seeds.setdefault(game_id, seed)
        if previous != seed:
            raise CapturableDatasetError(
                f"{split} maps one game ID to multiple simulation seeds"
            )
    game_ids = set(game_seeds)
    simulation_seeds = set(game_seeds.values())
    if len(game_ids) != VALIDATION_GAME_COUNT:
        raise CapturableDatasetError(
            f"{split} must contain exactly 2500 games"
        )
    if len(simulation_seeds) != len(game_ids):
        raise CapturableDatasetError(
            f"{split} reuses a simulation seed across games"
        )
    converted = _mapping(
        ledger_split.get("converted"),
        f"{split} corpus-ledger converted identity",
    )
    if (
        sha256 != converted.get("sha256")
        or len(rows) != converted.get("rows")
        or len(game_ids) != converted.get("games")
        or game_ids != set(converted.get("gameIds", ()))
        or simulation_seeds != set(converted.get("simulationSeeds", ()))
    ):
        raise CapturableDatasetError(
            f"{split} does not match the authenticated corpus ledger"
        )
    expected_roots = list(SPLIT_SEED_ROOTS[split])
    if ledger_split.get("seedRoots") != expected_roots:
        raise CapturableDatasetError(
            f"{split} corpus-ledger schedule roots are inconsistent"
        )
    return {
        "file": _safe_basename(path.resolve().name, f"{split} file"),
        "sha256": _sha256(sha256, f"{split} SHA-256"),
        "rows": _positive_count(len(rows), f"{split} rows"),
        "games": len(game_ids),
        "gameIdSetSha256": converted["gameIdSetSha256"],
        "simulationSeedSetSha256": converted["simulationSeedSetSha256"],
        "scheduleId": ledger_split["scheduleId"],
        "seedRoots": ledger_split["seedRoots"],
    }


def _evaluate(
    loaded: tuple[Path, Any, Mapping[str, Any], str],
    rows: Sequence[CapturableDatasetRow],
) -> Mapping[str, Any]:
    checkpoint_path, model, metadata, checkpoint_sha256 = loaded
    mode = metadata.get("opportunityMode")
    config = _training_config_from_json(metadata["config"])
    return {
        "checkpoint": {
            "file": checkpoint_path.name,
            "sha256": checkpoint_sha256,
        },
        "selection": {
            "selectedEpoch": metadata["selectedEpoch"],
            "selectedFusionAlpha": metadata["selectedFusionAlpha"],
            "selectedPriorSmoothing": metadata["selectedPriorSmoothing"],
        },
        "metrics": evaluate_capturable(
            model,
            rows,
            tensorize(rows, opportunity_mode=mode),
            config,
            float(metadata["selectedFusionAlpha"]),
            float(metadata["selectedPriorSmoothing"]),
        ),
    }


def _paired_result(
    pair: _LoadedPair,
    rows: Sequence[CapturableDatasetRow],
) -> Mapping[str, Any]:
    control = _evaluate(pair.control, rows)
    treatment = _evaluate(pair.treatment, rows)
    control_metrics = _mapping(control["metrics"], "control metrics")
    treatment_metrics = _mapping(treatment["metrics"], "treatment metrics")
    synthetic_pair = _Pair(
        comparison_path=Path("evaluation"),
        comparison={
            "control": {"seed": pair.control[2]["config"]["seed"]},
        },
        comparison_sha256="0" * 64,
        control_report={"validation": control_metrics},
        treatment_report={"validation": treatment_metrics},
    )
    deltas = _delta(synthetic_pair)
    primary_confirmed = float(deltas["top1"]) > 0.0
    checks = blend_reliability_checks(
        control_metrics,
        treatment_metrics,
        primary_confirmed,
    )
    return {
        "control": control,
        "treatment": treatment,
        "deltas": deltas,
        "primaryDecision": (
            "confirm-treatment" if primary_confirmed else "reject-treatment"
        ),
        "reliabilityChecks": checks,
        "decision": (
            "promote-treatment"
            if all(checks.values())
            else "retain-control"
        ),
    }


def _assert_disjoint(
    rows: Sequence[CapturableDatasetRow],
    forbidden: frozenset[str],
    label: str,
) -> frozenset[str]:
    observed = frozenset(row.evaluation.game_id for row in rows)
    overlap = sorted(observed & forbidden)
    if overlap:
        raise CapturableDatasetError(
            f"{label} overlaps earlier experiment games: "
            + ", ".join(overlap[:5])
        )
    return observed


def _build_stage_b(
    stage_a_path: Path,
    validation_b_path: Path,
    corpus_ledger_path: Path,
    corpus_ledger_sha256: str,
) -> tuple[Mapping[str, Any], tuple[CapturableDatasetRow, ...]]:
    stage_a, stage_a_sha256 = load_stage_a(
        stage_a_path,
        corpus_ledger_path=corpus_ledger_path,
        corpus_ledger_sha256=corpus_ledger_sha256,
    )
    if stage_a.get("decision") != "promote-treatment":
        raise CapturableDatasetError(
            "Stage A did not authorize validation B"
        )
    corpus_ledger = _load_corpus_ledger(
        corpus_ledger_path,
        corpus_ledger_sha256,
    )
    pair = _load_frozen_pair(
        stage_a_path,
        stage_a,
        corpus_ledger.splits["train"],
    )
    rows, validation_sha256 = _load_stable_capturable_dataset(
        validation_b_path,
        "public-exact",
    )
    identity = _dataset_identity(
        validation_b_path,
        rows,
        validation_sha256,
        "validation-b",
        corpus_ledger.splits["validation-b"],
    )
    if validation_sha256 == _mapping(
        stage_a["validationAInput"],
        "validation A input",
    ).get("sha256"):
        raise CapturableDatasetError(
            "validation B reuses validation-A bytes"
        )
    _assert_disjoint(rows, pair.source_game_ids, "validation B")
    result = _paired_result(pair, rows)
    artifact = {
        "format": STAGE_B_FORMAT,
        "version": OPPORTUNITY_WORKFLOW_VERSION,
        "protocol": _protocol(),
        "corpusLedger": stage_a["corpusLedger"],
        "stageA": {
            "file": stage_a_path.resolve().name,
            "sha256": stage_a_sha256,
            "decision": stage_a["decision"],
        },
        "frozenPair": stage_a["selectedPair"],
        "validationBInput": identity,
        "result": result,
        "authorization": (
            "sealed-test-authorized"
            if result["decision"] == "promote-treatment"
            else "blocked"
        ),
        "sealedTestStatus": "unopened",
    }
    _assert_path_free_artifact(artifact)
    return artifact, rows


def run_stage_b(
    stage_a_path: Path,
    validation_b_path: Path,
    output_path: Path,
    *,
    corpus_ledger_path: Path,
    corpus_ledger_sha256: str,
) -> Mapping[str, Any]:
    """Evaluate only the Stage-A-frozen pair on validation B."""

    if output_path.exists():
        raise FileExistsError("Stage B output already exists")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stage_a_path = _same_parent_path(
        stage_a_path,
        output_path.parent,
        "Stage A",
    )
    corpus_ledger_path = _same_parent_path(
        corpus_ledger_path,
        output_path.parent,
        "corpus ledger",
    )
    artifact, _ = _build_stage_b(
        stage_a_path,
        validation_b_path,
        corpus_ledger_path,
        corpus_ledger_sha256,
    )
    payload = _canonical_json(artifact)
    publish_bytes_durable(output_path, payload)
    return {
        "artifactPath": output_path.name,
        "artifactSha256": hashlib.sha256(payload).hexdigest(),
        "decision": artifact["result"]["decision"],
        "authorization": artifact["authorization"],
    }


def _load_stage_b_context(
    path: Path,
    validation_b_path: Path,
    corpus_ledger_path: Path,
    corpus_ledger_sha256: str,
) -> tuple[
    Mapping[str, Any],
    str,
    tuple[CapturableDatasetRow, ...],
]:
    artifact, artifact_sha256 = _selection_report(path)
    _assert_path_free_artifact(artifact, f"{path.name} Stage B")
    if (
        set(artifact)
        != {
            "format",
            "version",
            "protocol",
            "corpusLedger",
            "stageA",
            "frozenPair",
            "validationBInput",
            "result",
            "authorization",
            "sealedTestStatus",
        }
        or artifact.get("format") != STAGE_B_FORMAT
        or type(artifact.get("version")) is not int
        or artifact.get("version") != OPPORTUNITY_WORKFLOW_VERSION
        or artifact.get("protocol") != _protocol()
        or artifact.get("sealedTestStatus") != "unopened"
    ):
        raise CapturableDatasetError(
            f"{path.name} is not a compatible Stage B artifact"
        )
    stage_a_reference = _mapping(artifact.get("stageA"), "Stage A reference")
    stage_a_name = _safe_basename(
        stage_a_reference.get("file"),
        "Stage B Stage A file",
    )
    stage_a_path = path.resolve().parent / stage_a_name
    corpus_ledger_path = _same_parent_path(
        corpus_ledger_path,
        path.resolve().parent,
        "corpus ledger",
    )
    expected, rows = _build_stage_b(
        stage_a_path,
        validation_b_path,
        corpus_ledger_path,
        corpus_ledger_sha256,
    )
    if not _json_equal(artifact, expected):
        raise CapturableDatasetError(
            f"{path.name} Stage B authorization is inconsistent"
        )
    return artifact, artifact_sha256, rows


def load_stage_b(
    path: Path,
    validation_b_path: Path,
    *,
    corpus_ledger_path: Path,
    corpus_ledger_sha256: str,
) -> tuple[Mapping[str, Any], str]:
    """Replay Stage B from Stage A, checkpoints, and validation-B bytes."""

    artifact, artifact_sha256, _ = _load_stage_b_context(
        path,
        validation_b_path,
        corpus_ledger_path,
        corpus_ledger_sha256,
    )
    return artifact, artifact_sha256


def _build_sealed_result(
    stage_b_path: Path,
    stage_b: Mapping[str, Any],
    stage_b_sha256: str,
    validation_b_rows: Sequence[CapturableDatasetRow],
    test_path: Path,
    corpus_ledger_path: Path,
    corpus_ledger_sha256: str,
) -> Mapping[str, Any]:
    stage_a_name = _mapping(stage_b["stageA"], "Stage A reference")["file"]
    stage_a_path = stage_b_path.resolve().parent / str(stage_a_name)
    stage_a, _ = load_stage_a(
        stage_a_path,
        corpus_ledger_path=corpus_ledger_path,
        corpus_ledger_sha256=corpus_ledger_sha256,
    )
    corpus_ledger = _load_corpus_ledger(
        corpus_ledger_path,
        corpus_ledger_sha256,
    )
    pair = _load_frozen_pair(
        stage_a_path,
        stage_a,
        corpus_ledger.splits["train"],
    )
    test_rows, test_sha256 = _load_stable_capturable_dataset(
        test_path,
        "public-exact",
    )
    identity = _dataset_identity(
        test_path,
        test_rows,
        test_sha256,
        "sealed-test",
        corpus_ledger.splits["test"],
    )
    prior_shas = {
        _mapping(stage_a["validationAInput"], "validation A input").get(
            "sha256"
        ),
        _mapping(stage_b["validationBInput"], "validation B input").get(
            "sha256"
        ),
    }
    if test_sha256 in prior_shas:
        raise CapturableDatasetError(
            "sealed test reuses validation bytes"
        )
    validation_b_games = frozenset(
        row.evaluation.game_id for row in validation_b_rows
    )
    forbidden = pair.source_game_ids | validation_b_games
    _assert_disjoint(test_rows, forbidden, "sealed test")
    result = _paired_result(pair, test_rows)
    artifact = {
        "format": SEALED_TEST_FORMAT,
        "version": OPPORTUNITY_WORKFLOW_VERSION,
        "protocol": _protocol(),
        "corpusLedger": stage_b["corpusLedger"],
        "stageB": {
            "file": stage_b_path.resolve().name,
            "sha256": stage_b_sha256,
            "authorization": stage_b["authorization"],
        },
        "frozenPair": stage_b["frozenPair"],
        "testInput": identity,
        "result": result,
        "sealedTestStatus": "consumed",
    }
    _assert_path_free_artifact(artifact)
    return artifact


def consumption_marker_path(
    stage_b_path: Path,
    stage_b_sha256: str,
) -> Path:
    """Return the one marker name canonically keyed by Stage-B bytes."""

    digest = _sha256(stage_b_sha256, "Stage B SHA-256")
    return stage_b_path.with_name(
        f"sealed-test-consumption-{digest}.json"
    )


def _consumption_artifact(
    stage_b_path: Path,
    stage_b: Mapping[str, Any],
    stage_b_sha256: str,
) -> Mapping[str, Any]:
    artifact = {
        "format": CONSUMPTION_FORMAT,
        "version": OPPORTUNITY_WORKFLOW_VERSION,
        "protocol": _protocol(),
        "corpusLedger": stage_b["corpusLedger"],
        "stageB": {
            "sha256": stage_b_sha256,
            "authorization": stage_b["authorization"],
        },
        "frozenPair": stage_b["frozenPair"],
        "sealedTestStatus": "consumed",
    }
    _assert_path_free_artifact(artifact)
    return artifact


def run_sealed_test(
    stage_b_path: Path,
    validation_b_path: Path,
    test_path: Path,
    output_path: Path,
    *,
    corpus_ledger_path: Path,
    corpus_ledger_sha256: str,
) -> Mapping[str, Any]:
    """Consume the sealed test once after replaying Stage-B authorization."""

    if output_path.exists():
        raise FileExistsError("sealed test is already consumed")
    # Do not move any operation involving test_path above this authorization
    # boundary.  In particular, do not resolve or inspect that path early.
    stage_b, stage_b_sha256, validation_b_rows = _load_stage_b_context(
        stage_b_path,
        validation_b_path,
        corpus_ledger_path,
        corpus_ledger_sha256,
    )
    if stage_b.get("authorization") != "sealed-test-authorized":
        raise CapturableDatasetError(
            "Stage B did not authorize the sealed test"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stage_b_path = _same_parent_path(
        stage_b_path,
        output_path.parent,
        "Stage B",
    )
    corpus_ledger_path = _same_parent_path(
        corpus_ledger_path,
        output_path.parent,
        "corpus ledger",
    )
    marker_path = consumption_marker_path(
        stage_b_path,
        stage_b_sha256,
    )
    marker = _consumption_artifact(
        stage_b_path,
        stage_b,
        stage_b_sha256,
    )
    marker_payload = _canonical_json(marker)
    publish_bytes_durable(marker_path, marker_payload)
    marker_sha256 = hashlib.sha256(marker_payload).hexdigest()
    artifact = _build_sealed_result(
        stage_b_path,
        stage_b,
        stage_b_sha256,
        validation_b_rows,
        test_path,
        corpus_ledger_path,
        corpus_ledger_sha256,
    )
    _, current_stage_b_sha256 = _selection_report(stage_b_path)
    current_marker, current_marker_sha256 = _selection_report(marker_path)
    if (
        current_stage_b_sha256 != stage_b_sha256
        or not _json_equal(current_marker, marker)
        or current_marker_sha256 != marker_sha256
    ):
        raise CapturableDatasetError(
            "Stage B or consumption marker changed during evaluation"
        )
    artifact = {
        **artifact,
        "consumption": {
            "file": marker_path.name,
            "sha256": marker_sha256,
        },
    }
    payload = _canonical_json(artifact)
    publish_bytes_durable(output_path, payload)
    return {
        "artifactPath": output_path.name,
        "artifactSha256": hashlib.sha256(payload).hexdigest(),
        "decision": artifact["result"]["decision"],
        "consumptionMarker": marker_path.name,
        "consumptionMarkerSha256": marker_sha256,
        "sealedTestStatus": "consumed",
    }


def load_sealed_test(
    path: Path,
    stage_b_path: Path,
    validation_b_path: Path,
    *,
    report_sha256: str,
    corpus_ledger_path: Path,
    corpus_ledger_sha256: str,
) -> tuple[Mapping[str, Any], str]:
    """Authenticate a consumed report without reopening the sealed test."""

    artifact, artifact_sha256 = _selection_report(path)
    expected_report_sha256 = _sha256(
        report_sha256,
        "caller-authenticated sealed report SHA-256",
    )
    if artifact_sha256 != expected_report_sha256:
        raise CapturableDatasetError(
            f"{path.name} does not match the caller-authenticated final "
            "report SHA-256"
        )
    _assert_path_free_artifact(artifact, f"{path.name} sealed report")
    if (
        set(artifact)
        != {
            "format",
            "version",
            "protocol",
            "corpusLedger",
            "stageB",
            "frozenPair",
            "testInput",
            "result",
            "consumption",
            "sealedTestStatus",
        }
        or artifact.get("format") != SEALED_TEST_FORMAT
        or type(artifact.get("version")) is not int
        or artifact.get("version") != OPPORTUNITY_WORKFLOW_VERSION
        or artifact.get("protocol") != _protocol()
        or artifact.get("sealedTestStatus") != "consumed"
    ):
        raise CapturableDatasetError(
            f"{path.name} is not a compatible consumed sealed test"
        )
    stage_b_path = _same_parent_path(
        stage_b_path,
        path.resolve().parent,
        "Stage B",
    )
    corpus_ledger_path = _same_parent_path(
        corpus_ledger_path,
        path.resolve().parent,
        "corpus ledger",
    )
    stage_b, stage_b_sha256 = load_stage_b(
        stage_b_path,
        validation_b_path,
        corpus_ledger_path=corpus_ledger_path,
        corpus_ledger_sha256=corpus_ledger_sha256,
    )
    if stage_b.get("authorization") != "sealed-test-authorized":
        raise CapturableDatasetError(
            "Stage B did not authorize the sealed test"
        )
    expected_marker = _consumption_artifact(
        stage_b_path,
        stage_b,
        stage_b_sha256,
    )
    consumption = _mapping(
        artifact.get("consumption"),
        "consumption marker reference",
    )
    _exact_keys(
        consumption,
        {"file", "sha256"},
        "consumption marker reference",
    )
    marker_name = _safe_basename(
        consumption.get("file"),
        "consumption marker filename",
    )
    canonical_marker = consumption_marker_path(
        stage_b_path,
        stage_b_sha256,
    )
    if marker_name != canonical_marker.name:
        raise CapturableDatasetError(
            f"{path.name} consumption marker filename is invalid"
        )
    marker, marker_sha256 = _selection_report(
        canonical_marker
    )
    _assert_path_free_artifact(marker, f"{marker_name} consumption marker")
    if (
        not _json_equal(marker, expected_marker)
        or marker_sha256 != consumption.get("sha256")
    ):
        raise CapturableDatasetError(
            f"{path.name} consumption marker is inconsistent"
        )
    stage_b_reference = _mapping(
        artifact.get("stageB"),
        "Stage B reference",
    )
    _exact_keys(
        stage_b_reference,
        {"file", "sha256", "authorization"},
        "Stage B reference",
    )
    stage_b_file = _safe_basename(
        stage_b_reference.get("file"),
        "sealed report Stage B file",
    )
    if (
        stage_b_path.resolve().name != stage_b_file
        or stage_b_sha256 != stage_b_reference.get("sha256")
        or stage_b_reference.get("authorization")
        != "sealed-test-authorized"
        or not _json_equal(
            artifact.get("corpusLedger"),
            stage_b.get("corpusLedger"),
        )
        or not _json_equal(
            artifact.get("frozenPair"),
            stage_b.get("frozenPair"),
        )
    ):
        raise CapturableDatasetError(
            f"{path.name} Stage B binding is inconsistent"
        )
    result = _mapping(artifact.get("result"), "sealed result")
    if set(result) != {
        "control",
        "treatment",
        "deltas",
        "primaryDecision",
        "reliabilityChecks",
        "decision",
    }:
        raise CapturableDatasetError(
            f"{path.name} sealed result fields are invalid"
        )
    control = _mapping(result.get("control"), "sealed control")
    treatment = _mapping(result.get("treatment"), "sealed treatment")
    frozen_pair = _mapping(artifact.get("frozenPair"), "frozen pair")
    frozen_control = _mapping(
        frozen_pair.get("control"),
        "frozen control",
    )
    frozen_treatment = _mapping(
        frozen_pair.get("treatment"),
        "frozen treatment",
    )
    expected_control_checkpoint = {
        "file": frozen_control.get("checkpointFile"),
        "sha256": frozen_control.get("checkpointSha256"),
    }
    expected_treatment_checkpoint = {
        "file": frozen_treatment.get("checkpointFile"),
        "sha256": frozen_treatment.get("checkpointSha256"),
    }
    stage_b_result = _mapping(stage_b.get("result"), "Stage B result")
    stage_b_control = _mapping(
        stage_b_result.get("control"),
        "Stage B control",
    )
    stage_b_treatment = _mapping(
        stage_b_result.get("treatment"),
        "Stage B treatment",
    )
    if (
        control.get("checkpoint") != expected_control_checkpoint
        or treatment.get("checkpoint") != expected_treatment_checkpoint
        or not _json_equal(
            control.get("selection"),
            stage_b_control.get("selection"),
        )
        or not _json_equal(
            treatment.get("selection"),
            stage_b_treatment.get("selection"),
        )
    ):
        raise CapturableDatasetError(
            f"{path.name} sealed checkpoints are not the frozen pair"
        )
    control_metrics = _mapping(control.get("metrics"), "sealed control metrics")
    treatment_metrics = _mapping(
        treatment.get("metrics"),
        "sealed treatment metrics",
    )
    synthetic = _Pair(
        comparison_path=Path("sealed"),
        comparison={"control": {"seed": 0}},
        comparison_sha256="0" * 64,
        control_report={"validation": control_metrics},
        treatment_report={"validation": treatment_metrics},
    )
    deltas = _delta(synthetic)
    primary = float(deltas["top1"]) > 0.0
    checks = blend_reliability_checks(
        control_metrics,
        treatment_metrics,
        primary,
    )
    expected_decision = (
        "promote-treatment" if all(checks.values()) else "retain-control"
    )
    if (
        not _json_equal(result.get("deltas"), deltas)
        or result.get("primaryDecision")
        != ("confirm-treatment" if primary else "reject-treatment")
        or not _json_equal(result.get("reliabilityChecks"), checks)
        or result.get("decision") != expected_decision
    ):
        raise CapturableDatasetError(
            f"{path.name} sealed decision is inconsistent"
        )
    test_input = _mapping(artifact.get("testInput"), "sealed test input")
    _exact_keys(
        test_input,
        {
            "file",
            "sha256",
            "rows",
            "games",
            "gameIdSetSha256",
            "simulationSeedSetSha256",
            "scheduleId",
            "seedRoots",
        },
        "sealed test input",
    )
    corpus_ledger = _load_corpus_ledger(
        corpus_ledger_path,
        corpus_ledger_sha256,
    )
    test_split = corpus_ledger.splits["test"]
    test_converted = _mapping(
        test_split.get("converted"),
        "corpus-ledger test identity",
    )
    if (
        test_input.get("sha256") != test_converted.get("sha256")
        or test_input.get("rows") != test_converted.get("rows")
        or test_input.get("games") != test_converted.get("games")
        or test_input.get("games") != VALIDATION_GAME_COUNT
        or test_input.get("gameIdSetSha256")
        != test_converted.get("gameIdSetSha256")
        or test_input.get("simulationSeedSetSha256")
        != test_converted.get("simulationSeedSetSha256")
        or test_input.get("scheduleId") != test_split.get("scheduleId")
        or test_input.get("seedRoots") != test_split.get("seedRoots")
    ):
        raise CapturableDatasetError(
            f"{path.name} sealed input identity is invalid"
        )
    _safe_basename(test_input.get("file"), "sealed test input file")
    _sha256(test_input.get("sha256"), "sealed test SHA-256")
    return artifact, artifact_sha256


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Run the schema-9 opportunity validation state machine.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    stage_a = commands.add_parser("stage-a")
    stage_a.add_argument(
        "--comparison",
        type=Path,
        action="append",
        required=True,
    )
    stage_a.add_argument("--corpus-ledger", type=Path, required=True)
    stage_a.add_argument("--corpus-ledger-sha256", required=True)
    stage_a.add_argument("--output", type=Path, required=True)
    stage_b = commands.add_parser("stage-b")
    stage_b.add_argument("--stage-a", type=Path, required=True)
    stage_b.add_argument("--validation-b", type=Path, required=True)
    stage_b.add_argument("--corpus-ledger", type=Path, required=True)
    stage_b.add_argument("--corpus-ledger-sha256", required=True)
    stage_b.add_argument("--output", type=Path, required=True)
    sealed = commands.add_parser("sealed-test")
    sealed.add_argument("--stage-b", type=Path, required=True)
    sealed.add_argument("--validation-b", type=Path, required=True)
    sealed.add_argument("--test", type=Path, required=True)
    sealed.add_argument("--corpus-ledger", type=Path, required=True)
    sealed.add_argument("--corpus-ledger-sha256", required=True)
    sealed.add_argument("--output", type=Path, required=True)
    verify_a = commands.add_parser("verify-stage-a")
    verify_a.add_argument("--stage-a", type=Path, required=True)
    verify_a.add_argument("--corpus-ledger", type=Path, required=True)
    verify_a.add_argument("--corpus-ledger-sha256", required=True)
    verify_b = commands.add_parser("verify-stage-b")
    verify_b.add_argument("--stage-b", type=Path, required=True)
    verify_b.add_argument("--validation-b", type=Path, required=True)
    verify_b.add_argument("--corpus-ledger", type=Path, required=True)
    verify_b.add_argument("--corpus-ledger-sha256", required=True)
    verify_test = commands.add_parser("verify-sealed-test")
    verify_test.add_argument("--report", type=Path, required=True)
    verify_test.add_argument("--report-sha256", required=True)
    verify_test.add_argument("--stage-b", type=Path, required=True)
    verify_test.add_argument("--validation-b", type=Path, required=True)
    verify_test.add_argument("--corpus-ledger", type=Path, required=True)
    verify_test.add_argument("--corpus-ledger-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    options = _parser().parse_args(argv)
    if options.command == "stage-a":
        result = run_stage_a(
            options.comparison,
            options.output,
            corpus_ledger_path=options.corpus_ledger,
            corpus_ledger_sha256=options.corpus_ledger_sha256,
        )
    elif options.command == "stage-b":
        result = run_stage_b(
            options.stage_a,
            options.validation_b,
            options.output,
            corpus_ledger_path=options.corpus_ledger,
            corpus_ledger_sha256=options.corpus_ledger_sha256,
        )
    elif options.command == "sealed-test":
        result = run_sealed_test(
            options.stage_b,
            options.validation_b,
            options.test,
            options.output,
            corpus_ledger_path=options.corpus_ledger,
            corpus_ledger_sha256=options.corpus_ledger_sha256,
        )
    elif options.command == "verify-stage-a":
        artifact, digest = load_stage_a(
            options.stage_a,
            corpus_ledger_path=options.corpus_ledger,
            corpus_ledger_sha256=options.corpus_ledger_sha256,
        )
        result = {"artifactSha256": digest, "decision": artifact["decision"]}
    elif options.command == "verify-stage-b":
        artifact, digest = load_stage_b(
            options.stage_b,
            options.validation_b,
            corpus_ledger_path=options.corpus_ledger,
            corpus_ledger_sha256=options.corpus_ledger_sha256,
        )
        result = {
            "artifactSha256": digest,
            "authorization": artifact["authorization"],
        }
    else:
        artifact, digest = load_sealed_test(
            options.report,
            options.stage_b,
            options.validation_b,
            report_sha256=options.report_sha256,
            corpus_ledger_path=options.corpus_ledger,
            corpus_ledger_sha256=options.corpus_ledger_sha256,
        )
        result = {
            "artifactSha256": digest,
            "sealedTestStatus": artifact["sealedTestStatus"],
        }
    _assert_path_free_artifact(result, "CLI result")
    print(_canonical_json(result).decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
