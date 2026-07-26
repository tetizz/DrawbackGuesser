"""Content-addressed identity for the frozen release training corpus set.

This module deliberately knows nothing about filesystem locations, dataset
readers, or model code.  It combines already-authenticated corpus identities
and rejects any set that is not the one-primary/six-supplement release shape.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Mapping, Sequence


FORMAT = "drawbacktrainer-training-corpus-set"
VERSION = 1
SCHEMA_VERSION = 6
SYMBOLIC_FEATURE_VERSION = 6
MAX_PLIES = 80
OBSERVATION_POLICY = "single-attempt-allow-partial-v1"
EVALUATOR_POLICY_ID = "stockfish-bestmove-v1"
EVALUATOR_POLICY_VERSION = 1
EVALUATOR_NODES = 10_000
AGENT_DOMAIN = (
    "random-legal",
    "greedy-material",
    "human-like-weak",
    "human-like-medium",
    "human-like-strong",
)
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
SOURCE_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
MAX_SAFE_INTEGER = 9_007_199_254_740_991

FROZEN_SUPPLEMENT_PROFILES = (
    (101, "checkers-pacman", ("checkers", "pacman")),
    (102, "truant-spice-of-life", ("truant", "spice-of-life")),
    (103, "oddball-even-keeled", ("oddball", "even-keeled")),
    (
        104,
        "quit-horsing-around-forward-march",
        ("quit-horsing-around", "forward-march"),
    ),
    (
        105,
        "horse-tranquilizer-conscientious-objectors",
        ("horse-tranquilizer", "conscientious-objectors"),
    ),
    (106, "gambler-truant", ("gambler", "truant")),
)

_CONTENT_KEYS = frozenset(
    {
        "outcomes_sha256",
        "dataset_sha256",
        "dataset_bytes",
        "games",
        "rows",
        "schema_version",
        "symbolic_feature_version",
        "max_plies",
        "observation_policy",
        "evaluator_policy_id",
        "evaluator_policy_version",
        "evaluator_nodes",
        "engine_binary_sha256",
        "engine_fingerprint",
        "agent_domain",
    }
)
_PRIMARY_KEYS = _CONTENT_KEYS | frozenset(
    {
        "release_root_sha256",
        "corpus_run_id",
        "private_train_manifest_sha256",
    }
)
_SUPPLEMENT_KEYS = _CONTENT_KEYS | frozenset(
    {
        "source_revision",
        "generation_run_id",
        "manifest_sha256",
        "plan_sha256",
        "profile_offset",
        "profile_id",
        "rule_ids",
        "profile_sha256",
    }
)
_SET_KEYS = frozenset({"format", "version", "primary", "supplements", "sha256"})


class TrainingCorpusSetError(ValueError):
    """Raised when aggregate training provenance is malformed or incompatible."""


@dataclass(frozen=True)
class CorpusIdentity:
    """Immutable identity of the primary authenticated training corpus."""

    release_root_sha256: str
    corpus_run_id: str
    private_train_manifest_sha256: str
    outcomes_sha256: str
    dataset_sha256: str
    dataset_bytes: int
    games: int
    rows: int
    schema_version: int
    symbolic_feature_version: int
    max_plies: int
    observation_policy: str
    evaluator_policy_id: str
    evaluator_policy_version: int
    evaluator_nodes: int
    engine_binary_sha256: str
    engine_fingerprint: str
    agent_domain: tuple[str, ...]


@dataclass(frozen=True)
class SupplementIdentity:
    """Immutable identity of one frozen hard-negative supplement."""

    source_revision: str
    generation_run_id: str
    manifest_sha256: str
    plan_sha256: str
    outcomes_sha256: str
    dataset_sha256: str
    dataset_bytes: int
    games: int
    rows: int
    schema_version: int
    symbolic_feature_version: int
    max_plies: int
    observation_policy: str
    evaluator_policy_id: str
    evaluator_policy_version: int
    evaluator_nodes: int
    engine_binary_sha256: str
    engine_fingerprint: str
    agent_domain: tuple[str, ...]
    profile_offset: int
    profile_id: str
    rule_ids: tuple[str, str]
    profile_sha256: str


def _exact_keys(
    value: Mapping[str, Any], expected: frozenset[str], label: str
) -> None:
    actual = set(value)
    missing = expected - actual
    unknown = actual - expected
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(sorted(missing)))
        if unknown:
            details.append("unknown " + ", ".join(sorted(unknown)))
        raise TrainingCorpusSetError(
            f"{label} fields are invalid: {'; '.join(details)}"
        )


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TrainingCorpusSetError(f"{label} must be an object")
    return value


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise TrainingCorpusSetError(f"{label} must be a lowercase SHA-256")
    return value


def _positive_int(value: object, label: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
        or value > MAX_SAFE_INTEGER
    ):
        raise TrainingCorpusSetError(
            f"{label} must be a positive interoperable integer"
        )
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise TrainingCorpusSetError(f"{label} must be a non-empty canonical string")
    return value


def _content_mapping(
    identity: CorpusIdentity | SupplementIdentity,
) -> dict[str, object]:
    return {
        "outcomes_sha256": identity.outcomes_sha256,
        "dataset_sha256": identity.dataset_sha256,
        "dataset_bytes": identity.dataset_bytes,
        "games": identity.games,
        "rows": identity.rows,
        "schema_version": identity.schema_version,
        "symbolic_feature_version": identity.symbolic_feature_version,
        "max_plies": identity.max_plies,
        "observation_policy": identity.observation_policy,
        "evaluator_policy_id": identity.evaluator_policy_id,
        "evaluator_policy_version": identity.evaluator_policy_version,
        "evaluator_nodes": identity.evaluator_nodes,
        "engine_binary_sha256": identity.engine_binary_sha256,
        "engine_fingerprint": identity.engine_fingerprint,
        "agent_domain": list(identity.agent_domain),
    }


def _primary_mapping(identity: CorpusIdentity) -> dict[str, object]:
    return {
        "release_root_sha256": identity.release_root_sha256,
        "corpus_run_id": identity.corpus_run_id,
        "private_train_manifest_sha256": identity.private_train_manifest_sha256,
        **_content_mapping(identity),
    }


def _supplement_mapping(identity: SupplementIdentity) -> dict[str, object]:
    return {
        "source_revision": identity.source_revision,
        "generation_run_id": identity.generation_run_id,
        "manifest_sha256": identity.manifest_sha256,
        "plan_sha256": identity.plan_sha256,
        **_content_mapping(identity),
        "profile_offset": identity.profile_offset,
        "profile_id": identity.profile_id,
        "rule_ids": list(identity.rule_ids),
        "profile_sha256": identity.profile_sha256,
    }


def _parse_content(
    value: Mapping[str, Any], label: str
) -> dict[str, object]:
    agent_value = value["agent_domain"]
    if not isinstance(agent_value, list) or any(
        not isinstance(item, str) for item in agent_value
    ):
        raise TrainingCorpusSetError(f"{label}.agent_domain must be a string array")
    agent_domain = tuple(agent_value)
    if agent_domain != AGENT_DOMAIN:
        raise TrainingCorpusSetError(
            f"{label}.agent_domain does not match the frozen release domain"
        )
    parsed: dict[str, object] = {
        "outcomes_sha256": _sha256(
            value["outcomes_sha256"], f"{label}.outcomes_sha256"
        ),
        "dataset_sha256": _sha256(
            value["dataset_sha256"], f"{label}.dataset_sha256"
        ),
        "dataset_bytes": _positive_int(
            value["dataset_bytes"], f"{label}.dataset_bytes"
        ),
        "games": _positive_int(value["games"], f"{label}.games"),
        "rows": _positive_int(value["rows"], f"{label}.rows"),
        "schema_version": value["schema_version"],
        "symbolic_feature_version": value["symbolic_feature_version"],
        "max_plies": value["max_plies"],
        "observation_policy": value["observation_policy"],
        "evaluator_policy_id": value["evaluator_policy_id"],
        "evaluator_policy_version": value["evaluator_policy_version"],
        "evaluator_nodes": value["evaluator_nodes"],
        "engine_binary_sha256": _sha256(
            value["engine_binary_sha256"], f"{label}.engine_binary_sha256"
        ),
        "engine_fingerprint": _text(
            value["engine_fingerprint"], f"{label}.engine_fingerprint"
        ),
        "agent_domain": agent_domain,
    }
    frozen = {
        "schema_version": SCHEMA_VERSION,
        "symbolic_feature_version": SYMBOLIC_FEATURE_VERSION,
        "max_plies": MAX_PLIES,
        "observation_policy": OBSERVATION_POLICY,
        "evaluator_policy_id": EVALUATOR_POLICY_ID,
        "evaluator_policy_version": EVALUATOR_POLICY_VERSION,
        "evaluator_nodes": EVALUATOR_NODES,
    }
    for field, expected in frozen.items():
        if parsed[field] != expected or isinstance(parsed[field], bool):
            raise TrainingCorpusSetError(
                f"{label}.{field} must equal the frozen value {expected!r}"
            )
    engine_fingerprint = parsed["engine_fingerprint"]
    assert isinstance(engine_fingerprint, str)
    if "/" in engine_fingerprint or "\\" in engine_fingerprint:
        raise TrainingCorpusSetError(
            f"{label}.engine_fingerprint must not contain a filesystem path"
        )
    return parsed


def _parse_primary(value: object) -> CorpusIdentity:
    mapping = _mapping(value, "primary")
    _exact_keys(mapping, _PRIMARY_KEYS, "primary")
    return CorpusIdentity(
        release_root_sha256=_sha256(
            mapping["release_root_sha256"], "primary.release_root_sha256"
        ),
        corpus_run_id=_sha256(mapping["corpus_run_id"], "primary.corpus_run_id"),
        private_train_manifest_sha256=_sha256(
            mapping["private_train_manifest_sha256"],
            "primary.private_train_manifest_sha256",
        ),
        **_parse_content(mapping, "primary"),  # type: ignore[arg-type]
    )


def _parse_supplement(
    value: object, index: int
) -> SupplementIdentity:
    label = f"supplements[{index}]"
    mapping = _mapping(value, label)
    _exact_keys(mapping, _SUPPLEMENT_KEYS, label)
    content = _parse_content(mapping, label)
    source_revision = _text(
        mapping["source_revision"], f"{label}.source_revision"
    )
    if SOURCE_REVISION_PATTERN.fullmatch(source_revision) is None:
        raise TrainingCorpusSetError(
            f"{label}.source_revision must be a full lowercase Git SHA"
        )
    offset = _positive_int(mapping["profile_offset"], f"{label}.profile_offset")
    profile_id = _text(mapping["profile_id"], f"{label}.profile_id")
    rule_value = mapping["rule_ids"]
    if (
        not isinstance(rule_value, list)
        or len(rule_value) != 2
        or any(not isinstance(item, str) or not item for item in rule_value)
    ):
        raise TrainingCorpusSetError(f"{label}.rule_ids must contain two rule IDs")
    return SupplementIdentity(
        source_revision=source_revision,
        generation_run_id=_sha256(
            mapping["generation_run_id"], f"{label}.generation_run_id"
        ),
        manifest_sha256=_sha256(
            mapping["manifest_sha256"], f"{label}.manifest_sha256"
        ),
        plan_sha256=_sha256(
            mapping["plan_sha256"], f"{label}.plan_sha256"
        ),
        **content,  # type: ignore[arg-type]
        profile_offset=offset,
        profile_id=profile_id,
        rule_ids=(rule_value[0], rule_value[1]),
        profile_sha256=_sha256(
            mapping["profile_sha256"], f"{label}.profile_sha256"
        ),
    )


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _payload(
    primary: CorpusIdentity,
    supplements: Sequence[SupplementIdentity],
) -> dict[str, object]:
    return {
        "format": FORMAT,
        "version": VERSION,
        "primary": _primary_mapping(primary),
        "supplements": [_supplement_mapping(item) for item in supplements],
    }


def _validate_set(
    primary: CorpusIdentity,
    supplements: Sequence[SupplementIdentity],
) -> tuple[SupplementIdentity, ...]:
    if len(supplements) != len(FROZEN_SUPPLEMENT_PROFILES):
        raise TrainingCorpusSetError("exactly six supplements are required")
    by_offset: dict[int, SupplementIdentity] = {}
    for supplement in supplements:
        if supplement.profile_offset in by_offset:
            raise TrainingCorpusSetError("supplement profile offsets must be unique")
        by_offset[supplement.profile_offset] = supplement
    ordered: list[SupplementIdentity] = []
    for offset, profile_id, rule_ids in FROZEN_SUPPLEMENT_PROFILES:
        supplement = by_offset.get(offset)
        if supplement is None:
            raise TrainingCorpusSetError(
                f"missing frozen supplement profile offset {offset}"
            )
        if (
            supplement.profile_id != profile_id
            or supplement.rule_ids != rule_ids
        ):
            raise TrainingCorpusSetError(
                f"supplement offset {offset} does not match its frozen profile"
            )
        ordered.append(supplement)

    unique_fields = (
        "generation_run_id",
        "manifest_sha256",
        "plan_sha256",
        "profile_sha256",
    )
    for field in unique_fields:
        values = [getattr(identity, field) for identity in ordered]
        if len(values) != len(set(values)):
            raise TrainingCorpusSetError(
                f"supplement {field} values must be unique"
            )
    dataset_values = [
        primary.dataset_sha256,
        *(supplement.dataset_sha256 for supplement in ordered),
    ]
    if len(dataset_values) != len(set(dataset_values)):
        raise TrainingCorpusSetError("dataset_sha256 values must be unique")

    compatibility_fields = (
        "schema_version",
        "symbolic_feature_version",
        "max_plies",
        "observation_policy",
        "evaluator_policy_id",
        "evaluator_policy_version",
        "evaluator_nodes",
        "engine_binary_sha256",
        "engine_fingerprint",
        "agent_domain",
    )
    for supplement in ordered:
        for field in compatibility_fields:
            if getattr(supplement, field) != getattr(primary, field):
                raise TrainingCorpusSetError(
                    f"supplement {supplement.profile_id} {field} "
                    "does not match primary"
                )
    return tuple(ordered)


def create_training_corpus_set(
    primary: CorpusIdentity,
    supplements: Sequence[SupplementIdentity],
) -> dict[str, object]:
    """Create the canonical aggregate, sorting supplements by frozen offset."""

    if not isinstance(primary, CorpusIdentity):
        raise TrainingCorpusSetError("primary must be a CorpusIdentity")
    if any(not isinstance(item, SupplementIdentity) for item in supplements):
        raise TrainingCorpusSetError(
            "supplements must contain SupplementIdentity values"
        )
    # Round-trip through the strict parser so manually constructed dataclasses
    # receive the same validation as serialized identities.
    parsed_primary = _parse_primary(_primary_mapping(primary))
    parsed_supplements = tuple(
        _parse_supplement(_supplement_mapping(item), index)
        for index, item in enumerate(supplements)
    )
    ordered = _validate_set(parsed_primary, parsed_supplements)
    payload = _payload(parsed_primary, ordered)
    return {
        **payload,
        "sha256": hashlib.sha256(_canonical_json(payload)).hexdigest(),
    }


def recompute_training_corpus_set_sha256(value: object) -> str:
    """Strictly parse an aggregate and recompute its outer content hash."""

    mapping = _mapping(value, "training corpus set")
    _exact_keys(mapping, _SET_KEYS, "training corpus set")
    if mapping["format"] != FORMAT or mapping["version"] != VERSION:
        raise TrainingCorpusSetError("training corpus set format/version is invalid")
    _sha256(mapping["sha256"], "training corpus set.sha256")
    primary = _parse_primary(mapping["primary"])
    supplements_value = mapping["supplements"]
    if not isinstance(supplements_value, list):
        raise TrainingCorpusSetError("supplements must be an array")
    supplements = tuple(
        _parse_supplement(item, index)
        for index, item in enumerate(supplements_value)
    )
    ordered = _validate_set(primary, supplements)
    if supplements != ordered:
        raise TrainingCorpusSetError(
            "supplements must use canonical profile order 101 through 106"
        )
    return hashlib.sha256(_canonical_json(_payload(primary, ordered))).hexdigest()


def verify_training_corpus_set(value: object) -> dict[str, object]:
    """Verify structure, compatibility, canonical order, and outer digest."""

    mapping = _mapping(value, "training corpus set")
    actual = recompute_training_corpus_set_sha256(mapping)
    if mapping["sha256"] != actual:
        raise TrainingCorpusSetError("training corpus set sha256 does not match")
    # Return a detached canonical mapping rather than caller-owned mutable data.
    primary = _parse_primary(mapping["primary"])
    supplements_value = mapping["supplements"]
    assert isinstance(supplements_value, list)
    supplements = tuple(
        _parse_supplement(item, index)
        for index, item in enumerate(supplements_value)
    )
    return create_training_corpus_set(primary, supplements)
